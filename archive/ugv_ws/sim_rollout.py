"""
sim_rollout.py — headless evaluation of the RF-seeking UGV.

Reproduces the notebook's deterministic evaluation INSIDE the container, using
the real checkpoints, with NO Gazebo/ROS. This validates that GNN_best.pth +
ppo_GNN.pth + the observation pipeline work together before we add Gazebo.

Code below (env, observation helpers, policy, evaluate_policy) is taken verbatim
from the training notebook; only imports, model loading, and main() are new.

Usage (inside the container, in ~/ugv_ws):
    python3 sim_rollout.py --data val.zip --episodes 12
    # --data accepts a .zip of parquet, a folder of parquet, or one .parquet
"""
import os, io, glob, zipfile, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from scipy.ndimage import gaussian_filter
import gymnasium as gym
from gymnasium import spaces
from radio_vig import ViGPatchModel, load_gnn

device = torch.device("cpu")


# ===================== verbatim from notebook =====================


def build_wnet_input(env, global_heatmap):
    """
    Build the 3-channel reconstruction input from ONLY data the UGV has
    collected — the building layout is fully hidden:
      ch0 : zeros   (no building map; the agent has no access to it)
      ch1 : zeros   (emitter location unknown)
      ch2 : sparse path-loss map (cells measured so far, normalised [0,1])
    Buildings still shape the GT field the agent samples (shadowing), but the
    mask itself never enters the pipeline. Returns numpy (3, 256, 256).
    """
    blank = np.zeros((256, 256), dtype=np.float32)
    return np.stack([blank, blank, global_heatmap], axis=0)


@torch.no_grad()
def reconstruct(model, x_tensor, res=256):
    """Run any reconstruction model -> (256,256) numpy path-loss map in [0,1].

    `res` < 256 lowers the model's working resolution: the 3-channel input is
    bilinearly downsampled to res x res, the model runs there, and its output
    is upsampled back to 256 for the shared planning frame. (All models now use
    res=256; goal-cue coarsening is done downstream in _goal_dir instead.)

    The reconstruction is built ONLY from the UGV's collected data (sparse
    samples); the building map is never an input nor a post-hoc mask, so argmin
    is purely the agent's data-driven guess of the emitter. Handles RadioWnet's
    [out1, out2] list."""
    x_in = x_tensor
    if res != x_tensor.shape[-1]:
        x_in = F.interpolate(x_tensor, size=(res, res),
                             mode='bilinear', align_corners=False)
    out = model(x_in)
    if isinstance(out, (list, tuple)):
        out = out[-1]
    if out.shape[-1] != 256:
        out = F.interpolate(out, size=(256, 256),
                            mode='bilinear', align_corners=False)
    return np.clip(out.squeeze(0).squeeze(0).cpu().numpy(), 0.0, 1.0)


class RadioMapDataset:
    def __init__(self, zip_path, limit=None):
        self.data = []
        if not os.path.exists(zip_path):
            print(f"Error: {zip_path} not found.")
            return
        with zipfile.ZipFile(zip_path, 'r') as z:
            files = [f for f in z.namelist() if f.endswith('.parquet')]
            if limit:
                files = files[:limit]
            print(f"Loading {len(files)} samples from {zip_path}...")
            for f in files:
                with z.open(f) as pf:
                    df  = pd.read_parquet(io.BytesIO(pf.read()))
                    row = df.iloc[0]
                    b_mask  = np.array(row['building_mask']).reshape(256, 256)
                    tx_mask = np.array(row['tx_origin']).reshape(256, 256)
                    pl_map  = np.array(row['path_loss']).reshape(256, 256)
                    tx_y, tx_x = np.unravel_index(np.argmax(tx_mask), (256, 256))
                    self.data.append({
                        'building_mask': b_mask.astype(np.float32),
                        'tx_pos': (int(tx_y), int(tx_x)),
                        'path_loss': pl_map.astype(np.float32),
                    })
        print("Done loading.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class RadioMapEnv(gym.Env):
    """
    Continuous (dy, dx) navigation on 256x256 URBAN radio maps.

    Observation (4, 256, 256) — kept for format compatibility; the policy now
    reads a compact state from the reconstruction + a local obstacle crop
    (see make_state), not this raw tensor:
      ch0 - sparse path-loss samples (free cells only, ~obs_frac of the disc)
      ch1 - building mask (real, static per episode)
      ch2 - explored mask
      ch3 - agent position one-hot

    The agent never sees the building layout. It discovers walls only by
    CONTACT: each cell it bumps is recorded into collision_mem, and an
    egocentric crop of that memory is what the policy uses to avoid obstacles
    (see make_state). Buildings still block movement (traced step + 45-degree
    escape) and shadow the GT field, but the mask is never an agent input.
    """

    def __init__(self, dataset, vis_radius=30, max_steps=300, step_size=5, obs_frac=0.15,
                 reach_dist=12.0, progress_weight=0.3, step_penalty=0.01,
                 pull_weight=0.15, pull_radius=15.0,
                 near_zone=25.0, near_penalty=0.05,
                 finish_bonus=50.0, shape_gamma=0.99, collision_penalty=0.15,
                 reversal_penalty=0.03):
        super().__init__()
        self.dataset    = dataset
        self.vis_radius = vis_radius
        self.max_steps  = max_steps
        self.step_size  = step_size
        self.obs_frac   = obs_frac
        self._route_cache: dict = {}

        # ── Reward-shaping constants (tunable ctor args) ──────────────────
        # Free-space tuning is kept (discounted pull, near-zone impatience,
        # large finish bonus); collision_penalty is ADDED for the urban task
        # so driving into / stalling against walls is strictly costly.
        self.reach_dist        = reach_dist
        self.progress_weight   = progress_weight
        self.step_penalty      = step_penalty
        self.pull_weight       = pull_weight
        self.pull_radius       = pull_radius
        self.near_zone         = near_zone
        self.near_penalty      = near_penalty
        self.finish_bonus      = finish_bonus
        self.shape_gamma       = shape_gamma
        self.collision_penalty = collision_penalty
        self.reversal_penalty  = reversal_penalty

        self.action_space      = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(4, 256, 256), dtype=np.float32)

        self.building_mask  = np.zeros((256, 256), dtype=np.float32)
        self.tx_pos         = (128, 128)
        self.path_loss      = np.zeros((256, 256), dtype=np.float32)
        self.norm_path_loss = np.zeros((256, 256), dtype=np.float32)
        self.route_distance = np.zeros((256, 256), dtype=np.float32)
        self.ugv_pos        = (0, 0)
        self.explored_mask  = np.zeros((256, 256), dtype=np.float32)
        self.observed_pl    = np.zeros((256, 256), dtype=np.float32)
        self.collision_mem  = np.zeros((256, 256), dtype=np.float32)
        self.last_action    = np.zeros(2, dtype=np.float32)
        self.current_step   = 0
        self.prev_dist      = 0.0

    # ── Route-distance precomputation (vectorized Dijkstra around buildings) ──
    def _compute_route_distance(self):
        """Shortest navigable route distance from the emitter to every cell,
        routing around buildings, via a vectorized SciPy Dijkstra over the
        8-connected free-space grid. Unreachable cells fall back to Euclidean so
        the progress gradient stays finite. With buildings present the route
        distance (not the straight line) is what the progress reward rewards.

        Replaces a pure-Python heap version that was both ~1000x slower and
        inaccurate: storing distances in float32 while keying the heap with
        float64 made the lazy-deletion test reject valid relaxations, inflating
        distances by >10 px even on an empty grid."""
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import dijkstra
        ty, tx = self.tx_pos
        yy, xx = np.ogrid[:256, :256]
        euc    = np.hypot(yy - ty, xx - tx).astype(np.float32)
        free   = (self.building_mask == 0)
        if not free[ty, tx]:
            return euc
        node_id = np.full((256, 256), -1, dtype=np.int64)
        fy, fx  = np.where(free)
        node_id[fy, fx] = np.arange(fy.size)
        rows, cols, wts = [], [], []
        for dy, dx, wt in [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                           (-1, -1, 1.414), (-1, 1, 1.414),
                           (1, -1, 1.414), (1, 1, 1.414)]:
            ny, nx = fy + dy, fx + dx
            ok = (ny >= 0) & (ny < 256) & (nx >= 0) & (nx < 256)
            ok &= free[np.clip(ny, 0, 255), np.clip(nx, 0, 255)]
            rows.append(node_id[fy[ok], fx[ok]])
            cols.append(node_id[ny[ok], nx[ok]])
            wts.append(np.full(int(ok.sum()), wt, dtype=np.float32))
        graph = csr_matrix((np.concatenate(wts),
                            (np.concatenate(rows), np.concatenate(cols))),
                           shape=(fy.size, fy.size))
        d   = dijkstra(graph, directed=False, indices=int(node_id[ty, tx]))
        out = euc.copy()
        out[fy, fx] = np.where(np.isinf(d), euc[fy, fx], d).astype(np.float32)
        return out

    # ── Episode reset ──────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if not self.dataset.data:
            return np.zeros((4, 256, 256), dtype=np.float32), {}

        idx                 = np.random.randint(len(self.dataset))
        sample              = self.dataset[idx]
        self.building_mask  = sample['building_mask'].astype(np.float32)
        self.tx_pos         = sample['tx_pos']
        self.path_loss      = sample['path_loss'].astype(np.float32)
        self.norm_path_loss = np.clip((self.path_loss - 50.0) / 100.0, 0.0, 1.0)
        # Buildings read as path-loss 'infinity' on the planning map.
        self.norm_path_loss[self.building_mask > 0] = 1.0

        if idx not in self._route_cache:
            self._route_cache[idx] = self._compute_route_distance()
        self.route_distance = self._route_cache[idx]

        free = np.argwhere(self.building_mask == 0)     # spawn only on free cells
        max_start_dist = (options or {}).get('max_start_dist', None)
        if max_start_dist is not None:
            d = np.hypot(free[:, 0] - self.tx_pos[0], free[:, 1] - self.tx_pos[1])
            cand = free[d <= max_start_dist]
            if len(cand) >= 5:
                free = cand

        self.ugv_pos       = tuple(int(v) for v in free[np.random.randint(len(free))])
        self.explored_mask = np.zeros((256, 256), dtype=np.float32)
        self.observed_pl   = np.zeros((256, 256), dtype=np.float32)
        self.collision_mem = np.zeros((256, 256), dtype=np.float32)
        self.last_action   = np.zeros(2, dtype=np.float32)
        self._goal_ema     = None
        self.current_step  = 0
        self.prev_dist     = float(self.route_distance[self.ugv_pos])

        self._update_visibility()
        return self._get_obs(), {}

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _dist_to_tx(self, pos):
        return float(np.hypot(pos[0] - self.tx_pos[0], pos[1] - self.tx_pos[1]))

    def _update_visibility(self):
        y, x   = self.ugv_pos
        yg, xg = np.ogrid[:256, :256]
        vis    = ((yg - y) ** 2 + (xg - x) ** 2) <= self.vis_radius ** 2
        self.explored_mask = np.maximum(self.explored_mask, vis.astype(np.float32))
        # Only free cells are measurable — no samples from inside buildings.
        candidate_idx = np.argwhere(vis & (self.building_mask == 0))
        if len(candidate_idx) > 0:
            n_sample = max(1, int(round(len(candidate_idx) * self.obs_frac)))
            chosen   = candidate_idx[
                np.random.choice(len(candidate_idx), n_sample, replace=False)
            ]
            cy, cx = chosen[:, 0], chosen[:, 1]
            self.observed_pl[cy, cx] = np.maximum(
                self.observed_pl[cy, cx], self.norm_path_loss[cy, cx]
            )

    def _get_obs(self):
        pos               = np.zeros((256, 256), dtype=np.float32)
        pos[self.ugv_pos] = 1.0
        return np.stack([self.observed_pl, self.collision_mem,
                         self.explored_mask, pos], axis=0)

    # ── Movement (wall-aware) ──────────────────────────────────────────────
    def _trace_step(self, dy_f, dx_f):
        """Walk pixel-by-pixel toward (pos+dy, pos+dx), stopping at the last
        free cell before any building. If the intended direction is fully
        blocked, rotate the action in 45-degree steps (+/-45, +/-90, +/-135,
        180) until a traversable direction is found, so the agent never freezes
        when the signal gradient points straight through a wall."""
        y0, x0 = self.ugv_pos

        def _try(dy, dx):
            y1 = int(np.clip(round(y0 + dy), 0, 255))
            x1 = int(np.clip(round(x0 + dx), 0, 255))
            n  = max(abs(y1 - y0), abs(x1 - x0))
            if n == 0:
                return y0, x0
            by, bx = y0, x0
            for i in range(1, n + 1):
                cy = int(round(y0 + (y1 - y0) * i / n))
                cx = int(round(x0 + (x1 - x0) * i / n))
                if self.building_mask[cy, cx]:
                    break
                by, bx = cy, cx
            return by, bx

        ny, nx = _try(dy_f, dx_f)
        if ny != y0 or nx != x0:
            return ny, nx

        mag = np.hypot(dy_f, dx_f)
        if mag < 0.5:
            return y0, x0
        base_angle = np.arctan2(dy_f, dx_f)
        for delta_deg in [45, -45, 90, -90, 135, -135, 180]:
            a = base_angle + np.radians(delta_deg)
            ny, nx = _try(mag * np.sin(a), mag * np.cos(a))
            if ny != y0 or nx != x0:
                return ny, nx
        return y0, x0   # completely surrounded (rare)

    # ── Contact-based obstacle discovery ───────────────────────────────────
    def _first_wall(self, start, dy_f, dx_f):
        """Ray-cast the INTENDED heading from `start`; return the first building
        cell within reach, or None. This is the wall the UGV physically contacts
        — the only obstacle information it ever learns (recorded into
        collision_mem). The mask is used here purely to resolve physics; the
        agent receives only the contacted cell, never the layout."""
        y0, x0 = start
        y1 = int(np.clip(round(y0 + dy_f), 0, 255))
        x1 = int(np.clip(round(x0 + dx_f), 0, 255))
        n  = max(abs(y1 - y0), abs(x1 - x0))
        for i in range(1, n + 1):
            cy = int(round(y0 + (y1 - y0) * i / n))
            cx = int(round(x0 + (x1 - x0) * i / n))
            if self.building_mask[cy, cx]:
                return cy, cx
        return None

    # ── Environment step ───────────────────────────────────────────────────
    def step(self, action):
        self.current_step += 1
        # Clip at the env boundary: the policy samples an unbounded Gaussian and
        # its stored log-probs describe the raw sample, so the bound is enforced
        # here rather than in the agent.
        a0 = float(np.clip(action[0], -1.0, 1.0))
        a1 = float(np.clip(action[1], -1.0, 1.0))
        dy = a0 * self.step_size
        dx = a1 * self.step_size
        cur_action = np.array([a0, a1], dtype=np.float32)

        old_pos      = self.ugv_pos
        new_y, new_x = self._trace_step(dy, dx)
        blocked      = (new_y == old_pos[0] and new_x == old_pos[1])

        # Contact discovery: if the intended heading runs into a building the
        # UGV bumps it and remembers that cell (small footprint). This is the
        # ONLY way the agent learns walls — it never sees the map.
        wall = self._first_wall(old_pos, dy, dx)
        if wall is not None:
            wy, wx = wall
            yg, xg = np.ogrid[:256, :256]
            self.collision_mem[((yg - wy) ** 2 + (xg - wx) ** 2) <= 4] = 1.0

        self.ugv_pos = (new_y, new_x)
        self._update_visibility()

        route_dist = float(self.route_distance[self.ugv_pos])
        euc_new    = self._dist_to_tx(self.ugv_pos)

        # Long-range: route-distance progress shaping routes around buildings.
        reward = (self.prev_dist - route_dist) * self.progress_weight - self.step_penalty

        # Collision: driving into / stalling against a wall is strictly costly.
        if blocked:
            reward -= self.collision_penalty

        # Near-zone / pull-in are now ROUTE-DISTANCE, not Euclidean: a spot that
        # is straight-line-close but walled off is route-far, so the pull no
        # longer drags the agent into the wall and detours are not punished.
        if route_dist < self.near_zone:
            reward -= self.near_penalty

        # Short-range: DISCOUNTED route-distance potential pull-in within pull_radius.
        phi_new  = self.pull_weight * max(0.0, self.pull_radius - route_dist)
        phi_prev = self.pull_weight * max(0.0, self.pull_radius - self.prev_dist)
        reward  += self.shape_gamma * phi_new - phi_prev

        # Reversal penalty: cost commanded heading flips so the policy is pushed
        # to commit to a consistent route around an obstacle rather than
        # oscillating at the wall face. Skipped when either action is ~zero
        # (undefined heading). cos in [-1,1] -> penalty in [0, 2*reversal_penalty].
        n_cur  = float(np.linalg.norm(cur_action))
        n_prev = float(np.linalg.norm(self.last_action))
        if n_cur > 1e-6 and n_prev > 1e-6:
            cos = float(np.dot(cur_action, self.last_action) / (n_cur * n_prev))
            reward -= self.reversal_penalty * (1.0 - cos)
        self.last_action = cur_action

        # Terminate on ROUTE-DISTANCE proximity so being straight-line-close but
        # separated by a building does NOT count as reaching the emitter.
        terminated = route_dist < self.reach_dist
        truncated  = self.current_step >= self.max_steps
        if terminated:
            reward += self.finish_bonus
        self.prev_dist = route_dist

        info = {
            'dist_to_tx' : euc_new,
            'route_to_tx': route_dist,
            'hit_wall'   : blocked,
            'coverage'   : float(self.explored_mask.mean()),
            'signal'     : float(self.norm_path_loss[self.ugv_pos]),
        }
        return self._get_obs(), float(reward), terminated, truncated, info


# Goal-target stabilization. The raw per-step argmin(reconstruction) swings hard
# while the UGV detours around walls (it samples shadow zones whose path-loss is
# non-monotone), flipping the heading so the agent circles. We blur the
# reconstruction before argmin (kills lone spurious-minimum pixels) and EMA the
# target LOCATION across steps (filters the high-frequency swings, still tracks
# genuine drift). Tunable: higher sigma / lower alpha = steadier but laggier.
GOAL_BLUR_SIGMA = 6.0
GOAL_EMA_ALPHA  = 0.2
# Goal-cue coarsening (per backbone): before argmin, the 256x256 reconstruction is
# bilinearly downsampled to GOAL_COARSE_RES then upsampled back, washing out
# high-fidelity spurious minima so the emitter estimate is as coarse as the GNN's
# 32x32 patch grid (which navigates best). The GNN is ALREADY that coarse, so it
# stays at native res (256 = disabled) -- coarsening it again is redundant. An
# unknown model (or value >= 256) disables coarsening.
GOAL_COARSE_RES = {'WNet': 32, 'CNN': 32, 'PartialConvMAE': 32, 'GNN': 256}

# Explore-then-commit: for the first EXPLORE_STEPS of every EXPLORE_PERIOD steps
# the UGV runs a short EXPLORATION phase -- it acts with inflated policy noise
# (EXPLORE_STD_SCALE) so it spreads out and gathers path-loss samples over a
# region (the env samples on every move), shrinking the emitter-estimate
# uncertainty before it COMMITS to that estimate for the remaining steps. This
# replaces the in-place dwell scan; exploration moves rather than stops.
# Applied identically in training and eval rollouts.
EXPLORE_STEPS     = 5
EXPLORE_PERIOD    = 20
EXPLORE_STD_SCALE = 1.5

# ── Compact navigation state (map-free) ─────────────────────────────────────
#   Goal cue : straight-line direction + distance from the UGV to
#     argmin(reconstruction) (the estimated emitter). The reconstruction is built
#     ONLY from the agent's collected path-loss, so this cue carries no
#     privileged map information.
#   Obstacle cue : an egocentric crop of the UGV's COLLISION MEMORY — cells it
#     has bumped into this episode (contact feedback, NOT the building map). The
#     agent discovers walls by interacting and learns to route around the ones it
#     has found; unknown cells read as free (0), so it must explore to learn.

DIAG_PX   = float(np.hypot(255.0, 255.0))
LOCAL_WIN = 64    # px window (centered on the agent) cropped from the collision map
LOCAL_RES = 32    # window is downsampled to LOCAL_RES x LOCAL_RES for the CNN
GOAL_DIM  = 3     # goal cue: unit direction (2) + dist (1)
ACT_DIM   = 2     # previous action (heading) fed back as memory -> commitment
VEC_DIM   = GOAL_DIM + ACT_DIM
STATE_DIM = VEC_DIM + LOCAL_RES * LOCAL_RES


def _goal_dir(reconstruction, env):
    """Goal cue: unit direction + normalized distance from the UGV to the
    STABILIZED argmin(reconstruction) -- the data-driven emitter estimate (built
    only from the agent's collected path-loss, so no map info). Blurs before
    argmin (kills lone spurious-minimum pixels) and EMA-smooths the target
    LOCATION across steps (GOAL_BLUR_SIGMA / GOAL_EMA_ALPHA) so the heading does
    not flip as the agent samples non-monotone shadow zones. EMA state lives on
    env, reset per episode."""
    cr = GOAL_COARSE_RES.get(getattr(env, '_model_name', None), 256)
    rc = reconstruction
    if cr < rc.shape[-1]:
        t  = torch.from_numpy(np.ascontiguousarray(rc))[None, None]
        t  = F.interpolate(t, size=(cr, cr), mode='bilinear', align_corners=False)
        t  = F.interpolate(t, size=rc.shape, mode='bilinear', align_corners=False)
        rc = t.squeeze(0).squeeze(0).numpy()
    rb     = gaussian_filter(rc, GOAL_BLUR_SIGMA)
    gy, gx = np.unravel_index(int(np.argmin(rb)), rb.shape)
    env._goal_raw = (float(gy), float(gx))   # pre-EMA target, for stability logging

    # EMA seeded AFTER settle. During the initial commit/settle phase
    # (current_step < EXPLORE_PERIOD) the reconstruction is data-starved and its
    # argmin is fragile, so re-seed the target to the current argmin every step
    # rather than latching onto the step-0 frame. Once settled, switch to EMA
    # smoothing -- the seed is then the data-rich settled estimate. This stops a
    # hallucinated early minimum from being latched and steering the agent.
    settling = getattr(env, 'current_step', 0) < EXPLORE_PERIOD
    if getattr(env, '_goal_ema', None) is None or settling:
        env._goal_ema = (float(gy), float(gx))
    else:
        ey, ex = env._goal_ema
        a = GOAL_EMA_ALPHA
        env._goal_ema = ((1.0 - a) * ey + a * gy, (1.0 - a) * ex + a * gx)

    ty, tx = env._goal_ema
    dy   = ty - env.ugv_pos[0]; dx = tx - env.ugv_pos[1]
    dist = float(np.hypot(dy, dx)); inv = 1.0 / (dist + 1e-6)
    return np.array([dy * inv, dx * inv, dist / DIAG_PX], dtype=np.float32)


def _obstacle_crop(occupancy, ugv_pos, win=LOCAL_WIN, res=LOCAL_RES):
    """Egocentric crop of the UGV's collision-memory occupancy map, downsampled
    to res x res by MAX-pool (any discovered wall in a block marks the coarse
    cell). Out-of-map area pads as 0 (unknown/unexplored — NOT assumed a wall),
    consistent with the agent only knowing what it has touched."""
    half = win // 2
    padded = np.zeros((256 + win, 256 + win), dtype=np.float32)
    padded[half:half + 256, half:half + 256] = occupancy
    y, x = ugv_pos
    crop = padded[y:y + win, x:x + win]
    f = win // res
    return crop.reshape(res, f, res, f).max(axis=(1, 3)).astype(np.float32)


def make_state(reconstruction, env):
    """Flat state: [dir_y, dir_x, dist_norm, prev_ay, prev_ax,
                    <res*res collision-memory crop>].
    Cues use ONLY data the UGV has collected: the reconstruction (from its sparse
    samples) gives the goal direction; the collision-memory crop gives locally
    discovered walls. The PREVIOUS action (heading) is fed back so the policy has
    temporal state -- it can commit to rounding an obstacle one way instead of
    oscillating at the wall face where attraction (goal) and repulsion (crop)
    cancel. `env.last_action` is the last clipped action, zeros at episode start.
    Flat so the PPO buffer is unchanged; the network splits it back into the
    direction+heading vector and the obstacle crop."""
    return np.concatenate([
        _goal_dir(reconstruction, env),
        np.asarray(env.last_action, dtype=np.float32),
        _obstacle_crop(env.collision_mem, env.ugv_pos).ravel(),
    ]).astype(np.float32)


# ── Actor-Critic: direction MLP + obstacle-crop CNN, fused ─────────────────
class ActorCritic(nn.Module):
    def __init__(self, obs_dim=STATE_DIM, action_dim=2, crop_res=LOCAL_RES):
        super().__init__()
        self.crop_res = crop_res
        # Local obstacle encoder: 1x32x32 -> 32x4x4 = 512.
        self.local_enc = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(inplace=True),   # 32->16
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),  # 16->8
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),  # 8->4
            nn.Flatten(),
        )
        self.dir_enc = nn.Sequential(nn.Linear(VEC_DIM, 64), nn.Tanh())
        self.trunk   = nn.Sequential(
            nn.Linear(512 + 64, 128), nn.Tanh(),
            nn.Linear(128, 128),      nn.Tanh(),
        )
        self.actor_mean    = nn.Linear(128, action_dim)   # raw, pre-squash mean
        self.actor_log_std = nn.Parameter(torch.full((action_dim,), -0.5))
        self.critic        = nn.Linear(128, 1)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.zeros_(self.actor_mean.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def _split(self, x):
        d = x[:, :VEC_DIM]
        c = x[:, VEC_DIM:].reshape(-1, 1, self.crop_res, self.crop_res)
        return d, c

    def forward(self, x):
        d, c = self._split(x)
        h = self.trunk(torch.cat([self.local_enc(c), self.dir_enc(d)], dim=1))
        return self.actor_mean(h), self.critic(h)

    def get_dist_and_value(self, x):
        mean, val = self.forward(x)
        std = self.actor_log_std.exp().clamp(1e-3, 2.0)
        return Normal(mean, std), val


def _tanh_log_prob(dist, stored_action):
    """Log-prob of the tanh-squashed action a = tanh(stored_action), with the
    change-of-variables (log-det-Jacobian) correction so the ratio is measured
    in the bounded action space the env actually executes -- not the latent
    u-space. Without this term the policy can satisfy the surrogate/entropy by
    pushing samples into tanh saturation at no behavioral cost; the correction
    sends log-prob -> -inf at the +/-1 edges, penalizing that exploit."""
    base = dist.log_prob(stored_action).sum(-1)
    corr = (2.0 * (np.log(2.0) - stored_action - F.softplus(-2.0 * stored_action))).sum(-1)
    return base - corr


class RunningMeanStd:
    """Welford running mean/variance, used to standardize value-function
    regression targets (returns) so the value-loss scale stays ~unit even as the
    curriculum inflates raw return magnitude."""
    def __init__(self, eps=1e-4):
        self.mean  = 0.0
        self.var   = 1.0
        self.count = eps

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        b_mean, b_var, b_count = x.mean(), x.var(), x.size
        delta = b_mean - self.mean
        tot   = self.count + b_count
        self.mean += delta * b_count / tot
        m_a = self.var * self.count
        m_b = b_var * b_count
        self.var   = (m_a + m_b + delta ** 2 * self.count * b_count / tot) / tot
        self.count = tot

    @property
    def std(self):
        return float(np.sqrt(self.var) + 1e-8)


# ── PPO with GAE (tanh-squashed Gaussian; ratio in bounded action space) ────
class PPOAgent:
    def __init__(self, obs_dim=STATE_DIM, action_dim=2, lr=3e-4, gamma=0.99, lam=0.95,
                 eps_clip=0.2, k_epochs=10, minibatch=256, ent_coef=0.01):
        self.policy     = ActorCritic(obs_dim, action_dim).to(device)
        self.policy_old = ActorCritic(obs_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optim      = optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)

        self.gamma, self.lam = gamma, lam
        self.eps_clip        = eps_clip
        self.k_epochs        = k_epochs
        self.minibatch       = minibatch
        self.ent_coef        = ent_coef
        self.ret_rms         = RunningMeanStd()   # value/return normalizer

    @torch.no_grad()
    def act(self, state, explore=False):
        t = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, val = self.policy_old.get_dist_and_value(t)
        if explore:
            # Exploration phase: widen the policy noise so the UGV samples a
            # broader region. The stored log-prob is taken under this inflated
            # distribution, so it remains the true behavior log-prob and the PPO
            # importance ratio stays valid.
            std  = (dist.stddev * EXPLORE_STD_SCALE).clamp(1e-3, 2.0)
            dist = Normal(dist.mean, std)
        stored_action = dist.sample()
        a  = torch.tanh(stored_action)
        lp = _tanh_log_prob(dist, stored_action)
        # critic predicts normalized returns -> de-normalize to raw reward scale
        # so GAE (and the truncation bootstrap) stay in raw space.
        val_raw = val.item() * self.ret_rms.std + self.ret_rms.mean
        return stored_action.cpu().numpy()[0], a.cpu().numpy()[0], lp.item(), val_raw

    def _gae(self, rewards, dones, values, last_val=0.0):
        n   = len(rewards)
        adv = np.zeros(n, dtype=np.float32)
        gae, nxt = 0.0, last_val
        for i in reversed(range(n)):
            d      = float(dones[i])
            delta  = rewards[i] + self.gamma * nxt * (1.0 - d) - values[i]
            gae    = delta + self.gamma * self.lam * (1.0 - d) * gae
            adv[i] = gae
            nxt    = values[i]
        return adv, adv + np.array(values, dtype=np.float32)

    def update(self, buf, last_val=0.0):
        adv, ret = self._gae(buf['rewards'], buf['dones'], buf['values'], last_val)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Value/return normalization: standardize the regression targets with a
        # running mean/std so the value loss stays ~unit-scale regardless of the
        # raw return magnitude (which grows as the curriculum lengthens routes).
        # GAE above used raw values (de-normalized in `act`), so only the target
        # is normalized here; the critic learns to predict standardized returns.
        self.ret_rms.update(ret)
        ret_norm = (ret - self.ret_rms.mean) / self.ret_rms.std

        ts   = torch.FloatTensor(np.stack(buf['obs'])).to(device)
        ta   = torch.FloatTensor(np.stack(buf['actions'])).to(device)
        tlp  = torch.FloatTensor(np.array(buf['log_probs'], np.float32)).to(device)
        tadv = torch.FloatTensor(adv).to(device)
        tret = torch.FloatTensor(ret_norm).to(device)

        n    = len(buf['rewards'])
        logs = {'pi': [], 'vf': [], 'ent': []}

        for _ in range(self.k_epochs):
            perm = torch.randperm(n)
            for s in range(0, n, self.minibatch):
                idx       = perm[s: s + self.minibatch]
                dist, val = self.policy.get_dist_and_value(ts[idx])
                nlp       = _tanh_log_prob(dist, ta[idx])
                ent       = dist.entropy().sum(-1)
                ratio     = (nlp - tlp[idx]).exp()
                pg        = -torch.min(
                    ratio * tadv[idx],
                    ratio.clamp(1 - self.eps_clip, 1 + self.eps_clip) * tadv[idx]
                ).mean()
                vl        = 0.5 * (val.squeeze(-1) - tret[idx]).pow(2).mean()
                el        = -self.ent_coef * ent.mean()
                loss      = pg + vl + el
                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optim.step()
                logs['pi'].append(pg.item())
                logs['vf'].append(vl.item())
                logs['ent'].append(ent.mean().item())

        self.policy_old.load_state_dict(self.policy.state_dict())
        return {k: float(np.mean(v)) for k, v in logs.items()}

def evaluate_policy(env, agent, reconstruction_model, res=256, n_episodes=12, seed0=None, name=None):
    """Run n_episodes deterministic rollouts. If seed0 is set, scenario i uses
    seed0+i so every model is evaluated on the SAME start/emitter/building
    layouts. `res` is the model's reconstruction working resolution."""
    env._model_name = name   # per-model goal-cue coarsening (GOAL_COARSE_RES)
    results = []
    for i in range(n_episodes):
        if seed0 is not None:
            np.random.seed(seed0 + i); torch.manual_seed(seed0 + i)
        env.reset()
        path = [env.ugv_pos]
        # Goal-target stability log: mean step-to-step jump (px) of the raw
        # argmin vs the EMA-smoothed target. ema_jump << raw_jump => the EMA is
        # damping the detour-induced swing that was making the agent circle.
        raw_jumps, ema_jumps = [], []
        prev_raw = prev_ema = None
        for step in range(env.max_steps):
            exploring      = step >= EXPLORE_PERIOD and (step % EXPLORE_PERIOD) < EXPLORE_STEPS   # match training (first window suppressed)
            wnet_in        = build_wnet_input(env, env.observed_pl)
            reconstruction = reconstruct(reconstruction_model,
                                         torch.FloatTensor(wnet_in).unsqueeze(0).to(device),
                                         res=res)
            state          = make_state(reconstruction, env)
            if prev_raw is not None:
                raw_jumps.append(float(np.hypot(env._goal_raw[0] - prev_raw[0],
                                                env._goal_raw[1] - prev_raw[1])))
                ema_jumps.append(float(np.hypot(env._goal_ema[0] - prev_ema[0],
                                                env._goal_ema[1] - prev_ema[1])))
            prev_raw, prev_ema = env._goal_raw, env._goal_ema
            t = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                mean, _ = agent.policy(t)
                if exploring:
                    std    = (agent.policy.actor_log_std.exp().clamp(1e-3, 2.0)
                              * EXPLORE_STD_SCALE)
                    action = torch.tanh(Normal(mean, std).sample()).cpu().numpy()[0]
                else:
                    action = torch.tanh(mean).cpu().numpy()[0]
            _, _, term, trunc, info = env.step(action)
            path.append(env.ugv_pos)
            if term or trunc:
                break
        pa = np.array(path)
        results.append(dict(success=info['route_to_tx'] < 12.0,
                            final_dist=info['route_to_tx'], n_steps=len(pa),
                            coverage=info['coverage'], path=pa,
                            raw_jump=float(np.mean(raw_jumps)) if raw_jumps else 0.0,
                            ema_jump=float(np.mean(ema_jumps)) if ema_jumps else 0.0,
                            tx_pos=env.tx_pos, path_loss=env.path_loss.copy(),
                            building_mask=env.building_mask.copy()))
    return results


# ── Figure 1: summary metric bars ──────────────────────────────────────────



# ===================== new: loading + main =====================
def load_dataset(path, limit=None):
    if path.endswith(".zip"):
        return RadioMapDataset(path, limit=limit)
    files = [path] if path.endswith(".parquet") else sorted(glob.glob(os.path.join(path, "*.parquet")))
    if limit:
        files = files[:limit]
    data = []
    for f in files:
        row = pd.read_parquet(f).iloc[0]
        b  = np.array(row["building_mask"]).reshape(256, 256)
        tx = np.array(row["tx_origin"]).reshape(256, 256)
        pl = np.array(row["path_loss"]).reshape(256, 256)
        ty, txx = np.unravel_index(int(np.argmax(tx)), (256, 256))
        data.append({"building_mask": b, "tx_pos": (ty, txx), "path_loss": pl})
    print(f"Loaded {len(data)} parquet sample(s) from {path}")
    class _DS:
        def __init__(s, d): s.data = d
        def __len__(s): return len(s.data)
        def __getitem__(s, i): return s.data[i]
    return _DS(data)


def plot_paths(results, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(results); cols = min(4, n); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    for k, r in enumerate(results):
        ax = axes[k // cols][k % cols]
        ax.imshow(r["path_loss"], cmap="viridis", origin="upper")
        p = np.asarray(r["path"])
        ax.plot(p[:, 1], p[:, 0], "-", color="#00d4ff", lw=1.5)
        ax.plot(p[0, 1], p[0, 0], "o", color="#44ff88", ms=7)
        ax.plot(p[-1, 1], p[-1, 0], "*", color="#ff4444", ms=13)
        ty, tx = r["tx_pos"]; ax.plot(tx, ty, "P", color="#FFD700", ms=11)
        ax.set_title(("reached" if r["success"] else "missed") + f"  d={r['final_dist']:.0f}px")
        ax.set_xticks([]); ax.set_yticks([])
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis("off")
    fig.tight_layout(); fig.savefig(out, dpi=130); print("Saved", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help=".zip / folder / .parquet of Sionna samples")
    ap.add_argument("--gnn", default="GNN_best.pth")
    ap.add_argument("--ppo", default="ppo_GNN.pth")
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="rollout_paths.png")
    a = ap.parse_args()

    ds = load_dataset(a.data, limit=a.episodes)
    env = RadioMapEnv(ds)

    recon = load_gnn(a.gnn, device)
    agent = PPOAgent(obs_dim=STATE_DIM, action_dim=2)
    agent.policy.load_state_dict(torch.load(a.ppo, map_location=device))
    agent.policy.eval()
    print("Models loaded. Running", a.episodes, "rollouts ...")

    results = evaluate_policy(env, agent, recon, res=256,
                              n_episodes=a.episodes, seed0=a.seed, name="GNN")
    succ = 100.0 * np.mean([r["success"] for r in results])
    dist = np.mean([r["final_dist"] for r in results])
    print(f"\nSuccess rate : {succ:.1f}%  ({sum(r['success'] for r in results)}/{len(results)})")
    print(f"Mean final distance to emitter: {dist:.1f} px ({dist*1.7:.1f} m)")
    plot_paths(results, a.out)


if __name__ == "__main__":
    main()