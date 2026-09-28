"""
seeker_brain.py — the GNN->PPO planner, driven by an EXTERNAL pose.

Reuses the validated pipeline from sim_rollout.py (RadioMapEnv as a state
container, reconstruct(), make_state(), the ActorCritic policy). Instead of the
env's idealized _trace_step movement, the rover's REAL pixel is fed in each tick
and the policy's chosen displacement is returned as the next waypoint pixel.

This keeps the policy exactly in-distribution (it sees the same pixel-space
state it trained on) while a real Gazebo rover executes the motion.
"""
import numpy as np
import torch

import sim_rollout as S


class SeekerBrain:
    def __init__(self, data_path, gnn_path="GNN_best.pth", ppo_path="ppo_GNN.pth"):
        self.ds = S.load_dataset(data_path, limit=1)
        self.env = S.RadioMapEnv(self.ds)
        self.env._model_name = "GNN"                       # goal-cue coarsening
        self.recon = S.load_gnn(gnn_path, S.device)
        self.agent = S.PPOAgent(obs_dim=S.STATE_DIM, action_dim=2)
        self.agent.policy.load_state_dict(torch.load(ppo_path, map_location=S.device))
        self.agent.policy.eval()
        self.step_size = self.env.step_size
        self.reach = getattr(self.env, "reach_dist", 12.0)
        self.last_recon = None   # latest GNN reconstruction (256x256, [0,1])

    def reset(self, start_pixel):
        """Initialise episode state at a specific start pixel (scene 0)."""
        self.env.reset()                                   # inits scene + buffers
        e = self.env
        e.observed_pl = np.zeros((256, 256), np.float32)
        e.explored_mask = np.zeros((256, 256), np.float32)
        if hasattr(e, "collision_mem"):
            e.collision_mem = np.zeros((256, 256), np.float32)
        e.last_action = np.zeros(2, np.float32)
        for attr in ("_goal_ema", "_goal_raw"):
            if hasattr(e, attr):
                setattr(e, attr, None)
        e.ugv_pos = (int(np.clip(start_pixel[0], 0, 255)),
                     int(np.clip(start_pixel[1], 0, 255)))
        e._update_visibility()
        return e.ugv_pos

    @property
    def tx_pixel(self):
        return self.env.tx_pos

    def _record_contacts(self, pix):
        """Mark nearby building cells as 'discovered by contact' (the env does
        this inside _trace_step; we approximate it from the real pose)."""
        if not hasattr(self.env, "collision_mem"):
            return
        y, x = pix
        y0, y1 = max(0, y - 2), min(256, y + 3)
        x0, x1 = max(0, x - 2), min(256, x + 3)
        b = self.env.building_mask[y0:y1, x0:x1]
        self.env.collision_mem[y0:y1, x0:x1] = np.maximum(
            self.env.collision_mem[y0:y1, x0:x1], b)

    @torch.no_grad()
    def plan(self, current_pixel):
        """Given the rover's current pixel, update sensing, run GNN->PPO, and
        return (target_pixel, done, info)."""
        e = self.env
        e.ugv_pos = (int(np.clip(current_pixel[0], 0, 255)),
                     int(np.clip(current_pixel[1], 0, 255)))
        e._update_visibility()
        self._record_contacts(e.ugv_pos)

        wnet_in = S.build_wnet_input(e, e.observed_pl)
        recon = S.reconstruct(self.recon,
                              torch.FloatTensor(wnet_in).unsqueeze(0).to(S.device),
                              res=256)
        self.last_recon = recon
        state = S.make_state(recon, e)
        t = torch.FloatTensor(state).unsqueeze(0).to(S.device)
        mean, _ = self.agent.policy(t)
        action = torch.tanh(mean).cpu().numpy()[0]
        e.last_action = action.astype(np.float32)

        # Where the policy wants to go, respecting walls (reuse the env's tracer).
        dy = float(action[0]) * self.step_size
        dx = float(action[1]) * self.step_size
        target = e._trace_step(dy, dx)

        dist = float(np.hypot(e.ugv_pos[0] - e.tx_pos[0], e.ugv_pos[1] - e.tx_pos[1]))
        done = dist < self.reach
        return target, done, {"dist_px": dist, "action": action,
                              "goal_est": getattr(e, "_goal_ema", None)}


if __name__ == "__main__":
    # Kinematic self-test: teleport the rover to each planned target (which
    # equals the env's own _trace_step), so this MUST reproduce sim_rollout.
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--episodes", type=int, default=3)
    a = ap.parse_args()
    brain = SeekerBrain(a.data)
    free = np.argwhere(brain.env.building_mask == 0) if False else None
    for ep in range(a.episodes):
        np.random.seed(ep); torch.manual_seed(ep)
        brain.env.reset()
        tx = brain.tx_pixel
        # start ~120 px from emitter
        f = np.argwhere(brain.env.building_mask == 0)
        d = np.hypot(f[:, 0] - tx[0], f[:, 1] - tx[1])
        start = tuple(f[np.abs(d - 120) < 10][0])
        pix = brain.reset(start)
        for step in range(brain.env.max_steps):
            target, done, info = brain.plan(pix)
            pix = target
            if done:
                break
        print(f"ep{ep}: start={start} -> end dist {info['dist_px']:.0f}px "
              f"{'REACHED' if done else 'missed'} in {step+1} steps")