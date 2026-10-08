#!/usr/bin/env python3
"""
ablation_inference_benchmark.py
===============================
Sequentially benchmark the four radio-map reconstruction models on the NVIDIA
Jetson Orin Nano against UNSEEN test data, measuring both *quality* and
*efficiency*, and produce per-model and combined comparison visualizations.

Models (the four you selected for deployment):
    CNN, WNet, PartialConvMAE, GNN

For each model the script:
  1. Loads the optimized TensorRT engine (built by onnx_to_engine.py).
     -> If TensorRT/pycuda is unavailable OR no engine exists, it transparently
        falls back to running the PyTorch checkpoint (so you can still get
        accuracy numbers anywhere).
  2. Runs inference over the unseen dataset.
  3. Records, DURING testing:
        - per-sample wall-clock latency (ms)
        - GPU-compute latency where the TRT engine reports it
        - live system telemetry via `tegrastats` (GPU%, RAM, power mW) sampled
          in a background thread for the duration of that model's run
  4. Records, AFTER testing, aggregate QUALITY metrics on unseen data:
        - MSE, RMSE, MAE   (on normalised [0,1] path-loss)
        - MSE / RMSE / MAE in dB (de-normalised, physically meaningful)
        - SSIM
        - "accuracy": fraction of pixels within a tolerance (default 1 dB),
          plus PSNR for completeness
        - throughput (FPS), mean/median/p95 latency
        - energy efficiency: mean power (W) and energy-per-inference (mJ)
  5. Saves:
        - per-model qualitative panels (input / GT / prediction / error)
        - combined comparison bar charts (accuracy + speed + efficiency)
        - latency distribution + quality-vs-speed scatter (Pareto view)
        - results.csv and results.json with every metric

DATA
----
The unseen dataset is the held-out `test` split of parquet files, with the same
schema the notebook used:
    columns: building_mask, tx_origin, path_loss   (each a flat 256*256 array)
Inputs are built identically to RadioMapDataset:
    ch0 = building_mask, ch1 = tx_origin, ch2 = sparse (random 1-10% of GT),
    target = normalised full path_loss.

USAGE
-----
    python3 ablation_inference_benchmark.py \
        --test-dir /data/dataset_256/test \
        --engines-dir ./engines \
        --weights-dir ./checkpoints \
        --out-dir ./benchmark_results \
        --max-samples 500

    # PyTorch-only (no TRT engines), still produces all quality metrics:
    python3 ablation_inference_benchmark.py --backend torch ...
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import threading
import time
from collections import defaultdict

import numpy as np

# Matplotlib without a display.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import docker_setup.jetson_radiomap.models_radiomap as M

MODELS = ["CNN", "WNet", "PartialConvMAE", "GNN"]
MODEL_COLORS = M.MODEL_COLORS
IMG = M.IMG_SIZE
MIN_DB, MAX_DB = M.MIN_DB, M.MAX_DB
DB_RANGE = MAX_DB - MIN_DB

CKPT_ALIASES = {
    "CNN": ["CNN_best.pth"],
    "WNet": ["WNet_best.pth", "WNET_best.pth"],
    "PartialConvMAE": ["PartialConvMAE_best.pth"],
    "GNN": ["GNN_best.pth"],
}


# ══════════════════════════════════════════════════════════════════════════════
#  Dataset (unseen test data) — mirrors RadioMapDataset from the notebook
# ══════════════════════════════════════════════════════════════════════════════
def build_sample(parquet_path, rng, sparse_low=655, sparse_high=6553):
    import pandas as pd
    df = pd.read_parquet(parquet_path, engine="pyarrow")
    row = df.iloc[0]
    env = np.array(row["building_mask"], dtype=np.float32).reshape(IMG, IMG)
    tx = np.array(row["tx_origin"], dtype=np.float32).reshape(IMG, IMG)
    radio = np.array(row["path_loss"], dtype=np.float32).reshape(IMG, IMG)

    radio_norm = np.clip((radio - MIN_DB) / DB_RANGE, 0.0, 1.0)

    n = rng.integers(sparse_low, sparse_high)
    xs = rng.integers(0, IMG, size=n)
    ys = rng.integers(0, IMG, size=n)
    sparse = np.zeros((IMG, IMG), dtype=np.float32)
    sparse[ys, xs] = radio_norm[ys, xs]

    inp = np.stack([env, tx, sparse], axis=0)           # (3,256,256)
    tgt = radio_norm[None, ...]                          # (1,256,256)
    return inp, tgt


def collect_test_files(test_dir, limit, seed):
    files = sorted(glob.glob(os.path.join(test_dir, "*.parquet")))
    rng = np.random.default_rng(seed)
    rng.shuffle(files)
    if limit:
        files = files[:limit]
    return files


# ══════════════════════════════════════════════════════════════════════════════
#  Quality metrics
# ══════════════════════════════════════════════════════════════════════════════
def _ssim_np(a, b, data_range=1.0):
    """Lightweight global SSIM (single-window) — dependency-free fallback."""
    a = a.astype(np.float64); b = b.astype(np.float64)
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) /
                 ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2)))


def quality_metrics(pred, tgt, tol_db=1.0):
    """pred, tgt : (1,256,256) normalised arrays. Returns dict of scalars."""
    p = np.clip(pred, 0.0, 1.0).astype(np.float32)
    t = np.clip(tgt, 0.0, 1.0).astype(np.float32)

    err = p - t
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(mse))

    # Physical (dB) scale
    err_db = err * DB_RANGE
    mae_db = float(np.mean(np.abs(err_db)))
    rmse_db = float(np.sqrt(np.mean(err_db ** 2)))

    # "accuracy" = fraction of pixels within tol_db dB of ground truth
    acc = float(np.mean(np.abs(err_db) <= tol_db))

    # PSNR (data range = 1.0 on normalised scale)
    psnr = float(20 * np.log10(1.0 / np.sqrt(mse))) if mse > 0 else 99.0

    ssim = _ssim_np(p[0], t[0], data_range=1.0)

    return dict(mse=mse, rmse=rmse, mae=mae,
                mae_db=mae_db, rmse_db=rmse_db,
                accuracy=acc, psnr=psnr, ssim=ssim)


# ══════════════════════════════════════════════════════════════════════════════
#  tegrastats telemetry sampler (GPU%, RAM, power) — Jetson specific
# ══════════════════════════════════════════════════════════════════════════════
class TegraStatsSampler:
    """Background sampler that parses `tegrastats` output for GPU/RAM/power.

    Gracefully no-ops if tegrastats is not present (e.g. running on a desktop).
    """

    GPU_RE = None  # compiled lazily

    def __init__(self, interval_ms=200):
        self.interval_ms = interval_ms
        self.proc = None
        self.thread = None
        self.samples = []  # list of dicts
        self._stop = threading.Event()
        self.available = self._has_tegrastats()

    @staticmethod
    def _has_tegrastats():
        from shutil import which
        return which("tegrastats") is not None

    def _reader(self):
        import re
        gpu_re = re.compile(r"GR3D_FREQ\s+(\d+)%")
        ram_re = re.compile(r"RAM\s+(\d+)/(\d+)MB")
        # power rails differ by carrier board; capture any "<RAIL> <cur>mW/<avg>mW"
        pow_re = re.compile(r"(VDD_\w+|POM_\w+|VIN_\w+)\s+(\d+)mW/(\d+)mW")
        for line in self.proc.stdout:
            if self._stop.is_set():
                break
            s = {}
            m = gpu_re.search(line)
            if m:
                s["gpu_pct"] = float(m.group(1))
            m = ram_re.search(line)
            if m:
                s["ram_used_mb"] = float(m.group(1))
                s["ram_total_mb"] = float(m.group(2))
            tot_cur = 0.0
            for rail, cur, avg in pow_re.findall(line):
                tot_cur += float(cur)
            if tot_cur > 0:
                s["power_mw"] = tot_cur
            if s:
                s["t"] = time.time()
                self.samples.append(s)

    def start(self):
        if not self.available:
            return
        self.samples = []
        self._stop.clear()
        self.proc = subprocess.Popen(
            ["tegrastats", "--interval", str(self.interval_ms)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def stop(self):
        if not self.available:
            return {}
        self._stop.set()
        try:
            self.proc.terminate()
        except Exception:
            pass
        if self.thread:
            self.thread.join(timeout=2)

        def col(key):
            return [s[key] for s in self.samples if key in s]

        gpu = col("gpu_pct")
        ram = col("ram_used_mb")
        pwr = col("power_mw")
        out = {}
        if gpu:
            out["gpu_pct_mean"] = float(np.mean(gpu))
            out["gpu_pct_max"] = float(np.max(gpu))
        if ram:
            out["ram_used_mb_mean"] = float(np.mean(ram))
            out["ram_used_mb_max"] = float(np.max(ram))
        if pwr:
            out["power_mw_mean"] = float(np.mean(pwr))
            out["power_mw_max"] = float(np.max(pwr))
        out["telemetry_samples"] = len(self.samples)
        return out


# ══════════════════════════════════════════════════════════════════════════════
#  Backends
# ══════════════════════════════════════════════════════════════════════════════
class TorchBackend:
    name = "torch"

    def __init__(self, model_name, weights_dir, device):
        import torch
        self.torch = torch
        self.device = device
        self.model = M.build_model(model_name).to(device)
        ckpt = None
        for fn in CKPT_ALIASES[model_name]:
            p = os.path.join(weights_dir, fn)
            if os.path.exists(p):
                ckpt = p
                break
        if ckpt is None:
            raise FileNotFoundError(
                f"No checkpoint for {model_name} in {weights_dir}")
        obj = torch.load(ckpt, map_location=device)
        if isinstance(obj, dict) and "state_dict" in obj:
            obj = obj["state_dict"]
        if any(k.startswith("module.") for k in obj):
            obj = {k[7:]: v for k, v in obj.items()}
        M.load_state_dict_flexible(self.model, obj)
        self.model.eval()
        self.ckpt = ckpt

    def infer(self, inp_np):
        torch = self.torch
        x = torch.from_numpy(inp_np[None, ...]).to(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            y = self.model(x)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1e3
        return y.detach().cpu().numpy()[0], ms, None


class TRTBackend:
    name = "tensorrt"

    def __init__(self, engine_path):
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401  (initializes CUDA context)
        self.trt = trt
        self.cuda = cuda

        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.engine_path = engine_path

        # Resolve I/O binding names across TRT versions.
        self._setup_bindings()

    def _setup_bindings(self):
        trt = self.trt
        self.input_name = "input"
        self.output_name = "output"
        # Allocate based on a fixed (1,3,256,256) -> (1,1,256,256) shape.
        in_shape = (1, 3, IMG, IMG)
        out_shape = (1, 1, IMG, IMG)
        try:
            self.context.set_input_shape(self.input_name, in_shape)
        except Exception:
            pass
        self.h_in = np.empty(in_shape, dtype=np.float32)
        self.h_out = np.empty(out_shape, dtype=np.float32)
        self.d_in = self.cuda.mem_alloc(self.h_in.nbytes)
        self.d_out = self.cuda.mem_alloc(self.h_out.nbytes)
        # tensor-address API (TRT 8.5+), fallback to bindings list.
        self._use_v3 = hasattr(self.context, "set_tensor_address")
        if self._use_v3:
            self.context.set_tensor_address(self.input_name, int(self.d_in))
            self.context.set_tensor_address(self.output_name, int(self.d_out))

    def infer(self, inp_np):
        cuda = self.cuda
        np.copyto(self.h_in, inp_np[None, ...])
        evt_start = cuda.Event(); evt_end = cuda.Event()
        cuda.memcpy_htod_async(self.d_in, self.h_in, self.stream)
        evt_start.record(self.stream)
        if self._use_v3:
            self.context.execute_async_v3(self.stream.handle)
        else:
            self.context.execute_async_v2(
                [int(self.d_in), int(self.d_out)], self.stream.handle)
        evt_end.record(self.stream)
        cuda.memcpy_dtoh_async(self.h_out, self.d_out, self.stream)
        self.stream.synchronize()
        gpu_ms = evt_end.time_since(evt_start)  # pure GPU compute time
        return self.h_out[0].copy(), gpu_ms, gpu_ms


def make_backend(model_name, args, device):
    """Pick TRT if requested+available, else fall back to torch."""
    engine_path = os.path.join(args.engines_dir, f"{model_name}.engine")
    want_trt = args.backend in ("auto", "tensorrt") and os.path.exists(engine_path)
    if want_trt:
        try:
            be = TRTBackend(engine_path)
            print(f"  backend    : TensorRT  ({engine_path})")
            return be
        except Exception as e:
            print(f"  backend    : TensorRT unavailable ({type(e).__name__}: "
                  f"{str(e)[:80]}) -> falling back to PyTorch")
            if args.backend == "tensorrt":
                raise
    be = TorchBackend(model_name, args.weights_dir, device)
    print(f"  backend    : PyTorch   ({be.ckpt})")
    return be


# ══════════════════════════════════════════════════════════════════════════════
#  Benchmark one model
# ══════════════════════════════════════════════════════════════════════════════
def benchmark_model(model_name, files, args, device, save_panel_for):
    print(f"\n{'='*70}")
    print(f"  BENCHMARK: {model_name}")
    print(f"{'='*70}")
    backend = make_backend(model_name, args, device)

    rng = np.random.default_rng(args.seed)  # same sparse pattern across models

    # Warm-up (not timed)
    warm_inp, _ = build_sample(files[0], np.random.default_rng(0),
                               args.sparse_low, args.sparse_high)
    for _ in range(args.warmup):
        backend.infer(warm_inp)

    latencies = []      # wall-clock ms per sample
    gpu_times = []       # GPU compute ms (TRT only)
    per_sample_q = defaultdict(list)
    panel_cache = None

    sampler = TegraStatsSampler(interval_ms=args.tegra_interval)
    sampler.start()

    for i, fpath in enumerate(files):
        inp, tgt = build_sample(fpath, rng, args.sparse_low, args.sparse_high)
        pred, ms, gpu_ms = backend.infer(inp)
        latencies.append(ms)
        if gpu_ms is not None:
            gpu_times.append(gpu_ms)

        q = quality_metrics(pred, tgt, tol_db=args.tol_db)
        for k, v in q.items():
            per_sample_q[k].append(v)

        if save_panel_for == model_name and panel_cache is None:
            panel_cache = (inp.copy(), tgt.copy(), pred.copy())

        if (i + 1) % max(1, len(files) // 10) == 0:
            print(f"    {i+1}/{len(files)}  "
                  f"lat={np.mean(latencies[-50:]):.2f}ms  "
                  f"rmse_db={np.mean(per_sample_q['rmse_db'][-50:]):.3f}")

    telemetry = sampler.stop()

    lat = np.array(latencies)
    agg = {
        "model": model_name,
        "backend": backend.name,
        "n_samples": len(files),
        # speed
        "latency_mean_ms": float(lat.mean()),
        "latency_median_ms": float(np.median(lat)),
        "latency_p95_ms": float(np.percentile(lat, 95)),
        "latency_std_ms": float(lat.std()),
        "fps": float(1000.0 / lat.mean()),
        # quality (averaged over unseen samples)
        **{k: float(np.mean(v)) for k, v in per_sample_q.items()},
    }
    if gpu_times:
        gt = np.array(gpu_times)
        agg["gpu_compute_mean_ms"] = float(gt.mean())
        agg["gpu_compute_median_ms"] = float(np.median(gt))
    agg.update(telemetry)

    # Energy efficiency (needs power telemetry)
    if "power_mw_mean" in agg:
        power_w = agg["power_mw_mean"] / 1000.0
        agg["power_w_mean"] = power_w
        agg["energy_per_inf_mj"] = power_w * agg["latency_mean_ms"]  # W * ms = mJ
        agg["inferences_per_joule"] = 1000.0 / agg["energy_per_inf_mj"] \
            if agg["energy_per_inf_mj"] > 0 else None

    # Pretty print
    print(f"\n  --- {model_name} results (unseen data, n={len(files)}) ---")
    print(f"    latency   : {agg['latency_mean_ms']:.3f} ms  "
          f"(median {agg['latency_median_ms']:.3f}, p95 {agg['latency_p95_ms']:.3f})")
    print(f"    throughput: {agg['fps']:.1f} FPS")
    if "gpu_compute_mean_ms" in agg:
        print(f"    gpu compute: {agg['gpu_compute_mean_ms']:.3f} ms")
    print(f"    MSE       : {agg['mse']:.5f}   RMSE: {agg['rmse']:.5f}   "
          f"MAE: {agg['mae']:.5f}")
    print(f"    RMSE(dB)  : {agg['rmse_db']:.3f}   MAE(dB): {agg['mae_db']:.3f}")
    print(f"    accuracy  : {agg['accuracy']*100:.2f}% (within {args.tol_db} dB)")
    print(f"    SSIM      : {agg['ssim']:.4f}   PSNR: {agg['psnr']:.2f} dB")
    if "gpu_pct_mean" in agg:
        print(f"    GPU util  : {agg['gpu_pct_mean']:.1f}% (max {agg.get('gpu_pct_max',0):.0f}%)")
    if "power_w_mean" in agg:
        print(f"    power     : {agg['power_w_mean']:.2f} W   "
              f"energy/inf: {agg['energy_per_inf_mj']:.1f} mJ")

    # release engine resources where possible
    del backend
    return agg, lat, panel_cache


# ══════════════════════════════════════════════════════════════════════════════
#  Visualizations
# ══════════════════════════════════════════════════════════════════════════════
def save_qualitative_panel(model_name, panel, out_dir):
    if panel is None:
        return
    inp, tgt, pred = panel
    pred = np.clip(pred, 0, 1)
    tgt = np.clip(tgt, 0, 1)
    err = np.abs(pred[0] - tgt[0])

    fig, ax = plt.subplots(1, 5, figsize=(22, 4.4))
    ax[0].imshow(inp[0], cmap="gray"); ax[0].set_title("Building mask")
    sparse = np.ma.masked_where(inp[2] == 0, inp[2])
    ax[1].imshow(inp[0], cmap="gray", alpha=0.3)
    ax[1].imshow(sparse, cmap="jet", vmin=0, vmax=1); ax[1].set_title("Sparse input")
    ax[2].imshow(tgt[0], cmap="jet", vmin=0, vmax=1); ax[2].set_title("Ground truth")
    ax[3].imshow(pred[0], cmap="jet", vmin=0, vmax=1); ax[3].set_title(f"{model_name} prediction")
    im = ax[4].imshow(err, cmap="magma", vmin=0, vmax=0.3); ax[4].set_title("|error|")
    for a in ax:
        a.axis("off")
    fig.colorbar(im, ax=ax[4], fraction=0.046, pad=0.04)
    fig.suptitle(f"{model_name} — qualitative reconstruction (unseen sample)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(out_dir, f"panel_{model_name}.png")
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved panel: {p}")


def color_for(name):
    return MODEL_COLORS.get(name, "#888888")


def save_comparison_charts(results, latencies, out_dir, tol_db):
    names = [r["model"] for r in results]
    colors = [color_for(n) for n in names]

    # ── 1. Quality comparison (RMSE_dB, MAE_dB, accuracy, SSIM) ──
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    metric_specs = [
        ("rmse_db", "RMSE (dB) — lower is better", False),
        ("mae_db", "MAE (dB) — lower is better", False),
        ("accuracy", f"Accuracy (% within {tol_db} dB) — higher is better", True),
        ("ssim", "SSIM — higher is better", True),
    ]
    for a, (key, title, higher_better) in zip(ax.ravel(), metric_specs):
        vals = [r.get(key, np.nan) for r in results]
        if key == "accuracy":
            vals = [v * 100 for v in vals]
        bars = a.bar(names, vals, color=colors, edgecolor="black", linewidth=0.6)
        a.set_title(title, fontsize=11, fontweight="bold")
        a.grid(axis="y", alpha=0.3)
        best = (max if higher_better else min)(range(len(vals)), key=lambda i: vals[i])
        for i, (b, v) in enumerate(zip(bars, vals)):
            a.text(b.get_x() + b.get_width()/2, v, f"{v:.3f}" if v < 10 else f"{v:.1f}",
                   ha="center", va="bottom", fontsize=9,
                   fontweight="bold" if i == best else "normal")
        bars[best].set_edgecolor("#000")
        bars[best].set_linewidth(2.2)
    fig.suptitle("Reconstruction quality on unseen data", fontsize=14, fontweight="bold")
    fig.tight_layout()
    p1 = os.path.join(out_dir, "compare_quality.png")
    fig.savefig(p1, dpi=130, bbox_inches="tight"); plt.close(fig)

    # ── 2. Speed + efficiency comparison ──
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    # latency
    lat = [r["latency_mean_ms"] for r in results]
    ax[0].bar(names, lat, color=colors, edgecolor="black")
    ax[0].set_title("Mean latency (ms) — lower better", fontweight="bold")
    ax[0].grid(axis="y", alpha=0.3)
    for i, v in enumerate(lat):
        ax[0].text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    # fps
    fps = [r["fps"] for r in results]
    ax[1].bar(names, fps, color=colors, edgecolor="black")
    ax[1].set_title("Throughput (FPS) — higher better", fontweight="bold")
    ax[1].grid(axis="y", alpha=0.3)
    for i, v in enumerate(fps):
        ax[1].text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    # energy or power
    if all("energy_per_inf_mj" in r for r in results):
        en = [r["energy_per_inf_mj"] for r in results]
        ax[2].bar(names, en, color=colors, edgecolor="black")
        ax[2].set_title("Energy / inference (mJ) — lower better", fontweight="bold")
        for i, v in enumerate(en):
            ax[2].text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    else:
        # fall back to GPU compute or just latency std
        if all("gpu_compute_mean_ms" in r for r in results):
            gc = [r["gpu_compute_mean_ms"] for r in results]
            ax[2].bar(names, gc, color=colors, edgecolor="black")
            ax[2].set_title("GPU compute (ms) — lower better", fontweight="bold")
        else:
            std = [r["latency_std_ms"] for r in results]
            ax[2].bar(names, std, color=colors, edgecolor="black")
            ax[2].set_title("Latency std (ms)", fontweight="bold")
    ax[2].grid(axis="y", alpha=0.3)
    fig.suptitle("Inference speed & efficiency on Jetson Orin Nano",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    p2 = os.path.join(out_dir, "compare_speed_efficiency.png")
    fig.savefig(p2, dpi=130, bbox_inches="tight"); plt.close(fig)

    # ── 3. Latency distributions (violin/box) ──
    fig, a = plt.subplots(figsize=(10, 5))
    data = [latencies[n] for n in names]
    try:
        bp = a.boxplot(data, tick_labels=names, patch_artist=True, showfliers=False)
    except TypeError:  # older matplotlib
        bp = a.boxplot(data, labels=names, patch_artist=True, showfliers=False)
    for patch, n in zip(bp["boxes"], names):
        patch.set_facecolor(color_for(n)); patch.set_alpha(0.8)
    a.set_ylabel("Latency (ms)")
    a.set_title("Per-sample latency distribution", fontweight="bold")
    a.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p3 = os.path.join(out_dir, "compare_latency_dist.png")
    fig.savefig(p3, dpi=130, bbox_inches="tight"); plt.close(fig)

    # ── 4. Quality vs Speed (Pareto) scatter ──
    fig, a = plt.subplots(figsize=(9, 7))
    for r in results:
        a.scatter(r["latency_mean_ms"], r["accuracy"] * 100,
                  s=260, color=color_for(r["model"]),
                  edgecolor="black", linewidth=1.2, zorder=3)
        a.annotate(r["model"],
                   (r["latency_mean_ms"], r["accuracy"] * 100),
                   textcoords="offset points", xytext=(8, 8), fontsize=11,
                   fontweight="bold")
    a.set_xlabel("Mean latency (ms)  —  faster ←")
    a.set_ylabel(f"Accuracy (% within {tol_db} dB)  —  ↑ better")
    a.set_title("Quality vs. Speed trade-off (top-left = ideal)",
                fontweight="bold")
    a.grid(alpha=0.3)
    fig.tight_layout()
    p4 = os.path.join(out_dir, "compare_pareto_quality_vs_speed.png")
    fig.savefig(p4, dpi=130, bbox_inches="tight"); plt.close(fig)

    print(f"  saved charts: {p1}, {p2}, {p3}, {p4}")
    return [p1, p2, p3, p4]


def save_tables(results, out_dir):
    import csv
    keys = ["model", "backend", "n_samples",
            "latency_mean_ms", "latency_median_ms", "latency_p95_ms", "fps",
            "gpu_compute_mean_ms",
            "mse", "rmse", "mae", "rmse_db", "mae_db", "accuracy", "psnr", "ssim",
            "gpu_pct_mean", "ram_used_mb_mean", "power_w_mean",
            "energy_per_inf_mj", "inferences_per_joule"]
    csv_path = os.path.join(out_dir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r)
    json_path = os.path.join(out_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  saved tables: {csv_path}, {json_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(
        description="Sequentially benchmark CNN/WNet/PartialConvMAE/GNN on "
                    "Jetson Orin Nano against unseen data.")
    ap.add_argument("--test-dir", required=True,
                    help="Directory of UNSEEN test .parquet files.")
    ap.add_argument("--engines-dir", default="./engines",
                    help="Directory of TensorRT .engine files.")
    ap.add_argument("--weights-dir", default="./checkpoints",
                    help="Directory of *_best.pth (PyTorch fallback).")
    ap.add_argument("--out-dir", default="./benchmark_results")
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--backend", choices=["auto", "tensorrt", "torch"], default="auto",
                    help="auto = TRT engine if present else PyTorch.")
    ap.add_argument("--max-samples", type=int, default=500,
                    help="Cap on number of unseen test files (None=all).")
    ap.add_argument("--warmup", type=int, default=10,
                    help="Warm-up inferences (not timed) per model.")
    ap.add_argument("--tol-db", type=float, default=1.0,
                    help="Tolerance in dB for the pixel 'accuracy' metric.")
    ap.add_argument("--sparse-low", type=int, default=655)
    ap.add_argument("--sparse-high", type=int, default=6553)
    ap.add_argument("--tegra-interval", type=int, default=200,
                    help="tegrastats sampling interval (ms).")
    ap.add_argument("--seed", type=int, default=42,
                    help="Seed for sparse sampling (identical across models for "
                         "a fair comparison).")
    ap.add_argument("--cpu", action="store_true",
                    help="Force CPU for the PyTorch fallback.")
    args = ap.parse_args()

    import torch
    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")

    os.makedirs(args.out_dir, exist_ok=True)
    files = collect_test_files(args.test_dir, args.max_samples, args.seed)
    if not files:
        print(f"ERROR: no .parquet files found in {args.test_dir}")
        sys.exit(1)

    print("="*70)
    print("  JETSON ORIN NANO — RADIO-MAP ABLATION INFERENCE BENCHMARK")
    print("="*70)
    print(f"  Device       : {device}")
    print(f"  Test dir     : {args.test_dir}")
    print(f"  Unseen files : {len(files)}")
    print(f"  Models       : {args.models}")
    print(f"  Backend      : {args.backend}")
    print(f"  Accuracy tol : {args.tol_db} dB")

    results, latencies, panels = [], {}, {}
    # save a qualitative panel for each model
    for name in args.models:
        agg, lat, panel = benchmark_model(name, files, args, device,
                                          save_panel_for=name)
        results.append(agg)
        latencies[name] = lat
        panels[name] = panel

    # ── Visualizations ──
    print(f"\n{'='*70}\n  GENERATING VISUALIZATIONS\n{'='*70}")
    for name in args.models:
        save_qualitative_panel(name, panels[name], args.out_dir)
    save_comparison_charts(results, latencies, args.out_dir, args.tol_db)
    save_tables(results, args.out_dir)

    # ── Final leaderboard ──
    print(f"\n{'='*70}\n  FINAL LEADERBOARD (unseen data)\n{'='*70}")
    print(f"  {'Model':16s} {'FPS':>7s} {'Lat(ms)':>9s} {'RMSE(dB)':>9s} "
          f"{'Acc%':>7s} {'SSIM':>7s}")
    for r in sorted(results, key=lambda x: x["rmse_db"]):
        print(f"  {r['model']:16s} {r['fps']:7.1f} {r['latency_mean_ms']:9.3f} "
              f"{r['rmse_db']:9.3f} {r['accuracy']*100:7.2f} {r['ssim']:7.4f}")
    print(f"\n  All outputs written to: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
