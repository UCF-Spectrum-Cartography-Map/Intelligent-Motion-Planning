#!/usr/bin/env python3
"""
onnx_to_engine.py
=================
Build highly optimized, Jetson-Orin-Nano-specific TensorRT `.engine` files from
the `.onnx` models produced by `pth_to_onnx.py`, using `trtexec` (shipped with
JetPack).

`trtexec` performs the heavy lifting we want for deployment:
    * strips training-only nodes (dropout, identity, constant-folded BN, etc.)
    * fuses layers (conv+bn+activation fusion, etc.)
    * runs the TensorRT auto-tuner to pick the fastest kernels (tactics) for the
      Orin Nano's Ampere GPU
    * serialises everything into a single .engine locked to THIS device/SM,
      and prints a full latency/throughput profile while it builds.

Designed to run inside the cloned `jetson-containers` dir, in a container that
has TensorRT / trtexec available (e.g. the l4t-tensorrt or l4t-pytorch images).

USAGE
-----
    # Build FP16 engines for every .onnx in ./onnx, write to ./engines
    python3 onnx_to_engine.py --onnx-dir ./onnx --out-dir ./engines

    # FP32 instead of FP16
    python3 onnx_to_engine.py --precision fp32

    # One model, custom workspace, more profiling iterations
    python3 onnx_to_engine.py --models GNN --workspace 2048 --iterations 500

PRECISION
---------
    fp16 (default) — best speed/accuracy trade-off on Orin Nano (DLA/GPU FP16).
    fp32           — reference accuracy, slowest.
    int8           — fastest, but needs calibration data; we expose --int8 but
                     it will run trtexec's (lower-accuracy) auto int8 unless you
                     supply a calibration cache. Use with care for regression
                     metrics; fp16 is recommended for the ablation study.

OUTPUT
------
For each model:
    ./engines/<MODEL>.engine          the serialized TensorRT engine
    ./engines/<MODEL>_build.log       full trtexec build + profile log
    ./engines/<MODEL>_profile.json    layer timing (from --exportProfile)
    ./engines/<MODEL>_times.json      per-iteration timing (from --exportTimes)
    ./engines/build_summary.json      parsed throughput/latency for all models
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

DEFAULT_MODELS = ["CNN", "WNet", "PartialConvMAE", "GNN"]

# Models that contain dynamic kNN / data-dependent ops. We keep these on a
# fully static shape so TensorRT can build cleanly.
DYNAMIC_OP_MODELS = {"GNN"}


def find_trtexec():
    """Locate the trtexec binary shipped with JetPack/TensorRT."""
    cand = shutil.which("trtexec")
    if cand:
        return cand
    for p in (
        "/usr/src/tensorrt/bin/trtexec",
        "/usr/local/tensorrt/bin/trtexec",
        "/opt/tensorrt/bin/trtexec",
    ):
        if os.path.exists(p):
            return p
    return None


def build_trtexec_cmd(trtexec, onnx_path, engine_path, profile_json, times_json,
                      args, static_shape):
    """Assemble the trtexec command line for one model."""
    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        # Profiling controls
        f"--iterations={args.iterations}",
        f"--warmUp={args.warmup}",          # ms of warm-up before timing
        f"--avgRuns={args.avg_runs}",
        f"--duration={args.duration}",      # min seconds to run
        "--useSpinWait",                    # lower-latency, accurate timing
        f"--exportProfile={profile_json}",  # per-layer timing
        f"--exportTimes={times_json}",      # per-iteration timing
        "--verbose",
    ]

    # Workspace / memory pool. Newer TRT prefers --memPoolSize; we pass both
    # forms and let trtexec ignore the one it doesn't recognise via fallback.
    cmd.append(f"--memPoolSize=workspace:{args.workspace}M")

    # Precision flags (these enable layer fusion + kernel autotuning for the mode)
    if args.precision == "fp16":
        cmd.append("--fp16")
    elif args.precision == "int8":
        cmd.append("--int8")
        # allow fp16 fallback for layers int8 can't handle
        cmd.append("--fp16")
        if args.calib_cache:
            cmd.append(f"--calib={args.calib_cache}")
    # fp32 -> no extra flag (default)

    # Shapes. For dynamic-op models we force a single static shape.
    # We build for explicit-batch ONNX; if the ONNX has a dynamic batch axis we
    # pin it with optimization profile shapes.
    shape = f"input:{args.batch}x3x256x256"
    if static_shape:
        cmd.append(f"--shapes={shape}")
    else:
        cmd += [
            f"--minShapes={shape}",
            f"--optShapes={shape}",
            f"--maxShapes=input:{args.max_batch}x3x256x256",
        ]

    # Build only (don't also load+infer) is NOT what we want — we want the
    # profile, so we let trtexec build AND benchmark in one shot.
    return cmd


def parse_build_log(log_text):
    """Extract throughput / latency numbers from a trtexec log."""
    out = {}
    patterns = {
        "throughput_qps": r"Throughput:\s*([\d.]+)\s*qps",
        "latency_mean_ms": r"Latency:.*?mean\s*=\s*([\d.]+)\s*ms",
        "latency_min_ms": r"Latency:.*?min\s*=\s*([\d.]+)\s*ms",
        "latency_max_ms": r"Latency:.*?max\s*=\s*([\d.]+)\s*ms",
        "latency_median_ms": r"Latency:.*?median\s*=\s*([\d.]+)\s*ms",
        "latency_p99_ms": r"Latency:.*?percentile\(99%\)\s*=\s*([\d.]+)\s*ms",
        "gpu_compute_mean_ms": r"GPU Compute Time:.*?mean\s*=\s*([\d.]+)\s*ms",
        "gpu_compute_median_ms": r"GPU Compute Time:.*?median\s*=\s*([\d.]+)\s*ms",
        "h2d_mean_ms": r"H2D Latency:.*?mean\s*=\s*([\d.]+)\s*ms",
        "d2h_mean_ms": r"D2H Latency:.*?mean\s*=\s*([\d.]+)\s*ms",
    }
    for key, pat in patterns.items():
        m = re.search(pat, log_text, re.IGNORECASE | re.DOTALL)
        if m:
            out[key] = float(m.group(1))
    return out


def build_one(model_name, onnx_dir, out_dir, trtexec, args):
    onnx_path = os.path.join(onnx_dir, f"{model_name}.onnx")
    if not os.path.exists(onnx_path):
        print(f"[SKIP] {model_name}: ONNX not found at {onnx_path}")
        return None

    engine_path = os.path.join(out_dir, f"{model_name}.engine")
    log_path = os.path.join(out_dir, f"{model_name}_build.log")
    profile_json = os.path.join(out_dir, f"{model_name}_profile.json")
    times_json = os.path.join(out_dir, f"{model_name}_times.json")

    static_shape = args.static or (model_name in DYNAMIC_OP_MODELS)
    cmd = build_trtexec_cmd(trtexec, onnx_path, engine_path, profile_json,
                            times_json, args, static_shape)

    print(f"\n{'='*70}")
    print(f"  BUILD + PROFILE: {model_name}  "
          f"({args.precision.upper()}, {'static' if static_shape else 'dynamic'} shape)")
    print(f"{'='*70}")
    print("  " + " ".join(cmd))

    t0 = time.time()
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True)
        logf.write(proc.stdout)
    build_secs = time.time() - t0

    # Echo the tail of the log so the user sees the profile inline.
    tail = "\n".join(proc.stdout.splitlines()[-25:])
    print(tail)

    if proc.returncode != 0:
        print(f"  [ERROR] trtexec exited {proc.returncode}. See {log_path}")
        return {
            "model": model_name,
            "status": "failed",
            "returncode": proc.returncode,
            "build_log": log_path,
        }

    metrics = parse_build_log(proc.stdout)
    engine_mb = (os.path.getsize(engine_path) / 1e6) if os.path.exists(engine_path) else None

    result = {
        "model": model_name,
        "status": "ok",
        "precision": args.precision,
        "static_shape": static_shape,
        "engine_path": engine_path,
        "engine_size_mb": engine_mb,
        "build_seconds": round(build_secs, 1),
        "build_log": log_path,
        "profile_json": profile_json if os.path.exists(profile_json) else None,
        "times_json": times_json if os.path.exists(times_json) else None,
        **metrics,
    }
    print(f"  ✓ engine     : {engine_path}  ({engine_mb:.1f} MB)" if engine_mb else
          f"  ✓ engine     : {engine_path}")
    if "throughput_qps" in metrics:
        print(f"  ✓ throughput : {metrics['throughput_qps']:.1f} qps")
    if "latency_mean_ms" in metrics:
        print(f"  ✓ latency    : {metrics['latency_mean_ms']:.3f} ms (mean)")
    if "gpu_compute_mean_ms" in metrics:
        print(f"  ✓ gpu compute: {metrics['gpu_compute_mean_ms']:.3f} ms (mean)")
    return result


def main():
    ap = argparse.ArgumentParser(
        description="Build + profile TensorRT engines from ONNX via trtexec.")
    ap.add_argument("--onnx-dir", default="./onnx",
                    help="Directory containing the .onnx files.")
    ap.add_argument("--out-dir", default="./engines",
                    help="Directory to write engines + logs.")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                    choices=DEFAULT_MODELS)
    ap.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp16")
    ap.add_argument("--calib-cache", default=None,
                    help="INT8 calibration cache file (only used with --precision int8).")
    ap.add_argument("--batch", type=int, default=1,
                    help="Batch size for the (opt) profile shape.")
    ap.add_argument("--max-batch", type=int, default=1,
                    help="Max batch for dynamic profile (ignored when static).")
    ap.add_argument("--static", action="store_true",
                    help="Force static shapes for ALL models.")
    ap.add_argument("--workspace", type=int, default=1024,
                    help="Workspace / memory pool size in MB. Orin Nano has 8GB "
                         "shared; 1024-2048 is usually safe.")
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--avg-runs", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=500, help="Warm-up time in ms.")
    ap.add_argument("--duration", type=int, default=10,
                    help="Minimum seconds of benchmarking per model.")
    ap.add_argument("--trtexec", default=None,
                    help="Explicit path to trtexec (auto-detected otherwise).")
    args = ap.parse_args()

    trtexec = args.trtexec or find_trtexec()
    if not trtexec:
        print("ERROR: trtexec not found. Run this inside a JetPack/TensorRT "
              "container (e.g. via jetson-containers run ... l4t-tensorrt) or "
              "pass --trtexec /path/to/trtexec.")
        sys.exit(1)
    print(f"trtexec    : {trtexec}")
    print(f"ONNX dir   : {os.path.abspath(args.onnx_dir)}")
    print(f"Out dir    : {os.path.abspath(args.out_dir)}")
    print(f"Precision  : {args.precision}")
    print(f"Models     : {args.models}")

    os.makedirs(args.out_dir, exist_ok=True)

    results = []
    for name in args.models:
        r = build_one(name, args.onnx_dir, args.out_dir, trtexec, args)
        if r is not None:
            results.append(r)

    summary_path = os.path.join(args.out_dir, "build_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print("  BUILD SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Model':16s} {'Status':8s} {'Lat(ms)':>9s} {'QPS':>9s} {'Engine(MB)':>11s}")
    for r in results:
        lat = r.get("latency_mean_ms", float("nan"))
        qps = r.get("throughput_qps", float("nan"))
        mb = r.get("engine_size_mb", float("nan"))
        print(f"  {r['model']:16s} {r['status']:8s} {lat:9.3f} {qps:9.1f} {mb:11.1f}")
    print(f"\nSummary JSON: {summary_path}")
    print("Next step: run ablation_inference_benchmark.py against unseen data.")


if __name__ == "__main__":
    main()
