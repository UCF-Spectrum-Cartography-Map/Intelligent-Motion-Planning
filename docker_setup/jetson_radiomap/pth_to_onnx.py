#!/usr/bin/env python3
"""
pth_to_onnx.py
==============
Convert the trained radio-map reconstruction checkpoints (*_best.pth) into the
Open Neural Network Exchange (.onnx) format on the NVIDIA Jetson Orin Nano.

Designed to live inside the cloned `jetson-containers` directory and be run
from inside a PyTorch-enabled Jetson container (e.g. the dustynv/l4t-pytorch
image launched via `jetson-containers run ...`).

Only the four models you benchmark are handled:
    CNN  -> CNN_best.pth
    WNet -> WNet_best.pth   (notebook name; "WNET" is also accepted)
    PartialConvMAE -> PartialConvMAE_best.pth
    GNN  -> GNN_best.pth

The model architectures are imported from `models_radiomap.py`, which must sit
next to this script.

USAGE
-----
    # Convert everything found in ./checkpoints into ./onnx
    python3 pth_to_onnx.py --weights-dir ./checkpoints --out-dir ./onnx

    # Convert a single model
    python3 pth_to_onnx.py --weights-dir ./checkpoints --models CNN

    # Use a larger fixed batch / static shape (recommended for TensorRT)
    python3 pth_to_onnx.py --batch 1 --static

NOTES
-----
* Input shape is (B, 3, 256, 256), output (B, 1, 256, 256).
* W-Net's training forward() returns (out1, out2); we export only the refined
  output `out2` via WNetExportWrapper so the graph has a single tensor output.
* The GNN (Vision-GNN) contains a dynamic kNN (cdist + topk + gather). It
  exports fine to ONNX, but for clean TensorRT engine building you should keep
  the batch dimension STATIC (use --static). onnx_to_engine.py defaults to a
  fixed shape for exactly this reason.
* We use the legacy TorchScript exporter (dynamo=False) because it produces the
  most TensorRT-compatible graphs on current Jetson torch builds.
"""

import argparse
import os
import sys

import torch

import docker_setup.jetson_radiomap.models_radiomap as M


DEFAULT_MODELS = ["CNN", "WNet", "PartialConvMAE", "GNN"]

# Map short model name -> list of acceptable checkpoint filenames (in priority order)
CKPT_ALIASES = {
    "CNN": ["CNN_best.pth"],
    "WNet": ["WNet_best.pth", "WNET_best.pth"],
    "PartialConvMAE": ["PartialConvMAE_best.pth"],
    "GNN": ["GNN_best.pth"],
}


def find_checkpoint(weights_dir, model_name):
    for fname in CKPT_ALIASES[model_name]:
        path = os.path.join(weights_dir, fname)
        if os.path.exists(path):
            return path
    return None


def load_weights(model, ckpt_path, device):
    """Load a checkpoint that may be a raw state_dict OR a dict with a nested key."""
    obj = torch.load(ckpt_path, map_location=device)
    # Some training loops wrap the state_dict; unwrap common containers.
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        state_dict = obj["state_dict"]
    elif isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        state_dict = obj["model"]
    else:
        state_dict = obj
    # Strip any DataParallel "module." prefixes.
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    missing, unexpected = M.load_state_dict_flexible(model, state_dict)
    return missing, unexpected


def export_one(model_name, ckpt_path, out_dir, batch, static, opset, device, verify):
    print(f"\n{'='*64}")
    print(f"  {model_name}")
    print(f"{'='*64}")
    print(f"  checkpoint : {ckpt_path}")

    model = M.build_model(model_name).to(device)
    missing, unexpected = load_weights(model, ckpt_path, device)
    n_params = M.count_params(model)
    print(f"  params     : {n_params:,}")
    if missing:
        print(f"  WARNING    : {len(missing)} missing keys (showing up to 5): {missing[:5]}")
    if unexpected:
        print(f"  WARNING    : {len(unexpected)} unexpected keys (showing up to 5): {unexpected[:5]}")
    model.eval()

    dummy = torch.randn(batch, 3, M.IMG_SIZE, M.IMG_SIZE, device=device)

    # Reference output (for optional verification).
    with torch.no_grad():
        ref = model(dummy).cpu()

    os.makedirs(out_dir, exist_ok=True)
    onnx_path = os.path.join(out_dir, f"{model_name}.onnx")

    dynamic_axes = None
    if not static:
        dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}

    print(f"  exporting  : opset={opset}  batch={batch}  "
          f"{'STATIC' if static else 'DYNAMIC batch'}")
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        dynamo=False,  # TorchScript exporter -> most TRT-friendly graph
    )
    size_mb = os.path.getsize(onnx_path) / 1e6
    print(f"  saved      : {onnx_path}  ({size_mb:.1f} MB)")

    if verify:
        try:
            import onnx
            onnx.checker.check_model(onnx.load(onnx_path))
            print("  onnx.check : PASSED")
        except ImportError:
            print("  onnx.check : skipped (onnx not installed)")
        except Exception as e:
            print(f"  onnx.check : FAILED -> {e}")

        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            ort_out = sess.run(["output"], {"input": dummy.cpu().numpy()})[0]
            import numpy as np
            max_diff = float(np.abs(ort_out - ref.numpy()).max())
            print(f"  parity     : max|torch-onnx| = {max_diff:.3e}")
            if max_diff > 1e-2:
                print("  parity     : WARNING — larger than expected; inspect graph.")
        except ImportError:
            print("  parity     : skipped (onnxruntime not installed)")
        except Exception as e:
            print(f"  parity     : skipped ({type(e).__name__}: {str(e)[:80]})")

    return onnx_path


def main():
    ap = argparse.ArgumentParser(description="Convert *_best.pth checkpoints to ONNX.")
    ap.add_argument("--weights-dir", default="./checkpoints",
                    help="Directory containing the *_best.pth files.")
    ap.add_argument("--out-dir", default="./onnx",
                    help="Directory to write the .onnx files into.")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                    choices=DEFAULT_MODELS,
                    help="Which models to export.")
    ap.add_argument("--batch", type=int, default=1,
                    help="Batch size baked into the dummy input.")
    ap.add_argument("--static", action="store_true",
                    help="Export with a fully static shape (recommended for TRT). "
                         "Without this flag the batch dim is dynamic.")
    ap.add_argument("--opset", type=int, default=17,
                    help="ONNX opset version.")
    ap.add_argument("--cpu", action="store_true",
                    help="Force CPU export even if CUDA is available.")
    ap.add_argument("--no-verify", action="store_true",
                    help="Skip onnx.checker / onnxruntime parity verification.")
    args = ap.parse_args()

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")
    print(f"Device          : {device}")
    print(f"Torch version   : {torch.__version__}")
    print(f"Weights dir     : {os.path.abspath(args.weights_dir)}")
    print(f"Output dir      : {os.path.abspath(args.out_dir)}")
    print(f"Models          : {args.models}")

    if not os.path.isdir(args.weights_dir):
        print(f"\nERROR: weights dir not found: {args.weights_dir}")
        sys.exit(1)

    converted, skipped = [], []
    for name in args.models:
        ckpt = find_checkpoint(args.weights_dir, name)
        if ckpt is None:
            print(f"\n[SKIP] {name}: no checkpoint found "
                  f"(looked for {CKPT_ALIASES[name]} in {args.weights_dir})")
            skipped.append(name)
            continue
        try:
            out = export_one(name, ckpt, args.out_dir, args.batch, args.static,
                             args.opset, device, verify=not args.no_verify)
            converted.append((name, out))
        except Exception as e:
            print(f"\n[ERROR] {name}: export failed -> {type(e).__name__}: {e}")
            skipped.append(name)

    print(f"\n{'='*64}")
    print("  SUMMARY")
    print(f"{'='*64}")
    for name, out in converted:
        print(f"  ✓ {name:16s} -> {out}")
    for name in skipped:
        print(f"  ✗ {name:16s} (skipped/failed)")
    print(f"\nDone. {len(converted)} converted, {len(skipped)} skipped.")
    if converted:
        print("Next step: build TensorRT engines with onnx_to_engine.py")


if __name__ == "__main__":
    main()
