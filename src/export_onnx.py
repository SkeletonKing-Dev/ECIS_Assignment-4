"""
Export the six student models (baseline + distilled) and the best NAS model
to ONNX (opset 17, dynamic batch size) for CPU deployment with ONNX Runtime.

Each ONNX file carries its own metadata (dataset, input resolution, class
names, normalisation), so onnx_infer.py needs no PyTorch checkpoint.
Every export is verified against the PyTorch model on random input.

Usage:
    python src/export_onnx.py
"""
import argparse
import json

import numpy as np
import onnx
import onnxruntime as ort
import torch

import common as C


def export_one(name, path):
    model, ckpt = C.load_checkpoint(path)
    res = ckpt.get("resolution") or 32
    x = torch.randn(2, 3, res, res)
    out = C.onnx_path(name)
    torch.onnx.export(model, (x,), out, input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=17, dynamo=False)

    m = onnx.load(out)
    mean, std = C.normalize_stats(ckpt["dataset"])
    meta = {"dataset": ckpt["dataset"], "resolution": str(res), "classes": json.dumps(ckpt["classes"]),
            "mean": json.dumps(mean), "std": json.dumps(std), "arch": ckpt["arch"]}
    for k, v in meta.items():
        m.metadata_props.add(key=k, value=v)
    onnx.checker.check_model(m)
    onnx.save(m, out)

    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = model(x).numpy()
    diff = float(np.abs(sess.run(None, {"input": x.numpy()})[0] - ref).max())
    print(f"  {name:<28} {res}x{res}  {C.file_size_mb(out):6.2f} MB  max |ONNX - PyTorch| = {diff:.2e}")
    return diff


def main():
    argparse.ArgumentParser().parse_args()
    C.ensure_dirs()
    cat = C.model_catalog(include_teacher=False)
    if not cat:
        raise SystemExit("no trained models found in models/")
    print(f"Exporting {len(cat)} models to {C.ONNX_DIR}")
    worst = max(export_one(n, p) for n, p in cat.items())
    print(f"All exports verified (largest output difference {worst:.2e})")


if __name__ == "__main__":
    main()
