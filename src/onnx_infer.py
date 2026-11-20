"""
CPU inference with ONNX Runtime for the exported student and NAS models.

Inference needs only onnxruntime, numpy and Pillow: the input resolution,
normalisation and class names are read from the ONNX file's metadata.
(PyTorch is imported only for --test-samples, to read the CIFAR test set.)

Usage:
    python src/onnx_infer.py --model all --test-samples 5
    python src/onnx_infer.py --model efficientnet_lite0_kd --image cat.jpg
    python src/onnx_infer.py --model nas_best --threads 1
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONNX_DIR = os.path.join(ROOT, "onnx_models")


class OnnxPredictor:
    def __init__(self, path, threads=0):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            opts.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
        meta = self.sess.get_modelmeta().custom_metadata_map
        self.dataset = meta["dataset"]
        self.resolution = int(meta["resolution"])
        self.classes = json.loads(meta["classes"])
        self.mean = np.array(json.loads(meta["mean"]), np.float32).reshape(1, 3, 1, 1)
        self.std = np.array(json.loads(meta["std"]), np.float32).reshape(1, 3, 1, 1)

    def preprocess(self, images_uint8):
        """uint8 [N,32,32,3] -> normalised float32 [N,3,R,R]."""
        if self.resolution != images_uint8.shape[1]:
            images_uint8 = np.stack([np.asarray(Image.fromarray(im).resize(
                (self.resolution, self.resolution), Image.BILINEAR)) for im in images_uint8])
        x = images_uint8.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        return (x - self.mean) / self.std

    def logits(self, x):
        return self.sess.run(None, {"input": x})[0]

    def predict(self, images_uint8):
        z = self.logits(self.preprocess(images_uint8))
        e = np.exp(z - z.max(1, keepdims=True))
        p = e / e.sum(1, keepdims=True)
        return p.argmax(1), p.max(1)


def available():
    return {os.path.basename(p)[:-5]: p for p in sorted(glob.glob(os.path.join(ONNX_DIR, "*.onnx")))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all")
    ap.add_argument("--image", nargs="*")
    ap.add_argument("--test-samples", type=int, default=8)
    ap.add_argument("--threads", type=int, default=0, help="0 = ONNX Runtime default")
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()

    models = available()
    if not models:
        raise SystemExit("no ONNX models found: run src/export_onnx.py first")
    names = list(models) if args.model == "all" else [args.model]
    for n in names:
        if n not in models:
            raise SystemExit(f"unknown model '{n}'. Available: {list(models)}")

    cache = {}
    for name in names:
        p = OnnxPredictor(models[name], args.threads)
        if args.image:
            imgs = np.stack([np.asarray(Image.open(f).convert("RGB").resize((32, 32), Image.BILINEAR))
                             for f in args.image])
            labels, ids = [None] * len(imgs), args.image
        else:
            if p.dataset not in cache:
                import common as C                      # only needed to read the CIFAR test set
                x, y, _ = C.load_arrays(p.dataset, False, args.fake)
                cache[p.dataset] = (x[:args.test_samples], y[:args.test_samples])
            imgs, labels = cache[p.dataset]
            ids = [f"test[{i}]" for i in range(len(labels))]
        p.predict(imgs[:1])                             # warm-up
        t0 = time.perf_counter()
        idx, conf = p.predict(imgs)
        ms = (time.perf_counter() - t0) * 1000
        print(f"\n== ONNX {name} ==  ({p.dataset}, {p.resolution}x{p.resolution}, {ms:.1f} ms for {len(imgs)} images)")
        for i in range(len(imgs)):
            pred = p.classes[idx[i]]
            if labels[i] is None:
                print(f"  {ids[i]:<12} pred: {pred:<15} conf {conf[i]:.3f}")
            else:
                mark = "ok" if idx[i] == labels[i] else "WRONG"
                print(f"  {ids[i]:<12} pred: {pred:<15} conf {conf[i]:.3f}  true: {p.classes[labels[i]]:<15} {mark}")


if __name__ == "__main__":
    main()
