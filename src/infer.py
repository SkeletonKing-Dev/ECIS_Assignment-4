"""
Unified PyTorch inference pipeline for all trained models:
  students (baseline and distilled, CIFAR-100), the teacher, and the best NAS
  model (CIFAR-10). Each checkpoint stores its dataset, input resolution and
  class names, so the same code serves every model.

Usage:
    python src/infer.py --list
    python src/infer.py --model mobilenetv2_kd --image cat.jpg
    python src/infer.py --model nas_best --test-samples 8
    python src/infer.py --model all --test-samples 5
"""
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import common as C


class Predictor:
    """Loads one checkpoint and classifies uint8 images [N,32,32,3] or image files."""

    def __init__(self, name):
        path = C.model_catalog()[name]
        self.name = name
        self.model, ckpt = C.load_checkpoint(path)
        self.dataset = ckpt["dataset"]
        self.classes = ckpt["classes"]
        self.resolution = ckpt.get("resolution") or 32

    @torch.no_grad()
    def predict(self, images_uint8, topk=1):
        x = C.preprocess_numpy(images_uint8, self.dataset, self.resolution)
        prob = F.softmax(self.model(x), 1)
        conf, idx = prob.topk(topk, 1)
        return idx.numpy(), conf.numpy()


def load_image_files(paths):
    """Any image -> RGB 32x32 uint8, the same format as CIFAR."""
    return np.stack([np.asarray(Image.open(p).convert("RGB").resize((32, 32), Image.BILINEAR)) for p in paths])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all", help="model name, 'all' or 'students' (see --list)")
    ap.add_argument("--image", nargs="*", help="image file(s) to classify")
    ap.add_argument("--test-samples", type=int, default=8, help="if no image: first N test images")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()

    cat = C.model_catalog()
    if args.list:
        for n, p in cat.items():
            print(f"  {n:<28} {p}")
        return
    if args.model == "all":
        names = list(cat)
    elif args.model == "students":
        names = [n for n in cat if n.endswith(("_baseline", "_kd"))]
    else:
        if args.model not in cat:
            raise SystemExit(f"unknown or untrained model '{args.model}'. Available: {list(cat)}")
        names = [args.model]

    cache = {}
    for name in names:
        p = Predictor(name)
        if args.image:
            imgs, labels, ids = load_image_files(args.image), [None] * len(args.image), args.image
        else:
            if p.dataset not in cache:
                x, y, _ = C.load_arrays(p.dataset, False, args.fake)
                cache[p.dataset] = (x[:args.test_samples], y[:args.test_samples])
            imgs, labels = cache[p.dataset]
            ids = [f"test[{i}]" for i in range(len(labels))]
        p.predict(imgs[:1])                                   # warm-up
        t0 = time.perf_counter()
        idx, conf = p.predict(imgs)
        ms = (time.perf_counter() - t0) * 1000
        print(f"\n== {name} ==  ({p.dataset}, {p.resolution}x{p.resolution}, {ms:.1f} ms for {len(imgs)} images)")
        for i in range(len(imgs)):
            pred = p.classes[idx[i, 0]]
            if labels[i] is None:
                print(f"  {ids[i]:<12} pred: {pred:<15} conf {conf[i, 0]:.3f}")
            else:
                true = p.classes[labels[i]]
                mark = "ok" if idx[i, 0] == labels[i] else "WRONG"
                print(f"  {ids[i]:<12} pred: {pred:<15} conf {conf[i, 0]:.3f}  true: {true:<15} {mark}")


if __name__ == "__main__":
    main()
