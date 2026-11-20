"""
Q1 step 2: baseline training of the students WITHOUT knowledge distillation
(plain cross-entropy on the hard labels). Same data, epochs, optimizer and
schedule as the distilled students, so the only difference is the teacher.

Usage:
    python src/train_student_baseline.py --arch all --epochs 30
    python src/train_student_baseline.py --arch mobilenetv2
"""
import argparse

import common as C
from networks import build_model


def train_one(arch, args, device, train, test, classes):
    C.set_seed(42)
    model = build_model(arch, len(classes)).to(device)
    print(f"\n=== Baseline {arch}: {sum(p.numel() for p in model.parameters()):,} parameters ===")
    acc = C.fit(model, train, test, device, args.epochs, args.lr, f"{arch} baseline")
    path = C.student_path(arch, "baseline")
    C.save_checkpoint(path, model, arch, "cifar100", classes, test_acc=acc, epochs=args.epochs,
                      training="baseline (cross-entropy only)")
    print(f"{arch} baseline final test accuracy: {acc:.2f}%  -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="all", choices=C.STUDENTS + ["all"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()

    C.ensure_dirs()
    device = C.get_device()
    print(f"Device: {device}")
    train, test, classes = C.get_loaders("cifar100", args.batch_size, device, args.fake)
    for arch in (C.STUDENTS if args.arch == "all" else [args.arch]):
        train_one(arch, args, device, train, test, classes)


if __name__ == "__main__":
    main()
