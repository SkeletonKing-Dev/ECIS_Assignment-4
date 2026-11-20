"""
Q1 step 1: train the ResNet-50 teacher on CIFAR-100.

Usage:
    python src/train_teacher.py --epochs 30
"""
import argparse

import common as C
from networks import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--fake", action="store_true", help="random data, for smoke tests only")
    args = ap.parse_args()

    C.set_seed(42)
    C.ensure_dirs()
    device = C.get_device()
    print(f"Device: {device}")
    train, test, classes = C.get_loaders("cifar100", args.batch_size, device, args.fake)

    model = build_model("resnet50", len(classes)).to(device)
    print(f"Teacher ResNet-50: {sum(p.numel() for p in model.parameters()):,} parameters")
    acc = C.fit(model, train, test, device, args.epochs, args.lr, "teacher")

    C.save_checkpoint(C.TEACHER_PATH, model, "resnet50", "cifar100", classes,
                      test_acc=acc, epochs=args.epochs)
    print(f"Final test accuracy: {acc:.2f}%  -> saved to {C.TEACHER_PATH}")


if __name__ == "__main__":
    main()
