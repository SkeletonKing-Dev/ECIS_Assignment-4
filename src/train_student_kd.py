"""
Q1 step 3: train the three students WITH knowledge distillation from the
ResNet-50 teacher (soft targets + hard labels + intermediate features).
Uses exactly the same data, epochs, optimizer and schedule as the baselines.

Usage:
    python src/train_student_kd.py --arch all --epochs 30 --T 4 --alpha 0.9 --beta 2 --mode both
    python src/train_student_kd.py --arch squeezenet --mode logit       # soft-target KD only
"""
import argparse

import common as C
from kd import Distiller
from networks import build_model


def train_one(arch, teacher, args, device, train, test, classes):
    C.set_seed(42)
    student = build_model(arch, len(classes))
    distiller = Distiller(student, teacher, args.T, args.alpha, args.beta, args.mode).to(device)
    n_adapt = sum(p.numel() for p in distiller.adapters.parameters())
    print(f"\n=== KD {arch}: {sum(p.numel() for p in student.parameters()):,} student parameters, "
          f"{n_adapt:,} adapter parameters (training only) ===")
    print(f"    T={args.T}  alpha={distiller.alpha}  beta={distiller.beta}  mode={args.mode}")

    acc = C.fit(distiller, train, test, device, args.epochs, args.lr, f"{arch} KD",
                loss_fn=lambda m, x, y: m.kd_loss(x, y), params=distiller.trainable_parameters(),
                epoch_hook=distiller.epoch_summary)

    path = C.student_path(arch, "kd")
    C.save_checkpoint(path, student, arch, "cifar100", classes, test_acc=acc, epochs=args.epochs,
                      training="knowledge distillation",
                      kd={"T": args.T, "alpha": distiller.alpha, "beta": distiller.beta, "mode": args.mode})
    print(f"{arch} KD final test accuracy: {acc:.2f}%  -> {path}  (adapters discarded)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="all", choices=C.STUDENTS + ["all"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--T", type=float, default=4.0, help="softmax temperature")
    ap.add_argument("--alpha", type=float, default=0.9, help="weight of the soft-target loss")
    ap.add_argument("--beta", type=float, default=2.0, help="weight of the feature loss")
    ap.add_argument("--mode", default="both", choices=["logit", "feature", "both"])
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()

    C.ensure_dirs()
    device = C.get_device()
    print(f"Device: {device}")
    train, test, classes = C.get_loaders("cifar100", args.batch_size, device, args.fake)

    teacher, ckpt = C.load_checkpoint(C.TEACHER_PATH)
    teacher = teacher.to(device).eval()
    print(f"Teacher loaded: test accuracy {C.evaluate(teacher, test):.2f}%")

    for arch in (C.STUDENTS if args.arch == "all" else [args.arch]):
        train_one(arch, teacher, args, device, train, test, classes)


if __name__ == "__main__":
    main()
