"""
Q2: train the best discovered architecture FROM SCRATCH (new random weights)
on the full CIFAR-10 training set, re-check the deployment constraints and
save the checkpoint.

Usage:
    python src/nas_train_best.py --epochs 30
"""
import argparse
import json

import common as C
import nas_space as S
from hw_metrics import measure_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()

    C.set_seed(42)
    C.ensure_dirs()
    device = C.get_device()
    best = json.load(open(C.NAS_BEST_JSON))
    cfg = best["config"]
    res = cfg["resolution"]
    print(f"Best architecture from search #{best['search_id']} (proxy val acc {best['val_acc_proxy']:.2f}%):")
    print(f"  {S.describe(cfg)}")

    train, test, classes = C.get_loaders("cifar10", args.batch_size, device, args.fake, resolution=res)
    model = S.build_from_config(cfg, len(classes)).to(device)      # fresh initialisation
    print(f"Training from scratch at {res}x{res}: {sum(p.numel() for p in model.parameters()):,} parameters")
    acc = C.fit(model, train, test, device, args.epochs, args.lr, "NAS best")

    hw = measure_model(model, (3, res, res), threads=1, runs=100)
    cons = best["constraints"]
    ok = (hw["latency_ms"] <= cons["max_latency_ms"] and hw["size_mb"] <= cons["max_size_mb"]
          and hw["peak_mem_mb"] <= cons["max_mem_mb"])
    print(f"Deployment check: latency {hw['latency_ms']:.2f} ms, size {hw['size_mb']:.2f} MB, "
          f"peak memory {hw['peak_mem_mb']:.1f} MB -> {'OK' if ok else 'VIOLATES CONSTRAINTS (re-measured)'}")

    C.save_checkpoint(C.NAS_BEST_PATH, model.cpu(), "nas", "cifar10", classes, test_acc=acc,
                      epochs=args.epochs, resolution=res, hardware=hw, constraints_ok=ok)
    print(f"Final test accuracy: {acc:.2f}%  -> saved to {C.NAS_BEST_PATH}")


if __name__ == "__main__":
    main()
