"""
Q2: hardware-aware evolutionary architecture search on CIFAR-10.

Algorithm: regularized (aging) evolution, Real et al. 2019.
  1. Fill a population of P architectures with random samples.
  2. Repeat until the training budget is used:
       - pick S random members of the population (tournament)
       - the one with the best validation accuracy is the parent
       - child = parent with ONE gene mutated
       - evaluate the child, add it to the population, remove the OLDEST member
     Removing the oldest (not the worst) member keeps the search exploring and
     stops one lucky early architecture from dominating.

Hardware-aware evaluation of every candidate (before any training):
  - CPU latency, batch 1, 1 thread, at the candidate's own input resolution
  - storage size of the FP32 weights
  - peak inference memory (PyTorch profiler)
  Candidates that violate a deployment constraint are REJECTED and never
  trained (saves search time); a new candidate is drawn instead.

Feasible candidates get a short proxy training on a subset of the CIFAR-10
training set and are scored on a held-out VALIDATION split (5,000 training
images); the test set is never used during the search.

Outputs:
  results/nas_history.csv / .json   every candidate (trained and rejected)
  results/nas_best_arch.json        best feasible architecture + its metrics

Usage:
    python src/nas_search.py --budget 30 --population 10 --sample 3 --proxy-epochs 4
    python src/nas_search.py --max-latency-ms 3 --max-size-mb 2 --max-mem-mb 6
"""
import argparse
import collections
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch

import common as C
import nas_space as S
from hw_metrics import measure_model


class Evaluator:
    def __init__(self, args, device):
        self.args, self.device = args, device
        x, y, self.classes = C.load_arrays("cifar10", True, args.fake)
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(y))
        n_val = min(5000, len(y) // 10)
        val_idx, tr_idx = idx[:n_val], idx[n_val:]
        tr_idx = tr_idx[:min(args.proxy_train_images, len(tr_idx))]
        self.train_xy = (x[tr_idx], y[tr_idx])
        self.val_xy = (x[val_idx], y[val_idx])
        self.loaders = {}
        print(f"Proxy training set: {len(tr_idx)} images, validation set: {n_val} images (held out from train)")

    def get_loaders(self, res):
        if res not in self.loaders:           # one on-device loader pair per resolution
            self.loaders[res] = (
                C.DeviceLoader(*self.train_xy, "cifar10", self.args.batch_size, True, self.device, res),
                C.DeviceLoader(*self.val_xy, "cifar10", 1000, False, self.device, res))
        return self.loaders[res]

    def hardware_check(self, config):
        model = S.build_from_config(config, 10)
        r = config["resolution"]
        hw = measure_model(model, (3, r, r), threads=1, runs=self.args.latency_runs)
        reasons = []
        if hw["latency_ms"] > self.args.max_latency_ms:
            reasons.append("latency")
        if hw["size_mb"] > self.args.max_size_mb:
            reasons.append("size")
        if hw["peak_mem_mb"] > self.args.max_mem_mb:
            reasons.append("memory")
        return hw, reasons

    def train_and_validate(self, config):
        torch.manual_seed(0)
        model = S.build_from_config(config, 10).to(self.device)
        train, val = self.get_loaders(config["resolution"])
        opt, sched = C.make_optimizer(model, self.args.lr, self.args.proxy_epochs, len(train))
        scaler = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda")
        for _ in range(self.args.proxy_epochs):
            C.train_epoch(model, train, opt, sched, self.device, scaler)
        return C.evaluate(model, val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=30, help="number of architectures to train")
    ap.add_argument("--population", type=int, default=10)
    ap.add_argument("--sample", type=int, default=3, help="tournament size")
    ap.add_argument("--proxy-epochs", type=int, default=4)
    ap.add_argument("--proxy-train-images", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--max-latency-ms", type=float, default=3.0, help="CPU, batch 1, 1 thread")
    ap.add_argument("--max-size-mb", type=float, default=2.0, help="FP32 weights on disk")
    ap.add_argument("--max-mem-mb", type=float, default=6.0, help="peak inference memory")
    ap.add_argument("--latency-runs", type=int, default=30)
    ap.add_argument("--max-tries", type=int, default=50, help="rejections allowed per new candidate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()

    C.ensure_dirs()
    C.set_seed(args.seed)
    rng = random.Random(args.seed)
    device = C.get_device()
    print(f"Device: {device}   search space: {S.space_size():.2e} architectures")
    print(f"Constraints: latency <= {args.max_latency_ms} ms, size <= {args.max_size_mb} MB, "
          f"peak memory <= {args.max_mem_mb} MB")
    ev = Evaluator(args, device)

    history, cache = [], {}
    t_start = time.time()

    def evaluate(config, origin, parent_id=None):
        """Hardware check, then (if feasible) proxy training. Returns the record."""
        key = S.config_key(config)
        rec = {"id": len(history), "origin": origin, "parent_id": parent_id,
               "arch": S.describe(config), "config": json.dumps(S.active_config(config))}
        if key in cache:                                         # already seen: reuse result
            rec.update({k: v for k, v in cache[key].items() if k not in rec})
            rec["status"] = cache[key]["status"] + " (duplicate)"
            history.append(rec)
            return rec
        t0 = time.time()
        hw, reasons = ev.hardware_check(config)
        rec.update({k: hw[k] for k in ("params", "macs_M", "size_mb", "latency_ms", "peak_mem_mb", "mem_method")})
        rec["resolution"] = config["resolution"]
        if reasons:
            rec.update(status="rejected", reject_reason="+".join(reasons), val_acc=np.nan)
        else:
            rec.update(status="trained", reject_reason="", val_acc=ev.train_and_validate(config))
        rec["eval_time_s"] = time.time() - t0
        cache[key] = {k: rec[k] for k in rec if k not in ("id", "origin", "parent_id")}
        history.append(rec)
        n_tr = sum(r["status"] == "trained" for r in history)
        if rec["status"] == "trained":
            print(f"#{rec['id']:3d} [{n_tr:2d}/{args.budget}] {origin:<9} val {rec['val_acc']:5.2f}%  "
                  f"lat {rec['latency_ms']:.2f}ms  size {rec['size_mb']:.2f}MB  mem {rec['peak_mem_mb']:.1f}MB  "
                  f"{rec['arch']}", flush=True)
        else:
            print(f"#{rec['id']:3d} rejected ({rec['reject_reason']}): lat {rec['latency_ms']:.2f}ms  "
                  f"size {rec['size_mb']:.2f}MB  mem {rec['peak_mem_mb']:.1f}MB", flush=True)
        return rec

    def new_candidate(make, origin, parent_id=None):
        """Draws candidates until one passes the hardware constraints."""
        for _ in range(args.max_tries):
            config, note = make()
            rec = evaluate(config, origin, parent_id)
            if note:
                history[-1]["mutation"] = note
            if rec["status"].startswith("trained"):
                return config, rec
        raise RuntimeError("too many rejected candidates: relax the constraints")

    # 1) random initial population
    population = collections.deque()
    print(f"\n--- Initial population ({args.population} random feasible architectures) ---")
    while len(population) < args.population:
        cfg, rec = new_candidate(lambda: (S.sample_config(rng), ""), "random")
        population.append((cfg, rec))

    # 2) aging evolution
    print(f"\n--- Evolution (tournament size {args.sample}) ---")
    while sum(r["status"] == "trained" for r in history) < args.budget:
        tournament = rng.sample(list(population), min(args.sample, len(population)))
        parent_cfg, parent_rec = max(tournament, key=lambda t: t[1]["val_acc"])
        cfg, rec = new_candidate(lambda: S.mutate(parent_cfg, rng), "mutation", parent_rec["id"])
        population.append((cfg, rec))
        population.popleft()                                    # remove the oldest

    # 3) results
    df = pd.DataFrame(history)
    df["parent_id"] = df["parent_id"].astype("Int64")       # 6 instead of 6.0 in the CSV
    df["best_so_far"] = df["val_acc"].where(df.status == "trained").cummax()
    df.to_csv(os.path.join(C.RESULT_DIR, "nas_history.csv"), index=False, float_format="%.4f")
    with open(os.path.join(C.RESULT_DIR, "nas_history.json"), "w") as f:
        json.dump(json.loads(df.to_json(orient="records")), f, indent=1)

    trained = df[df.status == "trained"]
    best = trained.loc[trained.val_acc.idxmax()]
    best_info = {
        "config": json.loads(best["config"]), "arch": best["arch"], "search_id": int(best["id"]),
        "val_acc_proxy": float(best["val_acc"]),
        "metrics": {k: float(best[k]) for k in ("params", "macs_M", "size_mb", "latency_ms", "peak_mem_mb")},
        "constraints": {"max_latency_ms": args.max_latency_ms, "max_size_mb": args.max_size_mb,
                        "max_mem_mb": args.max_mem_mb},
        "search": {"budget_trained": args.budget, "candidates_total": len(df),
                   "rejected": int((df.status == "rejected").sum()),
                   "population": args.population, "tournament": args.sample,
                   "proxy_epochs": args.proxy_epochs, "proxy_train_images": args.proxy_train_images,
                   "search_time_min": (time.time() - t_start) / 60, "space_size": S.space_size()},
    }
    with open(C.NAS_BEST_JSON, "w") as f:
        json.dump(best_info, f, indent=2)
    make_plots(df, args)

    print(f"\nSearch finished in {(time.time() - t_start) / 60:.1f} min: {len(df)} candidates, "
          f"{(df.status == 'trained').sum()} trained, {(df.status == 'rejected').sum()} rejected")
    print("Rejections by reason:", {k: int(v) for k, v in df[df.status == "rejected"].reject_reason.value_counts().items()})
    print(f"Best architecture (#{best_info['search_id']}): val {best['val_acc']:.2f}%  "
          f"lat {best['latency_ms']:.2f} ms  size {best['size_mb']:.2f} MB  mem {best['peak_mem_mb']:.1f} MB")
    print(f"  {best['arch']}")
    print(f"Saved -> {C.NAS_BEST_JSON}, results/nas_history.csv, results/nas_history.json")


def make_plots(df, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    tr, rj = df[df.status == "trained"], df[df.status == "rejected"]
    ax[0].scatter(tr.latency_ms, tr.val_acc, c=tr.id, cmap="viridis", s=35, label="trained (color = order)")
    ymin = tr.val_acc.min() if len(tr) else 0
    ax[0].scatter(rj.latency_ms, [ymin - 1] * len(rj), marker="x", c="red", s=25, label="rejected (not trained)")
    ax[0].axvline(args.max_latency_ms, ls="--", c="gray", label="latency limit")
    b = tr.loc[tr.val_acc.idxmax()]
    ax[0].scatter([b.latency_ms], [b.val_acc], s=160, facecolors="none", edgecolors="red", label="best")
    ax[0].set_xlabel("CPU latency, batch 1 (ms)")
    ax[0].set_ylabel("validation accuracy, proxy training (%)")
    ax[0].set_title("NAS candidates")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[1].plot(tr.id, tr.val_acc, "o", ms=4, alpha=0.6, label="candidate")
    ax[1].plot(tr.id, tr.best_so_far, "-", c="C3", label="best so far")
    ax[1].set_xlabel("candidate id (search order)")
    ax[1].set_ylabel("validation accuracy (%)")
    ax[1].set_title("Search progress")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(C.RESULT_DIR, "nas_search.png"), dpi=130)
    plt.close()


if __name__ == "__main__":
    main()
