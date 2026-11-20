"""
Automated benchmark of every trained model.

For each model (teacher, 3 baseline students, 3 distilled students, NAS model):
  * top-1 accuracy on the full test set of its dataset (CIFAR-100 / CIFAR-10)
  * parameters, MACs, checkpoint size, peak inference memory
  * CPU latency (batch 1, mean and p95) and throughput (batch 32), 1 thread
  * the same accuracy / latency / throughput with ONNX Runtime (exported models)
Q1 analysis: distillation gain per student, agreement with the teacher and a
McNemar test (is the KD student significantly better than its baseline?).

Outputs (results/):
  benchmark_kd.csv            teacher + students, PyTorch
  kd_summary.csv              per student: baseline vs KD
  benchmark_nas.csv           NAS model vs its constraints
  benchmark_onnx.csv          ONNX Runtime results
  benchmark_all.csv           all rows
  predictions_cifar100.csv    per-image predictions of the CIFAR-100 models
  predictions_cifar10_nas.csv per-image predictions of the NAS model
  benchmark_env.txt           CPU, threads, versions
  kd_vs_baseline.png, accuracy_vs_latency.png, latency_pytorch_vs_onnx.png

Usage:
    python src/benchmark.py --threads 1
"""
import argparse
import json
import math
import os
import platform
import time

import numpy as np
import pandas as pd
import torch

import common as C
from hw_metrics import latency_ms, measure_model

PRETTY = {"teacher_resnet50": "ResNet-50 teacher", "nas_best": "NAS best"}
for a, n in (("mobilenetv2", "MobileNetV2"), ("squeezenet", "SqueezeNet"), ("efficientnet_lite0", "EfficientNet-Lite0")):
    PRETTY[f"{a}_baseline"], PRETTY[f"{a}_kd"] = f"{n} baseline", f"{n} KD"


def cpu_name():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def mcnemar(base_ok, new_ok):
    """Paired test on the same images. n01: base right / new wrong, n10: base wrong / new right."""
    n01 = int(((base_ok == 1) & (new_ok == 0)).sum())
    n10 = int(((base_ok == 0) & (new_ok == 1)).sum())
    if n01 + n10 == 0:
        return n01, n10, 1.0
    chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    return n01, n10, math.erfc(math.sqrt(chi2 / 2))


@torch.no_grad()
def predict_all(fn, x, batch=500):
    return np.concatenate([np.asarray(fn(x[i:i + batch])).argmax(1) for i in range(0, len(x), batch)])


@torch.no_grad()
def throughput(fn, x32, runs=5):
    fn(x32)
    t0 = time.perf_counter()
    for _ in range(runs):
        fn(x32)
    return runs * len(x32) / (time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=1, help="CPU threads for latency (1 = edge-like)")
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--no-onnx", action="store_true")
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()

    C.ensure_dirs()
    device = C.get_device()
    cat = C.model_catalog()
    print(f"CPU: {cpu_name()}   latency threads: {args.threads}   accuracy pass on: {device}")

    data = {}
    rows, onnx_rows, preds = [], [], {"cifar100": {}, "cifar10": {}}
    labels = {}
    for name, path in cat.items():
        model, ckpt = C.load_checkpoint(path)
        ds, res = ckpt["dataset"], ckpt.get("resolution") or 32
        if ds not in data:
            x, y, _ = C.load_arrays(ds, False, args.fake)
            data[ds] = C.preprocess_numpy(x, ds), y          # 32x32 normalised test set
            labels[ds] = y
        x_test, y_test = data[ds]
        x_in = C.resize(x_test, res) if res != 32 else x_test

        # accuracy (on GPU if available; FP32 accuracy does not depend on the device)
        m_acc = model.to(device)
        p = predict_all(lambda b: m_acc(b.to(device)).float().cpu().numpy(), x_in)
        acc = 100 * (p == y_test).mean()
        preds[ds][name] = p

        # hardware metrics on CPU
        torch.set_num_threads(args.threads)
        hw = measure_model(model.cpu(), (3, res, res), args.threads, args.runs)
        thr = throughput(lambda b: model(b), x_in[:32])
        torch.set_num_threads(os.cpu_count() or 1)
        row = {"model": PRETTY.get(name, name), "key": name, "dataset": ds, "resolution": res,
               "runtime": "PyTorch", "accuracy_%": acc, "params": hw["params"], "macs_M": hw["macs_M"],
               "size_mb": C.file_size_mb(path), "peak_mem_mb": hw["peak_mem_mb"],
               "latency_b1_ms": hw["latency_ms"], "latency_b1_p95_ms": hw["latency_p95_ms"],
               "throughput_img_s": thr}
        rows.append(row)
        print(f"  {name:<28} acc {acc:6.2f}%  lat {hw['latency_ms']:7.2f} ms  size {row['size_mb']:6.2f} MB  "
              f"mem {hw['peak_mem_mb']:6.1f} MB", flush=True)

        # ONNX Runtime
        op = C.onnx_path(name)
        if not args.no_onnx and os.path.exists(op):
            from onnx_infer import OnnxPredictor
            fast = OnnxPredictor(op)                                   # all threads, accuracy pass
            po = predict_all(lambda b: fast.logits(b.numpy()), x_in)
            one = OnnxPredictor(op, threads=args.threads)              # latency, edge-like
            x1 = x_in[:1].numpy()
            lat, p95 = latency_ms(one.logits, x1, runs=args.runs)
            thr_o = throughput(one.logits, x_in[:32].numpy())
            preds[ds][f"onnx_{name}"] = po
            onnx_rows.append({"model": PRETTY.get(name, name), "key": name, "dataset": ds, "resolution": res,
                              "runtime": "ONNX Runtime", "accuracy_%": 100 * (po == y_test).mean(),
                              "size_mb": C.file_size_mb(op), "latency_b1_ms": lat, "latency_b1_p95_ms": p95,
                              "throughput_img_s": thr_o,
                              "agreement_with_pytorch_%": 100 * (po == p).mean()})
            print(f"  {'  ONNX':<28} acc {onnx_rows[-1]['accuracy_%']:6.2f}%  lat {lat:7.2f} ms  "
                  f"agreement with PyTorch {onnx_rows[-1]['agreement_with_pytorch_%']:.2f}%", flush=True)

    df = pd.DataFrame(rows)
    kd = df[df.dataset == "cifar100"].copy()
    if "teacher_resnet50" in set(kd.key):
        t = kd[kd.key == "teacher_resnet50"].iloc[0]
        kd["compression_vs_teacher_x"] = t["params"] / kd["params"]
        kd["speedup_vs_teacher_x"] = t["latency_b1_ms"] / kd["latency_b1_ms"]
    kd.to_csv(os.path.join(C.RESULT_DIR, "benchmark_kd.csv"), index=False, float_format="%.4f")

    # Q1 summary: baseline vs KD for each student
    summary = []
    pc = preds["cifar100"]
    y100 = labels.get("cifar100")
    for arch in C.STUDENTS:
        b, k = f"{arch}_baseline", f"{arch}_kd"
        if b not in pc or k not in pc:
            continue
        ok_b, ok_k = (pc[b] == y100).astype(int), (pc[k] == y100).astype(int)
        n01, n10, pv = mcnemar(ok_b, ok_k)
        r = {"student": PRETTY[b].replace(" baseline", ""), "arch": arch, "params": int(df[df.key == b].params.iloc[0]),
             "baseline_acc_%": 100 * ok_b.mean(), "kd_acc_%": 100 * ok_k.mean(),
             "kd_gain_pts": 100 * (ok_k.mean() - ok_b.mean()),
             "baseline_wrong_kd_right": n10, "baseline_right_kd_wrong": n01, "mcnemar_p": pv,
             "significant_p<0.05": "yes" if pv < 0.05 else "no"}
        if "teacher_resnet50" in pc:
            r["baseline_agree_teacher_%"] = 100 * (pc[b] == pc["teacher_resnet50"]).mean()
            r["kd_agree_teacher_%"] = 100 * (pc[k] == pc["teacher_resnet50"]).mean()
        summary.append(r)
    sdf = pd.DataFrame(summary)
    sdf.to_csv(os.path.join(C.RESULT_DIR, "kd_summary.csv"), index=False, float_format="%.4f")

    nas = df[df.dataset == "cifar10"].copy()
    if len(nas) and os.path.exists(C.NAS_BEST_JSON):
        cons = json.load(open(C.NAS_BEST_JSON))["constraints"]
        nas["max_latency_ms"], nas["max_size_mb"], nas["max_mem_mb"] = (
            cons["max_latency_ms"], cons["max_size_mb"], cons["max_mem_mb"])
        # size constraint applies to the FP32 weights (state_dict), as in the search
        best = torch.load(C.NAS_BEST_PATH, weights_only=False)
        nas["weights_size_mb"] = best["hardware"]["size_mb"]
        nas["meets_constraints"] = ((nas.latency_b1_ms <= nas.max_latency_ms) &
                                    (nas.weights_size_mb <= nas.max_size_mb) &
                                    (nas.peak_mem_mb <= nas.max_mem_mb)).map({True: "yes", False: "no"})
    nas.to_csv(os.path.join(C.RESULT_DIR, "benchmark_nas.csv"), index=False, float_format="%.4f")

    odf = pd.DataFrame(onnx_rows)
    if len(odf):
        odf.to_csv(os.path.join(C.RESULT_DIR, "benchmark_onnx.csv"), index=False, float_format="%.4f")
    all_df = pd.concat([df, odf], ignore_index=True)
    for col in ("params",):
        all_df[col] = all_df[col].astype("Int64")
    all_df.to_csv(os.path.join(C.RESULT_DIR, "benchmark_all.csv"), index=False, float_format="%.4f")

    for ds, fname in (("cifar100", "predictions_cifar100.csv"), ("cifar10", "predictions_cifar10_nas.csv")):
        if preds[ds]:
            out = pd.DataFrame({"image_index": np.arange(len(labels[ds])), "true_label": labels[ds]})
            for k, v in preds[ds].items():
                out[f"pred_{k}"] = v
            out.to_csv(os.path.join(C.RESULT_DIR, fname), index=False)

    with open(os.path.join(C.RESULT_DIR, "benchmark_env.txt"), "w") as f:
        f.write(f"cpu: {cpu_name()}\nlatency_threads: {args.threads}\naccuracy_device: {device}\n"
                f"torch: {torch.__version__}\npython: {platform.python_version()}\nos: {platform.platform()}\n")
    make_plots(df, odf, sdf)

    pd.set_option("display.width", 220)
    cols = ["model", "runtime", "dataset", "accuracy_%", "params", "macs_M", "size_mb", "peak_mem_mb",
            "latency_b1_ms", "throughput_img_s"]
    print("\n" + all_df[[c for c in cols if c in all_df]].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    if len(sdf):
        print("\nKnowledge distillation summary:")
        print(sdf.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nCSV files and plots written to {C.RESULT_DIR}/")


def make_plots(df, odf, sdf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(sdf):
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        x = np.arange(len(sdf))
        b1 = ax.bar(x - 0.2, sdf["baseline_acc_%"], 0.4, label="baseline (no KD)", color="#8C8C8C")
        b2 = ax.bar(x + 0.2, sdf["kd_acc_%"], 0.4, label="knowledge distillation", color="#4C72B0")
        t = df[df.key == "teacher_resnet50"]
        if len(t):
            ax.axhline(t["accuracy_%"].iloc[0], ls="--", c="C3", label="ResNet-50 teacher")
        for bars in (b1, b2):
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.2f}",
                        ha="center", va="bottom", fontsize=8)
        lo = min(sdf["baseline_acc_%"].min(), sdf["kd_acc_%"].min())
        hi = max(sdf["kd_acc_%"].max(), t["accuracy_%"].iloc[0] if len(t) else 0)
        ax.set_ylim(max(0, lo - 5), hi + 3)
        ax.set_xticks(x, sdf["student"])
        ax.tick_params(axis="x", pad=2)
        ax.set_ylabel("CIFAR-100 test accuracy (%)")
        ax.set_title("Students: baseline vs knowledge distillation")
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(C.RESULT_DIR, "kd_vs_baseline.png"), dpi=130)
        plt.close()

    c100 = df[df.dataset == "cifar100"]
    if len(c100):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for _, r in c100.iterrows():
            ax.scatter(r["latency_b1_ms"], r["accuracy_%"], s=20 + r["params"] / 1e5,
                       c="C3" if "teacher" in r.key else ("C0" if r.key.endswith("_kd") else "C7"), alpha=0.8)
            ax.annotate(r["model"], (r["latency_b1_ms"], r["accuracy_%"]), textcoords="offset points",
                        xytext=(5, 4), fontsize=7)
        ax.set_xscale("log")
        ax.set_xlabel("CPU latency, batch 1, 1 thread (ms, log scale)")
        ax.set_ylabel("CIFAR-100 test accuracy (%)")
        ax.set_title("Accuracy vs latency (marker size = parameters)")
        ax.grid(alpha=0.3, which="both")
        plt.tight_layout()
        plt.savefig(os.path.join(C.RESULT_DIR, "accuracy_vs_latency.png"), dpi=130)
        plt.close()

    if len(odf):
        keys = list(odf.key)
        pt = [float(df.loc[df.key == k, "latency_b1_ms"].iloc[0]) for k in keys]
        ox = list(odf.latency_b1_ms)
        x = np.arange(len(keys))
        fig, ax = plt.subplots(figsize=(10, 4.2))
        ax.bar(x - 0.2, pt, 0.4, label="PyTorch", color="#4C72B0")
        ax.bar(x + 0.2, ox, 0.4, label="ONNX Runtime", color="#DD8452")
        ax.set_xticks(x, [PRETTY.get(k, k).replace(" ", "\n", 1) for k in keys], fontsize=8)
        ax.set_ylabel("ms / image (batch 1, 1 thread)")
        ax.set_title("CPU latency: PyTorch vs ONNX Runtime")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(C.RESULT_DIR, "latency_pytorch_vs_onnx.png"), dpi=130)
        plt.close()


if __name__ == "__main__":
    main()
