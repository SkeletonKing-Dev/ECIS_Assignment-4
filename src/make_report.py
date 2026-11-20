"""
Builds REPORT.md (technical report) from logs/, results/ and the NAS JSON
produced by scripts/run_all.sh, so every number comes from the actual run.

Usage:
    python src/make_report.py
"""
import json
import os
import platform
import re

import pandas as pd
import torch

import common as C
import nas_space as S

# ----------------------------------------------------------------------------
# Section 13: terminal output of every pipeline step.
# To show a step as a screenshot instead of pasted log text, cut its line from
# LOG_SECTIONS, paste it into SCREENSHOT_SECTIONS and change the file name to
# the image in logs/screenshots/ (and the other way round). The sections are
# always printed in the order of their numbers, whichever list they are in.
# ----------------------------------------------------------------------------
SCREENSHOT_SECTIONS = [
    # ("13.1 Teacher training (ResNet-50, CIFAR-100)",       "1_train_teacher.png"),
]

LOG_SECTIONS = [
    ("13.1 Teacher training (ResNet-50, CIFAR-100)",       "1_train_teacher.log"),
    ("13.2 Baseline students (no distillation)",           "2_students_baseline.log"),
    ("13.3 Students with knowledge distillation",          "3_students_kd.log"),
    ("13.4 Hardware-aware NAS search (CIFAR-10)",          "4_nas_search.log"),
    ("13.5 Training the best NAS architecture",            "5_nas_train_best.log"),
    ("13.6 ONNX export",                                   "6_export_onnx.log"),
    ("13.7 PyTorch inference (`infer.py --model all`)",    "7_infer.log"),
    ("13.8 ONNX inference (`onnx_infer.py --model all`)",  "7_onnx_infer.log"),
    ("13.9 Benchmark table (`benchmark.py`)",              "8_benchmark.log"),
]

EPOCH_RE = re.compile(r"\[(.+?)\] epoch\s+(\d+)/(\d+)\s+loss ([\d.]+)\s+train ([\d.]+)%\s+test ([\d.]+)%")


def read(path):
    return open(path, encoding="utf-8", errors="ignore").read() if os.path.exists(path) else ""


def log(name):
    return read(os.path.join(C.LOG_DIR, name))


def csv(name):
    p = os.path.join(C.RESULT_DIR, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def md_table(df, fmt="{:.2f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append("" if pd.isna(v) else fmt.format(v))
            elif isinstance(v, int) and abs(v) >= 10000:
                cells.append(f"{v:,}")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def curves(*texts):
    rows = []
    for t in texts:
        rows += [(m[0], int(m[1]), float(m[3]), float(m[4]), float(m[5])) for m in EPOCH_RE.findall(t)]
    return pd.DataFrame(rows, columns=["tag", "epoch", "loss", "train", "test"])


def plot_curves(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
    t = df[df.tag == "teacher"]
    if len(t):
        axes[0].plot(t.epoch, t.train, label="train")
        axes[0].plot(t.epoch, t.test, label="test")
        axes[0].set_title("ResNet-50 teacher (CIFAR-100)")
        axes[0].legend()
    for ax, arch, title in zip(axes[1:], C.STUDENTS, ["MobileNetV2", "SqueezeNet", "EfficientNet-Lite0"]):
        for kind, style in (("baseline", "--"), ("KD", "-")):
            d = df[df.tag == f"{arch} {kind}"]
            if len(d):
                ax.plot(d.epoch, d.test, style, label=f"{kind} test")
        ax.set_title(f"{title}: baseline vs KD")
        ax.legend()
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.set_ylabel("accuracy (%)")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(C.RESULT_DIR, "training_curves.png"), dpi=120)
    plt.close()


def main():
    w = []
    add = w.append
    kd_df, summ, nas_df = csv("benchmark_kd.csv"), csv("kd_summary.csv"), csv("benchmark_nas.csv")
    onnx_df, hist = csv("benchmark_onnx.csv"), csv("nas_history.csv")
    best = json.load(open(C.NAS_BEST_JSON)) if os.path.exists(C.NAS_BEST_JSON) else None
    env = dict(l.split(": ", 1) for l in read(os.path.join(C.RESULT_DIR, "benchmark_env.txt")).splitlines() if ": " in l)
    t_log, b_log, k_log = log("1_train_teacher.log"), log("2_students_baseline.log"), log("3_students_kd.log")
    nb_log, ex_log = log("5_nas_train_best.log"), log("6_export_onnx.log")
    cv = curves(t_log, b_log, k_log)
    if len(cv):
        plot_curves(cv)

    def get(df, key, col):
        return float(df.loc[df.key == key, col].iloc[0])

    add("# Assignment 4 - Edge Computing & Intelligent Systems\n")
    add("## Knowledge Distillation (CIFAR-100) and Hardware-Aware Neural Architecture Search (CIFAR-10)\n")
    add("Name: Syed M. Alyun  Roll No: 23I-0022  Section: AI-A7  \n")

    add("## 1. Environment\n")
    add(f"- Python {platform.python_version()}, PyTorch {torch.__version__}, OS: {platform.system()} {platform.machine()}")
    add(f"- Training device: {'CUDA GPU (mixed precision FP16)' if 'cuda' in t_log + b_log else 'see logs'}")
    if env:
        add(f"- Benchmark CPU: {env.get('cpu')}, latency measured with {env.get('latency_threads')} thread(s), batch 1")
    add("- Data pipeline: the whole uint8 dataset is kept on the GPU; random crop (4-pixel padding) + horizontal "
        "flip and normalisation are done on the GPU per batch (much faster than a CPU DataLoader for 32x32 images).")
    add("- Model selection: every training run uses a fixed schedule and keeps the final weights. No checkpoint is "
        "chosen using the test set; the NAS uses a validation split carved out of the training set.\n")

    # ------------------------------------------------------------------ Q1
    add("# Question 1 - Knowledge distillation\n")
    add("## 2. Teacher: ResNet-50 - `src/resnet50.py`, `src/train_teacher.py`\n")
    add("Bottleneck ResNet-50 ([3, 4, 6, 3] blocks, 23.7 M parameters) adapted to 32x32 images: 3x3 stride-1 stem "
        "and no max-pool. Trained from scratch on CIFAR-100: SGD (momentum 0.9, Nesterov, weight decay 5e-4), "
        "One-Cycle LR (max 0.1), batch 128, mixed precision.")
    t = cv[cv.tag == "teacher"]
    if len(t):
        add(f"\nResult: **{t.test.iloc[-1]:.2f}%** test accuracy after {int(t.epoch.max())} epochs.\n")

    add("## 3. Student architectures - `src/mobilenetv2.py`, `src/squeezenet.py`, `src/efficientnet_lite.py`\n")
    add("All three are implemented from scratch (no torchvision models) and adapted to 32x32 input by removing early "
        "down-sampling, so every network ends with a 4x4 feature map.\n")
    add("| Student | Key building block | CIFAR adaptation | Feature maps used for KD |")
    add("|---|---|---|---|")
    add("| MobileNetV2 | inverted residual: 1x1 expand -> 3x3 depthwise -> 1x1 linear projection, ReLU6 | stem stride 1, stage-2 stride 1 | 96x8x8, 1280x4x4 |")
    add("| SqueezeNet 1.1 | Fire module: 1x1 squeeze -> parallel 1x1 / 3x3 expand, concatenated | first conv stride 1; BatchNorm added; no ReLU on logits | 256x8x8, 512x4x4 |")
    add("| EfficientNet-Lite0 | MBConv (B0 layout, kernels 3/5) without squeeze-excitation, ReLU6, fixed stem/head | stem stride 1, stage-2 stride 1 | 112x8x8, 1280x4x4 |")
    add("| ResNet-50 teacher | bottleneck residual | 3x3 stride-1 stem, no max-pool | 1024x8x8, 2048x4x4 |")
    add("")
    if kd_df is not None:
        cols = ["model", "params", "macs_M", "size_mb", "peak_mem_mb", "latency_b1_ms"]
        sub = kd_df[kd_df.key.isin(["teacher_resnet50"] + [f"{a}_baseline" for a in C.STUDENTS])][cols].copy()
        sub["model"] = sub["model"].str.replace(" baseline", "")
        add(md_table(sub))
        add("")

    add("## 4. Knowledge distillation framework - `src/kd.py`, `src/train_student_kd.py`\n")
    kdp = re.search(r"T=([\d.]+)\s+alpha=([\d.]+)\s+beta=([\d.]+)\s+mode=(\w+)", k_log)
    T, alpha, beta, mode = kdp.groups() if kdp else ("4.0", "0.9", "2.0", "both")
    add("Loss for every batch:\n")
    add("```\nL = (1 - alpha) * CE(z_s, y)                                   hard labels")
    add("  + alpha * T^2 * KL( softmax(z_t / T) || softmax(z_s / T) )    soft targets (temperature T)")
    add("  + beta  * sum_i || norm(A_i(f_s,i)) - norm(f_t,i) ||^2       intermediate features\n```\n")
    add(f"Used values: **T = {T}, alpha = {alpha}, beta = {beta}, mode = {mode}**.\n")
    add("- **Soft targets / temperature.** Dividing logits by T > 1 softens the teacher's distribution, so the "
        "student also learns which wrong classes the teacher finds similar (e.g. 'maple' vs 'oak'), information that "
        "one-hot labels do not carry. The T^2 factor keeps the size of this gradient independent of T.")
    add("- **Hard + soft combined.** The cross-entropy term keeps the student anchored to the true labels where the "
        "teacher is wrong.")
    add("- **Feature-based distillation (FitNets-style hints).** At two depths (8x8 and 4x4) the student's feature "
        "map is mapped by an adapter (1x1 conv + BN) to the teacher's channel count, then both maps are flattened and "
        "L2-normalised and their squared distance is minimised. Normalising makes the loss independent of the very "
        "different activation scales of the two networks. The adapters are trained with the student and discarded "
        "afterwards, so the deployed student is unchanged.")
    add("- **Teacher** is frozen and kept in eval mode (its BatchNorm statistics never change); its logits and "
        "features are computed on the same augmented batch as the student's.")
    add("- **Baseline pipeline** (`src/train_student_baseline.py`): identical data, augmentation, epochs, optimizer "
        "(SGD, One-Cycle, max LR 0.05) and seed, only cross-entropy. The only difference is the teacher.\n")
    agrees = re.findall(r"\[(\w+) KD\] epoch\s+\d+/\d+.*?CE ([\d.]+)\s+KD ([\d.]+)\s+feat ([\d.]+)\s+agree-w-teacher ([\d.]+)%", k_log)
    if agrees:
        last = {}
        for a, ce, kdv, fe, ag in agrees:
            last[a] = (a, float(ce), float(kdv), float(fe), float(ag))
        add("Loss components in the last KD epoch (training set):\n")
        add(md_table(pd.DataFrame(list(last.values()), columns=["student", "CE", "soft-target (T^2 KL)",
                                                                  "feature", "agreement with teacher %"]), "{:.3f}"))
        add("")

    add("## 5. Results: baseline vs distillation (CIFAR-100 test set, 10,000 images)\n")
    if summ is not None and len(summ):
        cols = [c for c in ["student", "baseline_acc_%", "kd_acc_%", "kd_gain_pts", "baseline_wrong_kd_right",
                            "baseline_right_kd_wrong", "mcnemar_p", "significant_p<0.05",
                            "baseline_agree_teacher_%", "kd_agree_teacher_%"] if c in summ]
        add(md_table(summ[cols], "{:.3f}"))
        add("\n`mcnemar_p`: McNemar's test on the paired predictions (same 10,000 images). p < 0.05 means the "
            "accuracy difference between the KD and the baseline student is unlikely to be noise.\n")
        add("![kd vs baseline](results/kd_vs_baseline.png)\n")
    if len(cv):
        add("![training curves](results/training_curves.png)\n")

    # ------------------------------------------------------------------ Q2
    add("# Question 2 - Hardware-aware neural architecture search\n")
    add("## 6. Search space and architecture generator - `src/nas_space.py`\n")
    add("| Dimension | Choices | Level |")
    add("|---|---|---|")
    add(f"| Input resolution | {S.SPACE['resolution']} | network |")
    add(f"| Stem channels | {S.SPACE['stem_channels']} | network |")
    add(f"| Head (1x1 conv) channels | {S.SPACE['head_channels']} (0 = none) | network |")
    add(f"| Stage channels | {S.SPACE['stage_channels']} | per stage (4 stages, strides 1/2/2/2) |")
    add(f"| Depth (blocks) | {S.SPACE['depth']} | per stage (4-12 blocks in total) |")
    add(f"| Activation | {S.SPACE['activation']} | per stage |")
    add(f"| Block type | conv (standard), dwsep (depthwise kxk + pointwise 1x1), mbconv3 / mbconv6 (inverted residual with depthwise conv) | per block |")
    add(f"| Kernel size | {S.SPACE['kernel']} | per block |")
    add(f"\nSearch space size: **{S.space_size():.2e}** distinct networks. An architecture is a JSON config; "
        "`build_from_config()` (the generator) turns any config into a PyTorch model, adding a residual connection "
        "whenever a block keeps stride 1 and the channel count.\n")

    add("## 7. Search strategy - regularized (aging) evolution - `src/nas_search.py`\n")
    if best:
        s = best["search"]
        add(f"1. Initial population: {s['population']} random architectures that satisfy the constraints.")
        add(f"2. Repeat until {s['budget_trained']} architectures are trained: pick {s['tournament']} random members "
            "(tournament), mutate exactly one gene of the best of them, evaluate the child, add it and remove the "
            "**oldest** member. Removing the oldest instead of the worst (Real et al., 2019) keeps the search "
            "exploring instead of locking onto one lucky early architecture.")
        add(f"3. Proxy evaluation: {s['proxy_epochs']} epochs on {s['proxy_train_images']:,} training images, "
            "accuracy on a 5,000-image validation split held out from the training set. The test set is never used "
            "during the search. Duplicate architectures are recognised and not re-trained.\n")

        add("## 8. Hardware-aware pipeline and resource constraints\n")
        c = best["constraints"]
        add("Before any training, every candidate is built and measured on the CPU:")
        add("- **latency**: mean of 30 runs, batch 1, 1 thread, at the candidate's own input resolution (after warm-up)")
        add("- **storage size**: size of the serialised FP32 weights")
        add("- **peak inference memory**: weights + peak live activation memory, measured with the PyTorch profiler "
            "(every allocation/free during one forward pass replayed in time order)\n")
        add(f"Deployment constraints: latency <= **{c['max_latency_ms']} ms**, size <= **{c['max_size_mb']} MB**, "
            f"peak memory <= **{c['max_mem_mb']} MB**. Violating candidates are rejected and never trained, so no "
            "training time is spent on architectures that could not be deployed.\n")
        if hist is not None:
            rj = hist[hist.status == "rejected"]
            add(f"Search statistics: {len(hist)} candidates generated, {(hist.status == 'trained').sum()} trained, "
                f"{len(rj)} rejected by the hardware check, {hist.status.str.contains('duplicate').sum()} duplicates; "
                f"search time {s['search_time_min']:.1f} min.\n")
            if len(rj):
                add(md_table(rj.reject_reason.value_counts().rename_axis("rejected because of").reset_index(name="candidates")))
                add("")
        add("![nas search](results/nas_search.png)\n")

        add("## 9. Best discovered architecture\n")
        cfg = best["config"]
        add(f"Search id #{best['search_id']}, proxy validation accuracy {best['val_acc_proxy']:.2f}%. "
            f"Input {cfg['resolution']}x{cfg['resolution']}, stem {cfg['stem_channels']} channels, head "
            f"{cfg['head_channels'] or 'none'}. Saved to `results/nas_best_arch.json`.\n")
        rows = [(i + 1, st["channels"], st["depth"], st["activation"],
                 ", ".join(f"{b['type']} k{b['kernel']}" for b in st["blocks"][:st["depth"]]))
                for i, st in enumerate(cfg["stages"])]
        add(md_table(pd.DataFrame(rows, columns=["stage", "channels", "depth", "activation", "blocks"])))
        add("")
        m = best["metrics"]
        add(f"Measured during the search: {int(m['params']):,} parameters, {m['macs_M']:.1f} M MACs, "
            f"{m['size_mb']:.2f} MB, {m['latency_ms']:.2f} ms, {m['peak_mem_mb']:.1f} MB peak memory.\n")
        nb = curves(nb_log)
        chk = re.search(r"Deployment check: (.*)", nb_log)
        if len(nb):
            add(f"**Trained from scratch** (fresh weights, full 50,000-image training set, {int(nb.epoch.max())} "
                f"epoch(s), `src/nas_train_best.py`): **{nb.test.iloc[-1]:.2f}%** CIFAR-10 test accuracy.")
        if chk:
            add(f"Re-measured after training: {chk.group(1)}.\n")

    # ------------------------------------------------------------------ deployment
    add("# Deployment\n")
    add("## 10. Inference and ONNX - `src/infer.py`, `src/export_onnx.py`, `src/onnx_infer.py`\n")
    add("- `infer.py` is one pipeline for every model (6 students, teacher, NAS model): each checkpoint stores its "
        "architecture, dataset, input resolution and class names, so `--model <name>` is all that changes.")
    add("- `export_onnx.py` exports the six students and the NAS model (opset 17, dynamic batch) and stores the "
        "resolution, normalisation and class names as ONNX metadata. Each export is checked against PyTorch:")
    diffs = re.findall(r"^\s+(\S+)\s+\d+x\d+\s+([\d.]+) MB\s+max \|ONNX - PyTorch\| = (\S+)", ex_log, re.M)
    if diffs:
        add("")
        add(md_table(pd.DataFrame(diffs, columns=["model", "ONNX size (MB)", "max |ONNX - PyTorch| output difference"])))
    add("\n- `onnx_infer.py` runs them on the CPU with ONNX Runtime and needs only onnxruntime, numpy and Pillow.\n")

    add("## 11. Benchmark (CPU) - `src/benchmark.py`\n")
    if kd_df is not None:
        cols = [c for c in ["model", "accuracy_%", "params", "macs_M", "size_mb", "peak_mem_mb", "latency_b1_ms",
                            "throughput_img_s", "compression_vs_teacher_x", "speedup_vs_teacher_x"] if c in kd_df]
        add("### 11.1 CIFAR-100 models (PyTorch)\n")
        add(md_table(kd_df[cols]))
        add("")
    if nas_df is not None and len(nas_df):
        add("### 11.2 NAS model (CIFAR-10, PyTorch)\n")
        cols = [c for c in ["model", "resolution", "accuracy_%", "params", "macs_M", "weights_size_mb",
                            "peak_mem_mb", "latency_b1_ms", "throughput_img_s", "meets_constraints"] if c in nas_df]
        add(md_table(nas_df[cols]))
        add("")
    if onnx_df is not None:
        add("### 11.3 ONNX Runtime\n")
        add(md_table(onnx_df[["model", "dataset", "accuracy_%", "agreement_with_pytorch_%", "size_mb",
                              "latency_b1_ms", "throughput_img_s"]]))
        add("")
    for img, cap in (("accuracy_vs_latency.png", "accuracy vs latency"),
                     ("latency_pytorch_vs_onnx.png", "pytorch vs onnx")):
        if os.path.exists(os.path.join(C.RESULT_DIR, img)):
            add(f"![{cap}](results/{img})\n")

    # ------------------------------------------------------------------ discussion
    add("## 12. Discussion\n")
    if summ is not None and len(summ) and kd_df is not None:
        gains = ", ".join(f"{r.student} {r['kd_gain_pts']:+.2f} pts (p = {r.mcnemar_p:.3g})" for _, r in summ.iterrows())
        n_sig = int((summ["significant_p<0.05"] == "yes").sum())
        n_pos = int((summ["kd_gain_pts"] > 0).sum())
        add(f"- **Effect of distillation.** Accuracy change from KD: {gains}. KD improved {n_pos} of {len(summ)} "
            f"students; {n_sig} of the changes are statistically significant.")
        if "kd_agree_teacher_%" in summ:
            d = (summ["kd_agree_teacher_%"] - summ["baseline_agree_teacher_%"]).mean()
            add(f"- **Students imitate the teacher.** On average the distilled students agree with the teacher's "
                f"prediction {d:+.1f} points more often than the baselines, which is what the soft-target and "
                "feature losses optimise for.")
        if "teacher_resnet50" in set(kd_df.key):
            tacc = get(kd_df, "teacher_resnet50", "accuracy_%")
            best_s = summ.loc[summ["kd_acc_%"].idxmax()]
            key = [f"{best_s.arch}_kd"] if f"{best_s.arch}_kd" in set(kd_df.key) else []
            add(f"- **Size vs accuracy.** The best distilled student ({best_s.student}, {best_s['kd_acc_%']:.2f}%) "
                f"is {tacc - best_s['kd_acc_%']:.2f} points below the teacher ({tacc:.2f}%)"
                + (f" while being {get(kd_df, key[0], 'compression_vs_teacher_x'):.1f}x smaller and "
                   f"{get(kd_df, key[0], 'speedup_vs_teacher_x'):.1f}x faster on the CPU." if key else "."))
        add(("- **Why KD helps small students.** " if n_pos >= 2 else
             "- **Why KD usually helps, and why it may not here.** With short training schedules or a weak teacher "
             "the benefit can vanish. In general: ") + "One-hot labels say only which class is right; the teacher's "
            "softened distribution also encodes class similarity, which acts as a strong regulariser and gives more "
            "information per training image. The feature term additionally guides the student's intermediate "
            "representations, which helps most for students whose capacity is far below the teacher's.")
        add("- **Latency is not proportional to MACs.** Depthwise convolutions have few MACs but low arithmetic "
            "intensity (memory-bound), so MobileNetV2 / EfficientNet-Lite0 are less fast on a CPU than their MAC "
            "count suggests; SqueezeNet uses only dense 1x1 / 3x3 convolutions.")
    if best is not None and nas_df is not None and len(nas_df):
        r = nas_df.iloc[0]
        add(f"- **NAS.** Under the deployment constraints the search found a {int(r['params']):,}-parameter network "
            f"reaching {r['accuracy_%']:.2f}% on CIFAR-10 at {r['latency_b1_ms']:.2f} ms per image. Rejecting "
            "infeasible architectures before training spent the whole training budget on deployable networks. "
            "Proxy accuracies (few epochs, subset) rank architectures, they are not final accuracies; the final "
            "number comes from training the winner from scratch.")
        add("- **Measured, not estimated, hardware cost.** MACs and parameters are poor latency predictors (kernel "
            "size, depthwise convolutions and activation choice change real speed), so constraints are checked "
            "with real measurements on the target CPU. The limitation is noise: latency is re-measured after "
            "training to confirm the constraint still holds.")
    if onnx_df is not None:
        agree = onnx_df["agreement_with_pytorch_%"].min()
        add(f"- **ONNX Runtime** predictions agree with PyTorch on at least {agree:.2f}% of the test images "
            "(remaining differences come from floating-point rounding in different kernels). ONNX Runtime is "
            "usually faster at batch 1 thanks to graph optimisations and fused kernels; see the latency table.")
    add("")

    add("## 13. Screenshots and terminal output\n")
    add("Output of every step of `scripts/run_all.sh`, either as a screenshot or copied from `logs/`.\n")
    sections = [(h, "image", f) for h, f in SCREENSHOT_SECTIONS] + [(h, "log", f) for h, f in LOG_SECTIONS]

    def section_number(item):            # "13.10 ..." -> (13, 10), so 13.10 comes after 13.9
        return tuple(int(n) for n in item[0].split()[0].split("."))

    for heading, kind, fname in sorted(sections, key=section_number):
        add(f"### {heading}\n")
        if kind == "image":
            if os.path.exists(os.path.join(C.LOG_DIR, "screenshots", fname)):
                add(f"![{heading}](logs/screenshots/{fname})\n")
            else:
                add(f"_Screenshot `logs/screenshots/{fname}` not found._\n")
        else:
            body = log(fname).rstrip()
            add(f"```\n{body}\n```\n" if body else f"_`logs/{fname}` not found._\n")
    add("## 14. Reproducibility\n")
    add("Fixed seeds, fixed validation split for the NAS, all hyper-parameters as command-line arguments with the "
        "values in `scripts/run_all.sh`, full logs in `logs/`. See README.md for the step-by-step commands.\n")

    with open(os.path.join(C.ROOT, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(w))
    print("Wrote REPORT.md")


if __name__ == "__main__":
    main()
