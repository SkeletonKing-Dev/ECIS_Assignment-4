# Assignment 4: Knowledge Distillation and Hardware-Aware NAS for Edge Devices

This project covers:

- Q1: a ResNet-50 teacher trained on CIFAR-100, distilled into three lightweight students (MobileNetV2, SqueezeNet 1.1, EfficientNet-Lite0). It uses soft targets with temperature, hard labels and intermediate-feature distillation, and compares each student against the same student trained without distillation.
- Q2: a hardware-aware evolutionary neural architecture search on CIFAR-10. Every candidate's CPU latency, storage size and peak inference memory are measured, and candidates that violate the deployment limits are rejected. The best architecture is then trained from scratch.
- ONNX export with CPU inference in ONNX Runtime, an automated benchmark (CSV files and plots) and an auto-generated technical report.

## 1. Project structure

```
.
├── data/                       datasets (not submitted)
│   ├── cifar10/                CIFAR-10  (cifar-10-batches-py); a symlink works
│   └── cifar100/               CIFAR-100 (cifar-100-python); downloaded automatically
├── logs/                       log of every pipeline step
│   └── screenshots/            screenshots for the report
├── models/                     PyTorch checkpoints (teacher, 3 baseline + 3 KD students, NAS model)
├── onnx_models/                ONNX models (6 students + NAS model)
├── results/                    benchmark CSV files, NAS history / best architecture, plots
├── scripts/
│   ├── run_all.sh              reproducible end-to-end pipeline
│   └── make_submission.sh      optional ZIP packaging
├── src/
│   ├── common.py               paths, datasets, on-GPU data loader, training loop, checkpoints
│   ├── networks.py             model registry
│   ├── resnet50.py             teacher
│   ├── mobilenetv2.py          student 1
│   ├── squeezenet.py           student 2
│   ├── efficientnet_lite.py    student 3
│   ├── train_teacher.py        Q1: teacher training
│   ├── train_student_baseline.py  Q1: students without distillation
│   ├── kd.py                   Q1: distillation framework (losses, adapters, Distiller)
│   ├── train_student_kd.py     Q1: students with distillation
│   ├── nas_space.py            Q2: search space + architecture generator
│   ├── hw_metrics.py           Q2/benchmark: latency, size, peak memory, MACs
│   ├── nas_search.py           Q2: hardware-aware evolutionary search
│   ├── nas_train_best.py       Q2: train the best architecture from scratch
│   ├── export_onnx.py          ONNX export (+ verification against PyTorch)
│   ├── infer.py                unified PyTorch inference for all models
│   ├── onnx_infer.py           ONNX Runtime CPU inference
│   ├── benchmark.py            automated benchmark, CSV results and plots
│   └── make_report.py          generates REPORT.md from logs/ and results/
├── README.md
├── REPORT.md / REPORT.pdf      technical report
└── requirements.txt
```

## 2. Installation (uv)

Requirements: Python 3.9 or newer and [uv](https://docs.astral.sh/uv/). An NVIDIA GPU is needed for training in reasonable time. Inference, the NAS hardware measurements and the benchmark run on the CPU.

```
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

On a machine without an NVIDIA GPU, install the CPU build of PyTorch first:

```
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install -r requirements.txt
```

## 3. Datasets

For each dataset the code looks for an existing copy in this order, and skips the download if it finds one:

| Dataset | Used for | Locations checked |
|---|---|---|
| CIFAR-100 | Q1 (distillation) | `$CIFAR100_ROOT`, `data/cifar100/`, `data/` (folder `cifar-100-python`) |
| CIFAR-10 | Q2 (NAS) | `$CIFAR10_ROOT`, `data/cifar10/`, `data/` (folder `cifar-10-batches-py`) |

A symlink such as `data/cifar10 -> ../../Assignment-2/data/cifar10` works. CIFAR-100 (about 170 MB) downloads to `data/cifar100/` on the first run.

## 4. Run everything (one command)

Run from the project root. The scripts use `.venv/bin/python` automatically.

```
bash scripts/run_all.sh                 # full run
bash scripts/run_all.sh --quick         # half the epochs and a smaller search, about half the time
bash scripts/run_all.sh --fake          # 7 min smoke test on random data (checks the code only)
```

Other options (can be combined):

| Option | Effect |
|---|---|
| `--skip-teacher` | reuse `models/teacher_resnet50_cifar100.pth` |
| `--only q1` | steps 1-3 (teacher, baseline students, KD students) |
| `--only q2` | steps 4-5 (NAS search, train best architecture) |
| `--only final` | steps 6-9 (ONNX export, inference, benchmark, report) |
| `--parallel` | run the NAS at the same time as Q1 on the same GPU. Faster overall, but the NAS latency measurements get a little noisier because the CPU is shared. |

Every training log prints the elapsed minutes after each epoch, so after the first epoch you can estimate the total and switch to `--quick` if needed. The teacher (ResNet-50) is the slowest part, followed by the six student trainings (the KD runs also run the teacher on each batch).

## 5. Step-by-step execution

Activate the environment first (`source .venv/bin/activate`) and run from the project root.

| Step | Command | Output |
|---|---|---|
| 1. Teacher | `python src/train_teacher.py --epochs 30` | `models/teacher_resnet50_cifar100.pth` |
| 2. Baseline students | `python src/train_student_baseline.py --arch all --epochs 30` | `models/student_<arch>_baseline_cifar100.pth` |
| 3. KD students | `python src/train_student_kd.py --arch all --epochs 30 --T 4 --alpha 0.9 --beta 2 --mode both` | `models/student_<arch>_kd_cifar100.pth` |
| 4. NAS search | `python src/nas_search.py --budget 30 --population 10 --max-latency-ms 3 --max-size-mb 2 --max-mem-mb 6` | `results/nas_history.csv/.json`, `results/nas_best_arch.json` |
| 5. Train best NAS model | `python src/nas_train_best.py --epochs 30` | `models/nas_best_cifar10.pth` |
| 6. ONNX export | `python src/export_onnx.py` | `onnx_models/*.onnx` |
| 7a. Inference (PyTorch) | `python src/infer.py --model all --test-samples 5` | console |
| 7b. Inference (ONNX, CPU) | `python src/onnx_infer.py --model all --test-samples 5` | console |
| 8. Benchmark | `python src/benchmark.py --threads 1` | `results/*.csv`, `results/*.png` |
| 9. Report | `python src/make_report.py` | `REPORT.md` |

`<arch>` is `mobilenetv2`, `squeezenet` or `efficientnet_lite0`. Steps 2 and 4 don't depend on the teacher, so they can run before or alongside step 1. Without activating the environment, use `uv run python src/<script>.py`.

Useful extras:

```
python src/infer.py --list                                  # names of all trained models
python src/infer.py --model squeezenet_kd --image data/test.jpg
python src/onnx_infer.py --model nas_best --image data/test.jpg --threads 1
python src/train_student_kd.py --arch mobilenetv2 --mode logit    # soft-target KD only (ablation)
python src/nas_space.py                                     # print search-space size and a sample
```

### NAS deployment constraints

The defaults (latency ≤ 3 ms, size ≤ 2 MB, peak memory ≤ 6 MB) were chosen so that roughly half of the random architectures are rejected on a typical desktop CPU. Latency depends on your CPU: if almost every candidate is rejected, the search stops with "too many rejected candidates". In that case raise `--max-latency-ms` in step 4 (or in `scripts/run_all.sh`).

## 6. Result files

- `results/kd_summary.csv`: per student, baseline vs KD accuracy, gain, McNemar p-value and agreement with the teacher
- `results/benchmark_kd.csv`: teacher and students: accuracy, parameters, MACs, size, peak memory, latency, throughput, compression and speed-up vs the teacher
- `results/benchmark_nas.csv`: the NAS model against its constraints
- `results/benchmark_onnx.csv`: the same models in ONNX Runtime, including agreement with PyTorch
- `results/benchmark_all.csv`: all of the above in one table
- `results/nas_history.csv` / `nas_history.json`: every NAS candidate (trained or rejected, with the reason) and its measurements
- `results/nas_best_arch.json`: the best architecture's configuration, metrics, constraints and search settings
- `results/predictions_cifar100.csv`, `results/predictions_cifar10_nas.csv`: per-image predictions of every model
- `results/benchmark_env.txt`: CPU, thread count and library versions
- Plots: `kd_vs_baseline.png`, `training_curves.png`, `accuracy_vs_latency.png`, `nas_search.png`, `latency_pytorch_vs_onnx.png`

## 7. Submission

You need to submit two things:

1. **GitHub repo** with the source code, trained models (`models/`, `onnx_models/`) and results (`results/`, `logs/`, `REPORT.md`).
2. **Documentation PDF** (the technical report).

### 7.1 Push to GitHub

Create an empty repository on https://github.com/new (no README, no .gitignore), then run from the project root:

```
git init
git add .
git commit -m "Assignment 4: knowledge distillation and hardware-aware NAS"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

`.gitignore` leaves out `data/` and `.venv/` but keeps the models and results. The largest checkpoint is the teacher at about 91 MB, just under GitHub's 100 MB per-file limit, so Git LFS isn't needed. If the repo is private, add your instructor as a collaborator (Settings > Collaborators).

### 7.2 Documentation PDF

Add the screenshots from `logs/screenshots/` to section 13 of `REPORT.md` (for example `![teacher](logs/screenshots/1_train_teacher.png)`), fill in your name, roll number, section and repo link, then export it to PDF. Any of these works:

- VS Code: install the "Markdown PDF" extension, open `REPORT.md`, right-click > "Markdown PDF: Export (pdf)".
- Pandoc: `pandoc REPORT.md -o REPORT.pdf` (needs a LaTeX installation).

If a ZIP is also needed: `bash scripts/make_submission.sh 23i-XXXX_A`.

## 8. Running on Google Colab (step by step)

Use this if your own machine has no NVIDIA GPU. Type each line into a notebook cell and press **Shift+Enter**. Lines that start with `!` run shell commands, and `%cd` changes the folder.

1. On your computer, zip the code: 
   ```
   zip -r Assignment-4.zip src scripts README.md requirements.txt
   ```
2. Go to https://colab.research.google.com, click **New notebook**, then **Runtime > Change runtime type > T4 GPU > Save**. Check the GPU with `!nvidia-smi`.
3. Upload and unzip:
   ```
   from google.colab import files
   files.upload()
   ```
   ```
   !unzip -q -o assignment4.zip -d Assignment4
   %cd /content/Assignment4
   ```
4. Install the ONNX packages (Colab already has PyTorch with GPU support, so don't use uv here): `!pip install -q onnx onnxruntime onnxscript`
5. Run the pipeline (the datasets download automatically): `!bash scripts/run_all.sh` (or `--quick`). Keep the tab open: **all files are deleted when the session ends**.
6. Download the results:
   ```
   !zip -r -q outputs.zip models onnx_models results logs REPORT.md
   from google.colab import files
   files.download("outputs.zip")
   ```
   If the browser download fails, copy it to Google Drive instead: `from google.colab import drive; drive.mount("/content/drive")`, then `!cp outputs.zip /content/drive/MyDrive/`.
7. Extract `outputs.zip` into your project root, then follow section 7.

Colab's CPU is slow and shared, so with the default limits the NAS may reject almost everything there. Add `--max-latency-ms 6` to the `nas_search.py` line in `scripts/run_all.sh` if that happens.

- **Quick:**
```
from google.colab import files
files.upload()
```
```
!unzip -q -o Assignment-4.zip -d Assignment-4
%cd /content/Assignment-4
!pip install -q onnx onnxruntime onnxscript
!bash scripts/run_all.sh
!zip -r -q outputs.zip models onnx_models results logs REPORT.md
from google.colab import files
files.download("outputs.zip")
```
## 9. Notes

- Model selection: no checkpoint is chosen using the test set. Every training run uses a fixed schedule and keeps the final weights; the NAS ranks candidates on a validation split carved out of the training set.
- Training uses mixed precision (FP16) on the GPU and keeps the whole dataset on the GPU, with augmentation (random crop + flip) done there per batch.
- Latency is measured on the CPU with 1 thread and batch size 1, after warm-up. Peak memory is measured with the PyTorch profiler; if a PyTorch version doesn't expose memory events, an estimate from layer output sizes is used (column `mem_method` in the NAS history).
- The KD feature adapters exist only during training. Distilled students have exactly the same architecture and size as the baselines.
