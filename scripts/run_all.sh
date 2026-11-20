#!/usr/bin/env bash
# Reproducible end-to-end pipeline for Assignment 4. Logs go to logs/.
# Usage (from any directory):
#   bash scripts/run_all.sh                  full run
#   bash scripts/run_all.sh --quick          fewer epochs / smaller search, about half the time
#   bash scripts/run_all.sh --fake           smoke test on random data (checks the code only)
#   bash scripts/run_all.sh --skip-teacher   reuse models/teacher_resnet50_cifar100.pth
#   bash scripts/run_all.sh --only q1        only knowledge distillation (steps 1-3)
#   bash scripts/run_all.sh --only q2        only NAS (steps 4-5)
#   bash scripts/run_all.sh --only final     only export, inference, benchmark, report (steps 6-9)
#   bash scripts/run_all.sh --parallel       run the NAS (Q2) at the same time as Q1 on the same GPU
#                                            (faster overall; NAS latency measurements get a bit noisier)
# Flags can be combined, e.g.  --quick --skip-teacher
set -e -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Python from the uv virtual environment (.venv), unless PY is set
if [ -z "$PY" ]; then
  if [ -x ".venv/bin/python" ]; then PY=".venv/bin/python"; else PY="python"; fi
fi

echo "Using Python: $PY"
$PY -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

FAKE=""; SKIP_TEACHER=0; ONLY="all"; PARALLEL=0
T_EP=30; S_EP=30; NAS_BUDGET=30; NAS_POP=10; NAS_EP=4; NAS_IMG=20000; BEST_EP=30
while [ $# -gt 0 ]; do
  case "$1" in
    --quick) T_EP=15; S_EP=15; NAS_BUDGET=16; NAS_POP=6; NAS_EP=3; NAS_IMG=15000; BEST_EP=15 ;;
    --fake)  FAKE="--fake"; T_EP=1; S_EP=1; NAS_BUDGET=4; NAS_POP=2; NAS_EP=1; NAS_IMG=500; BEST_EP=1 ;;
    --skip-teacher) SKIP_TEACHER=1 ;;
    --parallel) PARALLEL=1 ;;
    --only) shift; ONLY="$1" ;;
    *) echo "unknown option: $1"; exit 1 ;;
  esac
  shift
done
export PYTHONWARNINGS=ignore
mkdir -p logs/screenshots models onnx_models results

run_q1() {
  if [ "$SKIP_TEACHER" == "1" ]; then
    [ -f models/teacher_resnet50_cifar100.pth ] || { echo "teacher checkpoint not found, run without --skip-teacher"; exit 1; }
    echo "== 1. Teacher ResNet-50 == (skipped, using models/teacher_resnet50_cifar100.pth)"
  else
    echo "== 1. Teacher ResNet-50 on CIFAR-100 =="
    $PY src/train_teacher.py --epochs $T_EP $FAKE | tee logs/1_train_teacher.log
  fi
  echo "== 2. Student baselines (no distillation) =="
  $PY src/train_student_baseline.py --arch all --epochs $S_EP $FAKE | tee logs/2_students_baseline.log
  echo "== 3. Students with knowledge distillation =="
  $PY src/train_student_kd.py --arch all --epochs $S_EP $FAKE | tee logs/3_students_kd.log
}

run_q2() {
  echo "== 4. Hardware-aware evolutionary NAS on CIFAR-10 =="
  $PY src/nas_search.py --budget $NAS_BUDGET --population $NAS_POP --proxy-epochs $NAS_EP \
      --proxy-train-images $NAS_IMG --max-latency-ms 6 $FAKE | tee logs/4_nas_search.log
  echo "== 5. Train best NAS architecture from scratch =="
  $PY src/nas_train_best.py --epochs $BEST_EP $FAKE | tee logs/5_nas_train_best.log
}

run_final() {
  echo "== 6. ONNX export ==";        $PY src/export_onnx.py | tee logs/6_export_onnx.log
  echo "== 7. Inference demo ==";     $PY src/infer.py --model all --test-samples 5 $FAKE | tee logs/7_infer.log
                                      $PY src/onnx_infer.py --model all --test-samples 5 $FAKE | tee logs/7_onnx_infer.log
  echo "== 8. Benchmark ==";          $PY src/benchmark.py --threads 1 $FAKE | tee logs/8_benchmark.log
  echo "== 9. Report ==";             $PY src/make_report.py | tee logs/9_report.log
}

case "$ONLY" in
  q1) run_q1 ;;
  q2) run_q2 ;;
  final) run_final ;;
  all)
    if [ "$PARALLEL" == "1" ]; then
      echo "Running Q2 (NAS) in the background, output in logs/4_nas_search.log and logs/5_nas_train_best.log"
      ( run_q2 > /dev/null ) &
      Q2_PID=$!
      run_q1
      echo "Q1 done, waiting for Q2 ..."
      wait $Q2_PID
    else
      run_q1
      run_q2
    fi
    run_final ;;
  *) echo "--only must be q1, q2, final or all"; exit 1 ;;
esac
echo "Done. See results/, logs/ and REPORT.md."
