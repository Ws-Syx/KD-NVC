#!/usr/bin/env bash
# KD-NVC-T (tiny) 1080p decoding speed benchmark on RTX 5060.
#
# Single protocol configuration, identical to the DCVC-RT reference test script:
#   --force_intra_period -1   only one I-frame, at frame 0
#   --reset_interval 64       feature-adaptor I refresh every 64 frames
#
# Fixed workload: 300 frames, 1080p, QP=32, random weights (speed-only, no
# checkpoint needed), CUDA custom kernels. Every run discards the first 10% of
# the decoded frames (30 of 300) as warm-up before averaging.
#
# REPEATS runs are executed back to back with a fixed cool-down between them;
# each one writes its own log
#   logs/cuda_300f_run01.log ... logs/cuda_300f_run05.log
# and the final report averages the steady-state P-frame latency over all runs.
#
# Usage:
#   bash test.sh                        # 5 repeats x 300 frames, 30 s apart
#   REPEATS=3 bash test.sh              # fewer repeats
#   SLEEP_BETWEEN=0 bash test.sh        # no cool-down between runs
#   FRAMES=120 bash test.sh             # shorter measurement window
#   PYTHON=/path/to/python bash test.sh # explicit interpreter
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python}"
QP=32
FRAMES="${FRAMES:-300}"
REPEATS="${REPEATS:-5}"
WARMUP_FRAC="${WARMUP_FRAC:-0.1}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-30}"   # cool-down between runs (seconds)

INTRA_PERIOD=-1     # no periodic I-frame (single I-frame at frame 0)
RESET_INTERVAL=64   # feature-adaptor I refresh period

echo "================================================"
echo " KD-NVC-T (tiny) | RTX 5060 | 1080p | ${FRAMES} frames | QP=${QP}"
echo " random_weights=1 (no checkpoint; speed-only)"
echo " intra_period=${INTRA_PERIOD}  reset_interval=${RESET_INTERVAL}"
echo " repeats=${REPEATS}  gap=${SLEEP_BETWEEN}s  warm-up dropped=${WARMUP_FRAC} ($((FRAMES / 10)) frames/run)"
echo " interpreter: ${PYTHON}"
echo "================================================"

"$PYTHON" - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda,
      "| available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU", torch.cuda.get_device_name(0))
import MLCodec_extensions_cpp
import inference_extensions_cuda
from src.layers.cuda_inference import CUSTOMIZED_CUDA_INFERENCE
print("extensions OK | CUSTOMIZED_CUDA_INFERENCE =", CUSTOMIZED_CUDA_INFERENCE)
if not CUSTOMIZED_CUDA_INFERENCE:
    raise SystemExit("CUDA inference extension not active; rebuild it (see README Sec. 3).")
PY

mkdir -p logs
LOG_FILES=()

for ((i = 1; i <= REPEATS; i++)); do
    RUN="$(printf '%02d' "$i")"
    OUT="output_tiny_cuda_1080p_qp${QP}_${FRAMES}f_run${RUN}.json"
    LOG="logs/cuda_${FRAMES}f_run${RUN}.log"
    STREAMS="out_bin_cuda"

    if [ "$i" -gt 1 ] && [ "$SLEEP_BETWEEN" -gt 0 ]; then
        echo ""
        echo "--- cool-down ${SLEEP_BETWEEN} s before run ${RUN} ---"
        sleep "$SLEEP_BETWEEN"
    fi

    echo ""
    echo "================================================"
    echo " run ${RUN}/${REPEATS}: intra_period=${INTRA_PERIOD} reset_interval=${RESET_INTERVAL}"
    echo " log    : ${LOG}"
    echo "================================================"

    rm -rf "$STREAMS" "$OUT"

    "$PYTHON" test_video.py \
        --model_type tiny \
        --test_config dataset_config_single1080p_yuv420.json \
        --output_path "$OUT" \
        --qp "$QP" \
        --force_frame_num "$FRAMES" \
        --force_intra_period "$INTRA_PERIOD" \
        --reset_interval "$RESET_INTERVAL" \
        --random_weights 1 \
        --cuda 1 \
        --write_stream 1 \
        --verbose 2 \
        --stream_path "$STREAMS" | tee "$LOG"

    LOG_FILES+=("$LOG")
done

echo ""
echo "================================================"
echo " Steady-state decode over ${REPEATS} runs"
echo " (first ${WARMUP_FRAC} of every run dropped as warm-up)"
echo "================================================"
"$PYTHON" analyze_log.py "${LOG_FILES[@]}" --warmup-frac "$WARMUP_FRAC"
