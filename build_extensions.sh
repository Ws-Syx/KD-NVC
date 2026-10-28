#!/usr/bin/env bash
# Build both inference extensions from source and install them into the active
# interpreter's site-packages.
#
#   Project 1: MLCodec_extensions_cpp   (rANS entropy coder, C++/pybind11)
#   Project 2: inference_extensions_cuda (custom CUDA kernels)
#
# Prerequisites: CUDA Toolkit 13.0 (nvcc, sm_120), gcc, torch(cu130),
#                pybind11, ninja.
#
# Usage: CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=12.0 bash build_extensions.sh
#        (override the interpreter with PYTHON=/path/to/python)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python}"
SITE="$("$PYTHON" -c 'import site; print(site.getsitepackages()[0])')"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"

echo ">>> python: $PYTHON"
echo ">>> site-packages: $SITE"
echo ">>> CUDA_HOME: $CUDA_HOME  TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"

# 1) rANS entropy coder extension (CPU)
echo ">>> [1/2] build MLCodec_extensions_cpp"
(cd src/cpp && "$PYTHON" setup.py build_ext --inplace)
find src/cpp -maxdepth 1 -name 'MLCodec_extensions_cpp*.so' -exec cp {} "$SITE/" \;

# 2) CUDA inference extension (GPU)
echo ">>> [2/2] build inference_extensions_cuda"
(cd src/layers/extensions/inference && "$PYTHON" setup.py build_ext --inplace)
find src/layers/extensions/inference -maxdepth 1 -name 'inference_extensions_cuda*.so' -exec cp {} "$SITE/" \;

echo ">>> verify"
"$PYTHON" -c 'import MLCodec_extensions_cpp, inference_extensions_cuda; print("extensions OK")'
