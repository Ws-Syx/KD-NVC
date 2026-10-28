# KD-NVC-T | ~70 FPS 1080p Decoding on RTX 5060 (CUDA inference)

> **This repository is released for TMM (IEEE Transactions on Multimedia) review.**
>
> **Weight is not released at this moment.** The trained checkpoints (`.pth.tar`) are
> intentionally excluded. For speed benchmarking, use random weights
> (`--random_weights 1`); timing does not depend on weight values.

Supports the paper claim:

> **KD-NVC-T (tiny) decodes 1080p P-frames at 14.23 ms/frame ~ 70.3 FPS on an NVIDIA RTX 5060.**
>
> Protocol: 300 frames, DCVC-RT reference configuration (`intra_period=-1`,
> `reset_interval=64`), first 10% (30 frames) discarded as warm-up, P-frames only.
> Five repeats: **70.0-70.4 FPS** (14.20-14.28 ms/frame), stdev 0.03 ms.

This is the **CUDA inference version**: it relies on two compiled extensions
(`MLCodec_extensions_cpp` for rANS entropy coding and `inference_extensions_cuda`
for the custom CUDA kernels). Build instructions are in Sec. 3.

---

## 0. Reference machine

All numbers in this repository were measured on the machine below (single-GPU
desktop, native Ubuntu 24.04 - not WSL, not Docker).

| Component | Detail |
|---|---|
| CPU | AMD Ryzen 5 9600X (Zen 5, 6 cores / 12 threads, boost 5.49 GHz, 32 MiB L3) |
| CPU governor | `performance` (`amd-pstate`, ~5.4 GHz under load) |
| RAM | 32 GB (30 GiB usable, reported as 31,994,696 kB `MemTotal`) - nothing is offloaded to swap during the benchmark |
| Swap | 8 GiB (untouched: 0 B used) |
| GPU | NVIDIA GeForce RTX 5060, 8 GB GDDR7, Blackwell sm_120, 3,135 MHz max SM clock, 145 W board limit |
| GPU link | PCIe 4.0, currently 8 lanes (the GPU supports 16) |
| GPU driver | 580.126.09 |
| CUDA toolkit | 13.0 |
| OS | Ubuntu 24.04.4 LTS, kernel 6.17.0-23-generic, x86_64 |
| Python | CPython 3.12.13 (Miniconda, env `dcvc`) |
| PyTorch | 2.9.1+cu130 |
| Storage | system NVMe (48 GB root); the 1.9 GB 1080p YUV sequence is read from a second NVMe (`/mnt/windows_d`) |

---

## 1. What is / is not released

**Included:**

- `src/` - model architectures and the source of both extensions
  (`.py`, `.cu`, `.cpp`, `.h`), which are required to reproduce the speed.
- `logs/cuda_300f_run01.log` ... `logs/cuda_300f_run05.log` - the five speed
  repeats (300 frames each, warm-up already documented in Sec. 6).
- `test_video.py` - inference entry point with `--random_weights` for speed-only runs.
- `test.sh` - one-command benchmark (5 repeats, 30 s cool-down, warm-up trimmed).
- `analyze_log.py` - steady-state latency/FPS report for one or many logs.
- `compare_consistency.py` - optional CUDA vs PyTorch consistency check.
- `build_extensions.sh` - one-shot build of both extensions.
- `docs/EVIDENCE.md` - environment and measurement methodology.

**Not included:**

- Trained weights (`checkpoint/*.pth.tar`) - **weight is not released at this moment**.
- Prebuilt binaries (`*.so`) - build them from source, see Sec. 3.
- Test video data (`dataset/`) - bring your own YUV sequence, see Sec. 4.

---

## 2. Why random weights give the same speed

Encoding/decoding time is determined by the **model architecture + CUDA kernels**,
not by the numerical values of the weights. `--random_weights 1` keeps the randomly
initialized model (it never loads a checkpoint) and runs the full forward pass, so
the measured latency matches the trained model; only PSNR / bitrate are meaningless.

---

## 3. Build the two extensions (required for CUDA inference)

The pipeline depends on two compiled extensions:

| # | Extension | Purpose | Backend |
|---|---|---|---|
| 1 | `MLCodec_extensions_cpp` | rANS arithmetic entropy coding | C++ (pybind11) |
| 2 | `inference_extensions_cuda` | custom CUDA quantization/conv kernels | CUDA |

### 3.1 Prerequisites

- CUDA Toolkit 13.0 (`nvcc`, sm_120 for RTX 5060 Blackwell)
- Python 3.12 + PyTorch 2.9.1 built against CUDA 13.0
- Build deps: `pip install pybind11 ninja`

### 3.2 One-shot build

```bash
CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=12.0 bash build_extensions.sh
```

`build_extensions.sh` builds both extensions in place and copies the resulting
`.so` files into the active interpreter's `site-packages`.

### 3.3 Manual build (two projects)

**Project 1 - `MLCodec_extensions_cpp` (rANS entropy coder, CPU):**

```bash
cd src/cpp
python setup.py build_ext --inplace
python -c 'import site, glob, shutil, os; \
  [shutil.copy(f, site.getsitepackages()[0]) for f in glob.glob("MLCodec_extensions_cpp*.so")]'
cd ../..
```

**Project 2 - `inference_extensions_cuda` (custom CUDA kernels):**

```bash
cd src/layers/extensions/inference
CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=12.0 \
  python setup.py build_ext --inplace
python -c 'import site, glob, shutil, os; \
  [shutil.copy(f, site.getsitepackages()[0]) for f in glob.glob("inference_extensions_cuda*.so")]'
cd ../../../..
```

### 3.4 Verify CUDA inference is active

```bash
python - <<'PY'
import MLCodec_extensions_cpp
import inference_extensions_cuda
from src.layers.cuda_inference import CUSTOMIZED_CUDA_INFERENCE
print("extensions: OK")
print("CUSTOMIZED_CUDA_INFERENCE =", CUSTOMIZED_CUDA_INFERENCE)  # must be True
PY
```

If `CUSTOMIZED_CUDA_INFERENCE` is `False`, the code silently falls back to pure
PyTorch (correct but slower); rebuild the CUDA extension for your toolchain.

---

## 4. Prepare test data (not included)

```bash
mkdir -p dataset
# Place any 1080p yuv420 8-bit sequence (>= 300 frames) at:
#   dataset/Beauty_1920x1080_120fps_420_8bit_YUV.yuv
# or edit dataset_config_single1080p_yuv420.json accordingly.
```

---

## 5. Run the speed benchmark (random weights, CUDA)

One configuration only, identical to the DCVC-RT reference test script:

```bash
bash test.sh
```

- `--force_intra_period -1` - a single I-frame, at frame 0, no periodic refresh
- `--reset_interval 64`   - feature-adaptor I refresh every 64 frames
- 300 frames, 1080p, QP=32, `--random_weights 1`, CUDA custom kernels active
- 5 repeats, 30 s cool-down between runs (`REPEATS`, `SLEEP_BETWEEN` env vars)
- the first 10% of every run (30 of 300 decoded frames) is dropped as warm-up

Each repeat writes its own log and result file:

```
logs/cuda_300f_run01.log ... logs/cuda_300f_run05.log
output_tiny_cuda_1080p_qp32_300f_run01.json ... _run05.json
out_bin_cuda/                              # bitstreams (regenerated, not tracked)
```

`test.sh` ends by calling `analyze_log.py` on all five logs, which prints a
per-run table plus the aggregate over runs.

Equivalent single command:

```bash
python test_video.py \
  --model_type tiny \
  --test_config dataset_config_single1080p_yuv420.json \
  --output_path output_tiny_cuda_1080p_qp32_300f.json \
  --qp 32 --force_frame_num 300 --force_intra_period -1 --reset_interval 64 \
  --random_weights 1 --cuda 1 --write_stream 1 --verbose 2 \
  --stream_path out_bin_cuda
```

Expected per P-frame decode: 14.2-14.3 ms (median 14.20 ms, p95 14.87 ms,
min/max 13.55/16.08 ms over all 1350 steady P-frames of the five runs).

Decode is measured per frame with `torch.cuda.synchronize()` on both sides of the
decode call, so latency includes full GPU execution, not just kernel launch.

`--force_intra_period` defaults to the `intra_period` of `--test_config`
(`-1` in `dataset_config_single1080p_yuv420.json`), and `--reset_interval`
defaults to `64`, so the plain command above already matches the reference
configuration.

---

## 6. How the FPS number is computed

### 6.1 Protocol

- **300 frames** of 1080p (`Beauty_1920x1080_120fps_420_8bit_YUV.yuv`), QP=32,
  random weights, CUDA custom kernels active.
- **Configuration** (DCVC-RT reference): `intra_period=-1` (one I-frame, frame 0),
  `reset_interval=64` (feature-adaptor I refresh at frames 1 / 65 / 129 / 193 / 257).
- **Warm-up: the first 10% (30 frames) of every run is discarded.** That window
  contains the first I-frame (~62-64 ms) and the first P-frame (~16 ms,
  adaptive-I preparation), plus the frame-1 adaptor refresh.
- **Metric: steady-state P-frame decode latency** - frames 30-299, n=270 per run,
  all P-frames (no I-frame falls inside the steady window). The paper claim is
  about P-frame decoding throughput.
- **Aggregate: 5 repeats**, each preceded by a 30 s cool-down (1350 steady
  P-frames in total).

### 6.2 Results per run

| Log | Steady P-frames | P-frame FPS |
|---|---|---|
| `logs/cuda_300f_run01.log` | 14.20 ms/f (n=270) | 70.4 |
| `logs/cuda_300f_run02.log` | 14.21 ms/f (n=270) | 70.4 |
| `logs/cuda_300f_run03.log` | 14.24 ms/f (n=270) | 70.2 |
| `logs/cuda_300f_run04.log` | 14.28 ms/f (n=270) | 70.0 |
| `logs/cuda_300f_run05.log` | 14.22 ms/f (n=270) | 70.3 |
| **All five** | **14.23 ms/f (n=1350)** | **70.3** |

Aggregate over the five runs: mean 14.23 ms/f, median 14.22 ms/f, stdev 0.03 ms,
min/max run 14.20/14.28 ms/f -> **70.0-70.4 FPS**.

Per-frame distribution of the pooled steady P-frames (n=1350): median 14.20 ms,
p95 14.87 ms, min 13.55 ms, max 16.08 ms. The 5 adaptor refreshes
(frames 65/129/193/257 and the warm-up frame 1) show up as the 15.8-16.1 ms
outliers; they are included in the average but do not move it.

Encoding, for reference: 19.4-19.6 ms/f (51.1-51.4 FPS), bitrate 24,663 kbps at
30 fps for the random-weight model (PSNR values are meaningless with random
weights and are not reported).

**No warm-up excluded - all 300 frames counted.** The number above is a
steady-state P-frame metric. If nothing is dropped and every decoded frame of
every run is averaged, the result is:

| Log | All 300 frames | All-frame FPS |
|---|---|---|
| `logs/cuda_300f_run01.log` | 14.37 ms/f | 69.6 |
| `logs/cuda_300f_run02.log` | 14.39 ms/f | 69.5 |
| `logs/cuda_300f_run03.log` | 14.41 ms/f | 69.4 |
| `logs/cuda_300f_run04.log` | 14.45 ms/f | 69.2 |
| `logs/cuda_300f_run05.log` | 14.39 ms/f | 69.5 |
| **All five** | **14.40 ms/f (n=1500)** | **69.4** |

Pooling the raw per-frame times of all 5 runs (21.60 s of decode time over 1500
frames) gives the same 14.40 ms/f -> **69.4 FPS**. The gap comes from exactly two
frames per run: frame 0 (first I-frame plus CUDA context/allocator warm-up, mean
63.1 ms = 4.4x a steady frame) and frame 1 (first P-frame with adaptive-I
preparation, mean 16.0 ms). Together they add ~0.17 ms/f over 300 frames, i.e.
-0.9 FPS (-1.3%) versus the 70.3 FPS steady-state figure. So the headline number
is not flattered by discarding frames: the cold-start-inclusive average is still
69.4 FPS.

Reproduce:

```bash
python analyze_log.py logs/cuda_300f_run0*.log --warmup-frac 0.1
# all 300 frames, no warm-up exclusion:
python analyze_log.py logs/cuda_300f_run01.log   # "All-frame average" line
```

`analyze_log.py` with a single log path prints the detailed per-log report
(I-frame, first P-frame, all-frame average, steady window, P-only metric); with
several paths it prints the per-run table plus the aggregate above.

---

## 7. Directory layout

```
.
|-- src/
|   |-- models/                    # image / video_model / tiny / small / standard
|   |-- layers/                    # cuda_inference, layers, CUDA ext source (.cu/.cpp)
|   |-- cpp/                       # rANS entropy coder source (pybind11)
|   `-- utils/                     # YUV I/O, PSNR, bitstream helpers
|-- logs/
|   `-- cuda_300f_run01.log ... cuda_300f_run05.log   # the five speed repeats
|-- docs/EVIDENCE.md
|-- test_video.py                  # --random_weights for speed-only runs
|-- compare_consistency.py         # CUDA vs PyTorch consistency check (optional)
|-- analyze_log.py                 # steady-state FPS from one or many logs
|-- test.sh                        # one-command benchmark (5 repeats)
|-- build_extensions.sh            # build both extensions
|-- dataset_config_single1080p_yuv420.json
`-- requirements.txt
```

---

## 8. Requirements

```bash
conda create -n kdnvc python=3.12 -y
conda activate kdnvc
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt      # numpy scipy pillow
pip install pybind11 ninja           # build-time only
```

---

## 9. License

Adapted from Microsoft DCVC (MIT). Released under the MIT License, see `LICENSE`.
