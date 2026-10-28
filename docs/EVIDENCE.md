# Evidence: KD-NVC-T decodes 1080p P-frames at 14.23 ms/f ~ 70.3 FPS on RTX 5060

> Protocol: DCVC-RT reference configuration (`intra_period=-1`, `reset_interval=64`),
> 300 frames, first 10% (30 frames) of every run discarded as warm-up, P-frame metric.
> Five repeats with a 30 s cool-down: 70.0-70.4 FPS.

> Released for TMM review. Weight is not released at this moment; speed
> benchmarking uses random weights.

## Environment

- GPU: `NVIDIA GeForce RTX 5060` (Blackwell, sm_120), 8 GB GDDR7,
  PCIe 4.0 x8, 145 W board limit
- CPU: AMD Ryzen 5 9600X (6C/12T, Zen 5, boost 5.49 GHz), `performance` governor
- RAM: 32 GB, swap 8 GiB untouched
- OS: Ubuntu 24.04.4 LTS, kernel 6.17.0-23-generic, native install (not WSL)
- Idle policy: no suspend (GNOME `sleep-inactive-ac-type=nothing`), so no repeat
  is disturbed by a display/GPU power event
- Driver: 580.126.09, CUDA 13.0
- Python 3.12.13, `torch 2.9.1+cu130`
- Models: `kd-nvc-tiny-stage8-epoch26.pth.tar` (P-frame) + `cvpr2025_image.pth.tar`
  (I-frame); for the speed runs both are randomly initialized (`--random_weights 1`)
- Sequence: Beauty_1920x1080_120fps_420_8bit_YUV.yuv, 1920x1080, QP=32
- Path: CUDA custom kernels enabled (`CUSTOMIZED_CUDA_INFERENCE=True`)

## Protocol

Single configuration, identical to the DCVC-RT reference test script:

| Option | Value | Meaning |
|---|---|---|
| `--force_intra_period` | `-1` | one I-frame only, at frame 0 |
| `--reset_interval` | `64` | feature-adaptor I refresh at frames 1 / 65 / 129 / 193 / 257 |
| `--force_frame_num` | `300` | measurement window |
| `--random_weights` | `1` | speed-only, no checkpoint loaded |

1. Run 300 frames of 1080p at QP=32 on the CUDA path.
2. Discard the first 10% of the run (30 of 300 frames) as warm-up: it holds the
   first I-frame (~62-64 ms), the first P-frame (~16 ms, adaptive-I preparation)
   and the frame-1 adaptor refresh.
3. Average decode latency over the remaining **P-frames** (frames 30-299,
   n=270 per run; no I-frame falls inside the steady window).
4. Decode time is wall-clock with `torch.cuda.synchronize()` before and after the
   decode call, so it includes full GPU execution.
5. Repeat 5 times, 30 s cool-down between runs, and aggregate.

Reproduce everything:

```bash
bash test.sh                                                     # 5 repeats
python analyze_log.py logs/cuda_300f_run0*.log --warmup-frac 0.1  # report
```

## Results - five repeats

| Log | Steady P-frames | P-frame FPS |
|---|---|---|
| `logs/cuda_300f_run01.log` | 14.20 ms/f (n=270) | 70.4 |
| `logs/cuda_300f_run02.log` | 14.21 ms/f (n=270) | 70.4 |
| `logs/cuda_300f_run03.log` | 14.24 ms/f (n=270) | 70.2 |
| `logs/cuda_300f_run04.log` | 14.28 ms/f (n=270) | 70.0 |
| `logs/cuda_300f_run05.log` | 14.22 ms/f (n=270) | 70.3 |
| **Aggregate (n=1350)** | **14.23 ms/f** | **70.3** |

- Mean 14.23 ms/f, median 14.22 ms/f, stdev 0.03 ms, run-to-run spread
  14.20-14.28 ms/f -> **70.0-70.4 FPS**.
- Pooled per-frame P latency (n=1350): median 14.20 ms, p95 14.87 ms,
  min 13.55 ms, max 16.08 ms.
- The 5 adaptor refreshes (frames 65/129/193/257 plus warm-up frame 1) are the
  15.8-16.1 ms outliers; they stay in the average and move it by < 0.05 ms.
- Encoding side, for reference: 19.4-19.6 ms/f (51.1-51.4 FPS), 24,663 kbps at
  30 fps for the random-weight model. PSNR is meaningless with random weights and
  is not reported.
- Every run decodes the identical stream (30,829,202 bytes for 300 frames), so the
  spread is pure timing noise, not a bitstream difference.

## Why this is a fair P-frame number

- The steady window contains **no I-frame at all** (the only intra frame is frame 0,
  inside the discarded warm-up), so the P-frame average is not diluted by
  intra decoding.
- `reset_interval=64` is kept at the DCVC-RT default, so the feature-adaptor
  refresh cost is part of the measurement rather than hidden.
- Random weights cannot speed up the kernels: the same convolutions, quantization
  and rANS entropy coding run; only the bit values differ (see README Sec. 2).
- Latency is measured with `torch.cuda.synchronize()` on both sides of the decode
  call, so it is full GPU execution time, not kernel-launch time.

## Consistency reference

`compare_consistency.py` verifies that the CUDA path is not trading accuracy for
speed:

```bash
python compare_consistency.py                                            # operator level
python compare_consistency.py <cuda_run.log> <torch_run.log>             # + end-to-end
```

Level 1 compares the quantization / indexing / masking operators against the pure
PyTorch implementations (expected bit-exact) and the fused conv blocks (small
fp16 tolerance). Level 2 is optional: pass a CUDA-path run log and a PyTorch-path
run log (produced with `--cuda 0 --verbose 2`) to compare per-frame bit counts and
PSNR, including a head/tail drift check that would expose a cumulative bug.
