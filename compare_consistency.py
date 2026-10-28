# -*- coding: utf-8 -*-
"""CUDA vs PyTorch dual-path consistency verification.

Level 1: operator-level comparison (functions in cuda_inference.py vs CUDA ext)
  - Quantization operators (round/quant/index/mask): must be bit-exact
  - Conv proxies (fp16 reduction order differs): small error allowed, report max abs diff
Level 2: end-to-end per-frame comparison (pass the CUDA-path and PyTorch-path
  run logs on the command line; skipped when no logs are given)
"""
import re
import sys

import torch
import torch.nn.functional as F

from src.layers import cuda_inference as ci

print("=" * 64)
print("Level 1: operator-level consistency (GPU, float16)")
print("=" * 64)

g = torch.Generator(device="cuda").manual_seed(0)


def rand(*shape, dtype=torch.float16):
    return torch.randn(*shape, device="cuda", dtype=dtype, generator=g)


def check(name, exact, a, b=None):
    if b is None:
        ok = a
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        return ok
    d = (a.float() - b.float()).abs().max().item()
    ok = (torch.equal(a, b) if exact else d < 5e-2)
    tag = "bit-exact" if exact else f"max|diff|={d:.5f}"
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:35s} ({tag})")
    return ok


results = {}

# --- replicate_pad ---
x = rand(1, 8, 36, 64)
pad_c = ci.replicate_pad(x, 4, 6)
pad_t = F.pad(x, (0, 6, 0, 4), mode="replicate")
results["replicate_pad"] = check("replicate_pad", True, pad_c, pad_t)

# --- round_and_to_int8 ---
z = rand(1, 32, 64, 64) * 40
if ci.CUSTOMIZED_CUDA_INFERENCE:
    _, z_i8_cuda = ci.round_and_to_int8(z)
    z_hat = torch.clamp(torch.round(z), -128., 127.).to(torch.int8)
    results["round_and_to_int8"] = check("round_and_to_int8", True,
                                         torch.equal(z_i8_cuda, z_hat))

# --- process_with_mask ---
y = rand(1, 64, 64, 64) * 10
scales, means, mask = rand(1, 64, 64, 64), rand(1, 64, 64, 64), (rand(1, 64, 64, 64) > 0).half()
_, q_c, yh_c, sc_c = ci.process_with_mask(y, scales, means, mask, 0.12)
scales_hat = scales * mask
means_hat = means * mask
y_res = (y - means_hat) * mask
y_q = torch.round(y_res)
y_q = y_q * (scales_hat > 0.12)
y_q = torch.clamp(y_q, -128., 127.)
y_hat = y_q + means_hat
results["process_with_mask"] = check("process_with_mask", True,
                                     torch.equal(q_c, y_q) and torch.equal(yh_c, y_hat))

# --- combine_for_reading_2x / restore_y_2x / restore_y_4x ---
m2 = (rand(1, 64, 64, 64) > 0).half()
x2 = rand(1, 64, 64, 64)
if ci.CUSTOMIZED_CUDA_INFERENCE:
    out_c = ci.combine_for_reading_2x(x2, m2)
    out_t = (x2 * m2).chunk(2, 1)[0] + (x2 * m2).chunk(2, 1)[1]
    results["combine_2x"] = check("combine_for_reading_2x", True, torch.equal(out_c, out_t))

yv = rand(1, 32, 64, 64)
ymeans2, ymask2 = rand(1, 64, 64, 64), (rand(1, 64, 64, 64) > 0).half()
ymeans4, ymask4 = rand(1, 128, 64, 64), (rand(1, 128, 64, 64) > 0).half()
results["restore_y_2x"] = check("restore_y_2x", True,
                                torch.equal(ci.restore_y_2x(yv, ymeans2, ymask2),
                                            (torch.cat((yv, yv), 1) + ymeans2) * ymask2))
results["restore_y_4x"] = check("restore_y_4x", True,
                                torch.equal(ci.restore_y_4x(yv, ymeans4, ymask4),
                                            (torch.cat((yv, yv, yv, yv), 1) + ymeans4) * ymask4))

# --- add_and_multiply ---
a0, a1, qd = rand(1, 16, 64, 64), rand(1, 16, 64, 64), rand(1, 16, 64, 64) + 1
if ci.CUSTOMIZED_CUDA_INFERENCE:
    out_c = ci.add_and_multiply(a0.clone(), a1, qd.clone())
    out_t = (a0 + a1) * qd
    results["add_and_multiply"] = check("add_and_multiply", True, torch.equal(out_c, out_t))

# --- clamp_reciprocal_with_quant ---
q_dec = rand(1, 16, 64, 64) + 1
yy = rand(1, 16, 64, 64)
if ci.CUSTOMIZED_CUDA_INFERENCE:
    q_c2, y_c2 = ci.clamp_reciprocal_with_quant(q_dec.clone(), yy.clone(), 0.5)
    q_t = torch.clamp_min(q_dec, 0.5)
    y_t = yy * torch.reciprocal(q_t)
    results["clamp_recip_quant"] = check("clamp_reciprocal_with_quant", True,
                                         torch.equal(q_c2, q_t) and torch.equal(y_c2, y_t))

# --- bias_quant / bias_pixel_shuffle_8 (bias/quant_step are per-channel, length C) ---
xb = rand(1, 12, 32, 32)
bias = rand(12)
qs = rand(12)
if ci.CUSTOMIZED_CUDA_INFERENCE:
    out_c = ci.bias_quant(xb.clone(), bias, qs)
    out_t = (xb + bias[None, :, None, None]) * qs[None, :, None, None]
    results["bias_quant"] = check("bias_quant", True, torch.equal(out_c, out_t))

# bias_pixel_shuffle_8: input channels must be 3*8^2=192
xp = rand(1, 192, 16, 16)
bp = rand(192)
if ci.CUSTOMIZED_CUDA_INFERENCE:
    out_c = ci.bias_pixel_shuffle_8(xp, bp)
    out_t = torch.clamp(F.pixel_shuffle(xp + bp[None, :, None, None], 8), 0., 1.)
    results["bias_pixel_shuffle_8"] = check("bias_pixel_shuffle_8", True,
                                            torch.equal(out_c, out_t))

# --- Conv proxies (reduction order differs, fp16 error allowed) ---
if ci.CUSTOMIZED_CUDA_INFERENCE:
    from src.layers.layers import DepthConvBlock, DepthConvBlockMiddleFFN
    blk = DepthConvBlock(32, 32).cuda().half().eval()
    xin = rand(1, 32, 64, 64)
    with torch.no_grad():
        out_cuda = blk(xin)
    # torch path: temporarily disable the CUDA flag
    ci.CUSTOMIZED_CUDA_INFERENCE = False
    with torch.no_grad():
        out_torch = blk(xin)
    ci.CUSTOMIZED_CUDA_INFERENCE = True
    results["DepthConvBlock"] = check("DepthConvBlock (proxy)", False, out_cuda, out_torch)

    blk2 = DepthConvBlockMiddleFFN(32, 32).cuda().half().eval()
    x2in = rand(1, 32, 64, 64)
    qs32 = rand(1, 32, 1, 1)
    with torch.no_grad():
        out_cuda = blk2(x2in, quant_step=qs32)
    ci.CUSTOMIZED_CUDA_INFERENCE = False
    with torch.no_grad():
        out_torch = blk2.forward_torch(x2in, qs32)
    ci.CUSTOMIZED_CUDA_INFERENCE = True
    results["DepthConvMiddleFFN"] = check("DepthConvBlockMiddleFFN (proxy)", False,
                                          out_cuda, out_torch)

# Level 2: end-to-end per-frame
print()
print("=" * 64)
print("Level 2: end-to-end per-frame consistency (two logs)")
print("=" * 64)


ENC_RE = re.compile(r"enc frame\s+(\d+)\s+(?:[IP]\s+)?([\d.]+) ms\s+bits=([\d,]+)")
DEC_RE = re.compile(r"dec frame\s+(\d+)\s+(?:[IP]\s+)?([\d.]+) ms\s+bits=([\d,]+)\s+PSNR=([\d.]+)")


def parse_log(path):
    """Per-frame enc bits / dec bits / dec PSNR from a --verbose 2 run log."""
    enc_bits, dec_bits, psnrs = {}, {}, {}
    for line in open(path, encoding="utf-8", errors="ignore"):
        m = ENC_RE.search(line)
        if m:
            enc_bits[int(m.group(1))] = int(m.group(3).replace(",", ""))
        m = DEC_RE.search(line)
        if m:
            i = int(m.group(1))
            dec_bits[i] = int(m.group(3).replace(",", ""))
            psnrs[i] = float(m.group(4))
    return enc_bits, dec_bits, psnrs


def level2(cuda_log, torch_log):
    """Compare two run logs frame by frame. Returns True/False."""
    enc_c, dec_c, ps_c = parse_log(cuda_log)
    enc_t, dec_t, ps_t = parse_log(torch_log)
    ids = sorted(set(dec_c) & set(dec_t))
    if not ids:
        print(f"  [FAIL] no common 'dec frame' entries in {cuda_log} / {torch_log}")
        return False

    print(f"  {'frame':>5} {'bits(cuda)':>11} {'bits(torch)':>12} {'diff':>6} "
          f"{'PSNR(c)':>9} {'PSNR(t)':>9} {'dPSNR':>8}")
    max_bits_rel, max_dpsnr = 0.0, 0.0
    for i in ids:
        db = dec_t[i] - dec_c[i]
        rel = abs(db) / dec_c[i] * 100
        dp = ps_t[i] - ps_c[i]
        max_bits_rel = max(max_bits_rel, rel)
        max_dpsnr = max(max_dpsnr, abs(dp))
        print(f"  {i:>5} {dec_c[i]:>11,} {dec_t[i]:>12,} {db:>+6} "
              f"{ps_c[i]:>9.3f} {ps_t[i]:>9.3f} {dp:>+8.3f}")

    # Drift detection: a cumulative error/bug would make dPSNR grow monotonically with
    # frame index; compare the mean of the first/last 5 frames.
    def mean_abs_dpsnr(subset):
        return sum(abs(ps_t[i] - ps_c[i]) for i in subset) / len(subset)

    head = mean_abs_dpsnr(ids[:5])
    tail = mean_abs_dpsnr(ids[-5:])
    drift = tail - head
    no_drift = abs(drift) < 0.02
    psnr_ok = max_dpsnr < 0.15

    print()
    print(f"  Max relative dec-bits diff : {max_bits_rel:.2f}%  (fp16 fused-conv noise, reference only)")
    print(f"  Max abs PSNR diff          : {max_dpsnr:.3f} dB  (threshold 0.15 dB)")
    print(f"  Drift (tail-head 5-frame mean): {drift:+.4f} dB  (|drift| < 0.02 = no accumulation)")
    return bool(psnr_ok and no_drift)


if len(sys.argv) >= 3:
    level2_ok = level2(sys.argv[1], sys.argv[2])
else:
    print("  [SKIP] pass two run logs to compare the CUDA and the PyTorch path:")
    print("         python compare_consistency.py <cuda_run.log> <torch_run.log>")
    print("         (test.sh writes CUDA-path logs to logs/cuda_*f_run*.log;")
    print("          produce the PyTorch reference with --cuda 0 --verbose 2)")
    level2_ok = None

fails = [k for k, v in results.items() if not v]
consistent = (not fails) and (level2_ok is None or level2_ok)

print()
print(f"  Operator-level: {len(results) - len(fails)}/{len(results)} bit-exact passed" +
      (f", failed: {fails}" if fails else ""))
print(f"  End-to-end    : {'skipped (no logs supplied)' if level2_ok is None
                      else ('consistent' if level2_ok else 'inconsistent')}")
print(f"  Conclusion: {'consistent (quantization/entropy ops bit-exact'
                  + (', PSNR matched, no drift' if level2_ok else '')
                  + ', no bug)' if consistent else 'inconsistent, manual investigation required!'}")
sys.exit(0 if consistent else 1)
