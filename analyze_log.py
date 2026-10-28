#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract per-frame decode latency from test_video.py logs and compute the
steady-state decoding FPS.

Usage:
  python analyze_log.py [log ...] [--warmup-frac 0.1]

With one log a detailed report is printed; with several logs (e.g. the 10
repeats written by test.sh) a compact per-run table plus the aggregate
statistics across runs are printed.

--warmup-frac X  : drop the first X (0..1) fraction of decoded frames of every
                   run before computing the steady-state metric (0.1 = 10%).
                   Default 0 reproduces the frame-2-onward metric.
"""
import argparse
import glob
import math
import re
import statistics
import sys

# New logs:  "  dec frame   0  I   70.07 ms  bits=..."   (type present)
# Old logs:  "  dec frame   0   70.07 ms  bits=..."
NEW_RE = re.compile(r"\s*dec frame\s+(\d+)\s+([IP])\s+([\d.]+)\s+ms")
OLD_RE = re.compile(r"\s*dec frame\s+(\d+)\s+([\d.]+)\s+ms")

DEFAULT_GLOB = "logs/cuda_*f_run*.log"


def parse_dec_ms(path):
    """Return list of (index, type_or_None, ms)."""
    dec = []
    with open(path) as f:
        for line in f:
            m = NEW_RE.match(line)
            if m:
                dec.append((int(m.group(1)), m.group(2), float(m.group(3))))
                continue
            m = OLD_RE.match(line)
            if m:
                dec.append((int(m.group(1)), None, float(m.group(2))))
    return dec


def fps(avg_ms):
    return 1000.0 / avg_ms if avg_ms > 0 else float("inf")


def warmup_count(n, frac):
    """Number of leading frames to drop (same rule as --warmup-frac)."""
    if frac > 0:
        return min(int(math.ceil(n * frac)), n - 1) if n > 1 else 0
    return min(2, n - 1) if n > 1 else 0


def run_stats(dec, frac):
    """Steady-state statistics of one run."""
    n = len(dec)
    warm = warmup_count(n, frac)
    times = [d[2] for d in dec]
    types = [d[1] for d in dec]
    steady_t = times[warm:]
    steady_ty = types[warm:]
    p_t = [t for t, ty in zip(steady_t, steady_ty) if ty == 'P']
    i_t = [t for t, ty in zip(steady_t, steady_ty) if ty == 'I']
    return {
        'n': n,
        'warm': warm,
        'i0': times[0] if times else float('nan'),
        'p1': times[1] if n > 1 else float('nan'),
        'all_avg': sum(times) / n if n else float('nan'),
        'steady_avg': sum(steady_t) / len(steady_t) if steady_t else float('nan'),
        'p_avg': sum(p_t) / len(p_t) if p_t else float('nan'),
        'p_n': len(p_t),
        'p_times': p_t,
        'i_avg': sum(i_t) / len(i_t) if i_t else float('nan'),
        'i_n': len(i_t),
        'typed': all(t is not None for t in types),
    }


def report_single(path, st, frac):
    print(f"log: {path}  ({st['n']} frames)")
    print("-" * 66)
    print(f"I-frame (frame 0)             : {st['i0']:6.2f} ms")
    if st['n'] > 1:
        print(f"First P-frame (frame 1, warmup): {st['p1']:6.2f} ms")
    print("-" * 66)
    print(f"All-frame average             : {st['all_avg']:6.2f} ms/f  "
          f"({fps(st['all_avg']):5.1f} FPS)")
    print(f"Steady frames (frame {st['warm']}..{st['n'] - 1}, n={st['n'] - st['warm']}) : "
          f"{st['steady_avg']:6.2f} ms/f  ({fps(st['steady_avg']):5.1f} FPS)")
    if frac > 0:
        print(f"  ^ warm-up excluded: first {st['warm']}/{st['n']} frames "
              f"({100.0 * st['warm'] / st['n']:.1f}%, requested {100 * frac:.1f}%)")
    if st['typed']:
        print("-" * 66)
        if st['p_n']:
            print(f"Steady P-frames only (n={st['p_n']})   : {st['p_avg']:6.2f} ms/f  "
                  f"({fps(st['p_avg']):5.1f} FPS)  <- P-frame metric")
        if st['i_n']:
            print(f"Steady I-frames (n={st['i_n']})        : {st['i_avg']:6.2f} ms/f  "
                  f"({fps(st['i_avg']):5.1f} FPS)")
    print("-" * 66)


def report_multi(paths, stats, frac):
    print(f"{'run':<26}{'frames':>7}{'P n':>6}{'P ms/f':>9}{'P FPS':>8}"
          f"{'steady ms/f':>13}{'steady FPS':>12}")
    print("-" * 81)
    for path, st in zip(paths, stats):
        name = path.rsplit('/', 1)[-1]
        print(f"{name:<26}{st['n']:>7}{st['p_n']:>6}{st['p_avg']:>9.2f}"
              f"{fps(st['p_avg']):>8.1f}{st['steady_avg']:>13.2f}"
              f"{fps(st['steady_avg']):>12.1f}")
    print("-" * 81)

    p_avgs = [st['p_avg'] for st in stats if st['p_n']]
    if not p_avgs:
        return
    p_fps = [fps(v) for v in p_avgs]
    n_runs = len(p_avgs)
    print(f"{'ACROSS RUNS':<26}{'':>7}{'':>6}{'':>9}{'':>8}{'':>13}{'':>12}")
    print(f"  runs                 : {n_runs}")
    print(f"  steady P-frame avg   : mean {statistics.mean(p_avgs):6.2f} ms/f  "
          f"({fps(statistics.mean(p_avgs)):5.1f} FPS)")
    print(f"  ... median           :      {statistics.median(p_avgs):6.2f} ms/f  "
          f"({fps(statistics.median(p_avgs)):5.1f} FPS)")
    print(f"  ... stdev            :      {statistics.stdev(p_avgs):6.2f} ms/f"
          if n_runs > 1 else "  ... stdev            :      n/a")
    print(f"  ... min / max        :      {min(p_avgs):6.2f} / {max(p_avgs):.2f} ms/f  "
          f"({min(p_fps):5.1f} / {max(p_fps):.1f} FPS)")
    print(f"  per-run FPS range    : {min(p_fps):.1f} - {max(p_fps):.1f} FPS  "
          f"(mean {statistics.mean(p_fps):.1f} FPS)")

    pooled = [t for st in stats for t in st['p_times']]
    if pooled:
        print(f"  pooled P-frames      : n={len(pooled)}  "
              f"{statistics.mean(pooled):.2f} ms/f  ({fps(statistics.mean(pooled)):.1f} FPS)")
    steady = [st['steady_avg'] for st in stats]
    print(f"  all steady frames    : mean {statistics.mean(steady):.2f} ms/f  "
          f"({fps(statistics.mean(steady)):.1f} FPS)  [I-frames included]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="*",
                    help=f"log file(s); default: {DEFAULT_GLOB}")
    ap.add_argument("--warmup-frac", type=float, default=0.0,
                    help="drop first fraction of frames per run (e.g. 0.1)")
    a = ap.parse_args()

    paths = a.path if a.path else sorted(glob.glob(DEFAULT_GLOB))
    if not paths:
        print(f"[ERROR] no logs given and nothing matched {DEFAULT_GLOB}")
        sys.exit(1)

    stats = []
    for path in paths:
        dec = parse_dec_ms(path)
        if not dec:
            print(f"[ERROR] no 'dec frame' lines found in {path}")
            sys.exit(1)
        stats.append(run_stats(dec, a.warmup_frac))

    if len(paths) == 1:
        report_single(paths[0], stats[0], a.warmup_frac)
    else:
        report_multi(paths, stats, a.warmup_frac)


if __name__ == "__main__":
    main()
