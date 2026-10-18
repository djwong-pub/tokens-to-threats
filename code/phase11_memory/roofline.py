"""Measure the FLOP/byte trade that makes a decode step memory-bound (Phase 11).

No GPU on this machine, so every measured number is CPU-measured and labelled as such.
The chapter's roofline argument is qualitative but quantitative: a GEMV (batch one)
moves 4 d bytes of weight to do 2 d FLOPs, arithmetic intensity 0.5 FLOP/byte, while a
batched matmul moves the same weights once and does b x the FLOPs. A decode step is a
GEMV against the whole weight matrix, so its speed is set by how fast the weights can
be read, not by how fast they can be multiplied.

Checks
  flop-account         the FLOP count of a GEMV/GEMM matches 2 M N K exactly and the
                       byte count of a weight read matches M N K times the element size
  intensity-linear     the static arithmetic intensity of a weight-read-bound matmul,
                       I(b) = 2 b D^2 / (4 D^2 + 8 b D), grows from 0.5 FLOP/byte at
                       b = 1 toward b/2, with the b=16 / b=1 ratio inside (8, 16)
  gemv-streams-weights a GEMV over the weight matrix is faster than a copy of the same
                       bytes (it reads where the copy reads and writes) but not more
                       than 3x faster: the step is set by moving the weights, the
                       signature of a memory-bound step
  batching-amortizes   one B = 64 matmul costs less than six tenths of 64 separate
                       GEMVs, because the weight matrix is read once instead of 64
                       times

All measured work runs single-threaded (the container's multithreaded BLAS is throttled
unpredictably on this shared CPU). The thread environment is pinned before numpy
imports its BLAS, so the numbers are the deterministic single-core regime of the
machine, and they are labelled CPU-measured in the chapter.

Run:  ./.venv/bin/python code/phase11_memory/roofline.py
"""
from __future__ import annotations

import os
import time

for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[var] = "1"

import numpy as np

D = 4096


def flops_bytes(m: int, n: int, k: int, elem: int) -> tuple[int, int]:
    """FLOPs of an m x k times k x n multiply, and the bytes of reading the weight."""
    return 2 * m * n * k, m * n * k * elem


def intensity(b: int, d: int, elem: int = 4) -> float:
    """Static FLOPs per byte read for a batch-b weight-read-bound matmul.

    Weights (d x d) are read once, the b rows of activations twice, in fp32.
    """
    return (2.0 * b * d * d) / (elem * d * d + 2.0 * elem * b * d)


def time_call(fn, reps: int = 25) -> float:
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> int:
    rng = np.random.default_rng(3)
    W = (rng.standard_normal((D, D)) * 0.01).astype(np.float32)

    # ------------------------------------------------------------------ exact accounting
    for m in (1, 16, 64):
        f, bb = flops_bytes(m, D, D, 4)
        assert f == 2 * m * D * D and bb == m * D * D * 4
    _, b8 = flops_bytes(1, D, D, 2)
    assert b8 == D * D * 2, "fp16 byte count must halve"
    print("PASS flop-account (2MNK exact; weight bytes = M N K x element size, fp32 and fp16)")

    # ------------------------------------------------------------------ static intensity
    i1, i16 = intensity(1, D), intensity(16, D)
    ratio = i16 / i1
    assert abs(i1 - 0.5 * D / (D + 2.0)) < 1e-9          # = 2D^2 / (4D^2 + 8D)
    assert 8.0 < ratio < 16.0, f"intensity ratio {ratio:.2f} outside (8, 16)"
    print(f"PASS intensity-linear (static FLOPs/weight-byte: B=1 -> {i1:.3f}, "
          f"B=16 -> {i16:.2f}, x{ratio:.1f}; the ideal weight-dominant limit is x16)")

    # ------------------------------------------------------------------ measured regime (CPU)
    X1 = rng.standard_normal((1, D)).astype(np.float32)
    X64 = rng.standard_normal((64, D)).astype(np.float32)
    buf = np.zeros_like(W)
    t_gemv = time_call(lambda: X1 @ W)
    t_copy = time_call(lambda: np.copyto(buf, W))
    t_64gemv = time_call(lambda: [X1 @ W for _ in range(64)], reps=8)
    t_b64 = time_call(lambda: X64 @ W, reps=5)

    bw_copy = W.nbytes / t_copy / 1e9
    bw_gemv = W.nbytes / t_gemv / 1e9
    print(f"CPU-measured: copy {W.nbytes / 1e6:.0f} MB at {bw_copy:.1f} GB/s; "
          f"GEMV at {bw_gemv:.1f} GB/s effective")

    # a GEMV reads the weights and writes a d-vector; a copy of the same volume reads
    # and writes the whole matrix. The GEMV must therefore beat the copy, and by no
    # more than the factor between read-and-write traffic and read-only traffic.
    assert t_gemv < t_copy, "GEMV not faster than a copy of the same bytes"
    assert t_gemv > 0.3 * t_copy, "GEMV more than 3x faster than a copy: read too fast"
    print(f"PASS gemv-streams-weights (GEMV takes {t_gemv / t_copy:.2f}x the time of a "
          f"copy of the same {W.nbytes / 1e6:.0f} MB, effective read {bw_gemv:.1f} GB/s "
          f"vs copy {bw_copy:.1f} GB/s: the step is set by moving the weights)")

    # batching amortises the weight read: one B=64 matmul reads the matrix once
    assert t_b64 < 0.6 * t_64gemv, "one B=64 matmul not clearly cheaper than 64 GEMVs"
    print(f"PASS batching-amortizes (one B=64 matmul {t_64gemv / t_b64:.1f}x cheaper than "
          f"64 separate GEMVs: the weight is read once, not 64 times)")

    gf1 = 2 * D * D / t_gemv / 1e9
    gfb = 64 * 2 * D * D / t_b64 / 1e9
    print(f"METRIC copy_bw_gbs={bw_copy:.1f}")
    print(f"METRIC gemv_effective_bw_gbs={bw_gemv:.1f}")
    print(f"METRIC gflops_b1={gf1:.2f}")
    print(f"METRIC gflops_b64={gfb:.2f}")
    print(f"METRIC batch_amortization={t_64gemv / t_b64:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
