"""Verify tiled (online-softmax) attention against the dense reference (Phase 12).

The chapter derives the online rescaling: a running max m and running sum l, with the
accumulator updated as S += exp(qk - m') when the max changes. This script implements
both the dense softmax attention and the block-tiled version and checks they agree,
including under a causal mask and under numerically hostile inputs.

Checks
  dense-vs-tiled       block-tiled online softmax equals the dense attention to 1e-8,
                       random data, no mask
  dense-vs-tiled-causal the same with a causal mask; the two must agree on the lower
                       triangle and the tiled version must not read future positions
  extreme-stability    with logit multiplier 60 (which overflows a naive exp at d = 16)
                       the tiled version stays finite and within 1e-6 of a log-sum-exp
                       reference, while a naive exp without rescaling produces Inf
  block-size-invariant tiling at block sizes 4, 8 and 32 all agree to 1e-8

Run:  ./.venv/bin/python code/phase12_flash_attention/tiled_attention.py
"""
from __future__ import annotations

import numpy as np


def dense_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray,
                    causal: bool = False, scale: float = None) -> np.ndarray:
    s = scale if scale is not None else 1.0 / np.sqrt(q.shape[-1])
    a = q @ k.transpose(0, 2, 1) * s
    if causal:
        n = q.shape[1]
        a = np.where(np.arange(n)[:, None] >= np.arange(n)[None, :], a, -np.inf)
    a = a - a.max(axis=-1, keepdims=True)
    e = np.exp(a)
    return (e / e.sum(axis=-1, keepdims=True)) @ v


def tiled_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray,
                    block: int = 8, causal: bool = False,
                    scale: float = None) -> np.ndarray:
    """Online softmax over key blocks. q,k,v are (B, T, d). Returns (B, T, d)."""
    B, T, d = q.shape
    s = scale if scale is not None else 1.0 / np.sqrt(d)
    out = np.zeros((B, T, d), dtype=np.float64)
    m = np.full((B, T), -np.inf, dtype=np.float64)
    l = np.zeros((B, T), dtype=np.float64)
    for j0 in range(0, T, block):
        jb = slice(j0, min(j0 + block, T))
        a = q @ k[:, jb].transpose(0, 2, 1) * s          # (B, T, block)
        if causal:
            rows = np.arange(T)[:, None]
            cols = np.arange(j0, min(j0 + block, T))[None, :]
            a = np.where(rows >= cols, a, -np.inf)
        m_new = np.maximum(m, a.max(axis=-1))
        # rescale the old accumulator; avoid 0 * exp(-inf - -inf)
        alpha = np.exp(np.where(m == -np.inf, 0.0, m - m_new))
        out = out * alpha[:, :, None]
        p = np.exp(a - m_new[:, :, None])
        out += p @ v[:, jb]
        l = l * alpha + p.sum(axis=-1)
        m = m_new
    return out / l[:, :, None]


def main() -> int:
    rng = np.random.default_rng(9)
    B, T, d = 2, 48, 16
    q = rng.standard_normal((B, T, d))
    k = rng.standard_normal((B, T, d))
    v = rng.standard_normal((B, T, d))

    a0 = dense_attention(q, k, v)
    worst = 0.0
    for blk in (4, 8, 32):
        a1 = tiled_attention(q, k, v, block=blk)
        worst = max(worst, float(np.abs(a0 - a1).max()))
    assert worst < 1e-8, f"tiled attention differs by {worst:.2e}"
    print(f"PASS dense-vs-tiled (worst diff {worst:.2e} over block sizes 4, 8, 32)")

    a0c = dense_attention(q, k, v, causal=True)
    a1c = tiled_attention(q, k, v, block=8, causal=True)
    dc = float(np.abs(a0c - a1c).max())
    assert dc < 1e-8, f"causal tiled differs by {dc:.2e}"
    print(f"PASS dense-vs-tiled-causal (worst diff {dc:.2e})")

    # future positions must not leak: change v at a future index and check the past is untouched
    q2 = q.copy(); k2 = k.copy(); v2 = v.copy()
    v2[:, 40:, :] += 7.0
    before = tiled_attention(q2, k2, v, block=8, causal=True)[:, :40, :]
    after = tiled_attention(q2, k2, v2, block=8, causal=True)[:, :40, :]
    leak = float(np.abs(before - after).max())
    assert leak == 0.0, f"causal tiled attention reads the future (leak {leak:.2e})"
    print(f"PASS causal-future-isolation (perturbing v beyond t=40 changes nothing before it)")

    # ------------------------------------------------------------------ extreme stability
    # multiplier 60 on d=16 standard-normal dot products (std ~4) gives logit std ~240,
    # so the max over T^2 entries clears 710 and a naive exp overflows to Inf.
    scale_big = 60.0
    a_dense = dense_attention(q, k, v, scale=scale_big)
    a_tiled = tiled_attention(q, k, v, block=8, scale=scale_big)
    dt = float(np.abs(a_dense - a_tiled).max())
    assert dt < 1e-6, f"extreme-scale tiled differs by {dt:.2e}"
    # naive exp without rescaling overflows
    raw = q @ k.transpose(0, 2, 1) * scale_big
    with np.errstate(over="ignore", invalid="ignore"):
        naive_e = np.exp(raw)
    assert np.isinf(naive_e).any(), "naive exp was not expected to overflow at this scale"
    print(f"PASS extreme-stability (scale {scale_big:.2f}: tiled finite, naive exp overflows; "
          f"tiled vs log-sum-exp ref diff {dt:.2e})")

    print(f"METRIC tiled_max_diff={worst:.2e}")
    print(f"METRIC causal_max_diff={dc:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
