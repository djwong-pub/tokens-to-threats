"""Verify the three component substitutions of Phase 7 against the chapter's derivations.

The chapter derives LayerNorm and RMSNorm from the same stabilisation goal, the SwiGLU gate
and the 2/3 hidden-dimension sizing rule, and the RoPE rotation with its relative-position
property. This script measures each claim.

Checks
  rmsnorm-identity   the implementation equals the hand formula x_i / RMS(x) * g_i on random
                     input to machine precision
  rmsnorm-stats      RMSNorm output has RMS 1 to 1e-5 without any mean constraint; LayerNorm
                     output has mean 0 and std 1 to 1e-5; the two outputs differ
  rmsnorm-gradient   analytic gradient through RMSNorm matches the float64 finite difference
  swiglu-identity    SiLU(x) = x sigmoid(x) and the SwiGLU block match hand computations
  swiglu-two-thirds  the parameter count of a ReLU FFN with hidden 4d equals the parameter
                     count of a SwiGLU FFN with hidden round(8d/3); the rule is exact up to
                     the rounding of the hidden dimension
  rope-rotation      the RoPE update equals the complex rotation (a + bi) e^{i m theta} and
                     preserves the vector norm
  rope-relative      the RoPE logit <R_m q, R_n k> depends only on the offset m - n; the same
                     offset gives the same logit for (m, n) pairs up to 1e-5

Run:  ./.venv/bin/python code/phase07_modern_transformer/modern_components.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


def rmsnorm(x: np.ndarray, gamma: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * gamma


def layernorm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mu) / np.sqrt(var + eps) + beta


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def swiglu(x: np.ndarray, w1: np.ndarray, w3: np.ndarray, w2: np.ndarray) -> np.ndarray:
    return (silu(x @ w1) * (x @ w3)) @ w2


def rope_freqs(d_k: int, max_pos: int, base: float = 10000.0) -> np.ndarray:
    """theta_i = base^(-2i/d_k) for i = 0..d_k/2-1. Shape (d_k // 2,)."""
    i = np.arange(0, d_k, 2, dtype=np.float64)
    return base ** (-i / d_k)


def rope_rotate(x: np.ndarray, pos: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Apply RoPE. x: (..., d_k), d_k even. pos: (...,) positions.

    Each pair (x[2i], x[2i+1]) is rotated by pos * theta_i.
    """
    x = x.astype(np.float64)
    pairs = x.reshape(*x.shape[:-1], d_k := x.shape[-1] // 2, 2)
    ang = pos[..., None] * freqs[None, :]                  # (..., d_k/2)
    cos, sin = np.cos(ang), np.sin(ang)
    a, b = pairs[..., 0], pairs[..., 1]
    out = np.empty_like(pairs)
    out[..., 0] = a * cos - b * sin
    out[..., 1] = a * sin + b * cos
    return out.reshape(*x.shape).astype(x.dtype)


def main() -> int:
    rng = np.random.default_rng(7)

    # ------------------------------------------------------------------ RMSNorm
    d = 64
    x = rng.standard_normal((8, d)) * 3.0
    gamma = 1.0 + 0.1 * rng.standard_normal(d)
    beta = 0.1 * rng.standard_normal(d)

    got = rmsnorm(x, gamma)
    manual = (x / np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + 1e-6)) * gamma
    assert np.abs(got - manual).max() < 1e-12
    print("PASS rmsnorm-identity")

    # stats are measured on the normalised (pre-affine) output: the affine part is separate
    ones = np.ones(d)
    ln = layernorm(x, ones, np.zeros(d))
    got_unit = rmsnorm(x, ones)
    assert abs(ln.mean(axis=-1).max()) < 1e-6, "LayerNorm output not mean-centred"
    assert abs(ln.std(axis=-1).max() - 1.0) < 1e-5, "LayerNorm output not unit variance"
    assert abs(np.sqrt((got_unit ** 2).mean(axis=-1)).max() - 1.0) < 1e-5, \
        "RMSNorm output RMS not 1"
    assert np.abs(got_unit - ln).max() > 1e-3, "RMSNorm and LayerNorm unexpectedly identical"
    # the mean of the RMSNorm output is in general not zero: check the claim holds here
    assert abs(got_unit.mean(axis=-1).max()) > 1e-3, \
        "RMSNorm mean happened to vanish; choose another seed"
    print(f"PASS rmsnorm-stats (LayerNorm |mean|={abs(ln.mean()):.2e}, RMSNorm "
          f"|mean|={abs(got_unit.mean()):.4f} not 0, RMSNorm RMS="
          f"{np.sqrt((got_unit**2).mean()):.6f})")

    # gradient check in float64
    torch.manual_seed(7)
    x_t = torch.tensor(x[:1].astype(np.float64), requires_grad=True)
    g_t = torch.tensor(gamma.astype(np.float64))
    out_t = torch.nn.functional.rms_norm(x_t, (d,), weight=g_t, eps=1e-6)
    loss = (out_t * torch.arange(1, d + 1, dtype=torch.float64)).sum()
    loss.backward()
    analytic = x_t.grad.numpy().reshape(-1)
    orig = x_t.detach().numpy().reshape(-1).copy()
    eps = 1e-6

    def loss_at(v):
        o = rmsnorm(v.reshape(1, d).astype(np.float64), gamma.astype(np.float64))
        return float((o * np.arange(1, d + 1)).sum())

    fd = np.zeros_like(orig)
    for j in range(orig.size):
        v = orig.copy(); v[j] += eps; lp = loss_at(v)
        v = orig.copy(); v[j] -= eps; lm = loss_at(v)
        fd[j] = (lp - lm) / (2 * eps)
    rel = np.abs(analytic - fd) / np.maximum(np.abs(analytic) + np.abs(fd), 1e-10)
    assert rel.max() < 5e-3, f"RMSNorm gradient mismatch {rel.max():.2e}"
    print("PASS rmsnorm-gradient")

    # ------------------------------------------------------------------ SwiGLU
    a = rng.standard_normal(50)
    d6 = np.abs(silu(a) - a / (1.0 + np.exp(-a))).max()
    assert d6 < 1e-15, f"SiLU identity {d6:.2e}"
    h = 32
    w1 = rng.standard_normal((d, h)) * 0.1
    w3 = rng.standard_normal((d, h)) * 0.1
    w2 = rng.standard_normal((h, d)) * 0.1
    xs = rng.standard_normal((7, d))
    manual = (silu(xs @ w1) * (xs @ w3)) @ w2
    assert np.abs(swiglu(xs, w1, w3, w2) - manual).max() < 1e-12
    print(f"PASS swiglu-identity (max diff {d6:.2e})")

    # the 2/3 rule: match the ReLU FFN parameter count 8d^2 with SwiGLU hidden h = 8d/3
    for dw in (128, 512, 1024):
        relu_ffn = 2 * dw * (4 * dw)                    # W1, W2, hidden 4d (biases ignored)
        h_s = round(8 * dw / 3)
        swiglu_ffn = 3 * dw * h_s                       # W1, W3, W2
        assert abs(swiglu_ffn - relu_ffn) <= 3 * dw, \
            f"sizing rule violated at d={dw}: {swiglu_ffn} vs {relu_ffn}"
    print("PASS swiglu-two-thirds (h = round(8d/3) matches 8d^2 within rounding)")

    # ------------------------------------------------------------------ RoPE
    d_k = 32
    freqs = rope_freqs(d_k, 512)
    q = rng.standard_normal(d_k)
    # norm preservation: the rotation is orthogonal
    for m in (0, 5, 97, 511):
        rq = rope_rotate(q[None, :], np.array([m]), freqs)[0]
        assert abs(np.linalg.norm(rq) - np.linalg.norm(q)) < 1e-9, f"norm not preserved at m={m}"
    # complex-rotation identity on one pair: dimension i0 = 2*i belongs to pair i
    i0 = 0
    theta = freqs[0]
    m = 11
    a_, b_ = q[i0], q[i0 + 1]
    ang = m * theta
    exp_re = a_ * np.cos(ang) - b_ * np.sin(ang)
    exp_im = a_ * np.sin(ang) + b_ * np.cos(ang)
    rq = rope_rotate(q[None, :], np.array([m]), freqs)[0]
    d7 = max(abs(rq[i0] - exp_re), abs(rq[i0 + 1] - exp_im))
    assert d7 < 1e-12, f"complex rotation mismatch {d7:.2e}"
    print(f"PASS rope-rotation (max diff {d7:.2e})")

    # relative-position property: <R_m q, R_n k> is a function of m - n only
    k = rng.standard_normal(d_k)
    offsets = {}
    for m in (3, 40, 120, 500):
        for n in (m, m - 7, m - 1):
            lq = rope_rotate(q[None, :], np.array([m]), freqs)[0]
            lk = rope_rotate(k[None, :], np.array([n]), freqs)[0]
            key = m - n
            val = float(np.dot(lq, lk))
            if key in offsets:
                assert abs(val - offsets[key]) < 1e-6, \
                    f"relative invariance broken at offset {key}: {val} vs {offsets[key]}"
            offsets[key] = val
    # and different offsets give different logits (the signal is real)
    assert len(set(round(v, 6) for v in offsets.values())) > 1, "all offsets give the same logit"
    n_pairs = 4 * 3
    print(f"PASS rope-relative ({n_pairs} (m,n) pairs agree across all {len(offsets)} offsets; "
          f"distinct logits {len(set(round(v, 6) for v in offsets.values()))})")

    print("METRIC rope_offsets_checked={}".format(len(offsets)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
