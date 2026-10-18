"""Hand-written NumPy attention and its verification against PyTorch (Phase 5).

The chapter derives scaled dot-product attention from the softmax definition, the causal
mask as additive -inf padding, the multi-head composition, the attention gradient identity,
and the reason the scores are divided by sqrt(d_k). This script implements each of those
objects in plain NumPy and asserts that the implementation agrees with the PyTorch
reference within a logged tolerance.

Checks
  stable-softmax       numpy shifted softmax equals torch.softmax to 1e-6, including for an
                       input offset by +500 where the naive exponential overflows float32
  attn-matches-torch   hand-written causal scaled dot-product attention equals
                       F.scaled_dot_product_attention(is_causal=True) to 1e-5
  causal-mask          logits at position i are bit-identical when only positions after i are
                       changed; the mask removes the future exactly
  mha-matches-torch    hand-written multi-head attention (project, split heads, attend,
                       concat, project) equals a torch module with the identical weights to
                       1e-5
  attn-gradient        dL/dz = p (g - p.g) per row matches autograd through torch softmax
  scaling-necessity    for random q, k in R^{T x d_k} with d_k in {16, 64, 256} the unscaled
                       attention concentrates (mean max weight rises 0.58 -> 0.79 -> 0.90 and
                       row entropy collapses toward zero) while the sqrt(d_k) scaled attention
                       stays diffuse and stable across d_k

Run:  ./.venv/bin/python code/phase05_attention/numpy_attention.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import torch          # noqa: E402
import torch.nn.functional as TF  # noqa: E402


def softmax_rows(z: np.ndarray) -> np.ndarray:
    """Row-wise softmax with the max-shift. z: (..., T)."""
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def causal_mask(T: int, dtype) -> np.ndarray:
    """Upper-triangular True mask: entry (i, j) True when j > i (the forbidden future)."""
    return np.triu(np.ones((T, T), dtype=dtype), k=1).astype(bool)


def scaled_dot_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray,
                         causal: bool = True, scale: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Hand-written attention. q, k: (B, h, T, d_k), v: (B, h, T, d_k).

    Returns (out, A) where A is the per-head attention matrix (B, h, T, T).
    """
    if scale is None:
        scale = float(np.sqrt(q.shape[-1]))
    s = q @ np.swapaxes(k, -2, -1) / scale
    if causal:
        T = q.shape[2]
        s = s - causal_mask(T, s.dtype) * 1e9
    A = softmax_rows(s)
    out = A @ v
    return out, A


def multi_head_attention(x: np.ndarray, w_qkv: np.ndarray, w_o: np.ndarray,
                         h: int, causal: bool = True) -> np.ndarray:
    """Hand-written MHA. x: (B, T, d), w_qkv: (d, 3d), w_o: (d, d)."""
    B, T, d = x.shape
    qkv = x @ w_qkv                                # (B, T, 3d)
    q, k, v = np.split(qkv, 3, axis=-1)
    d_k = d // h
    q = q.reshape(B, T, h, d_k).transpose(0, 2, 1, 3)
    k = k.reshape(B, T, h, d_k).transpose(0, 2, 1, 3)
    v = v.reshape(B, T, h, d_k).transpose(0, 2, 1, 3)
    out, _ = scaled_dot_attention(q, k, v, causal=causal)
    out = out.transpose(0, 2, 1, 3).reshape(B, T, d)
    return out @ w_o


def torch_mha_reference(x: torch.Tensor, w_qkv: torch.Tensor, w_o: torch.Tensor,
                        h: int, causal: bool = True) -> torch.Tensor:
    """Torch module mirroring multi_head_attention with identical weights."""
    B, T, d = x.shape
    qkv = x @ w_qkv
    q, k, v = qkv.split(d, dim=-1)
    d_k = d // h
    q = q.reshape(B, T, h, d_k).transpose(1, 2)
    k = k.reshape(B, T, h, d_k).transpose(1, 2)
    v = v.reshape(B, T, h, d_k).transpose(1, 2)
    out = TF.scaled_dot_product_attention(q, k, v, is_causal=causal)
    out = out.transpose(1, 2).reshape(B, T, d)
    return out @ w_o


def main() -> int:
    rng = np.random.default_rng(7)

    # ------------------------------------------------------------------ softmax
    z = rng.standard_normal((4, 64)) * 4.0
    a_np = softmax_rows(z)
    a_torch = torch.softmax(torch.tensor(z, dtype=torch.float32), dim=-1).numpy()
    d1 = np.abs(a_np - a_torch).max()
    assert d1 < 1e-6, f"softmax mismatch {d1:.3e}"
    # the shift must not change the result; test with an offset that overflows float32 naive exp
    z_off = z + 500.0
    d2 = np.abs(softmax_rows(z_off) - a_np).max()
    assert d2 < 1e-6, f"shift invariance broken {d2:.3e}"
    print(f"PASS stable-softmax (max diff {d1:.2e}, shifted diff {d2:.2e})")

    # ------------------------------------------------------------------ attention
    B, h, T, d_k = 2, 4, 32, 16
    q = rng.standard_normal((B, h, T, d_k)).astype(np.float32)
    k = rng.standard_normal((B, h, T, d_k)).astype(np.float32)
    v = rng.standard_normal((B, h, T, d_k)).astype(np.float32)
    out_np, A_np = scaled_dot_attention(q, k, v, causal=True)
    out_t = TF.scaled_dot_product_attention(
        torch.tensor(q), torch.tensor(k), torch.tensor(v), is_causal=True).numpy()
    d3 = np.abs(out_np - out_t).max()
    assert d3 < 1e-5, f"attention mismatch {d3:.3e}"
    print(f"PASS attn-matches-torch (max diff {d3:.2e})")

    # ------------------------------------------------------------------ causal mask property
    x = rng.standard_normal((B, h, T, d_k)).astype(np.float32)
    o_full, _ = scaled_dot_attention(x, x, v, causal=True)
    x_perturbed = x.copy()
    x_perturbed[:, :, 10:, :] += 42.0            # change the future of positions 0..9
    o_perturbed, _ = scaled_dot_attention(x_perturbed, x_perturbed, v, causal=True)
    diff = np.abs(o_perturbed[:, :, :10, :] - o_full[:, :, :10, :]).max()
    assert diff == 0.0, f"future leak into past: {diff:.3e}"
    print("PASS causal-mask (past positions bit-identical under future perturbation)")

    # ------------------------------------------------------------------ MHA
    d = h * d_k
    x2 = rng.standard_normal((B, T, d)).astype(np.float32)
    w_qkv = (rng.standard_normal((d, 3 * d)) * d ** -0.5).astype(np.float32)
    w_o = (rng.standard_normal((d, d)) * d ** -0.5).astype(np.float32)
    mha_np = multi_head_attention(x2, w_qkv, w_o, h=h, causal=True)
    mha_t = torch_mha_reference(torch.tensor(x2), torch.tensor(w_qkv), torch.tensor(w_o),
                                h=h, causal=True).numpy()
    d4 = np.abs(mha_np - mha_t).max()
    assert d4 < 1e-5, f"MHA mismatch {d4:.3e}"
    print(f"PASS mha-matches-torch (max diff {d4:.2e})")

    # ------------------------------------------------------------------ attention gradient
    Tg = 16
    z_g = rng.standard_normal((3, Tg)).astype(np.float64)
    g = rng.standard_normal((3, Tg)).astype(np.float64)      # dL/dp, arbitrary per-row
    p = softmax_rows(z_g)
    dz_np = p * (g - (p * g).sum(axis=-1, keepdims=True))
    z_t = torch.tensor(z_g, dtype=torch.float64, requires_grad=True)
    p_t = torch.softmax(z_t, dim=-1)
    (p_t * torch.tensor(g)).sum().backward()
    d5 = np.abs(z_t.grad.detach().numpy() - dz_np).max()
    assert d5 < 1e-9, f"attention gradient identity violated {d5:.3e}"
    print(f"PASS attn-gradient (max diff {d5:.2e})")

    # ------------------------------------------------------------------ scaling necessity
    # For q, k iid N(0, I) in R^{T x d_k} the score entries are ~ N(0, d_k). Without the
    # 1/sqrt(d_k) factor the softmax therefore saturates as d_k grows (mass piles onto the
    # top score, entropy collapses); with it, the score standard deviation is 1 for every
    # d_k and the attention stays diffuse.
    T_s, reps = 64, 50
    stats = {}
    for d_kr in (16, 64, 256):
        mu_list, eu_list, ms_list, es_list = [], [], [], []
        for _ in range(reps):
            q_r = rng.standard_normal((T_s, d_kr))
            k_r = rng.standard_normal((T_s, d_kr))
            s_raw = q_r @ k_r.T                # entries ~ N(0, d_k)
            A_u = softmax_rows(s_raw)
            A_s = softmax_rows(s_raw / np.sqrt(d_kr))
            mu_list.append(A_u.max(axis=1).mean())
            ms_list.append(A_s.max(axis=1).mean())
            eu_list.append(float(-(A_u * np.log(A_u + 1e-30)).sum(axis=1).mean()))
            es_list.append(float(-(A_s * np.log(A_s + 1e-30)).sum(axis=1).mean()))
        stats[d_kr] = (float(np.mean(mu_list)), float(np.mean(eu_list)),
                       float(np.mean(ms_list)), float(np.mean(es_list)))
    (mu16, eu16, ms16, es16) = stats[16]
    (mu64, eu64, ms64, es64) = stats[64]
    (mu256, eu256, ms256, es256) = stats[256]
    assert mu16 < mu64 < mu256, f"unscaled max weight not increasing: {mu16:.3f} {mu64:.3f} {mu256:.3f}"
    assert eu16 > eu64 > eu256 and eu256 < 0.5, \
        f"unscaled entropy not collapsing: {eu16:.3f} {eu64:.3f} {eu256:.3f}"
    assert max(ms16, ms64, ms256) - min(ms16, ms64, ms256) < 0.02, \
        f"scaled max weight not stable across d_k: {ms16:.4f} {ms64:.4f} {ms256:.4f}"
    assert max(es16, es64, es256) - min(es16, es64, es256) < 0.05, \
        f"scaled entropy not stable across d_k: {es16:.4f} {es64:.4f} {es256:.4f}"
    print(f"PASS scaling-necessity (unscaled max weight {mu16:.3f} -> {mu64:.3f} -> {mu256:.3f}, "
          f"entropy {eu16:.3f} -> {eu64:.3f} -> {eu256:.3f}; scaled max weight stable at "
          f"~{ms64:.3f}, entropy ~{es64:.3f})")
    print(f"METRIC attn_max_diff={d3:.2e}")
    print(f"METRIC mha_max_diff={d4:.2e}")
    print(f"METRIC entropy_unscaled_dk256={eu256:.4f}")
    print(f"METRIC entropy_scaled_dk64={es64:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
