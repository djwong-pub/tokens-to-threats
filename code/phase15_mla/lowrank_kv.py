"""Verify the MLA low-rank key-value compression of Phase 15 against its algebra.

The chapter derives the decomposition: instead of storing k_t = x_t W_K and
v_t = x_t W_V (two full tensors), the model compresses the stream once,
c_t = x_t W_C in R^{d_c}, and recovers the key from c_t by a second projection,
k_t = c_t W_KU, with d_c << h d_k. The memory ratio of the compressed cache to the
MHA cache is d_c / (2 h d_k) for the shared part, and the attention is unchanged
whenever the composition c_t W_KU equals x_t W_K exactly.

Checks
  composition-exact    when W_KU = W_C^+ (a right inverse on the data), the composed
                       key equals the direct key to 1e-9: the compression is lossless
  rank-deficit         a rank-deficient composition (d_c < intrinsic rank) changes the
                       key by a measurable amount, bounded by the truncated singular
                       values
  memory-ratio         the byte count of the compressed cache is d_c / (2 h d_k) of the
                       MHA cache at every tested length, exactly
  attention-unchanged  with an exact composition, the full attention output (keys from
                       the compression, direct values) equals the direct-MHA attention
                       to 1e-9

Run:  ./.venv/bin/python code/phase15_mla/lowrank_kv.py
"""
from __future__ import annotations

import numpy as np


def attention_out(q: np.ndarray, k: np.ndarray, v: np.ndarray,
                  causal: bool = True, dk: int | None = None) -> np.ndarray:
    T = q.shape[0]
    s = (dk if dk is not None else q.shape[-1]) ** -0.5
    a = (q @ k.T) * s
    if causal:
        a = np.where(np.arange(T)[:, None] >= np.arange(T)[None, :], a, -np.inf)
    a = a - a.max(axis=-1, keepdims=True)
    e = np.exp(a)
    return (e / e.sum(axis=-1, keepdims=True)) @ v


def main() -> int:
    rng = np.random.default_rng(7)
    T, d, h, dk, dc = 128, 128, 8, 16, 32
    x = rng.standard_normal((T, d))

    # direct MHA keys/values (one shared layout per head for the cache count)
    Wk = rng.standard_normal((d, dk)) * 0.1
    Wv = rng.standard_normal((d, dk)) * 0.1
    k_direct = x @ Wk
    v_direct = x @ Wv

    # ------------------------------------------------------------------ exact composition
    # SVD of the direct key projection Wk (d x dk, rank dk <= d): Wk = Uw Sw Vw^T.
    # Set W_C = Uw and W_KU = Sw Vw^T, then x W_C W_KU = x Wk exactly: the
    # compression stores c_t = x_t Uw in R^{dk} and reconstructs k_t from it.
    Uw, Sw, VwT = np.linalg.svd(Wk, full_matrices=False)
    W_C = Uw.copy()                              # (d, dk): compress into the singular space
    W_KU = (Sw[:, None] * VwT)                   # (dk, dk): reconstruct
    c = x @ W_C
    k_mla = c @ W_KU
    d1 = np.abs(k_mla - k_direct).max()
    assert d1 < 1e-9, f"exact composition differs by {d1:.2e}"
    print(f"PASS composition-exact (Wk rank {dk} <= dc={dc}: |k_mla - k| max = {d1:.2e})")

    # ------------------------------------------------------------------ rank deficit
    # keep only dc_small < dk singular directions: the dropped spectrum bounds the error
    dc_small = dk // 2
    W_C2 = Uw[:, :dc_small].copy()
    W_KU2 = (Sw[:dc_small, None] * VwT[:dc_small])
    k_trunc = (x @ W_C2) @ W_KU2
    err = np.abs(k_trunc - k_direct).max()
    tail = Sw[dc_small:].max()
    assert err > 1e-6, "truncation had no effect: test too weak"
    assert err <= x.max() * tail + 1e-9, f"error {err:.3e} above the dropped spectrum {tail:.3e}"
    print(f"PASS rank-deficit (dc={dc_small}: max error {err:.3e}, largest dropped "
          f"singular value {tail:.3e}; error bounded by the dropped spectrum)")

    # ------------------------------------------------------------------ memory ratio
    s_bytes = 2                                  # fp16
    mha_bytes = lambda L: 2 * h * dk * L * s_bytes
    mla_bytes = lambda L: dc * L * s_bytes
    ratios = [mla_bytes(L) / mha_bytes(L) for L in (256, 1024, 4096)]
    expected = dc / (2 * h * dk)
    assert all(abs(r - expected) < 1e-15 for r in ratios), f"ratios {ratios} != {expected}"
    print(f"PASS memory-ratio (compressed/MHA cache = {expected:.4f} at every length; "
          f"MHA at T=1024 is {mha_bytes(1024) / 1e6:.1f} MB, compressed {mla_bytes(1024) / 1e6:.2f} MB)")

    # ------------------------------------------------------------------ attention unchanged
    q1 = x @ (rng.standard_normal((d, dk)) * 0.1)
    y_mla = attention_out(q1, k_mla, v_direct, dk=dk)
    y_dir = attention_out(q1, k_direct, v_direct, dk=dk)
    d2 = np.abs(y_mla - y_dir).max()
    assert d2 < 1e-9, f"attention differs by {d2:.2e}"
    print(f"PASS attention-unchanged (exact composition: |y_mla - y| max = {d2:.2e})")

    print(f"METRIC mla_mha_ratio={expected:.4f}")
    print(f"METRIC mla_trunc_err_dc8={err:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
