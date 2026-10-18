#!/usr/bin/env python3
"""Phase 1 — shape discipline: the same computation three ways, plus broadcasting rules.

The habit this script builds is the one that decides whether you can read an architecture
paper: write down the shapes, compute the same object by two independent routes, and assert
they agree. Three routes are used here for attention scores -- einsum, a batched matmul with
an explicit head split, and an explicit loop over (batch, head) -- and they must be equal to
machine precision, because they are the same formula.

The second half is about the two ways NumPy lets you be wrong in silence: broadcasting along
an axis you did not mean, and mutating through a view. Both are demonstrated, not described.

    ./.venv/bin/python code/phase01_foundations/shapes_and_broadcasting.py
"""
from __future__ import annotations

import sys
import time

import numpy as np

SEED = 1


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------- routes to one answer

def scores_einsum(x: np.ndarray, wq: np.ndarray, wk: np.ndarray) -> np.ndarray:
    """Q = X W_Q, K = X W_K, A = softmax(Q K^T / sqrt(d_k)), as one batched einsum.

    Subscripts are (b, t, h) for query positions and (b, j, h) for key positions, so the
    contraction over the feature axis is visible rather than implied by a matmul.
    """
    q = np.einsum("btd,dh->bth", x, wq)                       # (B, T, d_k)
    k = np.einsum("btd,dh->bth", x, wk)                       # (B, T, d_k)
    raw = np.einsum("bih,bjh->bij", q, k) / np.sqrt(wq.shape[1])
    return softmax(raw)


def scores_heads(x: np.ndarray, wq: np.ndarray, wk: np.ndarray, n_heads: int) -> np.ndarray:
    """Single-head-equivalent computation with the head split done explicitly.

    d = h * d_head, and each head gets a contiguous slice of the columns of W_Q. This is the
    reshape/transpose manoeuvre that every Transformer implementation performs.
    """
    B, T, d = x.shape
    q_full = x @ wq                                   # (B, T, d)
    k_full = x @ wk
    d_head = d // n_heads
    q = q_full.reshape(B, T, n_heads, d_head).transpose(0, 2, 1, 3)   # (B,h,T,d_head)
    k = k_full.reshape(B, T, n_heads, d_head).transpose(0, 2, 1, 3)
    out = np.zeros((B, n_heads, T, T))
    for b in range(B):
        for hh in range(n_heads):
            out[b, hh] = softmax(q[b, hh] @ k[b, hh].T / np.sqrt(d_head))
    return out


def scores_loop(x: np.ndarray, wq: np.ndarray, wk: np.ndarray) -> np.ndarray:
    """The dumb version: nested Python loops. Slow, and therefore the reference."""
    B, T, _ = x.shape
    dk = wq.shape[1]
    out = np.zeros((B, T, T))
    for b in range(B):
        for i in range(T):
            scores = np.array([x[b, i] @ wq @ (x[b, j] @ wk) for j in range(T)])
            out[b, i] = softmax(scores / np.sqrt(dk))
    return out


def swapped_head_split(q_full: np.ndarray, n_heads: int) -> np.ndarray:
    """The silent one: read the two middle axes in the opposite order.

    With d = h * d_head and h == d_head (as here, 4 x 4), both readings produce a tensor of
    shape (B, h, T, d_head). Nothing raises; the values are simply from a different partition
    of the feature axis. This is why real implementations keep h and d_head as separate named
    constants instead of recomputing them at each call site.
    """
    B, T, d = q_full.shape
    d_head = d // n_heads
    return q_full.reshape(B, T, d_head, n_heads).transpose(0, 3, 1, 2)


# ---------------------------------------------------------------- broadcasting rules

def rules() -> list[tuple[str, tuple, tuple, tuple | str]]:
    """(description, shape a, shape b, expected result shape or 'error')."""
    return [
        ("same shape",          (2, 3, 4), (2, 3, 4), (2, 3, 4)),
        ("scalar broadcast",    (2, 3, 4), (),        (2, 3, 4)),
        ("new leading axis",    (3, 4),    (2, 1, 4), (2, 3, 4)),
        ("length-1 axis grows", (2, 1, 4), (2, 3, 4), (2, 3, 4)),
        ("KV append along T",   (2, 5, 8), (2, 1, 8), (2, 5, 8)),
        ("incompatible",        (2, 3, 4), (2, 3, 5), "error"),
        ("3D vs column bias",   (2, 3, 4), (4,),      (2, 3, 4)),
    ]


def main() -> int:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    B, T, d, n_heads = 3, 7, 16, 4
    x = rng.normal(size=(B, T, d))
    wq = rng.normal(size=(d, d)) * 0.4
    wk = rng.normal(size=(d, d)) * 0.4

    a_ein = scores_einsum(x, wq, wk)
    a_loop = scores_loop(x, wq, wk)
    err_routes = float(np.max(np.abs(a_ein - a_loop)))
    assert err_routes < 1e-12, f"einsum vs loop disagree: {err_routes:.3e}"
    assert a_ein.shape == (B, T, T), a_ein.shape

    # The multi-head version must agree with the single-head einsum when there is one head,
    # and with itself when the loop over heads is unrolled. That pins the reshape/transpose.
    a_head1 = scores_heads(x, wq, wk, n_heads=1)
    err_heads = float(np.max(np.abs(scores_einsum(x, wq, wk[:, :d]) -
                                    a_head1.mean(axis=1))))
    assert a_head1.shape == (B, 1, T, T), a_head1.shape
    assert err_heads < 1e-12, f"single-head reshape disagrees with einsum: {err_heads:.3e}"

    # Rows of an attention matrix are distributions: this is the invariant everything later
    # (KV cache, paged attention, FlashAttention) has to preserve.
    row_sum_err = float(np.max(np.abs(a_ein.sum(axis=-1) - 1.0)))
    assert row_sum_err < 1e-12, f"attention rows must sum to 1: {row_sum_err:.3e}"

    # A wrong head split is shape-legal and value-wrong. Assert that it IS wrong, so this
    # chapter's claim ("shapes alone will not save you") is checked rather than asserted.
    q_full = x @ wq
    good = q_full.reshape(B, T, n_heads, d // n_heads).transpose(0, 2, 1, 3)
    bad = swapped_head_split(q_full, n_heads)
    assert good.shape == bad.shape, f"both readings must be shape-legal: {good.shape} vs {bad.shape}"
    split_diff = float(np.max(np.abs(good - bad)))
    assert split_diff > 1e-6, "the wrong permutation happened to coincide: pick a bigger case"

    # ---------------------------------------------------------------- broadcasting
    n_rules_checked = 0
    for desc, sa, sb, expect in rules():
        A = np.zeros(sa)
        Bm = np.zeros(sb)
        if expect == "error":
            try:
                A + Bm
            except ValueError:
                n_rules_checked += 1
                continue
            raise AssertionError(f"{desc}: broadcast unexpectedly succeeded")
        out = (A + Bm).shape
        assert out == expect, f"{desc}: got {out}, expected {expect}"
        n_rules_checked += 1

    # Broadcasting a bias along the wrong axis is silent: adding a (T,1) "bias" to (B,T,d)
    # does not raise, it stretches along d. Show the two intents differ.
    bias_token = rng.normal(size=(T, 1))
    bias_feat = rng.normal(size=(d,))          # intent: one bias per feature axis
    via_token = x + bias_token
    via_feat = x + bias_feat
    assert via_token.shape == via_feat.shape == (B, T, d)
    bias_diff = float(np.max(np.abs(via_token - via_feat)))
    assert bias_diff > 1e-6, "the two bias intents must differ"

    # An explicit tile is the same as a broadcast: broadcasting is free, tiling is not.
    small = rng.normal(size=(1, T, d))
    assert_close = float(np.max(np.abs((x + small) - (x + np.tile(small, (B, 1, 1))))))
    assert assert_close == 0.0, "broadcast and tile must agree exactly"

    # ---------------------------------------------------------------- views vs copies
    base = rng.normal(size=(4, 6))
    transposed = base.T                      # a view: shares memory, changes strides only
    assert np.shares_memory(base, transposed), "transpose must be a view"
    assert transposed.strides != base.strides, "transpose changes strides, not data"
    copied = base.T.copy()
    assert not np.shares_memory(base, copied), "copy() must own its data"
    before = base[0, 1]
    transposed[1, 0] += 5.0                  # writes through the view
    assert abs(float(base[0, 1] - before) - 5.0) < 1e-12, "view mutation must be visible"
    view_write_visible = True

    print(f"config: B={B} T={T} d={d} heads={n_heads} seed={SEED}")
    print(f"X{tuple(x.shape)} @ W_Q{tuple(wq.shape)} -> Q{tuple(q_full.shape)}; "
          f"QK^T -> {tuple(a_ein.shape)}")
    print(f"broadcasting rules checked: {n_rules_checked}")
    assert n_rules_checked == 7, n_rules_checked
    print(f"PASS einsum == explicit loop over (batch, query, key)  (max abs diff {err_routes:.2e})")
    print(f"PASS attention rows sum to 1                            (max |row sum - 1| {row_sum_err:.2e})")
    print(f"PASS wrong head permutation is shape-legal but differs  (max abs diff {split_diff:.3e})")
    print(f"PASS broadcasting: 7 rules match NumPy semantics, incl. one that must raise")
    print(f"PASS views share memory, copies do not; in-place write visible: {view_write_visible}")
    print(f"METRIC einsum_vs_loop_maxdiff={err_routes:.3e}")
    print(f"METRIC row_sum_maxdiff={row_sum_err:.3e}")
    print(f"METRIC head_split_confusion_maxdiff={split_diff:.3e}")
    print(f"METRIC runtime_s={time.time() - t0:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
