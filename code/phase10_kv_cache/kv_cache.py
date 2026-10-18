"""Verify the KV cache: cached decode must recompute exactly what the full pass computes.

The chapter derives the incremental cache update k_t = W_k x_t, v_t = W_v x_t appended to
the cache, and the claim that one cached decode step reads O(T) cached bytes and writes
O(1) new ones, so generation of T tokens costs O(T^2) attention work either way but only
O(T) *model evaluations* with the cache.

Checks
  cache-equivalence    a full causal pass over x[:t] and a prefill + incremental decode
                       produce logits equal to 1e-10 (float64, fixed weights)
  step-identity        the cached attention numerator/denominator after t steps equals
                       the row t of the dense T x T softmax attention, to 1e-10
  cache-bytes          the cache size tracks 2 h_kv d_k T s exactly at every step
  timing               measured table: naive recompute vs cached decode, lengths
                       32..512; the cached step time is roughly flat and the naive step
                       grows with T; logged as CPU-measured numbers

Run:  ./.venv/bin/python code/phase10_kv_cache/kv_cache.py
"""
from __future__ import annotations

import time

import numpy as np

D, H, DK, B = 64, 4, 16, 1


def make_weights(rng: np.random.Generator) -> dict:
    s = 0.4
    return {
        "Wq": rng.standard_normal((D, H * DK)) * s,
        "Wk": rng.standard_normal((D, DK)) * s,      # one key-value head (MQA layout)
        "Wv": rng.standard_normal((D, DK)) * s,
        "Wo": rng.standard_normal((H * DK, D)) * s,
    }


def softmax_rows(a: np.ndarray) -> np.ndarray:
    a = a - a.max(axis=-1, keepdims=True)
    e = np.exp(a)
    return e / e.sum(axis=-1, keepdims=True)


def full_pass(w: dict, x: np.ndarray) -> np.ndarray:
    """One CAUSAL attention block over x: (B, T, D) -> (B, T, D). No cache.

    The causal mask is what makes this the correct reference for the cached decode,
    which can only see past positions. Without it the dense pass would attend to the
    future and the two would legitimately disagree.
    """
    B_, T, _ = x.shape
    q = (x @ w["Wq"]).reshape(B_, T, H, DK).transpose(0, 2, 1, 3)
    k = (x @ w["Wk"]).reshape(B_, T, 1, DK).transpose(0, 2, 1, 3)
    v = (x @ w["Wv"]).reshape(B_, T, 1, DK).transpose(0, 2, 1, 3)
    a = q @ k.transpose(0, 1, 3, 2) / np.sqrt(DK)
    idx = np.arange(T)
    a = np.where(idx[:, None] >= idx[None, :], a, -np.inf)
    att = softmax_rows(a)
    o = (att @ v).transpose(0, 2, 1, 3).reshape(B_, T, H * DK)
    return o @ w["Wo"]


class Cache:
    def __init__(self) -> None:
        self.k = np.zeros((B, 1, 0, DK))
        self.v = np.zeros((B, 1, 0, DK))

    def append(self, k: np.ndarray, v: np.ndarray) -> None:
        self.k = np.concatenate([self.k, k], axis=2)
        self.v = np.concatenate([self.v, v], axis=2)

    def n_bytes(self, s: int = 8) -> int:
        return self.k.nbytes + self.v.nbytes - 0 * s

    def decode(self, w: dict, x_t: np.ndarray) -> np.ndarray:
        """One step: x_t (B,1,D) -> output (B,1,D) against the cached keys."""
        q = (x_t @ w["Wq"]).reshape(B, 1, H, DK).transpose(0, 2, 1, 3)
        att = softmax_rows(q @ self.k.transpose(0, 1, 3, 2) / np.sqrt(DK))
        o = (att @ self.v).transpose(0, 2, 1, 3).reshape(B, 1, H * DK)
        return o @ w["Wo"]


def cached_pass(w: dict, x: np.ndarray) -> np.ndarray:
    c = Cache()
    outs = []
    for t in range(x.shape[1]):
        xt = x[:, t:t + 1, :]
        k = (xt @ w["Wk"]).reshape(B, 1, 1, DK)
        v = (xt @ w["Wv"]).reshape(B, 1, 1, DK)
        c.append(k, v)
        outs.append(c.decode(w, xt))
    return np.concatenate(outs, axis=1)


def main() -> int:
    rng = np.random.default_rng(5)
    T = 24
    x = rng.standard_normal((B, T, D))
    w = make_weights(rng)

    dense = full_pass(w, x)
    cached = cached_pass(w, x)
    d1 = np.abs(dense - cached).max()
    assert d1 < 1e-10, f"cached decode differs from full pass by {d1:.2e}"
    print(f"PASS cache-equivalence (max |dense - cached| = {d1:.2e} over {T} steps)")

    # step identity: cached attention row vs dense causal attention row
    q = (x @ w["Wq"]).reshape(B, T, H, DK).transpose(0, 2, 1, 3)
    k = (x @ w["Wk"]).reshape(B, T, 1, DK).transpose(0, 2, 1, 3)
    a = q @ k.transpose(0, 1, 3, 2) / np.sqrt(DK)
    idx = np.arange(T)
    a = np.where(idx[:, None] >= idx[None, :], a, -np.inf)
    dense_att = softmax_rows(a)
    c = Cache()
    worst = 0.0
    for t in range(T):
        xt = x[:, t:t + 1, :]
        c.append((xt @ w["Wk"]).reshape(B, 1, 1, DK), (xt @ w["Wv"]).reshape(B, 1, 1, DK))
        qt = q[:, :, t:t + 1, :]
        step_att = softmax_rows(qt @ c.k.transpose(0, 1, 3, 2) / np.sqrt(DK))
        worst = max(worst, float(np.abs(step_att - dense_att[:, :, t:t + 1, :t + 1]).max()))
    assert worst < 1e-10, f"step attention differs by {worst:.2e}"
    print(f"PASS step-identity (worst cached-vs-dense attention row diff {worst:.2e})")

    # cache bytes: 2 * h_kv * d_k * T * s per batch row
    steps = (4, 8, 16)
    prev = 0
    for t in steps:
        cc = Cache()
        for i in range(t):
            xt = x[:, i:i + 1, :]
            cc.append((xt @ w["Wk"]).reshape(B, 1, 1, DK), (xt @ w["Wv"]).reshape(B, 1, 1, DK))
        expect = 2 * 1 * DK * t * 8 * B
        assert cc.n_bytes() == expect, f"cache bytes {cc.n_bytes()} != {expect} at t={t}"
        assert cc.n_bytes() > prev
        prev = cc.n_bytes()
    print("PASS cache-bytes (2 h_kv d_k T s exact at T = 4, 8, 16; float64)")

    # timing table: naive recompute vs cached decode (CPU-measured)
    rows = []
    for Tlen in (32, 64, 128, 256, 512):
        xx = rng.standard_normal((B, Tlen, D))
        t0 = time.perf_counter()
        _ = full_pass(w, xx)          # the prefill
        t_pre = time.perf_counter() - t0
        # naive decode: recompute the full pass for every one of 32 new tokens
        gen = rng.standard_normal((B, 32, D))
        seq = xx
        t0 = time.perf_counter()
        for _ in range(32):
            seq_full = np.concatenate([seq, gen[:, :1, :]], axis=1)
            _ = full_pass(w, seq_full)
        t_naive = time.perf_counter() - t0
        # cached decode
        seq = xx
        c = Cache()
        for i in range(Tlen):
            xt = seq[:, i:i + 1, :]
            c.append((xt @ w["Wk"]).reshape(B, 1, 1, DK), (xt @ w["Wv"]).reshape(B, 1, 1, DK))
        t0 = time.perf_counter()
        for _ in range(32):
            _ = c.decode(w, gen[:, :1, :])
        t_cached = time.perf_counter() - t0
        rows.append((Tlen, t_pre, t_naive / 32, t_cached / 32))

    print("\nCPU-measured decode cost, 32 generated tokens, B = 1, D = 64, h_kv = 1")
    print(f"{'T':>5} {'prefill s':>10} {'naive/step ms':>14} {'cached/step ms':>15} {'speedup':>8}")
    for Tlen, tp, tn, tc in rows:
        print(f"{Tlen:>5} {tp:>10.4f} {tn * 1e3:>14.3f} {tc * 1e3:>15.3f} {tn / tc:>8.1f}x")

    assert all(r[3] < r[2] for r in rows), "cached decode not faster at some length"
    naive_growth = rows[-1][2] / rows[0][2]
    cached_growth = rows[-1][3] / max(rows[0][3], 1e-9)
    assert naive_growth > cached_growth, "naive decode did not grow faster than cached"
    print(f"\nPASS timing (naive step grows x{naive_growth:.0f} from T=32 to T=512, "
          f"cached x{cached_growth:.1f}; both O(T^2) total, cached with a far smaller constant)")

    print(f"METRIC prefill_ms_T512={rows[-1][1] * 1e3:.2f}")
    print(f"METRIC speedup_T512={rows[-1][2] / rows[-1][3]:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
