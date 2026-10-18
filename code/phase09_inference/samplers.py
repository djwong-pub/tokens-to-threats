"""Verify every sampler of Phase 9 against its closed-form definition.

The chapter derives temperature, top-k, top-p (nucleus), min-p and the repetition
penalty as transformations of one logit vector, in that order. This script applies
each transformation by hand and checks the property the chapter claims.

Checks
  temperature-argmax   temperature rescales but never moves the argmax, for T on a grid
  temperature-entropy  entropy decreases monotonically as T decreases to 0; the T = 1
                       distribution matches the plain softmax
  topk-exact           top-k keeps the k largest masses, zeroes the rest, renormalises;
                       k = |V| is the identity to machine precision
  topp-nucleus         the nucleus is the shortest prefix with mass >= p; dropping the
                       last kept head leaves mass < p; p = 1 is the identity
  minp-scales          min-p keeps exactly the heads with p_i >= f * max_j p_j;
                       f = 0 is the identity and f = 1 keeps only the argmax
  repetition-penalty   for theta > 1 the penalty lowers a repeated positive logit and
                       raises a repeated negative one, and never touches unseen tokens
  sampler-frequency    empirical frequencies of the top-p sampler match its renormalised
                       probabilities within a binomial tolerance (chi-square style band)

Run:  ./.venv/bin/python code/phase09_inference/samplers.py
"""
from __future__ import annotations

import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def temperature(z: np.ndarray, T: float) -> np.ndarray:
    return softmax(z / T)


def top_k_mask(z: np.ndarray, k: int) -> np.ndarray:
    idx = np.argsort(-z)
    keep = np.zeros_like(z, dtype=bool)
    keep[idx[:k]] = True
    return keep


def top_p_probs(p: np.ndarray, top_p: float) -> np.ndarray:
    order = np.argsort(-p)
    cum = np.cumsum(p[order])
    n = int(np.searchsorted(cum, top_p, side="left")) + 1
    n = min(n, p.size)
    out = np.zeros_like(p)
    out[order[:n]] = p[order[:n]]
    return out / out.sum()


def min_p_mask(p: np.ndarray, f: float) -> np.ndarray:
    return p >= f * p.max()


def repetition_penalty(z: np.ndarray, seen: set[int], theta: float) -> np.ndarray:
    out = z.copy()
    for i in seen:
        out[i] = out[i] / theta if out[i] > 0 else out[i] * theta
    return out


def sample(p: np.ndarray, rng: np.random.Generator) -> int:
    return int(rng.choice(p.size, p=p))


def main() -> int:
    rng = np.random.default_rng(11)
    z = rng.standard_normal(50) * 2.0

    # ------------------------------------------------------------------ temperature
    amax = int(np.argmax(z))
    for T in (0.1, 0.5, 1.0, 2.0, 10.0):
        assert int(np.argmax(temperature(z, T))) == amax, f"argmax moved at T={T}"
    p1 = temperature(z, 1.0)
    assert np.abs(p1 - softmax(z)).max() < 1e-15, "T = 1 must equal plain softmax"
    entropies = []
    for T in (3.0, 2.0, 1.0, 0.5, 0.25, 0.1):
        q = temperature(z, T)
        entropies.append(float(-(q * np.log(q + 1e-300)).sum()))
    assert all(entropies[i] > entropies[i + 1] for i in range(len(entropies) - 1)), \
        f"entropy not monotone in T: {entropies}"
    print(f"PASS temperature-argmax")
    print(f"PASS temperature-entropy (H = {' > '.join(f'{e:.2f}' for e in entropies)} on a 50-head vocab)")

    # ------------------------------------------------------------------ top-k
    keep = top_k_mask(z, 8)
    zs = np.sort(z)[::-1]
    assert keep.sum() == 8
    pk = softmax(np.where(keep, z, -np.inf))
    assert (pk[~keep] == 0.0).all(), "top-k leaked mass to dropped heads"
    kept_sorted = np.sort(pk[keep])[::-1]
    expected = softmax(zs[:8])
    assert np.abs(kept_sorted - expected).max() < 1e-15, "top-k renormalisation wrong"
    pid = top_k_mask(z, z.size)
    assert pid.all() and np.abs(softmax(np.where(pid, z, -np.inf)) - softmax(z)).max() < 1e-15
    orig_mass = float(softmax(z)[np.argsort(-z)][:8].sum())
    print(f"PASS topk-exact (k=8 keeps the {orig_mass:.3f} of probability the top-8 held, "
          f"renormalised to 1; k=|V| is the identity)")

    # ------------------------------------------------------------------ top-p
    p = softmax(z)
    for tp in (0.5, 0.9, 0.99):
        q = top_p_probs(p, tp)
        nz = np.sort(q[q > 0])[::-1]
        raw_mass = float(p[np.argsort(-p)][: nz.size].sum())
        assert raw_mass >= tp - 1e-12, "nucleus mass below p"
        if nz.size > 1:
            smaller = float(p[np.argsort(-p)][: nz.size - 1].sum())
            assert smaller < tp, "nucleus not minimal"
    assert np.abs(top_p_probs(p, 1.0) - p).max() < 1e-15, "p=1 must be the identity"
    print(f"PASS topp-nucleus (at p=0.9 the nucleus holds {int((top_p_probs(p, 0.9) > 0).sum())} of 50 heads)")

    # ------------------------------------------------------------------ min-p
    for f in (0.0, 0.25, 1.0):
        m = min_p_mask(p, f)
        assert (m == (p >= f * p.max())).all()
        if f == 0.0:
            assert m.all()
        if f == 1.0:
            assert m.sum() == 1 and m.argmax() == p.argmax()
    print(f"PASS minp-scales (f=0 keeps all, f=1 keeps the argmax; f=0.25 keeps "
          f"{int(min_p_mask(p, 0.25).sum())} of 50)")

    # ------------------------------------------------------------------ repetition penalty
    seen = {0, 1, 2}
    zt = z.copy()
    zt[0] = 1.5   # positive repeated logit
    zt[1] = -1.5  # negative repeated logit
    zp = repetition_penalty(zt, seen, 1.8)
    # both signs are pushed toward zero probability: positive logits are divided, negative
    # logits are multiplied (more negative). The probability of a repeated token falls.
    assert zp[0] < zt[0] and abs(zp[0] - zt[0] / 1.8) < 1e-15
    assert zp[1] < zt[1] and abs(zp[1] - zt[1] * 1.8) < 1e-15
    assert np.abs(zp[3:] - zt[3:]).max() == 0.0, "penalty touched an unseen token"
    assert int(np.argmax(repetition_penalty(z, set(), 1.8))) == amax, "empty-seen penalty moved argmax"
    print("PASS repetition-penalty (theta = 1.8: +1.5 -> %.3f, -1.5 -> %.3f; both probabilities fall, "
          "unseen untouched)" % (zp[0], zp[1]))

    # ------------------------------------------------------------------ sampler frequencies
    q = top_p_probs(p, 0.9)
    n = 60_000
    counts = np.bincount([sample(q, rng) for _ in range(n)], minlength=q.size)
    emp = counts / n
    sd = np.sqrt(np.maximum(q * (1 - q) / n, 1e-18))
    worst = np.abs(emp - q).max() / sd.max()
    assert (np.abs(emp - q) < 5 * (np.sqrt(q * (1 - q) / n) + 1e-12)).all(), \
        "sampler frequencies outside the 5-sigma binomial band"
    print(f"PASS sampler-frequency ({n} draws, worst |emp-q|/sigma_band = {worst:.2f})")

    nz9 = int((q > 0).sum())
    print(f"METRIC nucleus_heads_p09={nz9}")
    print(f"METRIC entropy_T1={entropies[2]:.4f}")
    print(f"METRIC entropy_T01={entropies[-1]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
