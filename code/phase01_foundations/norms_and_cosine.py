#!/usr/bin/env python3
"""Phase 1 — norms, cosine similarity, and the inequality that forces the 1/sqrt(d) factor.

Two things an architecture paper expects you to do without thinking: read a norm as an
optimisation problem, and know how big a dot product of two random feature vectors is.

Checks performed:
  1. dual norms, exactly:      ||x||_1 = max_{||u||_inf<=1} u.x,  ||x||_2 = max_{||u||_2<=1} u.x,
                               ||x||_inf = max_{||u||_1<=1} u.x   (attained at the analytic u)
  2. Cauchy-Schwarz with a measured margin, and cosine invariance to rescaling;
  3. E[cos^2] = 1/d for isotropic random vectors, Monte Carlo (the reason attention scales
     logits by 1/sqrt(d_head): without it the softmax saturates as d grows);
  4. orthogonal projection: idempotent, symmetric, and optimal on its line;
  5. sub-multiplicativity of Frobenius against the spectral norm: ||AB||_F <= ||A||_F ||B||_2;
  6. gradient of cosine similarity against central differences.

    ./.venv/bin/python code/phase01_foundations/norms_and_cosine.py
"""
from __future__ import annotations

import sys
import time

import numpy as np

SEED = 2


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def cos_grad(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """d/da cos(a,b) = b/(|a||b|) - (a.b) a / (|a|^3 |b|)."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return b / (na * nb) - (a @ b) * a / (na**3 * nb)


def main() -> int:
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    # 1 ------------------------------------------------------------------ dual norms
    x = rng.normal(size=5)
    l1, l2, linf = float(np.abs(x).sum()), float(np.linalg.norm(x)), float(np.abs(x).max())
    u_l1 = np.sign(x)                                  # in the Linf ball
    u_l2 = x / l2                                      # in the L2 ball
    u_inf = np.zeros_like(x)
    u_inf[int(np.argmax(np.abs(x)))] = float(np.sign(x[np.argmax(np.abs(x))]))
    assert abs(u_l1 @ x - l1) < 1e-13, "sign(x) must attain ||x||_1"
    assert abs(u_l2 @ x - l2) < 1e-13, "x/||x|| must attain ||x||_2"
    assert abs(u_inf @ x - linf) < 1e-13, "a coordinate vector must attain ||x||_inf"
    # and nothing in the respective dual ball beats it (random search, worst violation printed)
    worst_viol = 0.0
    for _ in range(20000):
        u = rng.uniform(-1, 1, size=5)
        worst_viol = max(worst_viol, float(u @ x) - l1)
        v = rng.normal(size=5); v /= np.linalg.norm(v)
        worst_viol = max(worst_viol, float(v @ x) - l2)
        w = rng.dirichlet(np.ones(5)); w = w * rng.choice([-1.0, 1.0])
        worst_viol = max(worst_viol, float(w @ x) - linf)
    assert worst_viol <= 1e-12, f"found a dual vector beating the norm by {worst_viol:.2e}"

    # 2 ------------------------------------------------------------ Cauchy-Schwarz / cosine
    a, b = rng.normal(size=64), rng.normal(size=64)
    cs_slack = float(abs(a @ b) - np.linalg.norm(a) * np.linalg.norm(b))
    assert cs_slack <= 1e-12, f"Cauchy-Schwarz violated by {-cs_slack:.2e}"
    cos_ab = cosine(a, b)
    assert abs(cos_ab - cosine(a, 3.5 * b)) < 1e-12, "cosine must ignore rescaling"
    assert abs(cosine(a, a) - 1.0) < 1e-12 and abs(cosine(a, -a) + 1.0) < 1e-12

    # 3 ------------------------------------------- cos^2 concentration: why 1/sqrt(d) exists
    dims = [16, 64, 256, 1024]
    n_draw = 4000
    concentration = {}
    for d in dims:
        A = rng.normal(size=(n_draw, d))
        A /= np.linalg.norm(A, axis=1, keepdims=True)
        cs = (A @ A.T) ** 2
        off = cs[~np.eye(n_draw, dtype=bool)]
        mean_cs2 = float(off.mean())
        # E[cos^2] = 1/d exactly; Monte Carlo error here is a few percent at 4000 pairs.
        assert abs(mean_cs2 - 1.0 / d) / (1.0 / d) < 0.08, \
            f"d={d}: mean cos^2 {mean_cs2:.5f} vs 1/d {1/d:.5f}"
        concentration[d] = mean_cs2

    # The consequence for attention: an un-scaled dot product of unit-norm-scaled features has
    # std ~ sqrt(d) * (1/d) = 1/sqrt(d) per pair; dividing by sqrt(d) keeps the logit spread
    # O(1) instead of shrinking it. Compare spreads directly.
    d_head = 64
    q = rng.normal(size=(256, d_head)); k = rng.normal(size=(256, d_head))
    raw = (q @ k.T)
    scaled = raw / np.sqrt(d_head)
    assert abs(float(raw.std()) / float(scaled.std()) - np.sqrt(d_head)) < 1e-6, \
        "the 1/sqrt(d) divisor must rescale spread by exactly sqrt(d)"

    # 4 ------------------------------------------------------------------- projections
    V = rng.normal(size=(40, 6))
    Qm, _ = np.linalg.qr(V)
    P = Qm @ Qm.T
    assert float(np.max(np.abs(P @ P - P))) < 1e-12, "projection must be idempotent"
    assert float(np.max(np.abs(P - P.T))) < 1e-12, "projection must be symmetric"
    y = rng.normal(size=40)
    py = P @ y
    for _ in range(200):
        z = Qm @ rng.normal(size=6)
        assert float(np.linalg.norm(y - py)) <= float(np.linalg.norm(y - z)) + 1e-12, \
            "projection must be the closest point in the subspace"

    # 5 ------------------------------------------------------- mixed norm submultiplicativity
    A = rng.normal(size=(9, 5))
    B = rng.normal(size=(5, 7))
    lhs = float(np.linalg.norm(A @ B, "fro"))
    r1 = float(np.linalg.norm(A, "fro") * np.linalg.norm(B, 2))
    r2 = float(np.linalg.norm(A, 2) * np.linalg.norm(B, "fro"))
    assert lhs <= min(r1, r2) + 1e-12, f"submultiplicativity violated: {lhs:.4f}"
    spectral_le_fro = np.linalg.norm(A, 2) <= np.linalg.norm(A, "fro") + 1e-12
    assert spectral_le_fro, "||A||_2 <= ||A||_F always"

    # 6 ----------------------------------------------------- gradient of cosine, by finite diff
    eps = 1e-6
    num = np.array([(cosine(a + eps * np.eye(64)[i], b) - cosine(a - eps * np.eye(64)[i], b))
                    / (2 * eps) for i in range(64)])
    grad_err = float(np.max(np.abs(num - cos_grad(a, b))))
    assert grad_err < 1e-7, f"cosine gradient disagrees with finite differences: {grad_err:.2e}"

    print(f"seed={SEED}  ||x||_1={l1:.4f} ||x||_2={l2:.4f} ||x||_inf={linf:.4f}")
    print(f"cos(a,b)={cos_ab:+.4f}  Cauchy-Schwarz slack={cs_slack:.2e}  "
          f"dual-ball worst violation={worst_viol:.2e}")
    print("mean cos^2 by dimension (theory 1/d):")
    for d in dims:
        print(f"    d={d:5d}  measured {concentration[d]:.6f}   1/d {1/d:.6f}")
    print(f"attention logit spread: raw {raw.std():.2f} -> scaled {scaled.std():.2f} "
          f"(ratio {raw.std()/scaled.std():.3f}, sqrt(d)={np.sqrt(d_head):.1f})")
    print("PASS dual norms attained at the analytic vectors, never beaten by search")
    print("PASS Cauchy-Schwarz, cosine scale invariance, cosine gradient == central differences")
    print("PASS E[cos^2] = 1/d within Monte Carlo error for d in {16,64,256,1024}")
    print("PASS orthogonal projection is idempotent, symmetric and closest on its subspace")
    print("PASS ||AB||_F <= ||A||_F ||B||_2 and ||A||_2 <= ||A||_F")
    print(f"METRIC mean_cos2_d64={concentration[64]:.6f}")
    print(f"METRIC cos_grad_maxdiff={grad_err:.3e}")
    print(f"METRIC logit_spread_ratio={float(raw.std()/scaled.std()):.6f}")
    print(f"METRIC runtime_s={time.time() - t0:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
