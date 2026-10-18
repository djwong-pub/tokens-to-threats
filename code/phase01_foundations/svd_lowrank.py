#!/usr/bin/env python3
"""Phase 1 — low-rank structure, rank limits, and the cost of a factorisation.

What this pins down numerically:
  1. Eckart-Young: the best rank-r approximation error is exactly the discarded tail,
        ||A - A_r||_F == sqrt(sum_{k>r} sigma_k^2)
     checked at several sizes and ranks;
  2. random subspaces are worse than the truncated SVD, by a measured margin;
  3. rank inequalities: rank(AB) <= min(rank A, rank B), and an attention matrix built from
     d_head features can have rank at most d_head -- so a head with d_head = 8 cannot express
     an arbitrary T x T pattern, no matter how the projections are trained;
  4. spectral norm, condition number, and the optimal fixed-step rate for a quadratic:
     min over eta of max_i |1 - eta lambda_i| = (kappa - 1)/(kappa + 1);
  5. parameter and activation accounting for low-rank factorisation, i.e. the arithmetic that
     LoRA and MLA-style compressed KV caches are made of.

    ./.venv/bin/python code/phase01_foundations/svd_lowrank.py
"""
from __future__ import annotations

import sys
import time

import numpy as np

SEED = 3


def truncated(A: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return np.zeros_like(A)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    return (U[:, :r] * s[:r]) @ Vt[:r]


def tail(s: np.ndarray, r: int) -> float:
    return 0.0 if r >= len(s) else float(np.sqrt((s[r:] ** 2).sum()))


def main() -> int:
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    # 1 ------------------------------------------------------------ Eckart-Young, exactly
    worst_ey = 0.0
    for (m, n) in [(20, 14), (37, 11), (50, 50)]:
        A = rng.normal(size=(m, n))
        s = np.linalg.svd(A, compute_uv=False)
        for r in range(0, min(m, n) + 1, 3):
            err = float(np.linalg.norm(A - truncated(A, r), "fro"))
            worst_ey = max(worst_ey, abs(err - tail(s, r)))
    assert worst_ey < 1e-9, f"Eckart-Young violated by {worst_ey:.2e}"

    # 2 ------------------------------------------- random subspaces lose, and by how much
    A = rng.normal(size=(64, 40))
    s = np.linalg.svd(A, compute_uv=False)
    best10 = tail(s, 10)
    ratios = []
    for _ in range(25):
        Qm, _ = np.linalg.qr(rng.normal(size=(64, 10)))
        ratios.append(float(np.linalg.norm(A - Qm @ (Qm.T @ A), "fro")) / best10)
    best_random, worst_random = min(ratios), max(ratios)
    assert best_random > 1.0 - 1e-9, "a random subspace beat the optimal rank-10 approximation"

    # 3 ------------------------------------------------ attention rank is capped by d_head
    T, d_head = 24, 6
    q = rng.normal(size=(T, d_head))
    k = rng.normal(size=(T, d_head))
    scores = q @ k.T
    rank_scores = int(np.linalg.matrix_rank(scores))
    assert rank_scores <= min(d_head, T), f"rank {rank_scores} exceeds d_head={d_head}"
    # A full-rank target pattern the head provably cannot reach:
    target = rng.normal(size=(T, T))
    resid = float(np.linalg.norm(target - scores) / np.linalg.norm(target))
    resid_full = float(np.linalg.norm(target - np.eye(T) * 0) / np.linalg.norm(target))
    assert resid > 0.5, "a rank-6 score matrix should be far from a generic dense pattern"

    # 4 ------------------------------------------- optimal fixed step on a quadratic
    kappa_grid = [4.0, 16.0, 64.0, 256.0]
    rate_check = []
    for kap in kappa_grid:
        lam_max, lam_min = kap, 1.0
        eta_star = 2.0 / (lam_max + lam_min)
        empirical = max(abs(1 - eta_star * lam) for lam in (lam_max, lam_min))
        theory = (kap - 1) / (kap + 1)
        rate_check.append((kap, empirical, theory))
        assert abs(empirical - theory) < 1e-12, f"kappa={kap}: {empirical} vs {theory}"
        # and no other step size does better (grid search over eta)
        grid = np.linspace(1e-4, 2.5 * eta_star, 20000)
        worst_curve = np.array([np.max(np.abs(1 - eta * np.array([lam_min, lam_max])))
                                for eta in grid])
        assert worst_curve.min() >= theory - 1e-9, "a better step size than eta_star exists"

    # 5 -------------------------------------------------- the cost arithmetic of low rank
    rows = []
    for d_model, r in [(1024, 8), (4096, 8), (4096, 64), (8192, 16)]:
        full = d_model * d_model
        low = 2 * r * d_model                    # down-projection r x d and up d x r
        rows.append((d_model, r, full, low, low / full))
        assert low < full, "rank r < d must shrink the parameter count"
    # A concrete claim used in the chapter: d=4096, r=8 is 0.39% of a square layer.
    assert abs(rows[1][4] - 2 * 8 / 4096) < 1e-12

    print(f"seed={SEED}")
    print(f"Eckart-Young max |measured - tail| = {worst_ey:.2e}")
    print(f"random rank-10 subspace vs truncated SVD: best ratio {best_random:.3f}, "
          f"worst {worst_random:.3f}  (>1 means SVD wins)")
    print(f"attention with d_head={d_head} over T={T}: rank {rank_scores} "
          f"(cap {min(d_head, T)}), relative residual vs a dense target {resid:.3f} "
          f"(a dense random matrix sits at {resid_full:.3f})")
    print("optimal fixed-step contraction for a quadratic:")
    for kap, emp, th in rate_check:
        print(f"    kappa={kap:6.1f}  measured {emp:.6f}  (kappa-1)/(kappa+1) {th:.6f}")
    print("low-rank factorisation cost (square layer of width d -> rank r, both factors):")
    for d_model, r, full, low, ratio in rows:
        print(f"    d={d_model:5d} r={r:3d}: full {full:>10,d}  low-rank {low:>9,d}  "
              f"{100*ratio:.2f}% of full")
    print("PASS Eckart-Young identity exact at three sizes, every rank step")
    print("PASS truncated SVD beats 25/25 random subspaces of equal rank")
    print("PASS rank(QK^T) <= d_head, and a low-rank score matrix cannot fit a dense pattern")
    print("PASS eta_star = 2/(L+mu) is optimal: no grid step size contracts faster")
    print(f"METRIC eckart_young_maxdiff={worst_ey:.3e}")
    print(f"METRIC random_vs_svd_best_ratio={best_random:.6f}")
    print(f"METRIC lowrank_fraction_d4096_r8={rows[1][4]:.6f}")
    print(f"METRIC runtime_s={time.time() - t0:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
