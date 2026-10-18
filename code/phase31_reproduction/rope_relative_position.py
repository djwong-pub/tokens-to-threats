"""Phase 31 — Learn to reproduce papers: a fully worked toy reproduction.

The reproduced claim is Theorem-style result 1 of the RoFormer paper (Su et al.,
2021, arXiv:2104.09864): with the rotary position embedding R(m) applied at
position m, the attention score between a query at position m and a key at
position n depends only on the relative position n-m, not on the absolute
positions:

    <R(m) q, R(n) k>  =  <q, R(n-m) k>          for all m, n, q, k.

Equivalently, the rotation operators form a (continuous-parameter, here
discretized) group: R(m)^T R(n) = R(n-m), with each R(m) orthogonal.

This script follows the seven-step protocol of the chapter on a toy scale:
(1) read the claim, (2) predict the mechanism (a group of rotations), (3)
read the construction (paired 2D rotations with base-10000 frequencies),
(4) implement the simplified version below, (5) reproduce the property
numerically, (6) change one variable (the frequency base; the rotation
convention), and (7) explain the discrepancy of a controlled wrong
convention (rotate only the query) in terms of what the paper's proof needs.

Checks (self-test mode, the default)
  orthogonality        R(m)^T R(m) = I to floating-point precision
  group-law            R(m)^T R(n) = R(n-m) to floating-point precision
  relative-invariance  the theorem's inner-product identity on 500 random trials
  base-sweep           the property is invariant to the frequency base
  controlled-breakage  rotating only the query makes the score depend on the
                       absolute position, by a margin orders of magnitude above
                       the tolerance (the discrepancy the chapter explains)

Run:  ./.venv/bin/python code/phase31_reproduction/rope_relative_position.py
"""
from __future__ import annotations

import numpy as np

D = 64                      # head dimension (even, as required by RoPE)
MAX_POS = 512               # context length for the position sweep
N_TRIALS = 500              # random (m, n, q, k) trials
TOL = 1e-9                  # float64 tolerance for "equal to machine precision"
RNG_SEED = 31
BASES = (100.0, 10_000.0, 1_000_000.0)   # step 6: the variable we change


def rope_matrix(m: int, base: float = 10_000.0, d: int = D) -> np.ndarray:
    """The RoPE rotation R(m): block-diagonal 2x2 rotations.

    Pair i (coordinates 2i, 2i+1) is rotated by m * theta_i with
    theta_i = base ** (-2i / d), exactly the paper's construction.
    """
    i = np.arange(d // 2)
    theta = base ** (-2.0 * i / d)   # per-pair frequency, the paper's theta_i
    ang = m * theta                  # per-pair angle at position m (radians)
    C = np.diag(np.cos(ang))
    S = np.diag(np.sin(ang))
    M = np.zeros((d, d))
    M[0::2, 0::2] = C
    M[0::2, 1::2] = -S
    M[1::2, 0::2] = S
    M[1::2, 1::2] = C
    return M


def main() -> int:
    rng = np.random.default_rng(RNG_SEED)

    # ------------------------------------------------------ orthogonality
    errs = []
    for m in (0, 1, 7, 63, 251, MAX_POS):
        R = rope_matrix(m)
        errs.append(np.linalg.norm(R.T @ R - np.eye(D), "fro"))
    max_orth = max(errs)
    assert max_orth < TOL, f"R(m) not orthogonal: {max_orth}"
    print(f"PASS orthogonality (max ||R^T R - I||_F over 6 positions = "
          f"{max_orth:.2e})")

    # ---------------------------------------------------------- group-law
    gerrs = []
    for m in (0, 1, 37, 250):
        for n in (0, 1, 99, 511):
            lhs = rope_matrix(m).T @ rope_matrix(n)
            rhs = rope_matrix(n - m)
            gerrs.append(np.linalg.norm(lhs - rhs, "fro"))
    max_group = max(gerrs)
    assert max_group < TOL, f"group law broken: {max_group}"
    print(f"PASS group-law (max ||R(m)^T R(n) - R(n-m)||_F over "
          f"{len(gerrs)} position pairs = {max_group:.2e})")

    # --------------------------------------------- relative-invariance
    trials_err = []
    for _ in range(N_TRIALS):
        m = int(rng.integers(0, MAX_POS + 1))
        n = int(rng.integers(0, MAX_POS + 1))
        q = rng.standard_normal(D)
        k = rng.standard_normal(D)
        lhs = float(np.dot(rope_matrix(m) @ q, rope_matrix(n) @ k))
        rhs = float(np.dot(q, rope_matrix(n - m) @ k))
        trials_err.append(abs(lhs - rhs))
    max_rel = max(trials_err)
    assert max_rel < TOL, f"relative invariance broken: {max_rel}"
    print(f"PASS relative-invariance (max |<R(m)q, R(n)k> - <q, R(n-m)k>| "
          f"over {N_TRIALS} random trials = {max_rel:.2e})")

    # --------------------------------------------------------- base-sweep
    for base in BASES:
        worst = 0.0
        for _ in range(50):
            m = int(rng.integers(0, MAX_POS + 1))
            n = int(rng.integers(0, MAX_POS + 1))
            q = rng.standard_normal(D)
            k = rng.standard_normal(D)
            lhs = float(np.dot(rope_matrix(m, base) @ q, rope_matrix(n, base) @ k))
            rhs = float(np.dot(q, rope_matrix(n - m, base) @ k))
            worst = max(worst, abs(lhs - rhs))
        assert worst < TOL, f"invariance broke at base {base}: {worst}"
    print(f"PASS base-sweep (invariance holds for bases {BASES}; the property "
          f"is independent of the frequency base, which changes wavelengths, "
          f"not the group law)")

    # ---------------------------------------------- controlled-breakage
    # Step 6, second variable: rotate ONLY the query (a wrong convention).
    spreads = []
    for _ in range(200):
        n = int(rng.integers(0, MAX_POS + 1))
        q = rng.standard_normal(D)
        k = rng.standard_normal(D)
        scores = [float(np.dot(rope_matrix(m) @ q, k)) for m in range(0, 128, 7)]
        spreads.append(max(scores) - min(scores))
    wrong_spread = max(spreads)
    assert wrong_spread > 10 * TOL * 1e3, \
        f"wrong convention unexpectedly invariant: {wrong_spread}"
    print(f"PASS controlled-breakage (rotating only the query makes the score "
          f"depend on the absolute position m; max spread over m = "
          f"{wrong_spread:.4f}, i.e. O(1), against the {TOL:.0e} tolerance "
          f"of the true convention: the discrepancy is explained by the group "
          f"law, which one-sided rotation does not satisfy)")

    print(f"METRIC d={D} max_pos={MAX_POS} n_trials={N_TRIALS} "
          f"max_rel_invariance_err={max_rel:.2e} max_group_err={max_group:.2e} "
          f"wrong_convention_spread={wrong_spread:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
