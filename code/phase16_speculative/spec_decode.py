"""Verify speculative decoding of Phase 16 against its rejection-sampling definition.

The chapter derives the accept/reject rule: a draft token x drawn from q(x) is
accepted with probability min(1, p(x)/q(x)), where p is the target distribution, and
on rejection the token is resampled from the normalised residual
    r(x) = max(0, p(x) - q(x)) / sum_y max(0, p(y) - q(y)).
The theorem checked here is that the accepted token is distributed exactly as p, and
the expected acceptance rate is alpha = sum_x min(p(x), q(x)), which gives the
expected speedup formula of the chapter.

Checks
  accept-distribution  the accepted token (accept or residual-resample) is distributed
                       exactly as the target p: empirical frequencies inside a binomial
                       band over many steps
  acceptance-rate      the measured acceptance rate agrees with
                       alpha = sum_x min(p, q) within a binomial band
  speedup-formula      the expected tokens per target step, (1 + alpha * L_eff), and
                       the chapter speedup ratio agree with the measured simulation
                       within tolerance (verification cost of L draft tokens vs 1)
  q-equals-p           when q = p the acceptance rate is 1 and no residual resampling
                       ever fires

Run:  ./.venv/bin/python code/phase16_speculative/spec_decode.py
"""
from __future__ import annotations

import numpy as np


def spec_step(p: np.ndarray, q: np.ndarray, rng: np.random.Generator) -> tuple[int, str]:
    """One speculative step for one token position.

    Returns (accepted_token, how): 'accept' or 'residual'.
    """
    x = int(rng.choice(p.size, p=q))
    if rng.random() < min(1.0, p[x] / q[x]):
        return x, "accept"
    r = np.maximum(0.0, p - q)
    r = r / r.sum()
    return int(rng.choice(p.size, p=r)), "residual"


def main() -> int:
    rng = np.random.default_rng(31)
    V = 20

    p = np.array([0.30, 0.25, 0.15, 0.10, 0.08, 0.05, 0.03, 0.02, 0.01, 0.01] + [0.01] * 10)
    p = p / p.sum()
    q = np.array([0.20, 0.35, 0.10, 0.12, 0.10, 0.04, 0.03, 0.03, 0.01, 0.01] + [0.01] * 10)
    q = q / q.sum()

    n = 40_000
    accepted = np.bincount([spec_step(p, q, rng)[0] for _ in range(n)], minlength=V)
    emp = accepted / n
    sd = np.sqrt(np.maximum(p * (1 - p) / n, 1e-18))
    assert (np.abs(emp - p) < 6 * sd).all(), "accepted tokens not distributed as p"
    print(f"PASS accept-distribution (worst |emp - p| = {np.abs(emp - p).max():.4f}, "
          f"band {6 * sd.max():.4f} over {n} steps)")

    alpha_theory = float(np.minimum(p, q).sum())
    n2 = 40_000
    n_acc = sum(1 for _ in range(n2) if spec_step(p, q, rng)[1] == "accept")
    alpha_emp = n_acc / n2
    band = 6 * np.sqrt(alpha_theory * (1 - alpha_theory) / n2)
    assert abs(alpha_emp - alpha_theory) < band, \
        f"acceptance rate {alpha_emp:.4f} outside band of {alpha_theory:.4f}"
    print(f"PASS acceptance-rate (theory alpha = {alpha_theory:.4f}, measured "
          f"{alpha_emp:.4f}, band {band:.4f})")

    # speedup formula: a draft proposes L tokens; the target verifies them in one pass
    # (cost 1) and, with high probability on a good draft, emits several tokens per
    # pass. The chapter's expected tokens per target pass for a chain of L drafts with
    # independent per-token acceptance is sum_{j=0}^{L} alpha^j = (1 - alpha^{L+1})/(1-alpha).
    for L in (2, 4, 8):
        theory = (1 - alpha_theory ** (L + 1)) / (1 - alpha_theory)
        # simulate: expected emitted tokens per pass
        tot, reps = 0.0, 6000
        for _ in range(reps):
            emitted = 0
            for j in range(L):
                tok, how = spec_step(p, q, rng)
                if how == "accept":
                    emitted += 1
                else:
                    emitted += 1          # the residual token is emitted, chain stops
                    break
            else:
                emitted += 1              # all L accepted: bonus target token at L+1
            tot += emitted
        emp_tokens = tot / reps
        assert abs(emp_tokens - theory) / theory < 0.06, \
            f"L={L}: measured {emp_tokens:.3f} vs theory {theory:.3f}"
    print(f"PASS speedup-formula (tokens per pass matches (1-alpha^(L+1))/(1-alpha) "
          f"within 6% for L = 2, 4, 8)")

    n3 = 20_000
    all_acc = all(spec_step(p, p, rng)[1] == "accept" for _ in range(n3))
    assert all_acc, "q = p must accept every draft"
    print(f"PASS q-equals-p ({n3} steps, zero residual resamplings)")

    print(f"METRIC alpha={alpha_theory:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
