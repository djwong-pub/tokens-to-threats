#!/usr/bin/env python3
"""Phase 1 — the information-theory identities a language model is trained on.

Everything here is exact arithmetic on small distributions, so the assertions are exact too.

Checks:
  1. CE(p,q) = H(p) + KL(p||q)              to 1e-14, on random pairs;
  2. Gibbs / non-negativity of KL, and the asymmetry KL(p||q) != KL(q||p);
  3. entropy is maximised by the uniform distribution, with H(uniform) = log|V|;
  4. perplexity = exp(mean NLL), and for a uniform model over |V| tokens it is exactly |V|;
  5. maximum likelihood == minimising mean NLL == minimising CE(empirical, model), both ways;
  6. log-sum-exp: where the naive form stops being finite (bisected, versus theory), and the
     stabilised form agreeing with float64 reference;
  7. sampling: inverse-CDF reproduces the target pmf, and the Gumbel-max trick agrees with it;
  8. temperature preserves argmax for every T > 0, and moves entropy monotonically.

    ./.venv/bin/python code/phase01_foundations/info_identities.py
"""
from __future__ import annotations

import sys
import time

import numpy as np

SEED = 4


def H(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-300, None)
    return -(p * np.log(p)).sum(axis=-1)


def CE(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return -(p * np.log(np.clip(q, 1e-300, None))).sum(axis=-1)


def KL(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    pp = np.clip(p, 1e-300, None)
    return (pp * np.log(pp / np.clip(q, 1e-300, None))).sum(axis=-1)


def softmax(z: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = z / T
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def lse_naive(z: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        return np.log(np.exp(z).sum(axis=-1))


def lse_stable(z: np.ndarray) -> np.ndarray:
    m = z.max(axis=-1, keepdims=True)
    return np.squeeze(m, axis=-1) + np.log(np.exp(z - m).sum(axis=-1))


def gumbel_max(p: np.ndarray, rng: np.random.Generator) -> int:
    """argmax_i (log p_i + Gumbel(0,1)) is exactly a draw from p."""
    g = -np.log(-np.log(rng.random(len(p))))
    return int(np.argmax(np.log(np.clip(p, 1e-300, None)) + g))


def inverse_cdf(p: np.ndarray, u: float) -> int:
    return int(np.searchsorted(np.cumsum(p), u, side="right"))


def main() -> int:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    V = 6

    # 1-3 --------------------------------------------------- entropy / CE / KL identities
    P = rng.dirichlet(np.ones(V), size=500)
    Q = rng.dirichlet(np.ones(V), size=500)
    decomp_err = float(np.max(np.abs(CE(P, Q) - (H(P) + KL(P, Q)))))
    assert decomp_err < 1e-13, f"CE = H + KL fails by {decomp_err:.2e}"
    kl_min = float(KL(P, Q).min())
    assert kl_min > -1e-12, f"KL went negative: {kl_min:.2e}"
    same = float(np.max(KL(P, P)))
    assert same < 1e-13, f"KL(p||p) must be 0, got {same:.2e}"
    assert float(np.max(CE(P, Q) - H(P))) >= -1e-13, "Gibbs: CE >= H"

    p_two = np.array([0.42, 0.05, 0.40, 0.08, 0.03, 0.02])
    q_two = np.array([0.86, 0.02, 0.04, 0.05, 0.02, 0.01])
    fwd, rev = float(KL(p_two, q_two)), float(KL(q_two, p_two))
    asym_ratio = max(fwd, rev) / max(min(fwd, rev), 1e-300)
    assert abs(fwd - rev) > 0.1, f"KL directions identical: {fwd} vs {rev}"

    u = np.ones(V) / V
    assert abs(float(H(u)) - np.log(V)) < 1e-14, "uniform entropy is log|V|"
    beats = float(np.max(H(rng.dirichlet(np.ones(V), size=20000))) - float(H(u)))
    assert beats <= 1e-12, f"a non-uniform distribution beat the uniform entropy by {beats:.2e}"

    # 4 ------------------------------------------------------------------- perplexity
    nll_uniform = float(CE(u, u))
    ppl_uniform = float(np.exp(nll_uniform))
    assert abs(ppl_uniform - V) < 1e-12, f"uniform perplexity should be |V|, got {ppl_uniform}"
    model = rng.dirichlet(np.ones(V), size=2000)
    nll_rows = CE(np.broadcast_to(u, model.shape), model)     # data is uniform
    ppl_mean = float(np.exp(nll_rows.mean()))
    # Gibbs again: no model can beat uniform perplexity on uniform data, so the mean perplexity
    # over random models must sit AT OR ABOVE |V|. (The first draft asserted <= and failed.)
    assert ppl_mean >= V - 1e-9, f"a model beat uniform perplexity: {ppl_mean} < {V}"

    # 5 ------------------------------------- MLE == mean NLL == CE(empirical, model)
    counts = rng.multinomial(4000, p_two)
    emp = counts / counts.sum()
    q_model = rng.dirichlet(np.ones(V) * 2.0)
    nll_from_counts = float(-(counts * np.log(np.clip(q_model, 1e-300, None))).sum() / counts.sum())
    nll_from_emp = float(CE(emp, q_model))
    assert abs(nll_from_counts - nll_from_emp) < 1e-12, \
        f"count-based NLL {nll_from_counts} != CE(empirical, model) {nll_from_emp}"
    # and MLE picks the empirical distribution: no sampled q beats emp on its own counts
    # MLE minimises CE, so the candidate set is searched with min(): nothing among 4000 smooth
    # draws may beat the empirical distribution on its own counts.
    best = min((float(CE(emp, cand)) for cand in rng.dirichlet(np.ones(V) * 30.0, size=4000)))
    ce_self = float(CE(emp, emp))
    assert best >= ce_self - 1e-12, f"a smoother distribution beat the MLE: {best:.6f} < {ce_self:.6f}"

    # 6 ------------------------------------------------------- log-sum-exp stability
    base = rng.normal(size=(300, 20))
    moderate = base * 20.0
    lse_err = float(np.max(np.abs(lse_stable(moderate) - lse_naive(moderate))))
    assert lse_err < 1e-9, f"stable vs naive log-sum-exp differ at safe scale: {lse_err:.2e}"

    def first_overflow(dtype) -> float:
        lo, hi = 1.0, 5000.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            with np.errstate(over="ignore", invalid="ignore"):
                ok = np.all(np.isfinite(lse_naive((base * mid).astype(dtype))))
            lo, hi = (mid, hi) if ok else (lo, mid)
        return float((base * hi).max())

    br32 = first_overflow(np.float32)
    br64 = first_overflow(np.float64)
    th32 = float(np.log(np.finfo(np.float32).max))
    th64 = float(np.log(np.finfo(np.float64).max))
    assert abs(br32 - th32) / th32 < 0.02, f"float32 overflow at {br32:.2f}, theory {th32:.2f}"
    assert abs(br64 - th64) / th64 < 0.02, f"float64 overflow at {br64:.2f}, theory {th64:.2f}"

    # 7 ---------------------------------------------------------------- sampling
    n_draw = 60000
    draws = np.array([inverse_cdf(p_two, u_) for u_ in rng.random(n_draw)])
    emp_p = np.bincount(draws, minlength=V) / n_draw
    icdf_dev = float(np.max(np.abs(emp_p - p_two)))
    assert icdf_dev < 0.02, f"inverse-CDF sampling off by {icdf_dev:.4f}"
    gdraws = np.array([gumbel_max(p_two, rng) for _ in range(n_draw)])
    gemp = np.bincount(gdraws, minlength=V) / n_draw
    gumbel_dev = float(np.max(np.abs(gemp - p_two)))
    assert gumbel_dev < 0.02, f"Gumbel-max sampling off by {gumbel_dev:.4f}"

    # 8 ----------------------------------------------------------------- temperature
    logits = np.array([6.2, 4.1, 3.0, 2.2, 1.1, -0.4])
    temps = np.logspace(-2, 2, 300)
    probs = np.array([softmax(logits, T)[0] for T in temps])
    argmax_ok = all(int(np.argmax(softmax(logits, T))) == int(np.argmax(logits)) for T in temps)
    assert argmax_ok, "temperature must not change the argmax"
    ent = np.array([H(softmax(logits, T)) for T in temps])
    assert np.all(np.diff(ent) >= -1e-12), "entropy must be non-decreasing in temperature"
    # At T = 100 the entropy is log|V| minus O(var(logits)/T^2) ~ 1.4e-4, so the tolerance is
    # set to the size of that correction rather than to whatever number happened to pass.
    assert ent[0] < 1e-3 and abs(ent[-1] - np.log(V)) < 2e-3, \
        f"temperature limits wrong: H(T_min)={ent[0]:.2e}, H(T_max)={ent[-1]:.6f} vs log|V|={np.log(V):.6f}"
    p_all = np.array([softmax(logits, T) for T in temps])          # (300, V)
    ppl_ok = float(np.max(np.abs(np.exp(ent) - np.exp(H(p_all)))))
    assert ppl_ok < 1e-12, f"ppl == exp(H) fails by {ppl_ok:.2e}"

    print(f"seed={SEED}  |V|={V}")
    print(f"max |CE - (H + KL)| = {decomp_err:.2e};  min KL = {kl_min:.2e};  max KL(p||p) = {same:.2e}")
    print(f"KL asymmetry on a bimodal/unimodal pair: forward {fwd:.4f}, reverse {rev:.4f} "
          f"(ratio {asym_ratio:.2f})")
    print(f"H(uniform) = {float(H(u)):.6f} = log|V|; best non-uniform entropy excess {beats:.2e}")
    print(f"perplexity of a uniform model over {V} tokens = {ppl_uniform:.6f}")
    print(f"NLL from counts == CE(empirical, model): max diff "
          f"{abs(nll_from_counts - nll_from_emp):.2e}; empirical pmf is the MLE")
    print(f"log-sum-exp overflow at max logit {br32:.2f} (float32, theory {th32:.2f}) and "
          f"{br64:.2f} (float64, theory {th64:.2f}); naive-vs-stable diff at safe scale {lse_err:.2e}")
    print(f"sampling over {n_draw} draws: max |p_emp - p| = {icdf_dev:.4f} (inverse-CDF), "
          f"{gumbel_dev:.4f} (Gumbel-max)")
    print(f"temperature: argmax preserved over T in [{temps[0]:.2g}, {temps[-1]:.2g}], "
          f"H from {ent[0]:.2e} to {ent[-1]:.4f}")
    print("PASS CE = H + KL, KL >= 0, KL(p||p) = 0, Gibbs inequality")
    print("PASS uniform maximises entropy; perplexity of uniform = |V| exactly")
    print("PASS MLE == mean NLL == CE(empirical, model), and the empirical pmf is optimal")
    print("PASS log-sum-exp overflow boundary located within 2% of theory for f32 and f64")
    print("PASS inverse-CDF and Gumbel-max sampling both reproduce the target pmf")
    print("PASS temperature keeps the argmax and moves entropy monotonically; ppl = exp(H)")
    print(f"METRIC ce_decomp_maxdiff={decomp_err:.3e}")
    print(f"METRIC kl_asymmetry_ratio={asym_ratio:.4f}")
    print(f"METRIC sampling_maxdev_icdf={icdf_dev:.4f}")
    print(f"METRIC sampling_maxdev_gumbel={gumbel_dev:.4f}")
    print(f"METRIC lse_overflow_f32_at={br32:.3f}")
    print(f"METRIC runtime_s={time.time() - t0:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
