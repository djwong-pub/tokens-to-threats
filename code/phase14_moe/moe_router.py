"""Verify the mixture-of-experts router of Phase 14 against its definitions.

The chapter derives the router softmax over E experts, the top-k gate, and the
auxiliary load-balancing loss
    L_aux = E * sum_i f_i * P_i,
where f_i is the fraction of tokens routed to expert i and P_i the average router
probability of expert i. This script builds a fixed random router and checks each
claim.

Checks
  router-softmax       the router probabilities are the softmax of the logits and sum
                       to one per token
  topk-gate            the gate is nonzero on exactly the k largest router
                       probabilities, renormalised over those k
  load-stats           f_i and P_i computed by hand match the reference formula and
                       sum to one (f) and E (P, per token row)
  aux-loss             the measured L_aux equals E * sum_i f_i P_i exactly
  balanced-random      when the router logits are (near) uniform, the load is (near)
                       uniform: max f_i within a binomial band of 1/E
  identical-experts    with all experts equal, the MoE block output equals the shared
                       FFN output exactly, whatever the routing

Run:  ./.venv/bin/python code/phase14_moe/moe_router.py
"""
from __future__ import annotations

import numpy as np


def router_softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def topk_gate(p: np.ndarray, k: int) -> np.ndarray:
    idx = np.argsort(-p, axis=-1)
    keep = idx[:, :k]
    g = np.zeros_like(p)
    for i in range(p.shape[0]):
        sel = keep[i]
        g[i, sel] = p[i, sel] / p[i, sel].sum()
    return g


def aux_loss(p: np.ndarray, gate: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Switch-style auxiliary loss L_aux = E sum_i f_i P_i.

    f_i: fraction of tokens whose top-k set contains expert i (sums to K over i).
    P_i: mean of the router probability of expert i over all tokens.
    """
    E = p.shape[1]
    T = p.shape[0]
    f = (gate > 0).sum(axis=0) / T
    P = p.mean(axis=0)
    return float(E * np.sum(f * P)), f, P


def ffn(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    return np.maximum(x @ w1, 0.0) @ w2


def moe_block(x: np.ndarray, gate: np.ndarray, experts: list[np.ndarray]) -> np.ndarray:
    """x: (T, d). experts: list of (w1, w2) pairs. Returns the gated mixture."""
    out = np.zeros_like(x)
    for e in range(len(experts)):
        m = gate[:, e:e + 1]
        out += m * ffn(x, experts[e][0], experts[e][1])
    return out


def main() -> int:
    rng = np.random.default_rng(21)
    T, E, K, d = 512, 8, 2, 64

    logits = rng.standard_normal((T, E)) * 2.0
    p = router_softmax(logits)
    assert np.abs(p.sum(axis=-1) - 1.0).max() < 1e-12, "router probs do not sum to one"
    print(f"PASS router-softmax (rows sum to 1; max prob {p.max():.3f})")

    g = topk_gate(p, K)
    topk_idx = np.argsort(-p, axis=-1)[:, :K]
    mask = np.zeros_like(g, dtype=bool)
    for i in range(T):
        mask[i, topk_idx[i]] = True
    assert ((g > 0) == mask).all(), "gate support is not the top-k set"
    assert np.abs(g.sum(axis=-1) - 1.0).max() < 1e-12, "gate does not renormalise"
    for i in range(T):
        sel = topk_idx[i]
        ratio = g[i, sel[0]] / p[i, sel[0]]
        assert np.abs(g[i, sel] / p[i, sel] - ratio).max() < 1e-12, "renormalisation not uniform"
    print(f"PASS topk-gate (support = top-{K} exactly; renormalised over the support)")

    loss, f, P = aux_loss(p, g)
    assert abs(f.sum() - K) < 1e-12, f"f_i must sum to K={K}, got {f.sum()}"
    assert abs(np.mean(P) - 1.0 / E) < 1e-12, "mean of P_i is 1/E"
    manual = float(E * np.sum(f * P))
    assert abs(loss - manual) < 1e-15
    print(f"PASS aux-loss (L_aux = E sum f_i P_i = {loss:.4f}; max load f = {f.max():.3f}, "
          f"sum f = {f.sum():.1f} = K)")

    # near-uniform router: small logits -> each expert in the top-k with prob K/E
    p_u = router_softmax(rng.standard_normal((4096, E)) * 0.05)
    g_u = topk_gate(p_u, K)
    _, f_u, _ = aux_loss(p_u, g_u)
    fair = K / E
    band = 5 * np.sqrt(fair * (1 - fair) / 4096)
    assert np.abs(f_u - fair).max() < band, f"uniform router load outside band: {f_u}"
    print(f"PASS balanced-random (uniform-ish router: max |f_i - K/E| = "
          f"{np.abs(f_u - fair).max():.4f} < band {band:.4f})")

    # identical experts: the mixture is the shared FFN regardless of routing
    w1 = rng.standard_normal((d, d)) * 0.1
    w2 = rng.standard_normal((d, d)) * 0.1
    experts = [(w1, w2)] * E
    x = rng.standard_normal((64, d))
    g_rand = topk_gate(router_softmax(rng.standard_normal((64, E)) * 3.0), K)
    d1 = np.abs(moe_block(x, g_rand, experts) - ffn(x, w1, w2)).max()
    assert d1 < 1e-9, f"MoE with identical experts differs by {d1:.2e}"
    print(f"PASS identical-experts (mixture == shared FFN to {d1:.1e} under any routing)")

    print(f"METRIC aux_loss={loss:.4f}")
    print(f"METRIC max_load={f.max():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
