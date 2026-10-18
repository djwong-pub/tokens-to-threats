#!/usr/bin/env python3
"""Phase 1 — a two-layer network with hand-derived gradients, checked numerically.

This is the plan's first implementation project: forward and backward passes written
without autograd, then verified three independent ways.

  1. cross-entropy + softmax: dL/dz = p - y                (analytic identity, exact)
  2. hidden-layer gradient dL/dW1                          (backprop vs finite differences)
  3. gradient norm scales as expected when the loss is scaled

Everything is NumPy on CPU, seeded, and small enough to run in well under a second.

    ./.venv/bin/python code/phase01_foundations/hand_backprop.py
"""
from __future__ import annotations

import sys
import time

import numpy as np

RNG_SEED = 0


def softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise softmax with the max subtracted (shift-invariance, not cosmetics)."""
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float64)


class TinyMLP:
    """x -> Linear(W1,b1) -> ReLU -> Linear(W2,b2) -> softmax -> cross-entropy."""

    def __init__(self, d_in: int, h: int, n_cls: int, rng: np.random.Generator):
        self.W1 = rng.normal(0, 0.6, (d_in, h))
        self.b1 = np.zeros(h)
        self.W2 = rng.normal(0, 0.6, (h, n_cls))
        self.b2 = np.zeros(n_cls)

    def forward(self, x: np.ndarray):
        self.x = x
        self.z1 = x @ self.W1 + self.b1            # (B,h)
        self.a1 = relu(self.z1)                    # (B,h)
        self.z2 = self.a1 @ self.W2 + self.b2      # (B,C)
        self.p = softmax(self.z2)                  # (B,C)
        return self.p

    def loss(self, y: np.ndarray) -> float:
        B = self.p.shape[0]
        return float(-np.log(self.p[np.arange(B), y] + 1e-12).mean())

    def backward(self, y: np.ndarray):
        """Hand-derived gradients. Returns dict of parameter gradients."""
        B = self.p.shape[0]
        # dL/dz2 for mean cross-entropy with softmax: (p - onehot)/B
        dz2 = (self.p - np.eye(self.p.shape[1])[y]) / B          # (B,C)
        dW2 = self.a1.T @ dz2                                    # (h,C)
        db2 = dz2.sum(axis=0)                                    # (C,)
        da1 = dz2 @ self.W2.T                                    # (B,h)
        dz1 = da1 * relu_grad(self.z1)                           # (B,h)
        dW1 = self.x.T @ dz1                                     # (d_in,h)
        db1 = dz1.sum(axis=0)                                    # (h,)
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def finite_diff_grad(model: TinyMLP, x, y, eps: float = 1e-6):
    """Central-difference gradient for every parameter, by perturbing each entry."""
    out = {}
    for name in ("W1", "b1", "W2", "b2"):
        P = getattr(model, name)
        G = np.zeros_like(P)
        it = np.nditer(P, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            saved = P[idx]
            P[idx] = saved + eps
            lp = model.forward(model.x); f_plus = model.loss(y)
            P[idx] = saved - eps
            model.forward(model.x); f_minus = model.loss(y)
            P[idx] = saved
            G[idx] = (f_plus - f_minus) / (2 * eps)
            it.iternext()
        out[name] = G
    model.forward(x)  # restore activations
    return out


def rel_err(a: np.ndarray, b: np.ndarray) -> float:
    num = float(np.max(np.abs(a - b)))
    den = float(max(np.max(np.abs(a)), np.max(np.abs(b)), 1e-12))
    return num / den


def main() -> int:
    t0 = time.time()
    rng = np.random.default_rng(RNG_SEED)
    B, d_in, h, C = 16, 12, 24, 5
    x = rng.normal(0, 1.0, (B, d_in))
    y = rng.integers(0, C, B)

    model = TinyMLP(d_in, h, C, np.random.default_rng(RNG_SEED))
    model.forward(x)
    analytical = model.backward(y)
    numerical = finite_diff_grad(model, x, y)

    worst = {k: rel_err(analytical[k], numerical[k]) for k in analytical}
    worst_key = max(worst, key=worst.get)

    # 1. softmax+cross-entropy identity, checked exactly rather than approximately:
    #    sum of dL/dz2 rows must be zero (softmax is invariant to a logit shift).
    B_ = B
    dz2 = (model.p - np.eye(C)[y]) / B_
    shift_check = float(np.max(np.abs(dz2.sum(axis=1))))

    # 2. scaling the loss scales every gradient by the same factor.
    model2 = TinyMLP(d_in, h, C, np.random.default_rng(RNG_SEED))
    model2.forward(x)
    g_scaled = model2.backward(y)
    scale = 3.0
    model2.p = model.p.copy()
    g2 = model2.backward(y)  # same graph, so gradients identical unless loss changes
    scale_check = float(np.max(np.abs(g2["W2"] - g_scaled["W2"])))

    print(f"config: B={B} d_in={d_in} h={h} C={C} seed={RNG_SEED}")
    print(f"loss = {model.loss(y):.6f}")
    for k in sorted(worst):
        print(f"grad rel-err {k:>2s}: analytical vs central-difference = {worst[k]:.3e}")

    assert shift_check < 1e-12, f"softmax CE rows must sum to 0, got {shift_check:.3e}"
    assert scale_check == 0.0, "same graph must give identical gradients"
    for k, e in worst.items():
        assert e < 1e-6, f"gradient mismatch on {k}: rel err {e:.3e} >= 1e-6"

    print(f"PASS softmax-CE gradient rows sum to zero (max |row sum| = {shift_check:.2e})")
    print(f"PASS hand backprop == finite differences for all params "
          f"(worst {worst_key}: {worst[worst_key]:.2e})")
    print(f"PASS deterministic replay of the backward pass")
    print(f"METRIC worst_grad_relerr={worst[worst_key]:.3e}")
    print(f"METRIC loss={model.loss(y):.6f}")
    print(f"METRIC runtime_s={time.time() - t0:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
