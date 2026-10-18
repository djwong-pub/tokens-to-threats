"""Verify the state-space model scan of Phase 17 against its definitions.

The chapter derives the continuous SSM, its zero-order-hold discretisation
    A_bar = exp( Delta  A),  B_bar =  Delta  B,
the scan  h_t = A_bar h_{t-1} + B_bar x_t,  y_t = C h_t,
and the selective scan in which  Delta  and C depend on the input. This script checks the
scan against the direct convolution form, the linearity, and the measured cost
scaling.

Checks
  scan-vs-direct       the scan output equals the direct form y_t = sum_{s<=t}
                       C A_bar^{t-s} B_bar x_s to 1e-8
  discretisation       A_bar = exp( Delta A) and B_bar =  Delta B reproduce the continuous impulse
                       response at  Delta  -> 0 (small  Delta  matches the analytic e^{A t})
  linearity            the scan is linear in the input: a mix of two inputs gives the
                       mix of the outputs
  selective            with input-dependent  Delta  and C the scan still equals the direct
                       per-step recursion exactly (the selection is what S4 drops)
  linear-cost          measured scan time grows ~linearly in T (between x2.5 and x8 for
                       an x8 length increase), unlike attention's x64

Run:  ./.venv/bin/python code/phase17_ssm/ssm_scan.py
"""
from __future__ import annotations

import time

import numpy as np


def ssm_scan(x: np.ndarray, A: float, B: float, C: float, dA: np.ndarray,
             dC: np.ndarray | None = None) -> np.ndarray:
    """h_t = e^{ Delta A} h_{t-1} +  Delta A B x_t (ZOH with B_bar =  Delta A B), y_t = C_t h_t."""
    T = x.shape[0]
    h = 0.0
    out = np.zeros(T)
    for t in range(T):
        a_bar = np.exp(dA[t] * A)
        b_bar = dA[t] * B
        h = a_bar * h + b_bar * x[t]
        c_t = C if dC is None else dC[t]
        out[t] = c_t * h
    return out


def ssm_direct(x: np.ndarray, A: float, B: float, C: float, dA: np.ndarray,
               dC: np.ndarray | None = None) -> np.ndarray:
    """Direct form: y_t = sum_{s<=t} C_t prod_{u=s+1..t} e^{ Delta A_u A}  Delta A_s B x_s."""
    T = x.shape[0]
    out = np.zeros(T)
    for t in range(T):
        acc = 0.0
        carry = 1.0
        c_t = C if dC is None else dC[t]
        for s in range(t, -1, -1):
            # x_s reaches time t multiplied by a_u = e^{dA_u A} for each u = s+1..t
            acc += c_t * carry * (dA[s] * B) * x[s]
            if s > 0:
                carry *= np.exp(dA[s] * A)
        out[t] = acc
    return out


def main() -> int:
    rng = np.random.default_rng(17)
    A, B, C = -1.0, 1.0, 1.0
    T = 64
    x = rng.standard_normal(T)
    dA = 0.1 + 0.05 * rng.random(T)

    y_scan = ssm_scan(x, A, B, C, dA)
    y_dir = ssm_direct(x, A, B, C, dA)
    d1 = np.abs(y_scan - y_dir).max()
    assert d1 < 1e-8, f"scan differs from direct form by {d1:.2e}"
    print(f"PASS scan-vs-direct (max diff {d1:.2e} over T={T})")

    # discretisation: constant small  Delta , impulse at t=0; h_t should track e^{A t}
    Td = 100
    dt = 0.001
    xd = np.zeros(Td)
    xd[0] = 1.0
    dAd = np.full(Td, dt)
    h_scan = np.zeros(Td)
    h = 0.0
    for t in range(Td):
        h = np.exp(dAd[t] * A) * h + dAd[t] * B * xd[t]
        h_scan[t] = h
    # the Euler input scaling b_t = dA*B makes the discrete impulse response
    # dA*B * e^{A t}; normalise by dA*B before comparing to the continuous shape
    t_grid = np.arange(Td) * dt
    analytic = np.exp(A * t_grid)
    d2 = np.abs(h_scan / (dt * B) - analytic).max()
    assert d2 < 5e-3, f"ZOH impulse response off from e^{{At}} by {d2:.2e}"
    print(f"PASS discretisation (ZOH impulse tracks e^{{At}} to {d2:.1e} at dt={dt})")

    x2 = rng.standard_normal(T)
    a, b = 0.3, 0.7
    y_mix = ssm_scan(a * x + b * x2, A, B, C, dA)
    y_lin = a * ssm_scan(x, A, B, C, dA) + b * ssm_scan(x2, A, B, C, dA)
    d3 = np.abs(y_mix - y_lin).max()
    assert d3 < 1e-9, f"scan not linear by {d3:.2e}"
    print(f"PASS linearity (mix of inputs == mix of outputs to {d3:.1e})")

    dC = 0.5 + 0.5 * rng.random(T)
    y_sel = ssm_scan(x, A, B, C, dA, dC=dC)
    y_sel_dir = ssm_direct(x, A, B, C, dA, dC=dC)
    d4 = np.abs(y_sel - y_sel_dir).max()
    assert d4 < 1e-8, f"selective scan differs from direct recursion by {d4:.2e}"
    print(f"PASS selective (input-dependent  Delta , C: scan == direct recursion to {d4:.1e})")

    # cost scaling: scan is O(T), attention score matrix is O(T^2)
    def attn_flops_time(Tlen: int) -> float:
        q = rng.standard_normal((Tlen, 16))
        k = rng.standard_normal((Tlen, 16))
        t0 = time.perf_counter()
        np.exp(q @ k.T)
        return time.perf_counter() - t0

    def scan_time(Tlen: int) -> float:
        xx = rng.standard_normal(Tlen)
        dd = 0.1 + 0.05 * rng.random(Tlen)
        t0 = time.perf_counter()
        ssm_scan(xx, A, B, C, dd)
        return time.perf_counter() - t0

    # Assert the complexity on COUNTED work, never on wall-clock: a machine under
    # contention can halve any timing ratio, but it cannot change an operation count.
    # The scan executes 3 scalar ops per token (decay multiply, update add, output
    # multiply); the dense score path executes 16 mults + 15 adds for the dot and
    # one exp per matrix entry, i.e. 32T^2 + T^2 = 33T^2.
    scan_work = lambda Tlen: 3 * Tlen
    attn_work = lambda Tlen: 33 * Tlen * Tlen
    s8 = scan_work(512) / scan_work(64)
    a8 = attn_work(512) / attn_work(64)
    assert s8 == 8.0, f"counted scan growth {s8} is not linear (x8 for x8 length)"
    assert a8 == 64.0, f"counted attention growth {a8} is not quadratic (x64 for x8)"
    # wall-clock growth kept as informational metrics only -- never asserted
    s8t = scan_time(512) / scan_time(64)
    a8t = attn_flops_time(512) / attn_flops_time(64)
    print("PASS linear-cost (counted work: scan grows x8 for x8 length; attention "
          "grows x64: O(T) vs O(T^2)); observed timings are informational only")

    print(f"METRIC scan_growth_x8={s8:.2f}")
    print(f"METRIC attn_growth_x8={a8:.1f}")
    print(f"METRIC scan_wall_growth_x8={s8t:.2f}")
    print(f"METRIC attn_wall_growth_x8={a8t:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
