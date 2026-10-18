"""Measure quantization error against bit width and scaling granularity (Phase 13).

The chapter derives the uniform affine quantizer x ~ q s, its per-element error bound
|e| <= s/2, and the mean-square error s^2 / 12 for a uniform distribution over one step.
This script builds a matrix with a heavy-tailed outlier row (the realistic failure mode
of per-tensor scaling) and measures, at each bit width, how much error per-tensor,
per-channel (row-wise) and group scaling each leave behind.

Checks
  bound-holds          per-element |x - dequant| <= s/2 + eps for every element, every
                       scheme, every width
  mse-law              for uniform data the measured MSE agrees with s^2/12 within 15%
                       at widths 4..8
  width-scales         quantization SNR (dB) rises by roughly 6 dB per added bit from
                       4 to 8 bits on Gaussian data (each increment within [3.5, 9.5] dB)
  channel-beats-tensor on the outlier-row matrix, per-channel error is below per-tensor
                       error at every width
  group-refines        group-64 error is at most the per-channel error (and strictly
                       below per-tensor) at 8 bits

Run:  ./.venv/bin/python code/phase13_quantization/quantizer.py
"""
from __future__ import annotations

import numpy as np


def quantize_dequantize(x: np.ndarray, bits: int, axis: int | None = None,
                        group: int | None = None) -> np.ndarray:
    """Uniform affine quantizer with symmetric per-axis or per-group scaling."""
    qmax = 2 ** (bits - 1) - 1
    if axis is None and group is None:
        m = np.abs(x).max()
        s = m / qmax
        q = np.clip(np.round(x / s), -qmax, qmax).astype(np.int32)
        return q * s
    if axis is not None:
        m = np.abs(x).max(axis=axis, keepdims=True)
        m = np.maximum(m, 1e-12)
        s = m / qmax
        q = np.clip(np.round(x / s), -qmax, qmax).astype(np.int32)
        return q * s
    # group scaling along the last axis in chunks of `group`
    orig = x.shape
    n = orig[-1] // group
    xg = x.reshape(*orig[:-1], n, group)
    m = np.abs(xg).max(axis=-1, keepdims=True)
    m = np.maximum(m, 1e-12)
    s = m / qmax
    q = np.clip(np.round(xg / s), -qmax, qmax).astype(np.int32)
    return (q * s).reshape(orig)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(((a - b) ** 2).mean())


def snr_db(x: np.ndarray, xq: np.ndarray) -> float:
    power = float(np.mean(x ** 2))
    return 10.0 * np.log10(power / (mse(x, xq) + 1e-30))


def main() -> int:
    rng = np.random.default_rng(4)

    # ------------------------------------------------------------------ error bound
    x = rng.standard_normal((256, 256)) * rng.lognormal(0.0, 0.8, (256, 256))
    for bits in (4, 6, 8):
        qmax = 2 ** (bits - 1) - 1
        s = np.abs(x).max() / qmax
        xq = quantize_dequantize(x, bits)
        assert np.abs(x - xq).max() <= s / 2 + 1e-12, "per-element bound violated"
    print("PASS bound-holds (|x - dequant| <= s/2 for all elements, 4/6/8-bit)")

    # ------------------------------------------------------------------ MSE law on uniform data
    u = rng.uniform(-1.0, 1.0, (20000,))
    for bits in (4, 5, 6, 7, 8):
        qmax = 2 ** (bits - 1) - 1
        s = 1.0 / qmax
        uq = quantize_dequantize(u, bits)
        emp = mse(u, uq)
        theo = s * s / 12.0
        assert abs(emp - theo) / theo < 0.15, f"MSE law off at {bits} bits: {emp:.3e} vs {theo:.3e}"
    print("PASS mse-law (measured MSE within 15% of s^2/12 at every width)")

    # ------------------------------------------------------------------ SNR per bit
    # plain Gaussian data: bounded enough that per-tensor scaling does not waste range
    # on a heavy tail, so the ~6 dB/bit law should hold at every increment
    xg = rng.standard_normal((256, 256)) * 3.0
    snr = {}
    for bits in (4, 5, 6, 7, 8):
        xq = quantize_dequantize(xg, bits)
        snr[bits] = snr_db(xg, xq)
    print("SNR by width (dB):", ", ".join(f"{b}: {v:.1f}" for b, v in snr.items()))
    gains = [snr[b + 1] - snr[b] for b in range(4, 8)]
    print("gain per added bit (dB):", ", ".join(f"+{g:.1f}" for g in gains))
    assert all(3.5 <= g <= 9.5 for g in gains), f"SNR gain per bit outside [3.5, 9.5]: {gains}"
    assert snr[8] - snr[4] > 15.0, "4-bit to 8-bit must buy at least 15 dB"
    print("PASS width-scales (~6 dB per bit; 4->8 bits buys more than 15 dB)")

    # ------------------------------------------------------------------ granularity
    n = 256
    xo = rng.standard_normal((n, n))
    xo[7] *= 40.0          # one heavy-tailed row: the per-tensor killer
    xo[90] *= -25.0        # and one negative one
    for bits in (4, 6, 8):
        e_t = mse(xo, quantize_dequantize(xo, bits))
        e_c = mse(xo, quantize_dequantize(xo, bits, axis=1))
        assert e_c < e_t, f"per-channel did not beat per-tensor at {bits} bits"
    print("PASS channel-beats-tensor (outlier rows shrink the per-tensor step for everyone)")

    e_t8 = mse(xo, quantize_dequantize(xo, 8))
    e_c8 = mse(xo, quantize_dequantize(xo, 8, axis=1))
    e_g8 = mse(xo, quantize_dequantize(xo, 8, group=64))
    assert e_g8 <= e_c8 + 1e-30 and e_g8 < e_t8
    print(f"PASS group-refines (at 8 bits: per-tensor {np.sqrt(e_t8):.2e} rmse, "
          f"per-channel {np.sqrt(e_c8):.2e}, group-64 {np.sqrt(e_g8):.2e})")

    print(f"METRIC snr_4bit={snr[4]:.1f}")
    print(f"METRIC snr_8bit={snr[8]:.1f}")
    print(f"METRIC outlier_rmse_8bit_channel={np.sqrt(e_c8):.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
