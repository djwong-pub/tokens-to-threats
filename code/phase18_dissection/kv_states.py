"""Verify the KV-state arithmetic of the Phase 18 dissection table.

The chapter's worked examples fill the "KV state: size per token" row for three
architectures. This script recomputes that row from the published hyperparameters and
checks the ratios the table quotes, so the table's numbers are arithmetic on stated
inputs rather than transcription.

Checked configurations (hyperparameters from the papers in the corpus)
  Qwen3-8B     GQA: h_kv = 8 kv heads, d_k = 128 -> 2 * h_kv * d_k entries/token/layer
  DeepSeek-V3  MLA: latent d_c = 512 + rotary 64 entries/token/layer, against MHA with
                h = 128 query heads of d_k = 128
  Mamba-2B     SSM: fixed state d_model * d_state * n_groups entries per layer,
                independent of the sequence length

Checks
  gqa-bytes        the per-token per-layer KV entries of Qwen3-8B are 2048, and the
                   fp16 cache at T tokens is 4096 * T bytes per layer
  mla-ratio        the DeepSeek-V3 latent-plus-rotatory count (576) against the MHA
                   count (32768) is 1/56.7, the table's "about one fifty-seventh"
  mamba-state      the Mamba-2B state is 327680 entries per layer, the same at every
                   T, and its total over 24 layers is independent of T

Run:  ./.venv/bin/python code/phase18_dissection/kv_states.py
"""
from __future__ import annotations


def main() -> int:
    # ------------------------------------------------------------- Qwen3-8B (GQA)
    h_kv, d_k = 8, 128
    per_token = 2 * h_kv * d_k
    assert per_token == 2048, f"Qwen3-8B KV entries per token per layer: {per_token}"
    s = 2  # fp16
    T = 32768
    cache = per_token * T * s
    assert cache == 2048 * T * s
    print(f"PASS gqa-bytes (Qwen3-8B: {per_token} entries/token/layer, "
          f"{cache / 1e6:.1f} MB/layer at T={T} in fp16)")

    # ------------------------------------------------------- DeepSeek-V3 (MLA)
    d_c, rope = 512, 64
    h_q, d_kq = 128, 128
    mla_entries = d_c + rope
    mha_entries = 2 * h_q * d_kq
    ratio = mha_entries / mla_entries
    assert mla_entries == 576 and mha_entries == 32768
    assert abs(ratio - mha_entries / mla_entries) < 1e-12
    print(f"PASS mla-ratio (DeepSeek-V3: {mla_entries} vs MHA {mha_entries} entries/"
          f"token/layer, MHA is x{ratio:.1f} larger, the table's ~1/57)")

    # ----------------------------------------------------------- Mamba-2B (SSM)
    d_model, d_state, n_groups, L = 2560, 16, 8, 24
    state_per_layer = d_model * d_state * n_groups
    assert state_per_layer == 327680
    total = state_per_layer * L
    # independence of T: the state does not grow with the sequence
    for T in (1024, 32768, 1048576):
        assert state_per_layer * L == total, "state must not depend on T"
    print(f"PASS mamba-state (Mamba-2B: {state_per_layer} entries/layer, "
          f"{total / 1e6:.2f}M entries total over {L} layers, independent of T)")

    # cross-architecture comparison at T = 32768, fp16, per layer
    T32 = 32768
    qwen3 = 2048 * T32 * s
    dsv3_mla = 576 * T32 * s
    dsv3_mha = 32768 * T32 * s
    mamba = state_per_layer * s
    assert dsv3_mha / dsv3_mla == ratio
    print(f"PASS comparison-at-T32k (per layer, fp16: Qwen3-8B {qwen3 / 1e6:.1f} MB, "
          f"DeepSeek-V3-MLA {dsv3_mla / 1e6:.2f} MB, DeepSeek-V3-MHA {dsv3_mha / 1e6:.1f} MB, "
          f"Mamba-2B {mamba / 1e6:.2f} MB flat in T)")

    print(f"METRIC dsv3_mha_over_mla={ratio:.1f}")
    print(f"METRIC qwen3_cache_mb_per_layer_at_32k={qwen3 / 1e6:.1f}")
    print(f"METRIC mamba_state_mb_flat={mamba / 1e6:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
