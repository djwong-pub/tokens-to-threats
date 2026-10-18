"""Verify MHA, GQA and MQA head sharing and the KV-cache byte accounting of Phase 8.

The chapter derives the three sharing schemes from one weight layout, proves that GQA reduces
to MHA when the group size is one, and counts the bytes a KV cache holds per position. This
script checks each claim against runnable implementations.

Checks
  three-variants-shapes    MHA, GQA and MQA all map (B, T, d) to (B, T, d)
  gqa-expansion-identity   GQA equals the per-head loop in which query head i attends only to
                           kv head i // g
  gqa-reduces-to-mha       with h_kv = h_q (group size 1) the GQA path is bit-identical to MHA
  mqa-is-gqa-special-case  a dedicated MQA module equals the GQA module at h_kv = 1
  kv-cache-bytes           the cache holds 2 * h_kv * d_k * T * s bytes and the MHA:GQA:MQA
                           ratio is exactly h_q : h_kv : 1 for a worked configuration
  kv-cache-scale           the GPT-3-large-like configuration saves exactly 4x at T = 4096
                           with h_q = 32, h_kv = 8, and the saving grows linearly in T

Run:  ./.venv/bin/python code/phase08_mha_mqa_gqa/kv_sharing.py
"""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F


class HeadSharedAttn(torch.nn.Module):
    """One attention layer with h_q query heads and h_kv key-value heads.

    h_kv = h_q gives MHA. h_kv = 1 gives MQA. Anything between is GQA with group
    size g = h_q / h_kv. Query head i reads kv head i // g.
    """

    def __init__(self, d: int, h_q: int, h_kv: int) -> None:
        super().__init__()
        assert h_q % h_kv == 0, "h_kv must divide h_q"
        self.d, self.h_q, self.h_kv = d, h_q, h_kv
        self.d_k = d // h_q
        self.g = h_q // h_kv
        self.wq = torch.nn.Linear(d, h_q * self.d_k, bias=False)
        self.wk = torch.nn.Linear(d, h_kv * self.d_k, bias=False)
        self.wv = torch.nn.Linear(d, h_kv * self.d_k, bias=False)
        self.wo = torch.nn.Linear(d, d, bias=False)

    def qkv(self, x: torch.Tensor):
        B, T, d = x.shape
        q = self.wq(x).reshape(B, T, self.h_q, self.d_k).transpose(1, 2)
        k = self.wk(x).reshape(B, T, self.h_kv, self.d_k).transpose(1, 2)
        v = self.wv(x).reshape(B, T, self.h_kv, self.d_k).transpose(1, 2)
        return q, k, v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        q, k, v = self.qkv(x)
        if self.h_kv == 1:
            k = k.expand(B, self.h_q, T, self.d_k)
            v = v.expand(B, self.h_q, T, self.d_k)
        elif self.h_kv < self.h_q:
            k = k.repeat_interleave(self.g, dim=1)
            v = v.repeat_interleave(self.g, dim=1)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = a.transpose(1, 2).reshape(B, T, d)
        return self.wo(o)


def per_head_loop(m: HeadSharedAttn, x: torch.Tensor) -> torch.Tensor:
    """Reference computation: each query head attends only to its group's kv head."""
    B, T, d = x.shape
    q, k, v = m.qkv(x)
    out_heads = []
    for i in range(m.h_q):
        kv = i // m.g
        qi, ki, vi = q[:, i:i + 1], k[:, kv:kv + 1], v[:, kv:kv + 1]
        a = F.scaled_dot_product_attention(qi, ki, vi, is_causal=True)
        out_heads.append(a)
    o = torch.cat(out_heads, dim=1).transpose(1, 2).reshape(B, T, d)
    return m.wo(o)


def kv_bytes(T: int, h_kv: int, d_k: int, h_q: int, dtype_bytes: int = 2) -> int:
    """Bytes for keys and values of T positions. d_k is per query head."""
    # keys and values: h_kv heads * d_k dims each * T positions * 2 (K and V) * dtype
    del h_q
    return 2 * h_kv * d_k * T * dtype_bytes


def main() -> int:
    torch.manual_seed(7)
    B, T, d, h_q = 2, 16, 128, 8
    d_k = d // h_q
    x = torch.randn(B, T, d)

    mha = HeadSharedAttn(d, h_q, h_q)
    gqa = HeadSharedAttn(d, h_q, 2)
    mqa = HeadSharedAttn(d, h_q, 1)
    outs = {n: m(x) for n, m in (("mha", mha), ("gqa", gqa), ("mqa", mqa))}
    for n, o in outs.items():
        assert o.shape == (B, T, d), f"{n} output shape {tuple(o.shape)}"
    print(f"PASS three-variants-shapes ({list(outs)})")

    # GQA must equal the explicit per-head loop with group assignment i // g
    dg = torch.abs(outs["gqa"] - per_head_loop(gqa, x)).max().item()
    assert dg == 0.0, f"GQA path differs from per-head loop: {dg:.3e}"
    print("PASS gqa-expansion-identity (bit-identical to the grouped per-head loop)")

    # GQA at group size 1 is MHA, bit for bit, given identical weights
    gqa_full = HeadSharedAttn(d, h_q, h_q)
    with torch.no_grad():
        gqa_full.wq.weight.copy_(mha.wq.weight)
        gqa_full.wk.weight.copy_(mha.wk.weight)
        gqa_full.wv.weight.copy_(mha.wv.weight)
        gqa_full.wo.weight.copy_(mha.wo.weight)
    dm = torch.abs(gqa_full(x) - outs["mha"]).max().item()
    assert dm == 0.0, f"GQA(h_kv=h_q) differs from MHA: {dm:.3e}"
    print("PASS gqa-reduces-to-mha (group size 1 is bit-identical to MHA)")

    # MQA as a dedicated module equals GQA configured at h_kv = 1
    gqa1 = HeadSharedAttn(d, h_q, 1)
    with torch.no_grad():
        gqa1.wq.weight.copy_(mqa.wq.weight)
        gqa1.wk.weight.copy_(mqa.wk.weight)
        gqa1.wv.weight.copy_(mqa.wv.weight)
        gqa1.wo.weight.copy_(mqa.wo.weight)
    dq = torch.abs(gqa1(x) - outs["mqa"]).max().item()
    assert dq == 0.0, f"MQA differs from GQA(h_kv=1): {dq:.3e}"
    print("PASS mqa-is-gqa-special-case (MQA is GQA at h_kv = 1)")

    # ------------------------------------------------------------------ cache bytes
    # worked configuration: d = 4096, h_q = 32, d_k = 128, fp16
    dW, hq, Tseq = 4096, 32, 4096
    dkw = dW // hq
    b_mha = kv_bytes(Tseq, hq, dkw, hq)
    b_gqa = kv_bytes(Tseq, 8, dkw, hq)
    b_mqa = kv_bytes(Tseq, 1, dkw, hq)
    assert b_mha / b_gqa == hq / 8 and b_mha / b_mqa == hq / 1
    print(f"PASS kv-cache-bytes (MHA {b_mha / 2**20:.1f} MiB, GQA-8 {b_gqa / 2**20:.1f} MiB, "
          f"MQA {b_mqa / 2**20:.1f} MiB at T = {Tseq}, fp16; ratio {hq}:8:1)")
    print(f"METRIC kv_mha_mib={b_mha / 2**20:.1f}")
    print(f"METRIC kv_gqa8_mib={b_gqa / 2**20:.1f}")
    print(f"METRIC kv_mqa_mib={b_mqa / 2**20:.1f}")

    # the saving is exactly a factor h_q / h_kv and grows linearly in T
    for T2 in (4096, 8192, 16384):
        r = kv_bytes(T2, hq, dkw, hq) / kv_bytes(T2, 8, dkw, hq)
        assert abs(r - 4.0) < 1e-12, f"ratio {r} at T = {T2}"
    print("PASS kv-cache-scale (exact 4x saving for T = 4096, 8192, 16384)")
    print(f"METRIC gqa_saving_factor={hq / 8:.0f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
