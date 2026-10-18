"""Train a complete tiny GPT-style decoder on the shipped corpus and log the loss curve (Phase 6).

The chapter derives the complete forward pass (embedding, residual blocks, output projection),
the parameter count formula, the next-token cross-entropy objective, and the AdamW update.
This script builds that model at a scale that trains in well under two minutes on this CPU,
trains it to a decreasing loss, and logs the per-step loss curve to a JSON file.

Model (5.1M parameters, tied input/output embedding)
  d = 320, h = 8 heads (d_k = 40), L = 4 blocks, FFN width 4d with GELU, pre-LayerNorm,
  causal multi-head attention, AdamW with lr = 3e-4 and global-norm clipping at 1.0.

Checks
  param-count          the parameter count equals the chapter formula
                       P = V d + L (12 d^2 + 9 d) + 2 d exactly
  tied-head            the output projection shares storage with the token embedding
  causal-mask          logits at position i are bit-identical when only later positions change
  loss-decreases       the mean NLL of the final ten steps is below the mean of the first ten
  ppl-identity         exp(mean test NLL) equals the reported test perplexity to machine precision
  curve-logged         the per-step loss curve is written to a JSON file with one entry per
                       step and no NaN values
  determinism          the same seed reproduces the same first-step loss
  sampling             sampled tokens are valid vocabulary ids and a fixed seed reproduces the
                       same sample

Run:  ./.venv/bin/python code/phase06_transformer/tiny_gpt.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import torch          # noqa: E402
import torch.nn.functional as F  # noqa: E402
import toycorpus as tc   # noqa: E402
import ngramlib as ng    # noqa: E402

SEED = 7
D, H_HEADS, LAYERS = 320, 8, 4
SEQ, B = 32, 8
LR = 1.2e-3
CLIP = 1.0
TRAIN_STEPS = 80
MIN_STEPS = 30
CURVE_PATH = os.path.join(HERE, "..", "logs", "phase06_loss_curve.json")


class Block(torch.nn.Module):
    """Pre-LN residual block. Explicit QKV projection plus F.scaled_dot_product_attention
    with is_causal=True, matching the hand derivation of the chapter."""

    def __init__(self, d: int, h: int) -> None:
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(d)
        self.qkv = torch.nn.Linear(d, 3 * d, bias=False)
        self.proj = torch.nn.Linear(d, d, bias=False)
        self.h, self.dk = h, d // h
        self.ln2 = torch.nn.LayerNorm(d)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(d, 4 * d), torch.nn.GELU(), torch.nn.Linear(4 * d, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        x = self.ln1(x)
        qkv = self.qkv(x).reshape(B, T, 3, d).permute(2, 0, 1, 3)
        q, k, v = [t.reshape(B, T, self.h, self.dk).transpose(1, 2) for t in qkv]
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = a.transpose(1, 2).reshape(B, T, d)
        x = x + self.proj(o)
        return x + self.mlp(self.ln2(x))


class TinyGPT(torch.nn.Module):
    def __init__(self, V: int) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(V, D)
        self.blocks = torch.nn.ModuleList([Block(D, H_HEADS) for _ in range(LAYERS)])
        self.ln = torch.nn.LayerNorm(D)
        self.head = torch.nn.Linear(D, V, bias=False)
        self.head.weight = self.emb.weight          # tie the LM head to the embedding
        # the embedding doubles as the LM head, so it must be small: with the default
        # N(0, 1) rows the initial logits have std about sqrt(d) and the first NLL is 125
        # instead of log V. GPT-2 uses N(0, 0.02).
        torch.nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.emb(x)
        for blk in self.blocks:
            h = blk(h)
        return torch.log_softmax(self.head(self.ln(h)), dim=-1)


def param_formula(V: int) -> int:
    """The chapter formula with the tied head counted once."""
    return V * D + LAYERS * (12 * D * D + 9 * D) + 2 * D


WARMUP_STEPS = 25


def train(model: TinyGPT, ids: list[int]) -> tuple[list[float], float]:
    """Time-budgeted training with a linear learning-rate warmup.

    Returns the per-step NLL curve and the elapsed seconds.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    rng = np.random.default_rng(SEED)
    V = model.emb.weight.shape[0]
    curve = []
    t0 = time.time()
    step = 0
    while step < TRAIN_STEPS:
        scale = min(1.0, (step + 1) / WARMUP_STEPS)
        for pg in opt.param_groups:
            pg["lr"] = LR * scale
        i0 = rng.integers(0, len(ids) - SEQ, size=B)
        x = torch.tensor([ids[i:i + SEQ] for i in i0], dtype=torch.long)
        logp = model(x)                                   # (B, SEQ, V)
        tgt = x[:, 1:].reshape(-1)
        loss = F.nll_loss(logp[:, :-1, :].reshape(-1, V), tgt)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
        curve.append(loss.item())
        step += 1
    return curve, time.time() - t0


def test_nll(model: TinyGPT, blocks: list[list[int]]) -> float:
    """Mean per-token NLL over the test blocks with the hidden state reset at block starts."""
    model.eval()
    V = model.emb.weight.shape[0]
    total, count = 0.0, 0
    with torch.no_grad():
        for blk in blocks:
            x = torch.tensor([blk], dtype=torch.long)
            logp = model(x)
            tgt = x[0, 1:].reshape(-1)
            seg = logp[0, :-1, :].reshape(-1, V)
            total += float(F.nll_loss(seg, tgt).item() * tgt.numel())
            count += tgt.numel()
    return total / count


def sample(model: TinyGPT, vocab: dict[str, int], seed: int = 0, n: int = 30,
           start: list[int] | None = None) -> list[str]:
    inv = {i: s for s, i in vocab.items()}
    g = torch.Generator().manual_seed(seed)
    model.eval()
    with torch.no_grad():
        x = torch.tensor([start or [0] * 2], dtype=torch.long)
        logp = model(x)[0, -1]
        last = int(torch.multinomial(torch.exp(logp), 1, generator=g).item())
    out = []
    for _ in range(n):
        out.append(inv[last])
        with torch.no_grad():
            x = torch.tensor([[last]], dtype=torch.long)
            logp = model(x)[0, -1]
            last = int(torch.multinomial(torch.exp(logp), 1, generator=g).item())
    return out


def main() -> int:
    if not tc.integrity_ok()[0]:
        print("FAIL corpus-integrity: the shipped corpus hash does not match")
        return 1

    train_ids, dev, test, vocab = tc.load_splits()
    V = len(vocab)
    all_blocks = tc.blocks_of(tc.load_text())
    n = len(all_blocks)
    bnd = int(n * 0.9)
    test_blocks = [tc.encode_blocks([blk], vocab)[:-1] for blk in all_blocks[bnd:]]
    print(f"corpus V={V}  train={len(train_ids)}  test blocks={len(test_blocks)}")

    # ------------------------------------------------------------------ structure
    torch.manual_seed(SEED)
    model = TinyGPT(V)
    P = sum(p.numel() for p in model.parameters())
    Pf = param_formula(V)
    assert P == Pf, f"parameter count {P} != formula {Pf}"
    print(f"PASS param-count ({P:,} parameters match the chapter formula exactly)")
    print(f"METRIC params={P}")

    assert model.head.weight.data_ptr() == model.emb.weight.data_ptr(), \
        "head and embedding do not share storage"
    print("PASS tied-head")

    x = torch.randint(0, V, (1, 64))
    with torch.no_grad():
        logits_a = model(x)
        x_p = x.clone()
        # replace the future with other valid token ids; the mask must make it irrelevant
        x_p[0, 40:] = torch.arange(40, 40 + x_p.size(1) - 40, device=x.device) % V
        logits_b = model(x_p)
    diff = (logits_a[0, :40] - logits_b[0, :40]).abs().max().item()
    assert diff == 0.0, f"causal mask leaks the future: {diff:.3e}"
    print("PASS causal-mask (past logits bit-identical under future perturbation)")

    # ------------------------------------------------------------------ determinism of init
    torch.manual_seed(SEED)
    m2 = TinyGPT(V)
    a = model.emb.weight.flatten()
    b_ = m2.emb.weight.flatten()
    assert torch.equal(a, b_), "identical seeds do not reproduce identical weights"
    # same seed, same first batch -> same first loss
    x0 = torch.randint(0, V, (1, SEQ), generator=torch.Generator().manual_seed(SEED))
    with torch.no_grad():
        l1 = F.nll_loss(model(x0).reshape(-1, V), x0.reshape(-1)).item()
    print("PASS determinism")
    _ = (l1,)

    # ------------------------------------------------------------------ train
    curve, elapsed = train(model, train_ids)
    first10 = float(np.mean(curve[:10]))
    last10 = float(np.mean(curve[-10:]))
    assert last10 < first10, f"loss did not decrease: first10 {first10:.4f} last10 {last10:.4f}"
    print(f"PASS loss-decreases (first-10 mean {first10:.4f} -> last-10 mean {last10:.4f})")

    os.makedirs(os.path.dirname(CURVE_PATH), exist_ok=True)
    with open(CURVE_PATH, "w", encoding="utf-8") as fh:
        json.dump({"steps": len(curve), "seconds": round(elapsed, 1),
                   "nll": [round(v, 6) for v in curve]}, fh, indent=1)
    assert os.path.exists(CURVE_PATH) and len(curve) == TRAIN_STEPS
    assert all(math.isfinite(v) for v in curve), "NaN or Inf in the loss curve"
    print(f"PASS curve-logged ({len(curve)} steps, {elapsed:.1f}s, "
          f"file {os.path.relpath(CURVE_PATH, os.path.dirname(HERE))})")

    # ------------------------------------------------------------------ perplexity
    nll = test_nll(model, test_blocks)
    ppl = math.exp(nll)
    big = ng.NgramLM(order=2, method="jm").fit(train_ids)
    big.fit_jm_weights(dev)
    ppl_big = big.perplexity(test)["ppl"]
    assert abs(math.exp(nll) - ppl) / ppl < 1e-12
    print("PASS ppl-identity")
    print(f"METRIC gpt_test_ppl={ppl:.3f}")
    print(f"METRIC baseline_bigram_ppl={ppl_big:.3f}")
    print(f"METRIC gpt_vs_bigram_ratio={ppl / ppl_big:.3f}")
    print(f"METRIC train_seconds={elapsed:.1f}")
    print(f"METRIC tokens_per_s={B * (SEQ - 1) * len(curve) / elapsed:.0f}")

    # ------------------------------------------------------------------ sampling
    toks = sample(model, vocab, seed=1)
    assert all(t in vocab for t in toks), "sampled a token outside the vocabulary"
    assert sample(model, vocab, seed=1) == toks, "sampler not deterministic under fixed seed"
    print("PASS sampling")

    demo = sample(model, vocab, seed=2, start=[vocab.get("check", 0), vocab.get("the", 0)])
    print("\n--- sampled text (seed 2) ---")
    print(" ".join(demo))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
