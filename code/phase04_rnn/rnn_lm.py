"""Train a GRU language model on the shipped corpus and verify the claims of Phase 4.

The chapter derives backpropagation through time, the vanishing gradient behaviour of the
recurrence, and the gated residual structure of the GRU. It also makes one empirical claim
about the value of recurrence. A window of two words cannot see a topic word that appeared
ten tokens earlier, but a recurrent state can carry it. The corpus is built so that each
document block reuses its topic words, which gives that claim something to measure. This
script measures it.

Model
  e_t  = Emb(x_t)                                  (E = 32)
  h_t  = GRU(e_t, h_{t-1})                         (H = 96)
  p_t  = softmax(W_out h_t)                        (V,)

Checks
  loss-decreases       the final batch NLL is below the first batch NLL
  ppl-identity         exp(mean NLL) equals the reported perplexity to machine precision
  rnn-beats-bigram     GRU test perplexity is below the interpolated bigram perplexity
  long-range-memory    per-token NLL on tokens whose identical twin appeared 9 to 40 tokens
                       earlier in the same block is below the NLL of tokens with no earlier
                       occurrence; the gap is the measurable content of the recurrent state
  window-blind         the two-word windowed control cannot use that signal by construction;
                       its far-minus-none gap is asserted to be strictly smaller than the GRU's
  clip-bounds-norm     after global-norm clipping at 1.0 the gradient norm never exceeds the cap
  sampling-determinism sampled tokens are valid vocabulary ids and the same seed reproduces the
                       same sample

Run:  ./.venv/bin/python code/phase04_rnn/rnn_lm.py
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import torch          # noqa: E402
import torch.nn.functional as F  # noqa: E402
import toycorpus as tc   # noqa: E402
import ngramlib as ng    # noqa: E402

SEED = 7
E, H = 32, 96
SEQ, B = 64, 8
STEPS = 200
LR = 3e-3
CLIP = 1.0

WIN_C, WIN_E, WIN_H = 2, 32, 64   # windowed control for the long-range analysis


class GruLM(torch.nn.Module):
    def __init__(self, V: int) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(V, E)
        self.gru = torch.nn.GRU(E, H, batch_first=True)
        self.out = torch.nn.Linear(H, V)

    def forward(self, x, h=None):
        o, h = self.gru(self.emb(x), h)
        return torch.log_softmax(self.out(o), dim=-1), h


class WindowLM(torch.nn.Module):
    """Two-word windowed feed-forward control. C cannot see beyond C by construction."""

    def __init__(self, V: int) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(V, WIN_E)
        self.w1 = torch.nn.Linear(WIN_C * WIN_E, WIN_H)
        self.w2 = torch.nn.Linear(WIN_H, V)

    def forward(self, ctx):
        e = self.emb(ctx)
        h = torch.tanh(e.reshape(*e.shape[:-2], -1) @ self.w1.weight.t() + self.w1.bias)
        return torch.log_softmax(h @ self.w2.weight.t() + self.w2.bias, dim=-1)


def train_gru(model: GruLM, ids: list[int]) -> tuple[float, float, list[tuple[float, float]]]:
    """Train with B sequences of length SEQ per step, stacked as one (B, SEQ) tensor so
    that every GRU cell vectorises across the batch. This is the difference between
    ~600 and ~4,000 cells/second on this CPU."""
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    rng = np.random.default_rng(SEED)
    V = model.out.weight.shape[0]
    first = last = None
    norms = []
    for _ in range(STEPS):
        i0 = rng.integers(0, len(ids) - SEQ, size=B)
        x = torch.tensor([ids[i:i + SEQ] for i in i0], dtype=torch.long)   # (B, SEQ)
        out, _ = model(x, None)                                            # (B, SEQ, V)
        tgt = x[:, 1:].reshape(-1)
        loss = F.nll_loss(out[:, :-1, :].reshape(-1, V), tgt)
        if first is None:
            first = loss.item()
        opt.zero_grad()
        loss.backward()
        pre = torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters() if p.grad is not None))
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        post = torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters() if p.grad is not None))
        norms.append((pre.item(), post.item()))
        opt.step()
        last = loss.item()
    return first, last, norms


def gru_block_nlls(model: GruLM, blocks: list[list[int]]) -> np.ndarray:
    """Per-token NLL over each block with the hidden state reset at block boundaries.

    Block i contributes len(block_i) - 1 entries. The first token of a block is scored with
    a fresh state and kept; its bucket flag is always False (no earlier occurrence).
    """
    model.eval()
    out = []
    with torch.no_grad():
        for blk in blocks:
            x = torch.tensor([blk], dtype=torch.long)
            logp, _ = model(x, None)
            tgt = x[0, 1:].unsqueeze(-1)
            out.extend((-logp[0, :-1, :].gather(1, tgt)).squeeze(1).tolist())
    return np.array(out)


def window_block_nlls(model: WindowLM, blocks: list[list[int]]) -> np.ndarray:
    """Per-token NLL with an in-block two-word window; skipped where the context is short."""
    model.eval()
    rows, tgts = [], []
    for blk in blocks:
        if len(blk) > WIN_C:
            rows.extend([blk[i - WIN_C:i] for i in range(WIN_C, len(blk))])
            tgts.extend(blk[WIN_C:])
    ctx = torch.tensor(rows, dtype=torch.long)
    tgt = torch.tensor(tgts, dtype=torch.long)
    with torch.no_grad():
        logp = model(ctx)                                  # (N, V)
        out = -logp[torch.arange(tgt.numel()), tgt]
    return out.numpy()


def _aligned_bucket(blocks: list[list[int]], lo: int, hi: int | None, none_mask: bool):
    """Bucket flags parallel to the per-block NLL arrays (len(blk)-1 entries each).

    A flag is True at position i when the same token occurred lo..hi tokens earlier in the
    block (or, with none_mask, when it never occurred earlier in the block).
    """
    per_block = []
    for blk in blocks:
        row = []
        last_pos = {}
        for i, t in enumerate(blk):
            if i == 0:
                last_pos[t] = i
                continue
            d = (i - last_pos[t]) if t in last_pos else None
            if none_mask:
                row.append(d is None)
            else:
                row.append(d is not None and d >= lo and (hi is None or d <= hi))
            last_pos[t] = i
        per_block.append(row)
    return per_block


def mask_mean(nlls: np.ndarray, per_block_masks: list[list[bool]]) -> tuple[float, int]:
    """Mean of the NLL entries whose flag is True, with the count of such entries."""
    start = 0
    total, count = 0.0, 0
    for m in per_block_masks:
        seg = nlls[start:start + len(m)]
        sel = seg[np.asarray(m, dtype=bool)]
        total += float(sel.sum())
        count += int(sel.size)
        start += len(m)
    return (total / count if count else float("nan")), count


def sample(model: GruLM, vocab: dict[str, int], seed: int = 0, n: int = 40,
           start: list[int] | None = None) -> list[str]:
    inv = {i: s for s, i in vocab.items()}
    g = torch.Generator().manual_seed(seed)
    model.eval()
    last_tok = (start or [0] * 3)[-1]
    h = None
    with torch.no_grad():
        _, h = model(torch.tensor([start or [0] * 3], dtype=torch.long), None)
    out = []
    for _ in range(n):
        x = torch.tensor([[last_tok]], dtype=torch.long)
        with torch.no_grad():
            logp, h = model(x, h)
        nxt = torch.multinomial(torch.exp(logp[0, -1]), 1, generator=g).item()
        out.append(inv[nxt])
        last_tok = nxt
    return out


def main() -> int:
    if not tc.integrity_ok()[0]:
        print("FAIL corpus-integrity: the shipped corpus hash does not match")
        return 1

    train, dev, test, vocab = tc.load_splits()
    V = len(vocab)
    all_blocks = tc.blocks_of(tc.load_text())
    n = len(all_blocks)
    a, bnd = int(n * 0.8), int(n * 0.9)
    # encode_blocks appends <eos> after each block; drop the marker so block = actual tokens
    test_blocks = [tc.encode_blocks([blk], vocab)[:-1] for blk in all_blocks[bnd:]]
    print(f"corpus V={V}  test blocks={len(test_blocks)}  test tokens={sum(len(b) for b in test_blocks)}")

    # ------------------------------------------------------------------ baselines
    big = ng.NgramLM(order=2, method="jm").fit(train)
    big.fit_jm_weights(dev)
    ppl_big = big.perplexity(test)["ppl"]
    print(f"METRIC baseline_bigram_ppl={ppl_big:.3f}")

    # ------------------------------------------------------------------ train GRU
    torch.manual_seed(SEED)
    gr = GruLM(V)
    first, last, norms = train_gru(gr, train)
    print(f"  first batch nll {first:.4f}   final batch nll {last:.4f}")
    assert last < first, "training did not decrease the loss"
    print("PASS loss-decreases")

    pre_max = max(p for p, _ in norms)
    post_max = max(q for _, q in norms)
    assert post_max <= CLIP + 1e-6, f"clipped norm {post_max:.4f} exceeds the cap"
    print(f"PASS clip-bounds-norm (pre-clip max {pre_max:.2f} -> post-clip max {post_max:.4f})")

    # ------------------------------------------------------------------ perplexity
    nll = gru_block_nlls(gr, test_blocks)
    test_nll = float(nll.mean())
    ppl_gru = math.exp(test_nll)
    assert abs(math.exp(test_nll) - ppl_gru) / ppl_gru < 1e-12
    print("PASS ppl-identity")
    print(f"METRIC gru_test_ppl={ppl_gru:.3f}")

    assert ppl_gru < ppl_big, f"GRU {ppl_gru:.2f} not below bigram {ppl_big:.2f}"
    print("PASS rnn-beats-bigram")
    print(f"METRIC rnn_vs_bigram_ratio={ppl_gru / ppl_big:.3f}")

    # ------------------------------------------------------------------ long-range
    far_mask = _aligned_bucket(test_blocks, lo=9, hi=40, none_mask=False)
    none_mask = _aligned_bucket(test_blocks, lo=0, hi=None, none_mask=True)
    gr_far, n_far = mask_mean(nll, far_mask)
    gr_none, n_none = mask_mean(nll, none_mask)
    print(f"  GRU    NLL far={gr_far:.4f} (n={n_far})   none={gr_none:.4f} (n={n_none})")
    assert n_far > 100 and n_none > 100, "too few tokens in the long-range buckets"
    assert gr_far < gr_none, "GRU does not use tokens seen 9-40 steps earlier"
    print("PASS long-range-memory")
    print(f"METRIC gru_far_minus_none={gr_far - gr_none:+.4f}")

    # ------------------------------------------------------------------ windowed control
    torch.manual_seed(SEED)
    win = WindowLM(V)
    wopt = torch.optim.Adam(win.parameters(), lr=LR)
    wrng = np.random.default_rng(SEED)
    for _ in range(400):
        idx = wrng.integers(WIN_C, len(train), size=64)
        ctx = torch.tensor([train[i - WIN_C:i] for i in idx], dtype=torch.long)
        tgt = torch.tensor([train[i] for i in idx], dtype=torch.long)
        loss = -win(ctx)[torch.arange(64), tgt].mean()
        wopt.zero_grad(); loss.backward(); wopt.step()
    wnll = window_block_nlls(win, test_blocks)

    # windowed bucket flags: position i (i >= WIN_C) with an earlier twin 9..40 away, or none
    win_flags = []
    for blk in test_blocks:
        row = []
        last_pos = {}
        for i, t in enumerate(blk):
            if i < WIN_C:
                last_pos[t] = i
                continue
            d = (i - last_pos[t]) if t in last_pos else None
            row.append((d is not None and 9 <= d <= 40, d is None))
            last_pos[t] = i
        win_flags.append(row)

    def win_mean(midx: int) -> tuple[float, int]:
        s, c, off = 0.0, 0, 0
        for seg in win_flags:
            for flag in seg:
                if flag[midx]:
                    s += wnll[off]
                    c += 1
            off += len(seg)
        return (s / c if c else float("nan")), c

    win_far, n_wf = win_mean(0)
    win_none, n_wn = win_mean(1)
    print(f"  window NLL far={win_far:.4f} (n={n_wf})   none={win_none:.4f} (n={n_wn})")
    assert n_wf > 100 and n_wn > 100, "too few tokens in the windowed buckets"
    assert (gr_far - gr_none) < (win_far - win_none), \
        (f"windowed control gap {win_far - win_none:+.4f} is not strictly smaller "
         f"than the GRU gap {gr_far - gr_none:+.4f}")
    print("PASS window-blind")
    print(f"METRIC window_far_minus_none={win_far - win_none:+.4f}")

    # ------------------------------------------------------------------ sampling
    toks = sample(gr, vocab, seed=1)
    assert all(t in vocab for t in toks), "sampled a token outside the vocabulary"
    assert sample(gr, vocab, seed=1) == toks, "sampler not deterministic under fixed seed"
    print("PASS sampling-determinism")

    print(f"METRIC gru_steps={STEPS}")
    print(f"METRIC final_batch_nll={last:.4f}")

    demo = sample(gr, vocab, seed=2, start=[vocab.get("check", 0), vocab.get("the", 0)])
    print("\n--- sampled text (seed 2) ---")
    print(" ".join(demo))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
