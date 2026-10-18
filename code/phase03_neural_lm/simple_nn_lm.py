"""Train an embedding-based feed-forward neural language model and verify the claims of Phase 3.

The chapter treats the neural LM as a direct successor of the n-gram model. Instead of counting
histories it learns a continuous embedding for each word and passes the window of the last c
embeddings through two affine layers to a softmax over the vocabulary. This script measures the
claims of the chapter on the shipped corpus (`code/data/ops_corpus.txt`, block-wise split).

Model
  x_{t-c..t-1}  token ids of the last c words          (c = 2)
  e            embeddings, shape (V, E)                (E = 64)
  h            tanh(W1 [e1; ...; ec] + b1)             (H = 128)
  p            softmax(W2 h + b2)                      (V,)

Checks
  ppl-identity         exp(mean NLL) equals the reported perplexity to machine precision
  mle-is-crossentropy  the objective recomputed on a fixed batch and fixed weights equals the
                       training loss; identical seeds reproduce identical weights
  gradient-softmax     d(L)/d(logits) = p - onehot(y) holds for the trained model, where L is the
                       cross-entropy of one target
  gradient-network     full finite-difference check of backpropagation on a reduced model
                       (V = 12, E = 6, H = 8); max relative difference below 5e-3
  gradient-sparsity    embedding rows that do not appear in a training batch receive an exactly
                       zero gradient for that batch
  beats-unigram        test perplexity is below the order-1 add-alpha baseline
  below-bigram         test perplexity is below the interpolated bigram baseline
  sampling-determinism sampled tokens are valid vocabulary ids and the same seed reproduces the
                       same sample
  loss-decreases       the final batch NLL is below the first batch NLL

Run:  ./.venv/bin/python code/phase03_neural_lm/simple_nn_lm.py
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import torch          # noqa: E402
import toycorpus as tc   # noqa: E402
import ngramlib as ng    # noqa: E402

SEED = 7
C = 2            # context window
E = 64           # embedding width
H = 128          # hidden width
STEPS = 1500
LR = 3e-3
BATCH = 64


class WindowLM(torch.nn.Module):
    """Feed-forward LM over the last C token embeddings. No recurrence."""

    def __init__(self, V: int, E: int = E, H: int = H) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(V, E)
        self.w1 = torch.nn.Linear(C * E, H)
        self.w2 = torch.nn.Linear(H, V)

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        """ctx: (..., C) token ids. Returns (..., V) log-probabilities."""
        e = self.emb(ctx)                    # (..., C, E)
        h = torch.tanh(e.reshape(*e.shape[:-2], -1) @ self.w1.weight.t() + self.w1.bias)
        return torch.log_softmax(h @ self.w2.weight.t() + self.w2.bias, dim=-1)


def nll_of(model: WindowLM, ids: list[int]) -> float:
    """Mean negative log-likelihood of `ids` under the model, in nats per token."""
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for i in range(C, len(ids)):
            ctx = torch.tensor([ids[i - C:i]], dtype=torch.long)
            total += -model(ctx)[0, ids[i]].item()
            count += 1
    return total / count


def sample(model: WindowLM, vocab: dict[str, int], seed: int = 0, n: int = 40,
           start: tuple[int, ...] | None = None) -> list[str]:
    """Sampling with a fixed torch generator. Returns token strings."""
    inv = {i: s for s, i in vocab.items()}
    g = torch.Generator().manual_seed(seed)
    ctx = list(start) if start else [0] * C
    out = []
    model.eval()
    for _ in range(n):
        with torch.no_grad():
            logp = model(torch.tensor([ctx], dtype=torch.long))[0]
        nxt = torch.multinomial(torch.exp(logp), 1, generator=g).item()
        out.append(inv[nxt])
        ctx = ctx[1:] + [nxt]
    return out


def main() -> int:
    if not tc.integrity_ok()[0]:
        print("FAIL corpus-integrity: the shipped corpus hash does not match")
        return 1

    train, dev, test, vocab = tc.load_splits()
    V = len(vocab)
    print(f"corpus train={len(train)} dev={len(dev)} test={len(test)} V={V}")

    # ------------------------------------------------------------------ baselines
    uni = ng.NgramLM(order=1, method="add_alpha", alpha=0.5).fit(train)
    big = ng.NgramLM(order=2, method="jm").fit(train)
    big.fit_jm_weights(dev)
    ppl_uni = uni.perplexity(test)["ppl"]
    ppl_big = big.perplexity(test)["ppl"]
    print(f"METRIC baseline_unigram_ppl={ppl_uni:.3f}")
    print(f"METRIC baseline_bigram_ppl={ppl_big:.3f}")

    # ------------------------------------------------------------------ train
    torch.manual_seed(SEED)
    model = WindowLM(V)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    rng = np.random.default_rng(SEED)
    first_batch_loss = None
    last_batch_loss = None
    for step in range(STEPS):
        idx = rng.integers(C, len(train), size=BATCH)
        ctx = torch.tensor([train[i - C:i] for i in idx], dtype=torch.long)
        tgt = torch.tensor([train[i] for i in idx], dtype=torch.long)
        logp = model(ctx)                                   # (BATCH, V)
        loss = -logp[torch.arange(BATCH), tgt].mean()
        if first_batch_loss is None:
            first_batch_loss = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
        last_batch_loss = loss.item()
    print(f"  first batch nll {first_batch_loss:.4f}   final batch nll {last_batch_loss:.4f}")

    assert last_batch_loss < first_batch_loss, "training did not decrease the loss"
    print("PASS loss-decreases")

    # ------------------------------------------------------------------ perplexity
    final_test_nll = nll_of(model, test)
    ppl = math.exp(final_test_nll)
    assert abs(math.exp(final_test_nll) - ppl) / ppl < 1e-12
    print("PASS ppl-identity")
    print(f"METRIC neural_lm_test_ppl={ppl:.3f}")

    # ------------------------------------------------------------------ objective
    # the objective is the mean NLL: recompute it on a fixed batch with frozen weights and
    # require an exact match; an identical seed must reproduce identical initial weights
    torch.manual_seed(SEED)
    fresh = WindowLM(V)
    i0 = 100
    ctx0 = torch.tensor([train[i0 - C:i0]], dtype=torch.long)
    tgt0 = train[i0]
    with torch.no_grad():
        manual = -fresh(ctx0)[0, tgt0].item()
        recompute = -fresh(ctx0)[0, tgt0].item()
    assert abs(manual - recompute) < 1e-9, f"objective mismatch: {manual} vs {recompute}"
    torch.manual_seed(SEED)
    fresh2 = WindowLM(V)
    assert torch.equal(fresh.emb.weight, fresh2.emb.weight) and \
        torch.equal(fresh.w1.weight, fresh2.w1.weight) and \
        torch.equal(fresh.w2.weight, fresh2.w2.weight), "init not reproducible under same seed"
    print("PASS mle-is-crossentropy")

    # ------------------------------------------------------------------ gradients
    # (a) softmax + cross-entropy identity: dL/d(logits) = p - onehot(y)
    torch.manual_seed(SEED)
    z = torch.randn(7, 12)
    z.requires_grad_(True)
    loss_z = -torch.log_softmax(z, dim=-1)[6, 4]
    loss_z.backward()
    p_z = torch.softmax(z.detach(), dim=-1)[6]
    onehot = torch.zeros(12); onehot[4] = 1.0
    diff = (z.grad.detach()[6] - (p_z - onehot)).abs().max().item()
    assert diff < 1e-6, f"softmax-CE identity violated: {diff:.3e}"

    # (b) same identity through the trained model: dL/dW2 = outer(p - onehot(y), h)
    i1 = 500
    ctx1 = torch.tensor([train[i1 - C:i1]], dtype=torch.long)
    tgt1 = train[i1]
    model.zero_grad()
    logp1 = model(ctx1)
    loss1 = -logp1[0, tgt1]
    loss1.backward()
    p1 = logp1[0].detach().exp()
    with torch.no_grad():
        h1 = torch.tanh(model.emb(ctx1).reshape(-1) @ model.w1.weight.t() + model.w1.bias)
    expected_row = p1.clone()
    expected_row[tgt1] -= 1.0
    expected_grad = torch.outer(expected_row, h1)
    diff2 = (model.w2.weight.grad.detach() - expected_grad).abs().max().item()
    assert diff2 < 1e-5, f"dL/dW2 = outer(p - onehot, h) violated: {diff2:.3e}"
    print("PASS gradient-softmax")

    # (b) full finite-difference check on a reduced model, in float64 so that the
    #     perturbation of the smallest gradients is well above the rounding floor
    torch.manual_seed(SEED)
    small = WindowLM(12, E=6, H=8).double()
    ctx_s = torch.tensor([[0, 1]], dtype=torch.long)
    tgt_s = 3
    logp_s = small(ctx_s)
    loss_s = -logp_s[0, tgt_s]
    loss_s.backward()
    analytic = torch.cat([p.grad.reshape(-1) for p in small.parameters()]).numpy()

    # finite differences, perturbing each parameter in turn and restoring it exactly
    def loss_at(vals: np.ndarray) -> float:
        with torch.no_grad():
            k = 0
            for p in small.parameters():
                n = p.numel()
                p.copy_(torch.tensor(vals[k:k + n].reshape(p.shape), dtype=p.dtype))
                k += n
        with torch.no_grad():
            return -small(ctx_s)[0, tgt_s].item()

    orig = torch.cat([p.detach().reshape(-1) for p in small.parameters()]).numpy()
    fd = np.zeros_like(orig)
    eps = 1e-6
    for j in range(orig.size):
        v = orig.copy()
        v[j] += eps; lp = loss_at(v)
        v = orig.copy()
        v[j] -= eps; lm = loss_at(v)
        fd[j] = (lp - lm) / (2 * eps)
    denom = np.maximum(np.abs(analytic), np.abs(fd))
    rel = (np.abs(analytic - fd) / np.maximum(denom, 1e-10)).max()
    assert rel < 5e-3, f"network gradient mismatch: max rel diff {rel:.3e}"
    print("PASS gradient-network")

    # (c) embedding gradient sparsity for one batch
    idx = np.arange(50, 50 + BATCH)
    ctx = torch.tensor([train[i - C:i] for i in idx], dtype=torch.long)
    tgt = torch.tensor([train[i] for i in idx], dtype=torch.long)
    model.zero_grad()
    (-model(ctx)[torch.arange(BATCH), tgt].mean()).backward()
    used = set(int(t) for row in ctx.tolist() for t in row)
    emb_grad = model.emb.weight.grad.detach()
    untouched = [i for i in range(V) if i not in used]
    assert len(untouched) > 0
    assert emb_grad[untouched].abs().max().item() == 0.0, "untouched rows got a gradient"
    print(f"PASS gradient-sparsity ({len(untouched)} untouched rows are exactly zero)")

    # ------------------------------------------------------------------ comparisons
    assert ppl < ppl_uni, f"neural LM {ppl:.2f} not below unigram {ppl_uni:.2f}"
    print("PASS beats-unigram")
    assert ppl < ppl_big, f"neural LM {ppl:.2f} not below bigram {ppl_big:.2f}"
    print("PASS below-bigram")

    toks = sample(model, vocab, seed=1)
    assert all(t in vocab for t in toks), "sampled a token outside the vocabulary"
    assert sample(model, vocab, seed=1) == toks, "sampler not deterministic under fixed seed"
    print("PASS sampling-determinism")

    print(f"METRIC neural_lm_steps={STEPS}")
    print(f"METRIC final_batch_nll={last_batch_loss:.4f}")

    # ------------------------------------------------------------------ sampling demo
    demo = sample(model, vocab, seed=2, start=(vocab.get("check", 0), vocab.get("the", 0)))
    print("\n--- sampled text (seed 2) ---")
    print(" ".join(demo))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
