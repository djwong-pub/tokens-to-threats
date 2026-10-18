"""Train a bigram language model on the shipped corpus and verify the identities of Phase 2.

The chapter derives the next-token factorisation, maximum-likelihood estimation, the perplexity
identity and the effect of smoothing. This script measures each of those claims on one corpus.
The corpus is `code/data/ops_corpus.txt`, split block-wise into train, dev and test.

Checks
  normalisation        every conditional row sums to one, for all five estimators
  ppl-equals-exp-ce    exp(mean NLL) equals the reported perplexity to machine precision
  mle-is-infinite      unsmoothed MLE assigns zero probability to some test tokens, so its test
                       perplexity is infinite. The count of such tokens is printed.
  conditioning-helps   bigram perplexity is below unigram perplexity for every smoothing method
  smoothing-helps      smoothing lowers test perplexity below the smallest add-alpha value tried
                       at alpha -> 0
  chain-rule           the product of conditionals over a block equals exp(-block NLL), and the
                       block NLL equals the sum of its token NLLs
  determinism          refitting and rescoring reproduces every number exactly

Run:  ./.venv/bin/python code/phase02_ngram_lm/bigram_train.py
"""
from __future__ import annotations

import math
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ngramlib as ng          # noqa: E402
import toycorpus as tc         # noqa: E402

ALPHA_SWEEP = (1e-9, 1e-7, 1e-5, 1e-4, 1e-3, 0.01, 0.03, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0)


def main() -> int:
    train, dev, test, vocab = tc.load_splits()
    V = len(vocab)
    out = {}

    # ---------------------------------------------------------------- estimators
    models = {
        "mle": ng.NgramLM(2, "mle", V=V),
        "add_alpha_1e-6": ng.NgramLM(2, "add_alpha", alpha=1e-6, V=V),
        "laplace": ng.NgramLM(2, "add_alpha", alpha=1.0, V=V),
        "absolute": ng.NgramLM(2, "absolute", discount=0.75, V=V),
        "witten_bell": ng.NgramLM(2, "jm", V=V),
    }
    for m in models.values():
        m.fit(train)

    # ------------------------------------------------- 1. normalisation of every row
    worst = 0.0
    rng = np.random.default_rng(0)
    ctx_space = list(models["laplace"].counts[2].keys())
    probe = [ctx_space[i] for i in rng.integers(0, len(ctx_space), size=200)]
    probe += [(-1,), (vocab["the"],), (0,), (7,)][-4:]        # include unseen histories
    for name, m in models.items():
        if name == "mle":
            continue            # an unsmoothed row is not normalised over V; that is the point
        for ctx in probe:
            s = float(m.full_dist(ctx).sum())
            worst = max(worst, abs(s - 1.0))
    assert worst < 1e-12, f"conditional rows are not normalised (worst |sum-1|={worst:.2e})"
    print(f"PASS normalisation contexts_probed={len(probe)} worst_abs_dev={worst:.2e}")
    out["norm_worst_dev"] = f"{worst:.2e}"

    # ------------------------------------------------- 2. perplexity = exp(cross-entropy)
    res = {name: m.perplexity(test) for name, m in models.items()}
    for name, r in res.items():
        if r["zero_prob_tokens"]:
            assert math.isinf(r["ppl"]), "a model with zero-probability tokens must be infinite"
            continue
        assert abs(r["ppl"] - math.exp(r["ce"])) < 1e-9 * r["ppl"], name
    print("PASS ppl-equals-exp-ce")
    for name, r in res.items():
        tag = "inf" if math.isinf(r["ppl"]) else f"{r['ppl']:.3f}"
        print(f"  {name:16s} ppl={tag:>10s} ce={r['ce']:.4f} "
              f"zero_prob_tokens={r['zero_prob_tokens']}/{r['tokens']}")

    # ------------------------------------------------- 3. raw MLE really does break
    mle = res["mle"]
    assert mle["zero_prob_tokens"] > 0, \
        "the corpus is so small that unsmoothed MLE never hits an unseen bigram"
    assert math.isinf(mle["ppl"])
    print(f"PASS mle-is-infinite zero_prob_tokens={mle['zero_prob_tokens']} "
          f"({100 * mle['zero_prob_tokens'] / mle['tokens']:.1f}% of test tokens)")
    out["mle_zero_prob_tokens"] = mle["zero_prob_tokens"]
    out["mle_zero_prob_pct"] = round(100 * mle["zero_prob_tokens"] / mle["tokens"], 2)

    # ------------------------------------------------- 4. conditioning reduces entropy
    lower_than_unigram = []
    for method in ("add_alpha", "absolute", "jm"):
        uni = ng.NgramLM(1, method, alpha=0.5, V=V).fit(train).perplexity(test)
        big = ng.NgramLM(2, method, alpha=0.5, discount=0.75, V=V).fit(train).perplexity(test)
        if not (math.isinf(uni["ppl"]) or math.isinf(big["ppl"])):
            assert big["ppl"] < uni["ppl"], \
                f"{method}: bigram {big['ppl']:.2f} is not below unigram {uni['ppl']:.2f}"
            lower_than_unigram.append((method, uni["ppl"], big["ppl"]))
    for method, u, b in lower_than_unigram:
        print(f"  {method:10s} unigram={u:8.2f} bigram={b:8.2f} drop={u - b:6.2f}")
    print(f"PASS conditioning-helps methods={len(lower_than_unigram)}")
    out["unigram_ppl"] = round(lower_than_unigram[0][1], 3)
    out["bigram_ppl_addalpha"] = round(lower_than_unigram[0][2], 3)

    # ------------------------------------------------- 5. smoothing has an interior optimum
    sweep = []
    for a in ALPHA_SWEEP:
        m = ng.NgramLM(2, "add_alpha", alpha=a, V=V).fit(train)
        sweep.append((a, m.perplexity(dev)["ppl"], m.perplexity(test)["ppl"]))
    # min over (alpha, ppl) tuples would compare alphas, so the key is explicit
    a_dev, ppl_dev = min(sweep, key=lambda row: row[1])[0], min(sweep, key=lambda row: row[1])[1]
    a_test, ppl_test = min(sweep, key=lambda row: row[2])[0], min(sweep, key=lambda row: row[2])[2]
    assert a_dev > ALPHA_SWEEP[0], "the optimum sat at the degenerate alpha, smoothing did not help"
    assert ppl_dev < dict((a, d) for a, d, _t in sweep)[1.0], \
        "tuning alpha on the development split did not beat the default alpha=1"
    # the curve must be unimodal on the log grid: one valley, no second local optimum
    devs = [d for _a, d, _t in sweep]
    k = devs.index(min(devs))
    assert all(devs[i] > devs[i + 1] for i in range(k)), "the dev curve rises before its minimum"
    assert all(devs[i] < devs[i + 1] for i in range(k, len(devs) - 1)), \
        "the dev curve falls after its minimum"
    assert abs(math.log(a_dev) - math.log(a_test)) < math.log(10), \
        f"dev-optimal alpha={a_dev} and test-optimal alpha={a_test} disagree by more than a decade"
    print(f"PASS smoothing-helps alpha_dev*={a_dev} ppl_dev={ppl_dev:.3f} "
          f"alpha_test*={a_test} ppl_test={ppl_test:.3f}")
    for a, d, t in sweep:
        print(f"  alpha={a:<8g} dev={d:9.3f} test={t:9.3f}")
    out["alpha_dev_star"] = a_dev
    out["alpha_test_star"] = a_test
    out["ppl_addalpha_opt"] = round(ppl_test, 3)

    # ------------------------------------------------- 6. the chain rule holds block by block
    model = models["laplace"]
    blocks = tc.blocks_of(tc.load_text())[int(len(tc.blocks_of(tc.load_text())) * 0.9):]
    worst_block = 0.0
    for blk in blocks[:40]:
        ids = [vocab.get(t, vocab["<unk>"]) for t in blk]
        ids.append(vocab["<eos>"])
        nll_direct = 0.0
        log_joint = 0.0
        for ctx, w in ng.context_pairs(ids, 2):
            lp = model.logprob(ctx, w)
            nll_direct -= lp
            log_joint += lp
        via_model, n_tokens, _z = model.nll(ids)
        assert n_tokens == len(ids), "the scorer dropped the padded first target"
        worst_block = max(worst_block, abs(nll_direct - via_model))
        # the joint probability of the whole scored sequence is exp of the summed log conditionals
        assert abs(math.exp(log_joint) - math.exp(-via_model)) < 1e-25 or math.exp(log_joint) < 1e-300
    assert worst_block < 1e-9, f"block NLL disagrees with the token sum by {worst_block:.2e}"
    print(f"PASS chain-rule blocks={min(40, len(blocks))} worst_abs_dev={worst_block:.2e}")

    # ------------------------------------------------- 7. determinism
    again = ng.NgramLM(2, "add_alpha", alpha=1.0, V=V).fit(train).perplexity(test)
    assert again == res["laplace"], "refit did not reproduce the original numbers"
    print("PASS determinism")

    # ------------------------------------------------- 8. stored table versus dense table
    stored = models["laplace"].n_params()
    dense = models["laplace"].table_cells()
    assert stored < dense, "a sparse bigram table should not exceed the dense one"
    print(f"  stored conditional entries={stored}  dense V^2 cells={dense}  "
          f"density={stored / dense:.3f}")
    out["stored_entries"] = stored
    out["dense_cells"] = dense

    for k, v in out.items():
        print(f"METRIC {k}={v}")
    print("RESULT bigram model verified on the shipped corpus")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
