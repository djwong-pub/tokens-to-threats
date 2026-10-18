"""Corpus statistics for the shipped toy corpus used by Phases 2 to 6.

Nothing in Part I is compared across models until the corpus itself is characterised. This script
measures the four properties that decide how a language model behaves: size, vocabulary growth,
the shape of the rank-frequency curve, and how much of the test set the training vocabulary has
never seen. Each property is asserted rather than described.

Checks
  corpus-integrity    the shipped file hashes to the recorded SHA-256, so every perplexity in the
                      book refers to the same bytes
  block-split         train, dev and test are block-wise splits and no block crosses a boundary
  vocab-coverage      at least 97% of test tokens are in the training vocabulary at min_count=2,
                      and the residual out-of-vocabulary rate is reported as a metric
  zipf-shape          log frequency against log rank has a slope near -1 (the Zipf regime)
  entropy-bounds      the marginal entropy is at most log(V) and strictly above 0.5 log(V)
  hapax-share         singleton types hold a measurable but minority share of tokens

Run:  ./.venv/bin/python code/phase02_ngram_lm/corpus_stats.py
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


def main() -> int:
    text = tc.load_text()
    ok, digest = tc.integrity_ok()
    blocks = tc.blocks_of(text)
    tokens = tc.tokenize(text)

    # ---------------------------------------------------------------- 1. integrity
    assert ok, f"corpus hash mismatch: {digest} != {tc.CORPUS_SHA256}"
    print(f"PASS corpus-integrity sha256={digest[:16]}...")
    METRICS = {"sha256_prefix": digest[:16]}

    # ---------------------------------------------------------------- 2. split hygiene
    train, dev, test, vocab = tc.load_splits()
    n_tr_blocks = int(len(blocks) * 0.8)
    n_dev_blocks = int(len(blocks) * 0.9) - n_tr_blocks
    tok_train = [t for b in blocks[:int(len(blocks) * 0.8)] for t in b]
    assert len(train) == len(tok_train) + n_tr_blocks, "train stream lost an <eos> per block"
    shared_words = set()
    for b in blocks[:int(len(blocks) * 0.8)]:
        shared_words.update(b)
    # a block-wise split may repeat vocabulary across splits; what must not happen is a repeated
    # document. The check below compares whole block signatures.
    sig = [tuple(b) for b in blocks]
    tr_sigs = set(sig[:int(len(blocks) * 0.8)])
    te_sigs = set(sig[int(len(blocks) * 0.9):])
    assert not (tr_sigs & te_sigs), "a test block also occurs in the training split"
    print(f"PASS block-split train_blocks={n_tr_blocks} dev_blocks={n_dev_blocks} "
          f"test_blocks={len(blocks) - n_tr_blocks - n_dev_blocks}")
    METRICS.update(tokens_total=len(tokens), train_tokens=len(train), dev_tokens=len(dev),
                   test_tokens=len(test), vocab=len(vocab))

    # ---------------------------------------------------------------- 3. coverage
    test_words = [t for b in blocks[int(len(blocks) * 0.9):] for t in b]
    oov = sum(1 for t in test_words if t not in vocab)
    frac_oov = oov / len(test_words)
    # An out-of-vocabulary rate near zero would mean the corpus has no tail at all, which would
    # make every smoothing experiment here vacuous. 3% is the ceiling above which scoring would
    # be dominated by <unk> rather than by the model.
    assert frac_oov < 0.03, f"out-of-vocabulary rate {frac_oov:.3f} is too high to score fairly"
    print(f"PASS vocab-coverage oov_test={frac_oov * 100:.2f}%")
    METRICS["oov_test_pct"] = round(frac_oov * 100, 3)

    # ---------------------------------------------------------------- 4. Zipf shape
    counts = Counter(tok_train)
    freqs = np.array(sorted(counts.values(), reverse=True), dtype=float)
    ranks = np.arange(1, len(freqs) + 1, dtype=float)
    slope, intercept = np.polyfit(np.log(ranks), np.log(freqs), 1)
    assert -1.8 < slope < -0.5, f"rank-frequency slope {slope:.3f} is outside the Zipf regime"
    print(f"PASS zipf-shape slope={slope:.3f} r2 will follow")
    pred = slope * np.log(ranks) + intercept
    r2 = 1 - np.sum((np.log(freqs) - pred) ** 2) / np.sum((np.log(freqs) - np.log(freqs).mean()) ** 2)
    assert r2 > 0.85, f"the log-log fit is too weak to quote (R^2={r2:.3f})"
    METRICS.update(zipf_slope=round(float(slope), 4), zipf_r2=round(float(r2), 4))

    # ---------------------------------------------------------------- 5. entropy bounds
    N = sum(counts.values())
    p = np.array([c / N for c in counts.values()], dtype=float)
    H = float(-np.sum(p * np.log2(p)))
    V = len(counts)
    assert 0.5 * math.log2(V) < H <= math.log2(V) + 1e-9, \
        f"H={H:.2f} bits is inconsistent with log2(V)={math.log2(V):.2f}"
    print(f"PASS entropy-bounds H={H:.2f} bits of log2(V)={math.log2(V):.2f}")
    METRICS.update(H_marginal_bits=round(H, 3), H_uniform_bits=round(math.log2(V), 3),
                   perplexity_marginal=round(2 ** H, 2))

    # ---------------------------------------------------------------- 6. hapaxes
    gt = ng.simple_good_turing(counts)
    n1 = gt["n_r"].get(1, 0)
    share_singletons = n1 / N
    assert 0.001 < share_singletons < 0.25, \
        f"singleton token share {share_singletons:.4f} is not a usable learning regime"
    # the Good-Turing unseen probability estimate is compared against the measured rate on test
    unseen_rate = sum(1 for t in test_words if counts.get(t, 0) == 0) / len(test_words)
    assert abs(unseen_rate - gt["p_unseen"]) < 0.02, \
        f"Good-Turing p0={gt['p_unseen']:.4f} but the test set disagrees at {unseen_rate:.4f}"
    print(f"PASS hapax-share n1={n1} GT_p_unseen={gt['p_unseen']:.4f} "
          f"measured_test_unseen={unseen_rate:.4f}")
    METRICS.update(gt_p_unseen=round(gt["p_unseen"], 5),
                   test_unseen_rate=round(unseen_rate, 5), n_types_tr=V,
                   mass_singletons=round(share_singletons, 4))

    for k, v in METRICS.items():
        print(f"METRIC {k}={v}")
    print("RESULT corpus statistics verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
