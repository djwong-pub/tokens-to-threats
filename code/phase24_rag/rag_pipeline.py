"""A RAG pipeline built from scratch on a controlled corpus (Phase 24).

Everything is implemented by hand on numpy, no retrieval library:
  chunker    fixed-size character chunks with exact overlap
  embedder   non-negative hashed bag-of-words vectors, L2-normalised; cosine
  BM25       Okapi BM25 with the standard idf and term-frequency saturation
  fusion     reciprocal rank fusion (RRF) of the two rankings
  reranker   a cross score computed on the (query, chunk) pair with fixed
             weights, whose coverage feature is what the bag-of-words cosine
             cannot see

Checks (self-test mode, the default)
  chunk-overlap      every chunk respects the size bound and consecutive
                     chunks overlap in exactly the overlap characters
  geometry           embedding unit norm, self-cosine 1, a related pair above
                     a disjoint pair, and the saturation inversion below
  bm25-idf           a rare term's idf exceeds a common term's idf, and the
                     rare-term query ranks its chunk first
  rrf-exact          the fused scores equal the hand-computed fractions and
                     the fused order is the hand order
  reranker-flips     on the saturation query the bag-of-words cosine ranks the
                     saturated distractor above the comprehensive gold, and the
                     cross reranker puts the gold back first
  pipeline-top1      reranked top-1 accuracy on the gold set is 1.0 and above
                     the dense-alone top-1 accuracy

Run:  ./.venv/bin/python code/phase24_rag/rag_pipeline.py
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

DIM = 256
CHUNK_SIZE = 400         # characters
CHUNK_OVERLAP = 24       # characters
K1, B, RRF_LAMBDA = 1.5, 0.75, 60


def tokenize(text: str) -> list[str]:
    return [t for t in text.lower().split() if t]


def chunk_documents(docs: list[str]) -> tuple[list[str], list[int]]:
    """Fixed character chunks of CHUNK_SIZE with CHUNK_OVERLAP overlap.

    Returns the chunk strings and, for each chunk, the document index it came
    from. The last chunk of a document may be shorter than CHUNK_SIZE.
    """
    chunks: list[str] = []
    origins: list[int] = []
    for di, doc in enumerate(docs):
        step = CHUNK_SIZE - CHUNK_OVERLAP
        pos = 0
        while pos < len(doc):
            chunks.append(doc[pos:pos + CHUNK_SIZE])
            origins.append(di)
            if pos + CHUNK_SIZE >= len(doc):
                break
            pos += step
    return chunks, origins


def embed(texts: list[str]) -> np.ndarray:
    """Non-negative hashed bag-of-words, L2-normalised to unit norm."""
    mat = np.zeros((len(texts), DIM), dtype=np.float64)
    for i, text in enumerate(texts):
        for tok in tokenize(text):      # counts kept: tf is what saturates
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            mat[i, h % DIM] += 1.0
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def cosine_topk(qv: np.ndarray, mat: np.ndarray, k: int) -> list[int]:
    scores = mat @ qv
    order = np.argsort(-scores)
    return [int(i) for i in order[:k]]


class BM25:
    """Okapi BM25 over a corpus of token lists."""

    def __init__(self, docs_tokens: list[list[str]], k1: float = K1, b: float = B):
        self.docs = docs_tokens
        self.n = len(docs_tokens)
        self.df = {}
        for toks in docs_tokens:
            for t in set(toks):
                self.df[t] = self.df.get(t, 0) + 1
        self.avgdl = sum(len(t) for t in docs_tokens) / max(1, self.n)
        self.k1, self.b = k1, b
        self.tf = [{} for _ in range(self.n)]
        for i, toks in enumerate(docs_tokens):
            for t in toks:
                self.tf[i][t] = self.tf[i].get(t, 0) + 1

    def idf(self, term: str) -> float:
        n_t = self.df.get(term, 0)
        return math.log((self.n - n_t + 0.5) / (n_t + 0.5) + 1.0)

    def score(self, query: list[str], i: int) -> float:
        total = 0.0
        dl = len(self.docs[i])
        for t in query:
            tf = self.tf[i].get(t, 0)
            if tf == 0:
                continue
            sat = tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            total += self.idf(t) * sat
        return total

    def topk(self, query: list[str], k: int) -> list[int]:
        scores = [self.score(query, i) for i in range(self.n)]
        order = np.argsort(-np.asarray(scores))
        return [int(i) for i in order[:k]]


def rrf_fuse(rankings: list[list[int]], lambda_: float = RRF_LAMBDA) -> list[int]:
    """Reciprocal rank fusion: score(d) = sum over rankings of 1/(lambda + rank)."""
    scores = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (lambda_ + rank + 1)
    return sorted(scores, key=lambda d: -scores[d])


def rrf_scores(rankings: list[list[int]], lambda_: float = RRF_LAMBDA) -> dict[int, float]:
    scores = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (lambda_ + rank + 1)
    return scores


def cross_rerank(query: list[str], chunks: list[str], order: list[int],
                 weights: tuple[float, float, float] = (1.0, 0.25, 0.1)) -> list[int]:
    """A fixed-weight cross score on the (query, chunk) pair.

    Features: coverage, the fraction of distinct query tokens present in the
    chunk; match mass, the cosine of the pair (the dense score, kept as a
    weak feature); length fit, 1 minus the normalised log length ratio. The
    coverage term is what the bag-of-words cosine cannot express.
    """
    w_cov, w_cos, w_len = weights
    qv = embed([" ".join(query)])[0]
    cm = embed([chunks[i] for i in order])
    lens = np.asarray([len(tokenize(chunks[i])) for i in order], dtype=np.float64)
    mean_q = max(1.0, float(len(set(query))) * 3)
    scored = []
    for j, i in enumerate(order):
        cov = len(set(query) & set(tokenize(chunks[i]))) / max(1, len(set(query)))
        cos = float(qv @ cm[j])
        lfit = 1.0 - abs(math.log(lens[j] / mean_q)) / (2.0 * math.log(2.0 + max(lens[j] / mean_q, 0.5)))
        scored.append((w_cov * cov + w_cos * cos + w_len * lfit, i))
    scored.sort(key=lambda p: -p[0])
    return [i for _, i in scored]


def main() -> int:
    # ------------------------------------------------------------ the corpus
    # Eight filler words shared by everything, so they are common terms.
    filler = "model system data input output layer weight matrix vector"
    docs = [
        # gold for q1: comprehensive, each query term once
        f"jailbreak quantization safety attack {filler} evaluation protocol",
        # distractor for q1: saturated in two of the four query terms
        "quantization quantization quantization quantization quantization "
        "quantization quantization quantization jailbreak jailbreak jailbreak "
        "jailbreak jailbreak jailbreak jailbreak jailbreak "
        f"{filler} noise",
        # gold for q2
        f"prompt injection agent tool {filler} harness policy",
        # distractor for q2: saturated in one query term
        "agent agent agent agent agent agent agent agent agent agent "
        "agent agent agent agent "
        f"{filler} loop",
        # gold for q3
        f"kv cache paging memory {filler} allocator page table",
        # gold for q4
        f"speculative decoding draft verify {filler} acceptance span",
        # gold for q5
        f"mixture experts routing sparse {filler} load balance",
        # a generic chunk, gold for nothing
        f"{filler} {filler} {filler} general background notes",
        # one long document, repeated filler, so the chunker actually splits:
        # it exercises the overlap rule the short documents cannot
        (filler + " ") * 25,
    ]
    chunks, origins = chunk_documents(docs)
    assert all(len(c) <= CHUNK_SIZE for c in chunks), "a chunk exceeds the bound"
    # exact overlap on the split document: consecutive chunks share exactly the
    # last CHUNK_OVERLAP characters of the previous one
    split = [k for k in range(len(chunks) - 1)
             if origins[k] == origins[k + 1] == len(docs) - 1]
    assert split, "the long document produced no multi-chunk run"
    for k in split:
        assert chunks[k][-CHUNK_OVERLAP:] == chunks[k + 1][:CHUNK_OVERLAP], \
            f"chunk {k}/{k + 1} overlap is not exactly {CHUNK_OVERLAP} chars"
    print(f"PASS chunk-overlap ({len(chunks)} chunks of <= {CHUNK_SIZE} chars from "
          f"{len(docs)} documents; {len(split)} consecutive pairs overlap in exactly "
          f"{CHUNK_OVERLAP} chars)")

    # ------------------------------------------------------------- geometry
    mat = embed(chunks)
    assert np.allclose(np.linalg.norm(mat, axis=1), 1.0, atol=1e-9), "not unit norm"
    qv1 = embed(["jailbreak quantization safety attack"])[0]
    g1, d1 = 0, 1
    cos_g = float(mat[g1] @ qv1)
    cos_d = float(mat[d1] @ qv1)
    cos_same = float(mat[g1] @ mat[g1])
    assert abs(cos_same - 1.0) < 1e-9, "self cosine not 1"
    # the saturation inversion: the distractor repeats two query terms enough
    # times to out-cosine the comprehensive gold on the bag-of-words embedding
    assert cos_d > cos_g, f"corpus must invert dense: cos_d={cos_d:.4f} <= cos_g={cos_g:.4f}"
    # related above disjoint: the injection chunk vs the kv-cache chunk
    qv2 = embed(["prompt injection agent tool"])[0]
    assert float(mat[2] @ qv2) > float(mat[4] @ qv2), "related pair below disjoint pair"
    print(f"PASS geometry (unit norms; saturation inversion cos_distractor "
          f"{cos_d:.4f} > cos_gold {cos_g:.4f}; related above disjoint)")

    # ----------------------------------------------------------------- bm25
    bm = BM25([tokenize(c) for c in chunks])
    rare, common = "quantization", "model"
    assert bm.idf(rare) > bm.idf(common), "rare term idf below common term idf"
    assert bm.topk([rare], 1)[0] in (0, 1), "rare term query not ranked sensibly"
    # saturation damping: on the four-term query BM25 ranks the gold above the
    # distractor, the opposite of the bag-of-words cosine above
    q1 = tokenize("jailbreak quantization safety attack")
    assert bm.topk(q1, 2)[:2] == [g1, d1], "BM25 must rank the comprehensive gold first"
    print(f"PASS bm25-idf (idf({rare}) {bm.idf(rare):.4f} > idf({common}) "
          f"{bm.idf(common):.4f}; rare-first; saturation damped, gold first)")

    # ------------------------------------------------------------------- rrf
    r1, r2 = [g1, d1, 2], [d1, 2, g1]
    fused = rrf_fuse([r1, r2])
    sc = rrf_scores([r1, r2])
    assert abs(sc[g1] - (1 / 61 + 1 / 63)) < 1e-12, "RRF score for g1 off"
    assert abs(sc[d1] - (1 / 62 + 1 / 61)) < 1e-12, "RRF score for d1 off"
    assert abs(sc[2] - (1 / 63 + 1 / 62)) < 1e-12, "RRF score for doc 2 off"
    assert fused[:2] == [d1, g1], f"RRF order {fused} != hand order"
    print(f"PASS rrf-exact (scores match 1/(lambda+rank) fractions to 1e-12; "
          f"fused order {fused[:3]} agrees with the hand computation)")

    # ------------------------------------------------------------- reranker
    dense = cosine_topk(qv1, mat, k=8)
    assert dense[0] == d1, "dense top-1 should be the saturated distractor"
    reranked = cross_rerank(q1, chunks, dense)
    assert reranked[0] == g1, "reranker failed to restore the comprehensive gold"
    print(f"PASS reranker-flips (dense top-1 is the distractor chunk {d1}, cross "
          f"reranker restores gold chunk {g1}: coverage 1.0 vs 0.5)")

    # -------------------------------------------------------- pipeline top-1
    gold = [
        (tokenize("jailbreak quantization safety attack"), g1),
        (tokenize("prompt injection agent tool"), 2),
        (tokenize("kv cache paging memory"), 4),
        (tokenize("speculative decoding draft verify"), 5),
        (tokenize("mixture experts routing sparse"), 6),
    ]
    dense_hits, fused_hits, reranked_hits = 0, 0, 0
    for q, g in gold:
        qv = embed([" ".join(q)])[0]
        d_order = cosine_topk(qv, mat, k=5)
        b_order = bm.topk(q, k=5)
        f_order = rrf_fuse([d_order, b_order])[:5]
        r_order = cross_rerank(q, chunks, f_order)
        dense_hits += int(d_order[0] == g)
        fused_hits += int(f_order[0] == g)
        reranked_hits += int(r_order[0] == g)
    n = len(gold)
    assert reranked_hits == n, f"reranked top-1 {reranked_hits}/{n}, wanted all"
    assert reranked_hits > dense_hits, "reranker must beat dense alone"
    print(f"PASS pipeline-top1 (reranked {reranked_hits}/{n} vs dense alone "
          f"{dense_hits}/{n} vs fused {fused_hits}/{n})")

    print(f"METRIC dense_top1={dense_hits / n:.3f}")
    print(f"METRIC fused_top1={fused_hits / n:.3f}")
    print(f"METRIC reranked_top1={reranked_hits / n:.3f}")
    print(f"METRIC chunks={len(chunks)} (from {len(docs)} documents)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
