"""n-gram language models and the four smoothing methods used in Phase 2.

The chapter derives each estimator on paper; this module is the implementation those derivations
point at. Every estimator here returns a proper distribution over the vocabulary for every
context, which is asserted in `code/phase02_ngram_lm/smoothing_sweep.py`, because a smoothed
model that does not sum to one is not a probability model.

Methods
  mle          c(h,w) / c(h,*). No smoothing. An unseen bigram gets exactly zero probability, so
               the test perplexity is infinite. That is not a bug, it is the lesson.
  add_alpha    (c(h,w) + a) / (c(h,*) + a*V). Laplace smoothing at a = 1.
  absolute     max(c(h,w) - d, 0) / c(h,*) + lambda(h) * P_lower(w), with
               lambda(h) = d * |{w : c(h,w) > 0}| / c(h,*). Ney's absolute discounting.
  jm           Jelinek-Mercer interpolated backoff, sum over all shorter histories of
               product(lambdas) * P_mle(w|shorter h). The interpolation weights are fitted on the
               development split, not hand-picked.
  kn           interpolated Kneser-Ney. The lower-order distribution uses continuation counts, so
               a word that follows many different contexts keeps probability even when the current
               context is unseen.

Conventions
  * Token ids are integers from `toycorpus`. Contexts shorter than order-1 are padded with PAD,
    an id that never occurs in the corpus.
  * Counts are stored as one dict per order: history tuple -> Counter of following tokens.
  * `full_dist` is memoised per (order, history), which turns test-set scoring from quadratic to
    linear in the number of distinct contexts.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np

PAD = -1
LOG_ZERO = -1e30  # stand-in for log(0) so that sums stay finite while the count is reported


def context_pairs(ids, order: int):
    """Yield (history, target) with PAD padding so that the first targets are scored too."""
    n1 = order - 1
    padded = [PAD] * n1 + list(ids)
    for i in range(n1, len(padded)):
        yield tuple(padded[i - n1:i]), padded[i]


def suffixes(ctx):
    """Strictly shorter suffixes of a history, longest first. () comes last."""
    return [ctx[i:] for i in range(1, len(ctx) + 1)]


class NgramLM:
    def __init__(self, order: int = 2, method: str = "mle", alpha: float = 0.5,
                 discount: float = 0.75, V: int | None = None,
                 lam: list[float] | None = None):
        assert order >= 1, "an n-gram model of order < 1 is not a model"
        assert method in ("mle", "add_alpha", "absolute", "jm", "kn"), method
        self.order = order
        self.method = method
        self.alpha = alpha
        self.discount = discount
        self.lam = lam                      # interpolation weights, one per backoff level
        self.V = V
        self.counts: list[dict[tuple, Counter]] = [defaultdict(Counter) for _ in range(order + 1)]
        self.total: list[dict[tuple, int]] = [defaultdict(int) for _ in range(order + 1)]
        self.cont: list[dict[int, int]] = [dict() for _ in range(order + 1)]       # continuation
        self.followed: list[dict[tuple, int]] = [dict() for _ in range(order + 1)]  # types after h
        self.cont_total: list[int] = [0] * (order + 1)
        self._dist_cache: dict[tuple[int, tuple], np.ndarray] = {}
        self.fit_tokens = 0

    # ------------------------------------------------------------------ estimation
    def fit(self, ids) -> "NgramLM":
        ids = list(ids)
        self.fit_tokens = len(ids)
        if self.V is None:
            self.V = max([int(i) for i in ids] + [0]) + 1
        for o in range(1, self.order + 1):
            n1 = o - 1
            padded = [PAD] * n1 + ids
            seen_pairs: set[tuple[tuple, int]] = set()
            for i in range(n1, len(padded)):
                ctx = tuple(padded[i - n1:i])
                w = padded[i]
                self.counts[o][ctx][w] += 1
                self.total[o][ctx] += 1
                seen_pairs.add((ctx, w))
            # continuation count of w at order o: how many distinct histories precede it
            cc: Counter = Counter()
            for _ctx, w in seen_pairs:
                cc[w] += 1
            self.cont[o] = dict(cc)
            self.cont_total[o] = sum(cc.values())
            # number of distinct word types that follow each history
            self.followed[o] = {ctx: len(cnt) for ctx, cnt in self.counts[o].items()}
        # Freeze the tables into plain dicts. They are defaultdicts during fitting, and a
        # defaultdict that is still open at scoring time silently creates an entry for every
        # context the test set touches, which makes unseen_context_fraction() report zero.
        self.counts = [dict(d) for d in self.counts]
        self.total = [dict(d) for d in self.total]
        return self

    # ------------------------------------------------------------------ probabilities
    def _unigram_kn(self) -> np.ndarray:
        """Continuation-based unigram used as the KN base distribution."""
        p = np.zeros(self.V)
        tot = self.cont_total[2] if self.order >= 2 and self.cont_total[2] > 0 else 0
        if tot == 0:
            return np.full(self.V, 1.0 / self.V)
        for w, c in self.cont[2].items():
            p[w] = c
        s = p.sum()
        return p / s if s > 0 else np.full(self.V, 1.0 / self.V)

    def full_dist(self, ctx: tuple, order: int | None = None) -> np.ndarray:
        """Distribution over the vocabulary given `ctx`. Memoised."""
        order = self.order if order is None else order
        key = (order, ctx)
        cached = self._dist_cache.get(key)
        if cached is not None:
            return cached
        if order == 1 or len(ctx) == 0:
            p = self._base_dist()
        else:
            p = self._dist_at(order, ctx)
        self._dist_cache[key] = p
        return p

    def _base_dist(self) -> np.ndarray:
        """Order-1 marginal, in the form the current method requires."""
        if self.method == "kn":
            return self._unigram_kn()
        c1 = self.counts[1].get((), Counter())
        p = np.zeros(self.V)
        tot = sum(c1.values())
        for w, c in c1.items():
            p[w] = c
        if self.method == "add_alpha":
            p += self.alpha
        s = p.sum()
        return p / s if s > 0 else np.full(self.V, 1.0 / self.V)

    def _dist_at(self, order: int, ctx: tuple) -> np.ndarray:
        cnt = self.counts[order].get(ctx, Counter())
        tot = self.total[order].get(ctx, 0)
        V = self.V

        if self.method == "mle":
            p = np.zeros(V)
            if tot > 0 and cnt:
                for w, c in cnt.items():
                    p[w] = c / tot
                return p
            # unseen history: no information at this order, so consult the next shorter one.
            return self.full_dist(ctx[1:], order - 1) if len(ctx) > 1 else self._base_dist()

        if self.method == "add_alpha":
            p = np.full(V, self.alpha)
            for w, c in cnt.items():
                p[w] += c
            return p / (tot + self.alpha * V)

        lower = self.full_dist(ctx[1:], order - 1)

        if self.method in ("absolute", "kn"):
            d = self.discount
            p = np.zeros(V)
            for w, c in cnt.items():
                p[w] = max(c - d, 0.0) / tot if tot > 0 else 0.0
            n_types = self.followed[order].get(ctx, 0)
            lam = (d * n_types / tot) if tot > 0 else 1.0
            return p + lam * lower

        # Jelinek-Mercer: interpolate the MLE at this order with the shorter backoff chain.
        p_mle = np.zeros(V)
        if tot > 0:
            for w, c in cnt.items():
                p_mle[w] = c / tot
        lam_h = self._jm_lambda(order, ctx)
        return lam_h * p_mle + (1.0 - lam_h) * lower

    def _jm_lambda(self, order: int, ctx: tuple) -> float:
        """Held-out interpolation weight. Falls back to a count-proportional estimate."""
        if self.lam is not None:
            return float(self.lam[order - 2]) if order - 2 < len(self.lam) else 0.5
        c = self.total[order].get(ctx, 0)
        return c / (c + 1.0) if c else 0.0          # Witten-Bell style default when no dev fit was supplied

    def fit_jm_weights(self, dev_ids, grid=(0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9)) -> list[float]:
        """Pick one interpolation weight per backoff level by maximising dev likelihood.

        One weight per level, chosen greedily from the top order down. Greedy rather than joint
        because the objective is separable in practice and a joint search over seven levels costs
        a product of grid sizes.
        """
        assert self.method == "jm", "weights only apply to the interpolated backoff model"
        best = [0.3] * (self.order - 1)
        for level in range(self.order, 1, -1):        # order .. 2
            scores = []
            for g in grid:
                trial = list(best)
                trial[level - 2] = g
                self.lam = trial
                self._dist_cache.clear()
                scores.append((self.nll(dev_ids)[0], g))
            nll, g = min(scores)
            best[level - 2] = g
        self.lam = best
        self._dist_cache.clear()
        return best

    # ------------------------------------------------------------------ scoring
    def logprob(self, ctx: tuple, w: int) -> float:
        p = self.full_dist(ctx)[w]
        return math.log(p) if p > 0 else LOG_ZERO

    def nll(self, ids) -> tuple[float, int, int]:
        """Total NLL, token count, and how many test tokens received exactly zero probability."""
        tot = 0.0
        n = 0
        zeros = 0
        for ctx, w in context_pairs(ids, self.order):
            lp = self.logprob(ctx, w)
            if lp <= LOG_ZERO + 1.0:
                zeros += 1
            tot -= lp
            n += 1
        return tot, n, zeros

    def perplexity(self, ids) -> dict:
        tot, n, zeros = self.nll(ids)
        ce = tot / n if n else float("inf")
        ppl = float("inf") if zeros else math.exp(ce)
        return {"nll": tot, "tokens": n, "ce": ce, "ppl": ppl, "zero_prob_tokens": zeros}

    def unseen_context_fraction(self, ids) -> float:
        seen = self.total[self.order]
        n = z = 0
        for ctx, _w in context_pairs(ids, self.order):
            n += 1
            if ctx not in seen:
                z += 1
        return z / n if n else 0.0

    def n_unseen_bigram_types(self, ids) -> int:
        """Distinct (h,w) pairs in `ids` that were never observed in training."""
        train_pairs = {(ctx, w) for ctx, cnt in self.counts[self.order].items() for w in cnt}
        return sum(1 for ctx, w in context_pairs(ids, self.order) if (ctx, w) not in train_pairs)

    # ------------------------------------------------------------------ reporting
    def n_params(self) -> int:
        """Free parameters actually stored: nonzero conditional rows minus one per history."""
        total = 0
        for o in range(1, self.order + 1):
            for _ctx, cnt in self.counts[o].items():
                total += max(len(cnt) - 1, 0)
        return total

    def table_cells(self) -> int:
        """Cells a dense table would need, V^(order). Compared against n_params()."""
        return self.V ** self.order


def frequency_spectrum(counts: Counter) -> dict[int, int]:
    """n_r for the observed counts: how many distinct items occurred exactly r times."""
    out: Counter = Counter()
    for _item, r in counts.items():
        out[r] += 1
    return dict(out)


def simple_good_turing(counts: Counter) -> dict:
    """Simple Good-Turing re-estimates.

    For an item observed r times the smoothed count is

        r* = (r + 1) * n_{r+1} / n_r

    and the probability reserved for items never seen is p0 = n_1 / N. Applied over the whole
    frequency spectrum the estimates sum to one, which is asserted in the Phase 2 sweep script
    rather than assumed here.

    Returns {"p_unseen": p0, "rstar": {r: r*}, "n_r": {r: n_r}, "N": N}.
    """
    n_r = Counter()
    for _item, r in counts.items():
        n_r[r] += 1
    N = sum(counts.values())
    n1 = n_r.get(1, 0)
    rstar = {}
    for r in sorted(n_r):
        nxt = n_r.get(r + 1, 0)
        rstar[r] = (r + 1) * nxt / n_r[r] if n_r[r] else float(r)
    return {"p_unseen": (n1 / N) if N else 0.0, "rstar": rstar,
            "n_r": dict(n_r), "N": N}


def top_tokens(ids, vocab_map: dict[str, int], k: int = 10) -> list[tuple[str, int]]:
    """Most frequent token ids with their counts, for the corpus table in the chapter."""
    inv = {i: t for t, i in vocab_map.items()}
    c = Counter(ids)
    return [(inv.get(i, str(i)), n) for i, n in c.most_common(k)]
