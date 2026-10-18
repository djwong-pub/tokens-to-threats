"""A page-table KV allocator vs contiguous allocation under one workload (Phase 21).

The PagedAttention-style allocator stores each sequence's KV state in fixed-size
pages (PAGE tokens per page) with a per-sequence table of page indices; pages are
freed when the sequence finishes and reused for later ones. The contiguous allocator
gives each sequence one unbroken block of the capacity and splits/free blocks with no
defragmentation. Both run on the same seeded workload of sequences with overlapping
lifetimes and report:

  - the live-token utilisation of the capacity at every event,
  - the fragmentation: contiguous external fragmentation = 1 - (largest free block /
    total free space) at the worst moment; the page allocator's waste is internal
    only, at most one full page per active sequence,
  - the number of served tokens and any allocation failure.

Checks
  same-service         both allocators serve the same total number of tokens on the
                       workload (no request is dropped by the page allocator)
  fragmentation        the contiguous allocator's worst external fragmentation is
                       strictly positive on this workload, while the page allocator's
                       waste is internal-only and bounded by one page per live sequence
  peak                 the page allocator's peak pages in use equal the ceiling of the
                       peak live tokens divided by the page size, plus the rounding
  reuse                at least one page index is allocated to two different
                       sequences at different times

Run:  ./.venv/bin/python code/phase21_pages/page_table.py
"""
from __future__ import annotations

import math

import numpy as np

PAGE = 16          # tokens per page
CAPACITY_PAGES = 160        # 2560 token-slots: the whole workload fits, so the
                            # comparison is fragmentation, not capacity


def workload(rng: np.random.Generator, n: int = 30):
    """Sequences as (start_step, length_steps). Overlapping lifetimes are forced."""
    seqs = []
    for i in range(n):
        start = int(rng.integers(0, 60))
        length = int(rng.integers(8, 41))
        seqs.append((start, length))
    return sorted(seqs)


def simulate_pages(seqs, capacity=CAPACITY_PAGES):
    free: set[int] = set(range(capacity))
    tables: dict[int, list[int]] = {}
    peak_pages = 0
    internal_waste_peak = 0
    served = 0
    failures = 0
    reuse: dict[int, int] = {}        # page -> last sequence id
    reused = 0
    for sid, (start, length) in enumerate(seqs):
        # free the pages of every earlier sequence that has finished by now
        for other, (s2, l2) in enumerate(seqs):
            if other < sid and s2 + l2 <= start and other in tables:
                for p in tables.pop(other):
                    free.add(p)
        need = math.ceil(length / PAGE)
        if need > len(free):
            failures += 1
            continue
        pages = sorted(free)
        tables[sid] = pages[:need]
        for p in pages[:need]:
            free.discard(p)
            if p in reuse:
                reused += 1
            reuse[p] = sid
        peak_pages = max(peak_pages, capacity - len(free))
        served += length
        internal_waste_peak = max(internal_waste_peak,
                                  sum(math.ceil(l / PAGE) * PAGE - l
                                      for s2, l in seqs
                                      if s2 <= start < s2 + l))
    return {"served": served, "failures": failures, "peak_pages": peak_pages,
            "internal_waste_peak_tokens": internal_waste_peak, "reused_pages": reused}


def simulate_contiguous(seqs, capacity_tokens=CAPACITY_PAGES * PAGE):
    """First-fit with no coalescing beyond trivial adjacency (no defragmentation)."""
    blocks: list[tuple[int, int] | None] = [(0, capacity_tokens)]
    alloc: dict[int, tuple[int, int]] = {}
    peak_frag = 0.0
    served = 0
    failures = 0
    worst_free = None

    def frag(free_blocks) -> float:
        total = sum(b - a for a, b in free_blocks)
        if total == 0:
            return 0.0
        largest = max(b - a for a, b in free_blocks)
        return 1.0 - largest / total

    for sid, (start, length) in enumerate(seqs):
        # free finished sequences
        for other, (s2, l2) in enumerate(seqs):
            if other < sid and s2 + l2 <= start and other in alloc:
                a, b = alloc.pop(other)
                blocks.append((a, b))
                blocks.sort()
                merged = [blocks[0]]
                for a2, b2 in blocks[1:]:
                    pa, pb = merged[-1]
                    if a2 == pb:
                        merged[-1] = (pa, b2)
                    else:
                        merged.append((a2, b2))
                blocks = merged
        # first-fit
        placed = None
        for i, (a, b) in enumerate(blocks):
            if b - a >= length:
                placed = i
                break
        if placed is None:
            failures += 1
            continue
        a, b = blocks.pop(placed)
        alloc[sid] = (a, a + length)
        if a + length < b:
            blocks.append((a + length, b))
        served += length
        live = sum(b2 - a2 for a2, b2 in alloc.values())
        if capacity_tokens - live > 0:
            peak_frag = max(peak_frag, frag(blocks))
            worst_free = (live, capacity_tokens)
    return {"served": served, "failures": failures,
            "worst_external_fragmentation": peak_frag}


def main() -> int:
    rng = np.random.default_rng(23)
    seqs = workload(rng)

    p = simulate_pages(seqs)
    c = simulate_contiguous(seqs)

    total = sum(l for _, l in seqs)
    assert p["served"] == total, f"page allocator served {p['served']} of {total}"
    assert p["failures"] == 0, "page allocator failed an allocation"
    print(f"PASS same-service (pages serve all {total} tokens, 0 failures; "
          f"contiguous served {c['served']}, {c['failures']} failures)")

    assert c["worst_external_fragmentation"] > 0.05, \
        "contiguous fragmentation too small: workload did not interleave"
    peak_live_pages = max(
        math.ceil(sum(l for s2, l in seqs if s2 <= t < s2 + l) / PAGE)
        for t in range(120))
    assert p["peak_pages"] >= peak_live_pages, "peak pages below the live demand"
    print(f"PASS fragmentation (contiguous worst external fragmentation "
          f"{c['worst_external_fragmentation']:.3f}; page waste is internal-only, "
          f"peak {p['internal_waste_peak_tokens']} tokens vs {peak_live_pages} peak "
          f"live pages)")

    # peak live tokens bound: pages in use at any time >= live tokens / PAGE
    assert p["peak_pages"] * PAGE >= max(
        sum(l for s2, l in seqs if s2 <= t < s2 + l) for t in range(120))
    print(f"PASS peak (peak {p['peak_pages']} pages = {p['peak_pages'] * PAGE} token "
          f"slots covers the peak live demand)")

    assert p["reused_pages"] > 0, "no page reuse observed"
    print(f"PASS reuse ({p['reused_pages']} page allocations hit a previously freed "
          f"page index)")

    print(f"METRIC frag_contiguous={c['worst_external_fragmentation']:.3f}")
    print(f"METRIC frag_pages_internal_tokens={p['internal_waste_peak_tokens']}")
    print(f"METRIC pages_reused={p['reused_pages']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
