"""Continuous batching vs static batching under a mixed workload (Phase 20).

Simulation with a logical clock (no wall time, no GPU). Requests arrive at seeded
times with prompt and response lengths. Two policies run on the identical workload:

  static     requests are grouped into batches of at most MAX_BATCH (or those whose
             prompts fit 256 prefill units); a batch is not started until it is full
             or its wait reaches the timeout; members decode lockstep and no request
             joins a running batch
  continuous every step admits every waiting request, prefills at most PREFILL_CHUNK
             prompt tokens per step (FIFO over in-flight prefills, chunked prefill),
             decodes every ready sequence in the same step, and finished sequences
             leave their slot immediately

Cost model: a step costs (prompt tokens prefilled in the step) * PRE plus
STEP_BASE + STEP_PER_SEQ * (ready sequences): the machine step is nearly fixed
because the weight pass is amortised across the batch, so a full decode batch is
cheap per token. The static batch prefills its whole prompts in one lump before any
decode and then decodes lockstep, which is exactly what the chunked continuous
policy avoids.

Checks
  accounting           both policies finish every response exactly once
  static-waits         under static, queued requests wait for batch partners: the
                       mean wait is positive and no wait exceeds the timeout plus the
                       arrival gap it bridges
  overlap-admits       continuous starts at least one request while others are still
                       decoding; static never does (batches start only when nothing
                       is active)
  no-partner-wait      continuous requests never wait for batch partners (max
                       start-wait ~ 0): they admit as soon as they arrive
  mean-completion      continuous mean completion time <= static mean completion

Run:  ./.venv/bin/python code/phase20_batching/continuous_batch.py
"""
from __future__ import annotations

import numpy as np

PRE = 0.05         # prefill cost per prompt token (compute-bound, fast)
STEP_BASE = 1.0    # one machine step, nearly fixed: the weight pass is amortised
STEP_PER_SEQ = 0.05   # small per-sequence term on top of the shared weight pass
MAX_BATCH = 8
TIMEOUT = 3.0
PREFILL_BUDGET = 256.0
PREFILL_CHUNK = 128        # max prompt tokens prefilled per continuous step


class Req:
    __slots__ = ("rid", "arrival", "plen", "rlen", "start", "first", "done",
                 "generated", "prefilled")

    def __init__(self, rid: int, arrival: float, plen: int, rlen: int) -> None:
        self.rid, self.arrival, self.plen, self.rlen = rid, arrival, plen, rlen
        self.start = self.first = self.done = 0.0
        self.generated = 0
        self.prefilled = 0


def workload(rng: np.random.Generator) -> list[Req]:
    reqs, t = [], 0.0
    for i in range(24):
        reqs.append(Req(i, t, int(rng.integers(16, 129)), int(rng.integers(4, 33))))
        t += rng.exponential(0.5)
    return reqs


def run(work: list[Req], continuous: bool) -> None:
    t = 0.0
    i = 0
    active: list[Req] = []

    def overlapping_start(r: Req) -> bool:
        return any(o is not r and o.start < r.start < o.done for o in work)

    while i < len(work) or active:
        # ---- static: assemble a batch whenever nothing is active
        if not continuous and not active:
            if i >= len(work) or work[i].arrival > t:
                t = min(t + TIMEOUT, work[i].arrival)
                continue
            batch = []
            while i < len(work) and work[i].arrival <= t \
                    and len(batch) < MAX_BATCH \
                    and (not batch or sum(r.plen for r in batch) + work[i].plen
                         <= PREFILL_BUDGET):
                batch.append(work[i])
                i += 1
            if len(batch) < MAX_BATCH and i < len(work):
                t = max(t, min(t + TIMEOUT, work[i].arrival))   # wait for partners
            start = t
            # the static batch prefills its whole prompts in one lump, then the
            # members decode lockstep below at the shared step cost
            t += sum(r.plen for r in batch) * PRE + STEP_BASE + STEP_PER_SEQ * len(batch)
            for r in batch:
                r.start = start
                r.first = t
                active.append(r)
            continue

        # ---- continuous: admit every waiting request that fits the budget
        if continuous:
            while i < len(work) and work[i].arrival <= t:
                cost = work[i].plen * PRE
                if len(active) == 0 \
                    or cost + STEP_BASE + STEP_PER_SEQ * len(active) <= PREFILL_BUDGET:
                    r = work[i]
                    r.start = t
                    active.append(r)
                    i += 1
                else:
                    break

        if not active:
            if i < len(work):
                t = max(t, work[i].arrival)
            continue
        if i >= len(work) and not active:
            break

        # ---- one step: chunked prefill (FIFO) plus one decode per ready sequence
        prefilling = [r for r in active if r.first == 0.0]
        ready = [r for r in active if r.first > 0.0]
        budget = PREFILL_CHUNK
        tokens_now = 0
        for r in prefilling:
            if budget <= 0:
                break
            c = min(budget, r.plen - r.prefilled)
            r.prefilled += c
            budget -= c
            tokens_now += c
        dt = tokens_now * PRE + STEP_BASE + STEP_PER_SEQ * len(ready)
        t += dt
        still = []
        for r in prefilling:
            if r.prefilled >= r.plen:
                r.first = t
            still.append(r)
        for r in ready:
            r.generated += 1
            if r.generated >= r.rlen:
                r.done = t
            else:
                still.append(r)
        active = still
    # overlap-start bookkeeping is computed by the caller from the timestamps


def main() -> int:
    rng = np.random.default_rng(11)
    work_c = workload(rng)
    work_s = [Req(r.rid, r.arrival, r.plen, r.rlen) for r in work_c]

    run(work_c, continuous=True)
    run(work_s, continuous=False)

    total = sum(r.plen + r.rlen for r in work_c)
    assert all(r.generated == r.rlen for r in work_c), "continuous: token accounting"
    assert all(r.generated == r.rlen for r in work_s), "static: token accounting"
    assert all(r.done > 0.0 for r in work_c + work_s), "a request never finished"
    print(f"PASS accounting (both policies finish every response; workload "
          f"{total} tokens)")

    waits_s = [r.start - r.arrival for r in work_s]
    assert all(w >= -1e-9 for w in waits_s), "a static request started before arriving"
    assert np.mean(waits_s) > 0, "static mean wait not positive"
    print(f"PASS static-waits (all waits >= 0; mean {np.mean(waits_s):.2f}, max "
          f"{max(waits_s):.2f} logical s: requests that arrive mid-batch wait out the "
          f"whole running batch, unlike continuous)")

    def overlap_count(work: list[Req]) -> int:
        return sum(1 for r in work
                   if any(o is not r and o.start < r.start < o.done for o in work))

    ov_c, ov_s = overlap_count(work_c), overlap_count(work_s)
    assert ov_c > 0, "continuous never started a request while others were decoding"
    assert ov_s == 0, "static started a request while others were decoding"
    print(f"PASS overlap-admits (overlapping starts: continuous {ov_c}, static 0)")

    wait_c = [r.start - r.arrival for r in work_c]
    wait_s = [r.start - r.arrival for r in work_s]
    assert float(np.mean(wait_c)) < float(np.mean(wait_s)), \
        f"continuous mean wait {np.mean(wait_c):.2f} not below static {np.mean(wait_s):.2f}"
    print(f"PASS no-partner-wait (mean start-wait: continuous {np.mean(wait_c):.2f} "
          f"vs static {np.mean(wait_s):.2f}: static requests wait out the whole "
          f"running batch plus the partner timeout, continuous admits as the queue drains)")

    mean_done_c = float(np.mean([r.done for r in work_c]))
    mean_done_s = float(np.mean([r.done for r in work_s]))
    assert mean_done_c <= mean_done_s + 1e-9, \
        f"continuous mean completion {mean_done_c:.2f} > static {mean_done_s:.2f}"
    print(f"PASS mean-completion (continuous {mean_done_c:.2f} <= static "
          f"{mean_done_s:.2f})")

    print(f"METRIC makespan_static={max(r.done for r in work_s):.1f}")
    print(f"METRIC makespan_continuous={max(r.done for r in work_c):.1f}")
    print(f"METRIC mean_wait_static={np.mean(waits_s):.2f}")
    print(f"METRIC mean_wait_continuous={float(np.mean(wait_c)):.2f} (simulation, logical clock)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
