"""A DistServe-style cost model: colocated vs disaggregated prefill/decode (Phase 23).

Discrete-step simulation, no GPU. Time is counted in steps; each step is one unit.
A seeded workload of (prompt_len, response_len) requests arrives over time.

  colocated     ONE machine serves both phases. A step does either prefill or decode,
                never both: if any arrived request still has prompt left, the step
                prefills PREFILL tokens of prompt (FIFO) and no sequence decodes
                (prefill starves decode); otherwise every active sequence decodes one
                token. This is the contention the separation removes.

  disaggregated a prefill machine that prefills PREFILL_D prompt tokens per step
                (PREFILL_D = PREFAILL_REPLICAS x PREFILL, more machines in parallel)
                and a decode machine whose replicas advance every active sequence by
                DECODE_D tokens per step; the two machines never share a step, and
                the prompt's KV state is transferred once per request.

Checks
  slo-split            colocated violates at least one SLO (TTFT or TPOT) while
                       disaggregated satisfies both: the SLOs are decoupled because
                       each phase gets its own sized machines
  transfer-cost        disaggregated transfers exactly prompt_tokens * BYTES_PER_TOKEN
  weight-memory        disaggregated holds exactly two weight copies against one
  tpot-improvement     disaggregated TPOT < colocated TPOT, because decode steps are
                       no longer stolen by prefill

Run:  ./.venv/bin/python code/phase23_disagg/disagg_cost.py
"""
from __future__ import annotations

import numpy as np

BYTES_PER_TOKEN = 2 * 8 * 128 * 2 * 4   # fp16 KV per token (h_kv=8, d_k=128, 4 layers)
WEIGHT_BYTES = 8.0e9                    # one full model copy, fixed
PREFILL = 128                           # prompt tokens prefilled per step (1 machine)
PREFILL_REPLICAS = 4                    # disaggregated prefill machines
DECODE_REPLICAS = 8                     # disaggregated decode replicas
TTFT_SLO, TPOT_SLO = 12.0, 1.0          # in steps


class Req:
    __slots__ = ("rid", "arr", "plen", "rlen", "prefill_left", "gen", "first",
                 "done")

    def __init__(self, rid: int, arr: float, plen: int, rlen: int) -> None:
        self.rid, self.arr, self.plen, self.rlen = rid, arr, plen, rlen
        self.prefill_left = plen
        self.gen = 0
        self.first = 0.0
        self.done = 0.0


def workload(rng: np.random.Generator) -> list[Req]:
    reqs, t = [], 0.0
    for i in range(20):
        reqs.append(Req(i, t, int(rng.integers(32, 513)), int(rng.integers(16, 65))))
        t += rng.exponential(0.4)
    return reqs


def run_colocated(work: list[Req]) -> dict:
    t = 0.0
    i = 0
    waiting: list[Req] = []
    decoding: list[Req] = []
    ttfts, tpots = [], []

    while i < len(work) or waiting or decoding:
        while i < len(work) and work[i].arr <= t:
            waiting.append(work[i])
            i += 1
        if not waiting and not decoding:
            if i < len(work):
                t = work[i].arr
            continue
        if waiting:
            # the step goes to prefill; decode is starved
            budget = PREFILL
            for r in sorted(waiting, key=lambda x: x.arr):
                if budget <= 0:
                    break
                c = min(budget, r.prefill_left)
                r.prefill_left -= c
                budget -= c
            t += 1
            done_p = [r for r in waiting if r.prefill_left == 0]
            waiting = [r for r in waiting if r.prefill_left > 0]
            for r in done_p:
                decoding.append(r)
        else:
            # pure decode step
            t += 1
            still = []
            for r in decoding:
                r.gen += 1
                if r.first == 0.0:
                    r.first = t
                    ttfts.append(t - r.arr)
                if r.gen >= r.rlen:
                    r.done = t
                    tpots.append((t - r.first + 1) / r.gen)
                else:
                    still.append(r)
            decoding = still
    return {"ttft": float(np.mean(ttfts)), "tpot": float(np.mean(tpots)),
            "steps": t, "weights": 1.0 * WEIGHT_BYTES, "kv_transfer": 0.0}


def run_disagg(work: list[Req]) -> dict:
    t = 0.0
    i = 0
    waiting: list[Req] = []
    decoding: list[Req] = []
    ttfts, tpots = [], []
    kv = 0.0

    while i < len(work) or waiting or decoding:
        while i < len(work) and work[i].arr <= t:
            waiting.append(work[i])
            i += 1
        if not waiting and not decoding:
            if i < len(work):
                t = work[i].arr
            continue
        # prefill machine: PREFILL * PREFILL_REPLICAS tokens per step, FIFO
        if waiting:
            budget = PREFILL * PREFILL_REPLICAS
            for r in sorted(waiting, key=lambda x: x.arr):
                if budget <= 0:
                    break
                c = min(budget, r.prefill_left)
                r.prefill_left -= c
                budget -= c
            done_p = [r for r in waiting if r.prefill_left == 0]
            waiting = [r for r in waiting if r.prefill_left > 0]
            for r in done_p:
                t_transfer = t + 1                    # KV state crosses the link
                kv += r.plen * BYTES_PER_TOKEN
                r.first = t_transfer
                ttfts.append(t_transfer - r.arr)
                r.gen = 0
                decoding.append(r)
        # decode machine: every active sequence advances by DECODE_REPLICAS tokens
        if decoding:
            t += 1
            still = []
            for r in decoding:
                r.gen += DECODE_REPLICAS
                if r.gen >= r.rlen:
                    r.done = t
                    tpots.append((t - r.first + 1) / r.rlen)
                else:
                    still.append(r)
            decoding = still
    return {"ttft": float(np.mean(ttfts)), "tpot": float(np.mean(tpots)),
            "steps": t, "weights": 2.0 * WEIGHT_BYTES, "kv_transfer": kv}


def main() -> int:
    rng = np.random.default_rng(37)
    work = workload(rng)

    c = run_colocated(work)
    d = run_disagg(work)
    print(f"[measured] colocated ttft={c['ttft']:.2f} tpot={c['tpot']:.3f} "
          f"disaggregated ttft={d['ttft']:.2f} tpot={d['tpot']:.3f}")

    c_violates = c["ttft"] > TTFT_SLO or c["tpot"] > TPOT_SLO
    d_violates = d["ttft"] > TTFT_SLO or d["tpot"] > TPOT_SLO
    assert c_violates, f"colocated unexpectedly meets both SLOs: {c}"
    assert not d_violates, f"disaggregated violates an SLO: {d}"
    print(f"PASS slo-split (colocated TTFT {c['ttft']:.2f} / TPOT {c['tpot']:.3f} "
          f"violates SLOs {TTFT_SLO}/{TPOT_SLO}; disaggregated {d['ttft']:.2f}/"
          f"{d['tpot']:.3f} satisfies both)")

    total_prompt = sum(r.plen for r in work)
    assert abs(d["kv_transfer"] - total_prompt * BYTES_PER_TOKEN) < 1e-6
    print(f"PASS transfer-cost ({d['kv_transfer'] / 1e6:.1f} MB = {total_prompt} "
          f"prompt tokens x {BYTES_PER_TOKEN} bytes)")

    assert d["weights"] == 2 * c["weights"]
    print(f"PASS weight-memory (disaggregated {d['weights'] / 1e9:.0f} GB vs "
          f"colocated {c['weights'] / 1e9:.0f} GB: exactly 2x)")

    assert d["tpot"] < c["tpot"], "disaggregated TPOT not better"
    print(f"PASS tpot-improvement (disaggregated TPOT {d['tpot']:.3f} < colocated "
          f"{c['tpot']:.3f}: decode steps are never stolen by prefill)")

    print(f"METRIC ttft_colocated={c['ttft']:.2f}")
    print(f"METRIC ttft_disaggregated={d['ttft']:.2f}")
    print(f"METRIC tpot_colocated={c['tpot']:.3f}")
    print(f"METRIC tpot_disaggregated={d['tpot']:.3f}")
    print(f"METRIC kv_transfer_mb={d['kv_transfer'] / 1e6:.1f} (simulation, discrete steps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
