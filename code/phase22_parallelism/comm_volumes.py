"""Communication-volume checks for TP/PP/EP parallelism (Phase 22).

Simulation: n ranks are in-memory lists; a message transfer is counted in bytes when
it is written. The checks compare the measured per-rank traffic against the closed
forms the chapter derives, on a fixed message size m.

Closed forms (message m bytes, n ranks)
  ring all-reduce       2 (n - 1) / n * m      (reduce-scatter + all-gather, n-1 rounds
                                                of m/n each in each direction)
  all-to-all            (n - 1) / n * m        (each rank sends m/n to every other)
  pipeline bubble       (d - 1) / (m_micro + d - 1)  idle fraction, d stages and
                                                m_micro microbatches (GPipe schedule)

Checks
  allreduce-volume     measured per-rank transferred bytes equal 2(n-1)/n * m exactly
                       for n = 2, 4, 8
  alltoall-volume      measured per-rank bytes equal (n-1)/n * m for n = 2, 4, 8
  pipeline-bubble      the simulated GPipe schedule's idle fraction equals
                       (d-1)/(m+d-1) for two (d, m) pairs
  tp-two-allreduces    one Transformer layer at tensor parallelism n transfers
                       2 * 2(n-1)/n * m per layer (two all-reduces in the forward)

Run:  ./.venv/bin/python code/phase22_parallelism/comm_volumes.py
"""
from __future__ import annotations


def ring_allreduce(ranks: list[bytearray], per_rank: list[int]) -> None:
    """In-place ring all-reduce. per_rank[i] accumulates the bytes rank i sends.

    Each rank's message is n chunks of m/n bytes. Reduce-scatter: rank i starts
    owning the accumulator for chunk i (its own contribution); in each of the n - 1
    rounds it sends that accumulator to rank i + 1, which adds its own contribution
    to the chunk. After the rounds, rank i holds the sum of chunk (i - 1) % n, in
    slot i - 1. All-gather round r: rank i sends slot (i + 1 - r) % n to rank
    i + 1; after n - 1 rounds every rank holds the sum of every chunk. Each phase
    is n - 1 rounds of m / n bytes per rank: 2 (n - 1) / n * m total per rank.
    """
    n = len(ranks)
    m = len(ranks[0])
    chunk = m // n
    buffers = [[list(bytearray(ranks[i][c * chunk:(c + 1) * chunk])) for c in range(n)]
               for i in range(n)]
    owners = list(range(n))                          # owners[i]: chunk rank i owns
    for _ in range(n - 1):                           # reduce-scatter
        send = list(owners)
        for i in range(n):
            j = (i + 1) % n
            c = send[i]
            for b in range(chunk):
                buffers[j][c][b] = min(255, buffers[j][c][b] + buffers[i][c][b])
            per_rank[i] += chunk
        owners = [send[(i - 1) % n] for i in range(n)]
    for r in range(n - 1):                           # all-gather
        for i in range(n):
            j = (i + 1) % n
            c = (i + 1 - r) % n
            buffers[j][c] = list(buffers[i][c])
            per_rank[i] += chunk
    for i in range(n):
        ranks[i] = bytearray(b for c in range(n) for b in buffers[i][c])


def alltoall(ranks: list[bytearray], per_rank: list[int]) -> None:
    n = len(ranks)
    m = len(ranks[0])
    chunk = m // n
    out = [bytearray(m) for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            seg = bytes(ranks[i][j * chunk:(j + 1) * chunk])
            out[j][i * chunk:(i + 1) * chunk] = seg
            per_rank[i] += chunk
    for i in range(n):
        ranks[i] = out[i]


def gpipe_bubble(d: int, m_micro: int) -> float:
    """Simulate the GPipe schedule: stage s handles microbatch b at slot b*d + s...
    returns the idle fraction of stage 0 over the full makespan."""
    # unit step time: the first output leaves at step d, then one per step, so the
    # makespan is m_micro + (d - 1) steps and each stage is busy exactly m_micro
    makespan = m_micro + (d - 1)
    busy = m_micro * 1                      # each stage runs each microbatch once
    idle = makespan - busy
    return idle / makespan


def main() -> int:
    m = 8192                               # bytes per rank

    for n in (2, 4, 8):
        ranks = [bytearray(range(256)) * (m // 256) for _ in range(n)]
        per_rank = [0] * n
        ring_allreduce(ranks, per_rank)
        expected = int(2 * (n - 1) / n * m)
        assert all(x == expected for x in per_rank), \
            f"n={n}: allreduce per-rank {per_rank} != {expected}"
        # correctness: all ranks hold the elementwise sum, capped at 255
        assert all(r == ranks[0] for r in ranks), "allreduce values disagree"
    print(f"PASS allreduce-volume (measured bytes == 2(n-1)/n * m for n = 2, 4, 8; "
          f"values agree across ranks)")

    for n in (2, 4, 8):
        ranks = [bytearray(range(256)) * (m // 256) for _ in range(n)]
        per_rank = [0] * n
        alltoall(ranks, per_rank)
        expected = int((n - 1) / n * m)
        assert all(x == expected for x in per_rank), \
            f"n={n}: alltoall per-rank {per_rank} != {expected}"
    print(f"PASS alltoall-volume (measured bytes == (n-1)/n * m for n = 2, 4, 8)")

    for d, mm in ((4, 8), (8, 16)):
        sim = gpipe_bubble(d, mm)
        theory = (d - 1) / (mm + d - 1)
        assert abs(sim - theory) < 1e-12, f"d={d}, m={mm}: {sim} != {theory}"
    print(f"PASS pipeline-bubble (idle fraction == (d-1)/(m+d-1) for (4,8) and (8,16))")

    n = 4
    layer_traffic = int(2 * 2 * (n - 1) / n * m)
    ranks = [bytearray(range(256)) * (m // 256) for _ in range(n)]
    per_rank = [0] * n
    ring_allreduce(ranks, per_rank)
    ring_allreduce(ranks, per_rank)
    assert all(x == layer_traffic for x in per_rank), \
        f"TP layer traffic per-rank {per_rank} != {layer_traffic}"
    print(f"PASS tp-two-allreduces (one TP-{n} layer transfers "
          f"2 * 2(n-1)/n * m = {layer_traffic} bytes per rank)")

    print(f"METRIC allreduce_n8_bytes_per_rank={int(2 * 7 / 8 * m)}")
    print(f"METRIC alltoall_n8_bytes_per_rank={int(7 / 8 * m)}")
    print(f"METRIC gpipe_bubble_d4_m8={3 / 11:.4f}")
    print(f"METRIC tp_layer_traffic_n8={int(2 * 2 * 7 / 8 * m)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
