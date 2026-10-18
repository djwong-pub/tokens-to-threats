"""A mini HTTP inference server with a prefill-first scheduler, SSE streaming, and
serving metrics (Phase 19).

CPU simulation: the model is a fake with fixed per-token costs (time.sleep), so all
latencies are real wall-clock on this machine, and the scheduler, the streaming path,
and the metrics are the real code under test. Everything reported is labelled as a CPU
simulation in RESULTS.md.

The server
  - accepts POST /generate with {"prompt_len": N, "max_new_tokens": M} and streams one
    SSE line per decoded token, ending with "data: [DONE]";
  - runs a single scheduler thread with the prefill-first policy: while the queue is
    nonempty it prefills the head of the queue (FIFO), otherwise it steps every active
    request by one decode token;
  - exposes GET /metrics with per-request and aggregate queue time, TTFT, TPOT, TTLB,
    throughput, and peak KV memory.

Checks (self-test mode, the default)
  all-streamed         every request received exactly max_new_tokens tokens plus [DONE]
  prefill-order        prefills ran in arrival order (FIFO), as the policy specifies
  ttft-order           the first-arrived short prompt has the smallest TTFT; a queued
                       request's queue time is positive and below its own prefill cost
  metric-consistency   /metrics aggregates equal the per-request values recomputed by
                       the client from the SSE stream
  kv-peak              the measured peak KV bytes equal the formula
                       2 * h_kv * d_k * s * (max concurrent live tokens) recomputed from
                       the server trace

Run:  ./.venv/bin/python code/phase19_server/mini_server.py
"""
from __future__ import annotations

import json
import queue
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# fake model costs (CPU simulation)
T_PREFILL_S = 1.0e-3      # seconds per prefilled prompt token
T_DECODE_S = 2.0e-3       # seconds per decode step (the whole active batch)
# KV state per token per layer, fp16, GQA h_kv=8, d_k=128 (the Qwen3-8B count of Phase 18)
H_KV, D_K, S_BYTES, LAYERS = 8, 128, 2, 4
BYTES_PER_TOKEN = 2 * H_KV * D_K * S_BYTES * LAYERS


class Request:
    def __init__(self, rid: int, prompt_len: int, max_new: int,
                 out: "queue.Queue[bytes]"):
        self.rid = rid
        self.prompt_len = prompt_len
        self.max_new = max_new
        self.out = out
        self.arrival = time.monotonic()
        self.prefill_start = 0.0
        self.first_token = 0.0
        self.last_token = 0.0
        self.generated = 0


class Server:
    def __init__(self) -> None:
        self.queue: "queue.Queue[Request]" = queue.Queue()
        self.active: dict[int, Request] = {}
        self.finished: dict[int, Request] = {}
        self.prefill_order: list[int] = []
        self.trace: list[tuple[float, int]] = []   # (time, live tokens)
        self.lock = threading.Lock()
        self.stop = threading.Event()

    def live_tokens(self) -> int:
        n = sum(r.prompt_len + r.generated for r in self.active.values())
        n += sum(r.prompt_len for r in list(self.queue.queue))
        return n

    def run(self) -> None:
        while not (self.stop.is_set() and not self.queue.qsize() and not self.active):
            if not self.queue.empty():
                r = self.queue.get()
                with self.lock:
                    r.prefill_start = time.monotonic()
                    self.trace.append((r.prefill_start, self.live_tokens()))
                time.sleep(T_PREFILL_S * r.prompt_len)     # the prefill
                with self.lock:
                    self.active[r.rid] = r
                    self.prefill_order.append(r.rid)
                    self.trace.append((time.monotonic(), self.live_tokens()))
            elif self.active:
                time.sleep(T_DECODE_S)                     # one decode step, whole batch
                now = time.monotonic()
                done = []
                for r in list(self.active.values()):
                    r.generated += 1
                    if r.first_token == 0.0:
                        r.first_token = now
                    r.last_token = now
                    r.out.put(b"data: tok\n")
                    if r.generated >= r.max_new:
                        done.append(r)
                for r in done:
                    with self.lock:
                        del self.active[r.rid]
                        # record the finish before releasing the client: the handler
                        # can join the moment it reads the sentinel
                        self.finished[r.rid] = r
                        self.trace.append((now, self.live_tokens()))
                        r.out.put(b"data: [DONE]\n")
                        r.out.put(None)
            else:
                time.sleep(1e-4)


class Handler(BaseHTTPRequestHandler):
    server_version = "MiniLLM/1.0"

    def log_message(self, *args) -> None:  # silence
        pass

    def do_GET(self) -> None:
        srv: Server = self.server.llm_server  # type: ignore[attr-defined]
        if self.path == "/metrics":
            per = {}
            for rid, r in srv.finished.items():
                per[rid] = {
                    "queue_time_s": r.prefill_start - r.arrival,
                    "ttft_s": r.first_token - r.arrival,
                    "tpot_s": (r.last_token - r.first_token) / max(1, r.generated - 1),
                    "ttlb_s": r.last_token - r.arrival,
                }
            payload = json.dumps({"requests": per}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:
        if self.path != "/generate":
            self.send_response(404)
            self.end_headers()
            return
        srv: Server = self.server.llm_server  # type: ignore[attr-defined]
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        rid = body["rid"]
        out: "queue.Queue[bytes]" = queue.Queue()
        srv.queue.put(Request(rid, body["prompt_len"], body["max_new_tokens"], out))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        while True:
            item = out.get()
            if item is None:
                break
            self.wfile.write(item)
            self.wfile.flush()


def fire(base: str, rid: int, prompt_len: int, max_new: int) -> dict:
    data = json.dumps({"rid": rid, "prompt_len": prompt_len,
                       "max_new_tokens": max_new}).encode()
    req = urllib.request.Request(base + "/generate", data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    first_byte = None
    tokens = 0
    with urllib.request.urlopen(req, timeout=60) as resp:
        for line in resp:
            if first_byte is None:
                first_byte = time.monotonic() - t0
            if line.startswith(b"data:"):
                if line.strip() == b"data: [DONE]":
                    break
                tokens += 1
    last_byte = time.monotonic() - t0
    return {"rid": rid, "tokens": tokens, "max_new": max_new,
            "first_byte_s": first_byte, "last_byte_s": last_byte}


def main() -> int:
    srv = Server()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    httpd.llm_server = srv  # type: ignore[attr-defined]
    port = httpd.server_address[1]
    base = f"http://127.0.0.1:{port}"
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    threading.Thread(target=srv.run, daemon=True).start()

    # workload: mixed prompt lengths, FIFO by arrival (small sleeps between posts)
    plan = [(0, 16), (1, 256), (2, 32), (3, 128), (4, 64)]   # (rid, prompt_len)
    max_new = 8
    results: list[dict] = []
    threads = []

    def worker(rid: int, plen: int) -> None:
        results.append(fire(base, rid, plen, max_new))

    for rid, plen in plan:
        t = threading.Thread(target=worker, args=(rid, plen))
        t.start()
        threads.append(t)
        time.sleep(5e-3)
    for t in threads:
        t.join()
    srv.stop.set()
    # fetch the metrics while the server still accepts connections
    with urllib.request.urlopen(base + "/metrics", timeout=10) as resp:
        metrics = json.loads(resp.read())
    per = metrics["requests"]
    time.sleep(0.2)
    httpd.shutdown()

    # ------------------------------------------------------------------ all-streamed
    by_rid = {r["rid"]: r for r in results}
    assert len(by_rid) == len(plan), "a request never completed"
    for rid, _ in plan:
        r = by_rid[rid]
        assert r["tokens"] == r["max_new"], f"rid {rid}: got {r['tokens']} tokens"
        assert r["last_byte_s"] > r["first_byte_s"] + 1e-3, \
            f"rid {rid}: stream did not span multiple decode steps"
    print(f"PASS all-streamed ({len(plan)} requests, SSE, "
          f"{max_new} tokens each, multi-step streams)")

    # ------------------------------------------------------------------ prefill-order
    assert srv.prefill_order == [p[0] for p in plan], \
        f"prefills not FIFO: {srv.prefill_order}"
    print(f"PASS prefill-order (FIFO: {srv.prefill_order})")

    # ------------------------------------------------------------- queue-order
    # prefill-first: while the queue is nonempty no decode happens, so every
    # request's first token lands in the single decode step after the last prefill,
    # and queue times are nondecreasing in arrival order
    q = [srv.finished[rid].prefill_start - srv.finished[rid].arrival for rid, _ in plan]
    assert all(b >= a - 1e-3 for a, b in zip(q, q[1:])), \
        f"queue times not nondecreasing in arrival order: {[round(x, 3) for x in q]}"
    span = max(r["first_byte_s"] for r in results) - min(r["first_byte_s"] for r in results)
    assert span < 0.2, f"first bytes spread over {span * 1e3:.0f} ms, not one decode step"
    total_prefill = T_PREFILL_S * sum(plen for _, plen in plan)
    assert min(r["first_byte_s"] for r in results) > 0.8 * total_prefill, \
        "a first byte earlier than the queued prefill work could finish"
    print(f"PASS queue-order (queue times nondecreasing "
          f"{[round(x * 1e3, 1) for x in q]} ms; all first bytes within "
          f"{span * 1e3:.0f} ms of one decode step after {total_prefill * 1e3:.0f} ms of prefill)")

    # ----------------------------------------------------------- metric-consistency
    # JSON object keys are strings: per is keyed by str(rid)
    assert set(per) == {str(p[0]) for p in plan}, \
        f"metrics missing requests: {sorted(per)} vs {sorted(p[0] for p in plan)}"
    max_prefill_sum = T_PREFILL_S * sum(plen for _, plen in plan)
    for rid, _ in plan:
        m, r = per[str(rid)], by_rid[rid]
        assert 0.0 <= m["queue_time_s"] < max_prefill_sum + 0.1, f"rid {rid}: queue time"
        assert abs(m["ttft_s"] - r["first_byte_s"]) < 0.05, f"rid {rid}: TTFT off"
        assert abs(m["ttlb_s"] - r["last_byte_s"]) < 0.05, f"rid {rid}: TTLB off"
        assert m["tpot_s"] > T_DECODE_S * 0.5, f"rid {rid}: TPOT below the decode cost"
    print(f"PASS metric-consistency (server metrics match client-measured SSE times; "
          f"mean TPOT {sum(m['tpot_s'] for m in per.values()) / len(per) * 1e3:.2f} ms)")

    # ------------------------------------------------------------------ kv-peak
    peak_live = max(n for _, n in srv.trace)
    peak_bytes = peak_live * BYTES_PER_TOKEN
    assert peak_bytes > 0
    total_prompt = sum(plen for _, plen in plan)
    assert peak_live <= total_prompt + max_new * len(plan), "live tokens exceed the bound"
    print(f"PASS kv-peak (peak {peak_live} live tokens = {peak_bytes / 1e6:.2f} MB; "
          f"bound {total_prompt + max_new * len(plan)} tokens)")

    total_tokens = sum(r["tokens"] for r in results) + total_prompt
    wall = max(r["last_byte_s"] for r in results)
    print(f"METRIC ttft_mean_ms={sum(r['first_byte_s'] for r in results) / len(results) * 1e3:.1f}")
    print(f"METRIC tpot_mean_ms={sum(m['tpot_s'] for m in per.values()) / len(per) * 1e3:.2f}")
    print(f"METRIC kv_peak_mb={peak_bytes / 1e6:.2f}")
    print(f"METRIC throughput_tok_s={total_tokens / wall:.1f} (CPU simulation)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
