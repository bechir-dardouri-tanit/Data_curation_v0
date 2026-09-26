"""Async load generator for the concurrency sweep.

Fires N chat-completion requests through a vLLM OpenAI server with at most
C in flight (asyncio semaphore), streaming so time-to-first-token is
measured per request. Emits one JSON aggregate line per invocation.

Also samples GPU utilization/memory every second for the duration of the
cell and reports them alongside request statistics.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import threading
import time

import aiohttp


class GpuSampler:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.util: list[float] = []
        self.mem: list[float] = []

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10,
                ).stdout.strip()
                utils, mems = [], []
                for line in out.splitlines():
                    u, m = line.split(",")
                    utils.append(float(u))
                    mems.append(float(m))
                if utils:
                    self.util.append(sum(utils) / len(utils))
                    self.mem.append(max(mems))
            except Exception:
                pass
            self._stop.wait(1.0)

    def stop(self) -> dict:
        self._stop.set()
        self._thread.join(timeout=5)
        return {
            "gpu_util_avg": round(sum(self.util) / len(self.util), 1) if self.util else None,
            "gpu_util_max": max(self.util) if self.util else None,
            "vram_peak_mb": int(max(self.mem)) if self.mem else None,
        }


def pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))
    return s[k]


def build_workload(pool: dict, n_requests: int, mix: str, long_only: bool) -> list[dict]:
    if long_only:
        entries = pool["long"]
        step = max(1, len(entries) // n_requests)
        return [entries[(i * step) % len(entries)] for i in range(n_requests)]
    # Mix pattern over an 8-slot cycle: short*1 medium*4 long*2 open*1
    parts = {"short": "short", "medium": "medium", "long": "long", "open": "open"}
    cycle: list[str] = []
    want = {"short": 1, "medium": 4, "long": 2, "open": 1}
    if mix != "default":
        w: dict[str, int] = {}
        for piece in mix.split(","):
            k, v = piece.split("=")
            w[k] = int(v)
        want = w
    for bucket, weight in want.items():
        cycle.extend([bucket] * weight)
    buckets = {b: list(pool[b]) for b in set(cycle)}
    idx = {b: 0 for b in buckets}
    out = []
    for i in range(n_requests):
        b = cycle[i % len(cycle)]
        entries = buckets[b]
        out.append(entries[idx[b] % len(entries)])
        idx[b] += 1
    return out


async def one_request(session, url: str, entry: dict, endpoint: str, level: int,
                      max_tokens: int, timeout_s: int, extra_body: dict, results: list,
                      ignore_eos: bool = False):
    t0 = time.perf_counter()
    rec = {"ok": False, "ttft": None, "latency": None, "out_tokens": 0, "in_tokens": 0}
    try:
        payload_base = {
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
            **extra_body,
        }
        if endpoint == "chat":
            payload_base["messages"] = entry["messages"]
            payload_base["stream_options"] = {"include_usage": True}
        else:
            payload_base["prompt"] = entry["prompt"]
            payload_base["stream_options"] = {"include_usage": True}
            if ignore_eos:
                payload_base["ignore_eos"] = True
        timeout = aiohttp.ClientTimeout(total=timeout_s)
        async with session.post(url, json=payload_base, timeout=timeout) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"HTTP {resp.status}: {body[:200]}")
            buf = ""
            usage = None
            async for chunk in resp.content:
                now = time.perf_counter()
                # SSE frames are separated by blank lines; parse data payloads.
                buf += chunk.decode("utf-8", errors="replace")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if not line.startswith("data:") or line == "data: [DONE]":
                        continue
                    try:
                        obj = json.loads(line[5:])
                    except json.JSONDecodeError:
                        continue
                    choices = obj.get("choices") or []
                    # Chat streams use delta.content/reasoning_content; raw
                    # completion streams put text directly on each choice.
                    has_token = any(
                        (c.get("delta") or {}).get("content")
                        or (c.get("delta") or {}).get("reasoning_content")
                        or c.get("text")
                        for c in choices
                    )
                    if has_token and rec["ttft"] is None:
                        rec["ttft"] = now - t0
                    if obj.get("usage"):
                        usage = obj["usage"]
            rec["latency"] = time.perf_counter() - t0
            if usage:
                rec["out_tokens"] = usage.get("completion_tokens", 0)
                rec["in_tokens"] = usage.get("prompt_tokens", 0)
            rec["ok"] = True
    except Exception as e:
        rec["error"] = str(e)[:300]
        rec["latency"] = time.perf_counter() - t0
    results.append(rec)


async def run_cell(args, pool: dict) -> dict:
    workload = build_workload(pool, args.requests, args.mix, args.long_only)
    decode_only = bool(getattr(args, "decode_only", False))
    if decode_only or args.endpoint_resolved == "completions":
        url = f"{args.url}/v1/completions"
    else:
        url = f"{args.url}/v1/chat/completions"
    endpoint = "completions" if (decode_only or args.endpoint_resolved == "completions") else "chat"
    results: list[dict] = []
    gpu = GpuSampler()

    conn = aiohttp.TCPConnector(limit=args.level + 8)
    sem = asyncio.Semaphore(args.level)

    async def gated(entry, i):
        async with sem:
            await one_request(session, url, entry, endpoint,
                              args.level, args.max_tokens, args.timeout_req,
                              {} if decode_only else args.extra_body, results,
                              ignore_eos=decode_only)
            return i

    async with aiohttp.ClientSession(connector=conn) as session:
        gpu.start()
        wall_start = time.perf_counter()
        await asyncio.gather(*[gated(e, i) for i, e in enumerate(workload)])
        wall = time.perf_counter() - wall_start
        gpu_stats = gpu.stop()

    ok_rows = [r for r in results if r["ok"]]
    errs = [r for r in results if not r["ok"]]
    ttfts = [r["ttft"] for r in ok_rows if r["ttft"] is not None]
    lats = [r["latency"] for r in ok_rows]
    out_tok = sum(r["out_tokens"] for r in ok_rows)
    in_tok = sum(r["in_tokens"] for r in ok_rows)

    agg = {
        "level": args.level,
        "tag": args.tag,
        "requests": len(results),
        "ok": len(ok_rows),
        "errors": len(errs),
        "wall_s": round(wall, 2),
        "out_tokens_total": out_tok,
        "in_tokens_total": in_tok,
        "out_tok_per_s": round(out_tok / wall, 1) if wall > 0 else None,
        "in_tok_per_s": round(in_tok / wall, 1) if wall > 0 else None,
        "req_per_min": round(len(ok_rows) / wall * 60, 1) if wall > 0 else None,
        "ttft_p50_ms": round(pct(ttfts, 50) * 1000, 1) if ttfts else None,
        "ttft_p95_ms": round(pct(ttfts, 95) * 1000, 1) if ttfts else None,
        "lat_p50_s": round(pct(lats, 50), 2) if lats else None,
        "lat_p95_s": round(pct(lats, 95), 2) if lats else None,
        "avg_completion_tokens": round(out_tok / len(ok_rows), 1) if ok_rows else 0,
        "endpoint": args.endpoint_resolved,
        "mode": "decode" if bool(getattr(args, "decode_only", False)) else "chat",
        "long_only": args.long_only,
        **gpu_stats,
    }
    if errs:
        agg["sample_error"] = errs[0].get("error", "")[:300]
    return agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--pool", required=True)
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--requests", type=int, default=None)
    ap.add_argument("--mix", default="default")
    ap.add_argument("--long-only", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--timeout-req", type=int, default=420)
    ap.add_argument("--thinking-off", action="store_true",
                    help="send enable_thinking=false (Qwen3+ chat templates)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--decode-only", action="store_true",
                    help="raw /v1/completions with ignore_eos: forced 256-token "
                         "decodes for steady-state batch-scaling throughput")
    ap.add_argument("--out-append")
    args = ap.parse_args()

    if args.requests is None:
        args.requests = min(2 * args.level + 8, 400)

    with open(args.pool) as f:
        pool = json.load(f)["pool"]

    args.extra_body = (
        {"chat_template_kwargs": {"enable_thinking": False}} if args.thinking_off else {}
    )
    args.endpoint_resolved = "chat"

    async def probe():
        """Preflight: pick endpoint + kwarg policy from a single cheap request."""
        async with aiohttp.ClientSession() as s:
            test_entry = pool["short"][0]
            attempts = [
                ("chat+kwargs", "chat", args.extra_body),
                ("chat-plain", "chat", {}),
                ("completions", "completions", {}),
            ]
            for name, ep, body in attempts:
                if name.endswith("kwargs") and not args.extra_body:
                    continue
                try:
                    payload = {
                        "max_tokens": 8, "temperature": 0.0, "stream": False, **body,
                    }
                    if ep == "chat":
                        payload["messages"] = test_entry["messages"]
                    else:
                        payload["prompt"] = test_entry["prompt"]
                    async with s.post(f"{args.url}/v1/{'chat/completions' if ep == 'chat' else 'completions'}",
                                      json=payload,
                                      timeout=aiohttp.ClientTimeout(total=60)) as r:
                        await r.read()
                        if r.status == 200:
                            return ep, body
                except Exception:
                    continue
        raise RuntimeError("preflight failed on all endpoint modes")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        ep, accepted_body = loop.run_until_complete(probe())
    except RuntimeError as e:
        print(json.dumps({"level": args.level, "tag": args.tag, "fatal": str(e)}))
        raise SystemExit(2)
    args.endpoint_resolved = ep
    # If the server rejected the thinking kwarg (accepted chat without it) or
    # forced the raw-completions endpoint, stream plain so the cell can run.
    if ep != "chat" or accepted_body == {}:
        args.extra_body = {}

    agg = loop.run_until_complete(run_cell(args, pool))

    if args.out_append:
        with open(args.out_append, "a") as f:
            f.write(json.dumps(agg) + "\n")
    print(json.dumps(agg))


if __name__ == "__main__":
    main()
