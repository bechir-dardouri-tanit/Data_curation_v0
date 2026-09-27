"""THE GPU gateway for every curation stage that talks to a model.

One module owns the serving lifecycle so the pipeline has exactly one place
that knows about vLLM: server startup/health, AIMD concurrency (additive-
increase, multiplicative-decrease -- the loadtest sweep showed fixed C=192 is
the knee, but AIMD adapts when the mix changes), per-request retry/timeout,
usage accounting, and a generation manifest pinning model + vLLM version +
seed base beside every output.

Design notes
------------
* Streaming chat completions, usage from the final SSE chunk (include_usage).
* AIMD adjusts a *target* that an admission gate enforces; the gate object is
  never replaced while requests are queued (replacing a live semaphore
  orphaned its waiters: the ramp never reached the backlog and the cap was
  only loosely enforced). The gate is rebuilt per event loop -- stages drive
  ``run_generation`` under fresh ``asyncio.run`` calls (S11 once per chunk),
  and loop-bound asyncio primitives raise when contended cross-loop.
* Failures cool down with ``asyncio.sleep``: a blocking ``time.sleep`` in an
  async path froze the whole event loop once per failure.
* Thinking-mode control is a SERVER concern (the judge serves with
  ``enable_thinking: false`` defaults); the gateway never per-request hacks it.
* Resume: ``run_generation`` skips keys already present in the output JSONL --
  but only when the key's LAST record is a success. A trailing error line is
  re-run (resume exists to finish interrupted runs, not to enshrine a transient
  outage: a permanently-skipped error silently shrinks pass@k samples).
* Bounded worker pool: tasks are created per worker (~max_concurrency), not
  per job -- S12 gathers 20M+ jobs at plan scale and a Task per job is tens of
  GB before the first response lands.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from medrl.curation.schema import utcnow


class ServingError(RuntimeError):
    """Gateway-level failure: server never came up, or fatal client config."""


@dataclass(slots=True)
class ServerHandle:
    """A running vLLM server this gateway talks to."""

    model: str
    port: int
    base_url: str = field(init=False)

    def __post_init__(self) -> None:
        self.base_url = f"http://127.0.0.1:{self.port}"


def wait_healthy(handle: ServerHandle, timeout_s: int = 900) -> None:
    """Block until /health returns 200; raise ServingError on timeout."""
    deadline = time.monotonic() + timeout_s
    with httpx.Client() as client:
        while time.monotonic() < deadline:
            try:
                if client.get(f"{handle.base_url}/health", timeout=5).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(3)
    raise ServingError(f"server on :{handle.port} not healthy after {timeout_s}s")


class Gateway:
    """Async request engine with AIMD concurrency and retries."""

    def __init__(
        self,
        handle: ServerHandle,
        max_concurrency: int = 192,
        timeout_s: float = 420.0,
        max_retries: int = 3,
        seed: int = 0,
    ) -> None:
        self.handle = handle
        self.max_concurrency = max_concurrency
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.rng = random.Random(seed)
        self._inflight = 8
        self._active = 0
        self._gate: asyncio.Condition | None = None
        self._gate_loop: asyncio.AbstractEventLoop | None = None
        self._successes = 0
        self._failures = 0

    def _loop_gate(self) -> asyncio.Condition:
        """The admission gate, (re)built when the running event loop changes.

        asyncio primitives bind to the loop that first awaits them under
        contention and raise ``... is bound to a different event loop`` when
        reused on another (verified on CPython 3.12.3); S11 legitimately drives
        ``run_generation`` once per chunk, each under a fresh ``asyncio.run``.
        Only the loop-bound primitive is rebuilt; AIMD counters and the target
        below persist across loops.
        """
        loop = asyncio.get_running_loop()
        if self._gate is None or self._gate_loop is not loop:
            self._gate = asyncio.Condition()
            self._gate_loop = loop
            self._active = 0
        return self._gate

    async def _admit(self) -> None:
        """Take one of ``_inflight`` slots, waiting while the gate is full."""
        gate = self._loop_gate()
        async with gate:
            while self._active >= self._inflight:
                await gate.wait()
            self._active += 1

    async def _release(self) -> None:
        gate = self._loop_gate()
        async with gate:
            self._active -= 1
            gate.notify_all()

    def _aimd_ok(self) -> None:
        """Additive increase after a success: +1 up to max, every 8 successes.

        Only the target moves; the gate re-reads it on every release, so queued
        waiters ride the ramp (the old code replaced the semaphore object,
        which orphaned everything already queued on it).
        """
        self._successes += 1
        if self._successes % 8 == 0 and self._inflight < self.max_concurrency:
            self._inflight = min(self.max_concurrency, self._inflight + 1)

    async def _aimd_fail(self) -> None:
        """Multiplicative decrease on failure: -25%, floor 1, brief cooldown."""
        self._failures += 1
        self._inflight = max(1, int(self._inflight * 0.75))
        await asyncio.sleep(0.2)

    async def chat(
        self,
        client: httpx.AsyncClient,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.6,
        top_p: float = 0.95,
        seed: int | None = None,
        extra: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """One chat completion (non-streaming); returns {'content', 'reasoning', 'usage'}."""
        payload: dict[str, Any] = {
            "model": model or self.handle.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            **({"seed": seed} if seed is not None else {}),
            **(extra or {}),
        }
        last_exc: Exception | None = None
        await self._admit()
        try:
            for attempt in range(self.max_retries):
                try:
                    r = await client.post(
                        f"{self.handle.base_url}/v1/chat/completions",
                        json=payload,
                        timeout=self.timeout_s,
                    )
                    if r.status_code in (429, 500, 502, 503) and attempt < self.max_retries - 1:
                        await self._aimd_fail()
                        await asyncio.sleep(2**attempt)
                        continue
                    r.raise_for_status()
                    obj = r.json()
                    self._aimd_ok()
                    msg = obj["choices"][0]["message"]
                    return {
                        "content": msg.get("content") or "",
                        "reasoning": msg.get("reasoning_content"),
                        "usage": obj.get("usage", {}),
                    }
                except (httpx.HTTPError, KeyError, IndexError) as exc:
                    last_exc = exc
                    await self._aimd_fail()
        finally:
            await self._release()
        raise ServingError(f"chat failed after {self.max_retries} attempts: {last_exc}")


async def run_generation(
    gateway: Gateway,
    jobs: list[dict[str, Any]],
    out_path: Path,
    *,
    concurrency_note: str | None = None,
    client_factory: Any | None = None,
) -> dict[str, Any]:
    """Execute chat jobs with resume; append JSONL lines; return a stats dict.

    Each job: {"key": "<item_id>::<repeat>", "messages": [...], plus chat kwargs}.
    A key already in out_path is skipped -- but only if its LAST record is a
    success; a trailing ``{"key", "error"}`` line is re-run (and the retry
    overwrites the verdict when read last-line-wins downstream). Output lines
    carry the key, the response fields, and wall time. ``client_factory`` lets
    callers inject a transport (tests use MockTransport; prod uses per-task
    plain clients).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    errored: dict[str, bool] = {}
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    key = rec["key"]
                except (json.JSONDecodeError, KeyError):
                    continue
                errored[key] = "error" in rec
    done = {k for k, err in errored.items() if not err}

    todo = [j for j in jobs if j["key"] not in done]
    stats = {"total": len(jobs), "resumed": len(jobs) - len(todo), "ran": 0, "failed": 0}
    if not todo:
        stats["concurrency_note"] = concurrency_note or f"aimd->{gateway._inflight}"
        return stats

    lock = asyncio.Lock()
    started = utcnow()

    make_client = client_factory or httpx.AsyncClient

    async def one(job: dict[str, Any]) -> None:
        async with make_client() as client:
            try:
                resp = await gateway.chat(
                    client,
                    job["messages"],
                    max_tokens=job.get("max_tokens", 1024),
                    temperature=job.get("temperature", 0.6),
                    top_p=job.get("top_p", 0.95),
                    seed=job.get("seed"),
                    extra=job.get("extra"),
                )
            except ServingError as exc:
                async with lock:
                    stats["failed"] += 1
                    with open(out_path, "a") as f:
                        f.write(json.dumps({"key": job["key"], "error": str(exc)[:200]}) + "\n")
                return
        record = {"key": job["key"], **resp, "wall_s": round((utcnow() - started).total_seconds(), 3)}
        async with lock:
            stats["ran"] += 1
            with open(out_path, "a") as f:
                f.write(json.dumps(record) + "\n")

    # Worker pool, not a Task per job: gather-all materialized one coroutine +
    # Task per remaining job up front (tens of GB at S12's 20M+ jobs, before
    # the first response). Workers ~= the concurrency cap; the gateway's AIMD
    # gate does the throttling, so extra workers would only queue.
    queue: asyncio.Queue = asyncio.Queue()
    for j in todo:
        queue.put_nowait(j)
    n_workers = max(1, min(len(todo), int(getattr(gateway, "max_concurrency", 64)) or 64))
    stop = asyncio.Event()

    async def worker() -> None:
        while not stop.is_set():
            try:
                job = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                await one(job)
            except BaseException:
                stop.set()  # siblings drain and exit; the exception still propagates
                raise

    await asyncio.gather(*(worker() for _ in range(n_workers)))
    stats["concurrency_note"] = concurrency_note or f"aimd->{gateway._inflight}"
    return stats


def write_generation_manifest(out_path: Path, model: str, vllm_version: str, seed_base: int, **kw: Any) -> None:
    """Pin model + version + seed base beside the JSONL (provenance rule)."""
    manifest = {"model": model, "vllm_version": vllm_version, "seed_base": seed_base,
                "written_at": utcnow().isoformat(), **kw}
    Path(str(out_path) + ".manifest.json").write_text(json.dumps(manifest, indent=2))


__all__ = ["Gateway", "ServerHandle", "ServingError", "run_generation", "wait_healthy", "write_generation_manifest"]
