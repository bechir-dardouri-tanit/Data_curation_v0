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
* AIMD: on success grow the in-flight cap by +1 up to ``max_concurrency``;
  on timeout/5xx shrink it 25% and cool down. Constant 192 is the fallback
  (and the measured knee -- see the plan section 1.5).
* Thinking-mode control is a SERVER concern (the judge serves with
  ``enable_thinking: false`` defaults); the gateway never per-request hacks it.
* Resume: ``run_generation`` skips (item_id, repeat) keys already present in
  the output JSONL -- the eval generate.py convention, loadtest-proven.
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
        self._sem = asyncio.Semaphore(self._inflight)
        self._successes = 0
        self._failures = 0

    def _aimd_ok(self) -> None:
        """Additive increase after a success: +1 up to max, every 8 successes."""
        self._successes += 1
        if self._successes % 8 == 0 and self._inflight < self.max_concurrency:
            self._inflight = min(self.max_concurrency, self._inflight + 1)
            self._sem = asyncio.Semaphore(self._inflight)  # replaced; old permits drain

    def _aimd_fail(self) -> None:
        """Multiplicative decrease on failure: -25%, floor 1, brief cooldown."""
        self._failures += 1
        self._inflight = max(1, int(self._inflight * 0.75))
        self._sem = asyncio.Semaphore(self._inflight)
        time.sleep(0.2)

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
        for attempt in range(self.max_retries):
            async with self._sem:
                try:
                    r = await client.post(
                        f"{self.handle.base_url}/v1/chat/completions",
                        json=payload,
                        timeout=self.timeout_s,
                    )
                    if r.status_code in (429, 500, 502, 503) and attempt < self.max_retries - 1:
                        self._aimd_fail()
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
                    self._aimd_fail()
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
    Lines already in out_path (by key) are skipped. Output lines carry the key,
    the response fields, and wall time. ``client_factory`` lets callers inject a
    transport (tests use MockTransport; prod uses per-task plain clients).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["key"])
                except (json.JSONDecodeError, KeyError):
                    continue

    todo = [j for j in jobs if j["key"] not in done]
    stats = {"total": len(jobs), "resumed": len(jobs) - len(todo), "ran": 0, "failed": 0}
    if not todo:
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

    async def bounded(job: dict[str, Any]) -> None:
        async with gateway._sem:
            await one(job)

    await asyncio.gather(*(bounded(j) for j in todo))
    stats["concurrency_note"] = concurrency_note or f"aimd->{gateway._inflight}"
    return stats


def write_generation_manifest(out_path: Path, model: str, vllm_version: str, seed_base: int, **kw: Any) -> None:
    """Pin model + version + seed base beside the JSONL (provenance rule)."""
    manifest = {"model": model, "vllm_version": vllm_version, "seed_base": seed_base,
                "written_at": utcnow().isoformat(), **kw}
    Path(str(out_path) + ".manifest.json").write_text(json.dumps(manifest, indent=2))


__all__ = ["Gateway", "ServerHandle", "ServingError", "run_generation", "wait_healthy", "write_generation_manifest"]
