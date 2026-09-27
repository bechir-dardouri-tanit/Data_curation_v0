"""Gateway unit tests — no server: httpx mock transport drives AIMD + resume paths.

The sync ``httpx.MockTransport`` handlers below never yield, so tasks run to
completion serially and hide interleaving bugs (the run_generation deadlock at
issue escaped exactly that way). The ``_YieldingTransport`` tests suspend the
task at the transport, forcing real concurrency through the AIMD gate.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from medrl.curation.serving import Gateway, ServerHandle, ServingError, run_generation


def _mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class _YieldingTransport(httpx.AsyncBaseTransport):
    """Async transport that truly suspends before answering, and counts in-flight."""

    def __init__(self, handler) -> None:
        self.handler = handler
        self.inflight = 0
        self.peak = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.inflight += 1
        self.peak = max(self.peak, self.inflight)
        try:
            await asyncio.sleep(0)  # yield: forces tasks to interleave
            return self.handler(request)
        finally:
            self.inflight -= 1


@pytest.mark.asyncio
async def test_chat_success_and_usage():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["messages"][0]["content"] == "hi"
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "hello", "reasoning_content": None}}],
            "usage": {"completion_tokens": 5},
        })

    gw = Gateway(ServerHandle(model="m", port=1), max_concurrency=4)
    out = await gw.chat(_mock_client(handler), [{"role": "user", "content": "hi"}])
    assert out["content"] == "hello" and out["usage"]["completion_tokens"] == 5


@pytest.mark.asyncio
async def test_chat_retries_on_503_then_succeeds():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 2:
            return httpx.Response(503)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}], "usage": {}})

    gw = Gateway(ServerHandle(model="m", port=1), max_retries=3)
    out = await gw.chat(_mock_client(handler), [{"role": "user", "content": "x"}])
    assert out["content"] == "ok" and calls["n"] == 2


@pytest.mark.asyncio
async def test_chat_fails_after_retries():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    gw = Gateway(ServerHandle(model="m", port=1), max_retries=2)
    with pytest.raises(ServingError):
        await gw.chat(_mock_client(handler), [{"role": "user", "content": "x"}])


@pytest.mark.asyncio
async def test_run_generation_resumes_and_records(tmp_path):
    out = tmp_path / "gen.jsonl"
    out.write_text(json.dumps({"key": "a::0", "content": "cached"}) + "\n")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "fresh"}}], "usage": {}})

    factory = lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))  # noqa: E731
    gw = Gateway(ServerHandle(model="m", port=1), max_concurrency=2)
    jobs = [
        {"key": "a::0", "messages": [{"role": "user", "content": "q"}]},  # resumed, not called
        {"key": "b::0", "messages": [{"role": "user", "content": "q"}]},
    ]
    stats = await run_generation(gw, jobs, out, client_factory=factory)
    assert stats["resumed"] == 1 and stats["ran"] == 1 and stats["failed"] == 0
    lines = [json.loads(l) for l in out.read_text().splitlines()]
    assert lines[0]["content"] == "cached" and lines[1]["content"] == "fresh"
    assert "error" not in lines[1]


@pytest.mark.asyncio
async def test_run_generation_records_errors_without_dying(tmp_path):
    out = tmp_path / "gen.jsonl"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    factory = lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))  # noqa: E731
    gw = Gateway(ServerHandle(model="m", port=1), max_retries=1)
    stats = await run_generation(
        gw, [{"key": "a::0", "messages": [{"role": "user", "content": "q"}]}], out,
        client_factory=factory,
    )
    assert stats["failed"] == 1
    rec = json.loads(out.read_text().splitlines()[0])
    assert "error" in rec


@pytest.mark.asyncio
async def test_run_generation_completes_under_interleaving_without_deadlock(tmp_path):
    """Regression: bounded() + chat() each acquired the SAME semaphore, so as
    soon as requests interleaved (any real network I/O yields), cap/2 tasks
    held one permit each and blocked on the second -- every permit held by a
    task waiting on a permit. 40 jobs at cap 8 hung forever; the whole gate
    now lives in one place (chat's admission slot)."""
    out = tmp_path / "gen.jsonl"
    transport = _YieldingTransport(
        lambda request: httpx.Response(
            200, json={"choices": [{"message": {"content": "ok"}}], "usage": {}}
        )
    )
    factory = lambda: httpx.AsyncClient(transport=transport)  # noqa: E731
    gw = Gateway(ServerHandle(model="m", port=1), max_concurrency=8)
    jobs = [{"key": f"j{i}::0", "messages": [{"role": "user", "content": "q"}]} for i in range(40)]
    stats = await asyncio.wait_for(
        run_generation(gw, jobs, out, client_factory=factory), timeout=30
    )
    assert stats["ran"] == 40 and stats["failed"] == 0
    assert transport.peak <= 8, "AIMD cap must hold under true interleaving"


@pytest.mark.asyncio
async def test_gateway_reusable_across_event_loops(tmp_path):
    """Regression: S11 calls asyncio.run(run_generation(...)) once per chunk
    with one Gateway. An asyncio primitive contended on loop 1 raises 'is bound
    to a different event loop' when contended on loop 2 (CPython 3.12); the
    gate must be per-loop."""
    transport = _YieldingTransport(
        lambda request: httpx.Response(
            200, json={"choices": [{"message": {"content": "ok"}}], "usage": {}}
        )
    )
    factory = lambda: httpx.AsyncClient(transport=transport)  # noqa: E731
    gw = Gateway(ServerHandle(model="m", port=1), max_concurrency=4)

    def make_jobs(tag: str) -> list[dict]:
        return [
            {"key": f"{tag}{i}::0", "messages": [{"role": "user", "content": "q"}]}
            for i in range(50)
        ]

    for chunk in ("a", "b", "c"):  # three asyncio.run passes over one gateway
        stats = await asyncio.wait_for(
            run_generation(gw, make_jobs(chunk), tmp_path / "gen.jsonl", client_factory=factory),
            timeout=30,
        )
        assert stats["ran"] == 50


@pytest.mark.asyncio
async def test_run_generation_reruns_errored_keys(tmp_path):
    """Regression: resume used to treat ANY existing key as done, so one
    transient outage permanently shrank pass@k samples (the error line was
    never retried). A trailing error record must re-run."""
    out = tmp_path / "gen.jsonl"
    out.write_text(json.dumps({"key": "a::0", "error": "boom"}) + "\n")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "fresh"}}], "usage": {}})

    factory = lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))  # noqa: E731
    gw = Gateway(ServerHandle(model="m", port=1))
    stats = await run_generation(
        gw, [{"key": "a::0", "messages": [{"role": "user", "content": "q"}]}], out,
        client_factory=factory,
    )
    assert stats["resumed"] == 0 and stats["ran"] == 1
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert records[-1]["content"] == "fresh" and "error" not in records[-1]


@pytest.mark.asyncio
async def test_run_generation_resumes_succeeded_keys(tmp_path):
    out = tmp_path / "gen.jsonl"
    out.write_text(json.dumps({"key": "a::0", "content": "cached"}) + "\n")
    out.open("a").write(json.dumps({"key": "b::0", "error": "older failure"}) + "\n")
    out.open("a").write(json.dumps({"key": "b::0", "content": "retried-ok"}) + "\n")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "x"}}], "usage": {}})

    factory = lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))  # noqa: E731
    gw = Gateway(ServerHandle(model="m", port=1))
    stats = await run_generation(
        gw,
        [
            {"key": "a::0", "messages": [{"role": "user", "content": "q"}]},
            {"key": "b::0", "messages": [{"role": "user", "content": "q"}]},  # last line: success
        ],
        out,
        client_factory=factory,
    )
    assert stats["resumed"] == 2 and stats["ran"] == 0
