"""Gateway unit tests — no server: httpx mock transport drives AIMD + resume paths."""

from __future__ import annotations

import json

import httpx
import pytest

from medrl.curation.serving import Gateway, ServerHandle, ServingError, run_generation


def _mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


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
