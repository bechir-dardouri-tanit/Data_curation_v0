"""Server lifecycle tests: the process-management contract, without GPUs.

The ``vllm serve`` command is replaced wholesale (the phase object is immutable data;
the lifecycle object only consumes a command list), so these tests exercise startup
detection, failure diagnosis, and teardown for real subprocesses.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from medrl.core.config import ModelConfig
from medrl.eval.serving.lifecycle import ServerError, VLMMServer
from medrl.eval.serving.vllm import ServePhase


def _phase(port: int) -> ServePhase:
    return ServePhase(
        role="policy", model=ModelConfig(hf_id="m"), gpu_ids=(0,),
        tensor_parallel=1, max_model_len=4096, gpu_memory_utilization=0.85, port=port,
    )


class _OK(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args: object) -> None:
        pass


def test_healthy_server_is_detected_and_stopped(tmp_path, monkeypatch) -> None:
    httpd = HTTPServer(("127.0.0.1", 0), _OK)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    # Stand in for `vllm serve`: a sleep that dies on SIGTERM.
    monkeypatch.setattr(
        ServePhase, "command",
        lambda self: ["sleep", "600"],
    )
    server = VLMMServer(_phase(port), run_dir=tmp_path)
    handle = server.start()
    assert handle.base_url.endswith(f":{port}/v1")
    server.stop()
    httpd.shutdown()


def test_dead_process_reports_log_tail(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ServePhase, "command", lambda self: ["false"])
    server = VLMMServer(_phase(64999), run_dir=tmp_path, startup_timeout_s=10)
    with pytest.raises(ServerError, match="exited rc=1"):
        server.start()


def test_timeout_stops_process_and_raises(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ServePhase, "command", lambda self: ["sleep", "600"])
    server = VLMMServer(_phase(64999), run_dir=tmp_path, startup_timeout_s=1.0)
    with pytest.raises(ServerError, match="not healthy"):
        server.start()
    assert server.proc is None  # timed-out start cleaned up after itself


def test_context_manager_stops_on_exception(tmp_path, monkeypatch) -> None:
    started: list[VLMMServer] = []
    monkeypatch.setattr(ServePhase, "command", lambda self: ["sleep", "600"])

    class _NeverHealthy(VLMMServer):
        def start(self):
            started.append(self)
            self.run_dir.mkdir(parents=True, exist_ok=True)
            import subprocess

            self.proc = subprocess.Popen(["sleep", "600"])
            return None

    server = _NeverHealthy(_phase(64999), run_dir=tmp_path)
    with pytest.raises(RuntimeError), server:
        raise RuntimeError("boom")
    assert server.proc is None  # __exit__ stopped it
