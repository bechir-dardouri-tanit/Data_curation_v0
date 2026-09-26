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
    # In production the health endpoint is the vllm process itself, which binds
    # the port AFTER start() claims it -- so the fake health server has to come
    # up inside the claim step, not before it (a pre-bound port reads as "taken
    # in the window" and gets re-drawn).
    bound: list[tuple[HTTPServer, int]] = []

    import medrl.eval.serving.lifecycle as lifecycle
    from medrl.eval.serving.vllm import free_port

    def _claim(phase: ServePhase) -> ServePhase:
        port = free_port()
        httpd = HTTPServer(("127.0.0.1", port), _OK)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        bound.append((httpd, port))
        return phase.model_copy(update={"port": port})

    monkeypatch.setattr(lifecycle, "claim_port", _claim)

    # Stand in for `vllm serve`: a sleep that dies on SIGTERM.
    monkeypatch.setattr(ServePhase, "command", lambda self: ["sleep", "600"])
    server = VLMMServer(_phase(64999), run_dir=tmp_path)
    handle = server.start()
    port = bound[0][1]
    assert handle.base_url.endswith(f":{port}/v1")
    assert server.pid_path.exists()  # the durable record for stale reaping
    assert handle.alive()  # liveness probe wired for mid-generation polling
    server.stop()
    assert not server.pid_path.exists()  # a clean stop clears the record
    bound[0][0].shutdown()


def test_stale_pid_file_is_reaped_before_start(tmp_path, monkeypatch) -> None:
    """A vLLM orphaned by a SIGKILLed runner must not OOM the next launch."""
    import subprocess

    orphan = subprocess.Popen(["sleep", "600"], start_new_session=True)
    tmp_path.joinpath("serve-policy.pid").write_text(str(orphan.pid))
    assert orphan.poll() is None

    bound: list[tuple[HTTPServer, int]] = []
    import medrl.eval.serving.lifecycle as lifecycle
    from medrl.eval.serving.vllm import free_port

    def _claim(phase: ServePhase) -> ServePhase:
        port = free_port()
        httpd = HTTPServer(("127.0.0.1", port), _OK)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        bound.append((httpd, port))
        return phase.model_copy(update={"port": port})

    monkeypatch.setattr(lifecycle, "claim_port", _claim)
    monkeypatch.setattr(ServePhase, "command", lambda self: ["sleep", "600"])

    server = VLMMServer(_phase(64999), run_dir=tmp_path)
    server.start()
    assert orphan.poll() is not None  # reaped, not raced
    server.stop()
    bound[0][0].shutdown()


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
