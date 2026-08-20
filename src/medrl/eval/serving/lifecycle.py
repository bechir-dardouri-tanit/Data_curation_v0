"""vLLM server process lifecycle: launch, health-poll, teardown.

The deployment *plan* (which GPUs, which argv) is pure data computed in
:mod:`medrl.eval.serving.vllm`; this module is the only place that turns a plan into a
running process. The contract is deliberately narrow: a server is healthy when
``GET /health`` returns 200, and teardown always terminates the process group so a
crashed run cannot leak GPUs onto the next one.

A 19 GB checkpoint loading over tp=2 can legitimately take many minutes, so startup
patience is generous and *all* server output is mirrored to a log file whose tail is
embedded in the failure message -- "vLLM did not start" is not a diagnosable error.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from medrl.core.logging import get_logger
from medrl.eval.serving.vllm import ServePhase

log = get_logger(__name__)

_HEALTH_TIMEOUT_S = 3600
_POLL_INTERVAL_S = 5.0
_TERM_GRACE_S = 60.0
_LOG_TAIL_LINES = 40


class ServerError(RuntimeError):
    """The server process failed to start or died mid-run."""


@dataclass
class ServerHandle:
    phase: ServePhase
    base_url: str
    log_path: Path


def _http_ok(url: str, timeout: float = 5.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return bool(resp.status == 200)
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


class VLMMServer:
    """Owns one ``vllm serve`` process for the duration of a phase."""

    def __init__(self, phase: ServePhase, *, run_dir: Path, startup_timeout_s: float = _HEALTH_TIMEOUT_S):
        self.phase = phase
        self.run_dir = Path(run_dir)
        self.startup_timeout_s = startup_timeout_s
        self.proc: subprocess.Popen[bytes] | None = None
        self.log_path = self.run_dir / f"serve-{phase.role}.log"
        self.base_url = f"http://127.0.0.1:{phase.port}"

    def start(self) -> ServerHandle:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, **self.phase.env()}
        log.info("starting %s server on GPUs %s (port %d; log %s)",
                 self.phase.role, list(self.phase.gpu_ids), self.phase.port, self.log_path)
        with self.log_path.open("ab") as logf:
            logf.write(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(self.phase.command())}\n".encode())
            logf.flush()
            self.proc = subprocess.Popen(
                self.phase.command(),
                stdout=logf,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,  # own process group: teardown kills workers too
            )
        deadline = time.monotonic() + self.startup_timeout_s
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise ServerError(
                    f"{self.phase.role} server exited rc={self.proc.returncode} during startup:\n"
                    f"{self._log_tail()}"
                )
            if _http_ok(f"{self.base_url}/health"):
                log.info("%s server healthy after %.0fs", self.phase.role, self.startup_timeout_s - (deadline - time.monotonic()))
                return ServerHandle(phase=self.phase, base_url=self.base_url + "/v1", log_path=self.log_path)
            time.sleep(_POLL_INTERVAL_S)
        self.stop()
        raise ServerError(
            f"{self.phase.role} server not healthy within {self.startup_timeout_s:.0f}s:\n{self._log_tail()}"
        )

    def stop(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            self._signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=_TERM_GRACE_S)
            except subprocess.TimeoutExpired:
                log.warning("%s server ignored SIGTERM; SIGKILL", self.phase.role)
                self._signal(signal.SIGKILL)
                self.proc.wait()
        log.info("%s server stopped (rc=%s)", self.phase.role, self.proc.returncode)
        self.proc = None

    def _signal(self, sig: int) -> None:
        """Signal the child's whole group -- but only when it *has* its own group.

        ``start()`` launches with ``start_new_session=True`` so vLLM's worker processes
        die with the API server. A child started without its own session (tests, embedders
        assigning ``proc`` directly) shares OUR process group, and signalling it would
        kill the runner -- fall back to signalling the child alone.
        """
        assert self.proc is not None
        try:
            own_group = os.getpgid(self.proc.pid) == os.getpgid(os.getpid())
        except ProcessLookupError:
            return
        try:
            if own_group:
                self.proc.send_signal(sig)
            else:
                os.killpg(self.proc.pid, sig)
        except ProcessLookupError:
            pass

    def _log_tail(self) -> str:
        try:
            lines = self.log_path.read_text(errors="replace").splitlines()
            return "\n".join(lines[-_LOG_TAIL_LINES:])
        except OSError:
            return f"(no log at {self.log_path})"

    def __enter__(self) -> ServerHandle:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()
