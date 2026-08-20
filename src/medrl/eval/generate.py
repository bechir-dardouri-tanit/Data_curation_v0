"""Generation against a running OpenAI-compatible (vLLM) server.

Design points that matter more than they look:

- **Completions live on disk, append-only, keyed by ``(benchmark, item_id, repeat)``.**
  A run interrupted at any point resumes without regenerating anything already written;
  the store is the resume boundary between the policy phase and the judge phase.
- **Per-request seeds derived from item identity** (not Python's hash), so a re-run with
  the same config samples the same completions. Reproducibility of a *sampled* eval is a
  config property, not luck.
- **Thinking control** goes through the chat template (``enable_thinking``) and, when
  prefilling, an assistant message with ``continue_final_message`` -- the two mechanisms
  vLLM actually supports, rather than string surgery on templates.
- **Constrained re-ask**: when an MCQA completion defeats extraction, the item is asked
  once more under a regex that *is* the answer contract. This is the cheap half of the
  plan's "guided decoding or judge fallback" -- it uses the policy itself, no extra
  serving phase, and the retry is recorded separately so extraction-fail statistics stay
  honest (the primary attempt is what's counted).
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from medrl.core.config import SamplingConfig, ThinkingConfig, ThinkingMode
from medrl.core.hashing import hash_text
from medrl.core.logging import get_logger
from medrl.eval.extraction import ExtractionPath, extract_mcqa
from medrl.eval.items import EvalItem

log = get_logger(__name__)

_REQUEST_TIMEOUT_S = 1200.0
_MAX_ATTEMPTS = 3
_BACKOFF_S = (5.0, 30.0)
_THINK_PREFILL = "<think>\n"


@dataclass(frozen=True)
class GenRecord:
    """One completion. Mirrored to JSONL; the grader's only input."""

    benchmark: str
    item_id: str
    repeat: int
    content: str | None
    reasoning: str | None
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    # Constrained re-ask output for items whose primary completion failed extraction.
    retry_content: str | None = None
    elapsed_s: float | None = None
    error: str | None = None


class CompletionStore:
    """Append-only JSONL with an in-memory key index for resume."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._keys: set[tuple[str, str, int]] = set()
        if self.path.exists():
            torn = 0
            with self.path.open(encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        # A torn final line is the signature of a killed run; skip it
                        # (it regenerates) but count it so silence is not assumed.
                        torn += 1
                        continue
                    if rec.get("error") and rec.get("content") is None:
                        # An errored record is NOT resume-complete: transient server
                        # failures must regenerate instead of permanently scoring 0.
                        continue
                    self._keys.add((rec["benchmark"], rec["item_id"], rec["repeat"]))
            if torn:
                log.warning("resuming: skipped %d torn lines in %s", torn, self.path)
            log.info("resuming: %d completions already in %s", len(self._keys), self.path)

    def has(self, benchmark: str, item_id: str, repeat: int) -> bool:
        return (benchmark, item_id, repeat) in self._keys

    def write(self, record: GenRecord) -> None:
        line = json.dumps(asdict(record), ensure_ascii=False)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            self._keys.add((record.benchmark, record.item_id, record.repeat))

    @classmethod
    def read_all(cls, path: Path) -> list[GenRecord]:
        records = [
            GenRecord(**json.loads(line))
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return records


def _seed_for(benchmark: str, item_id: str, repeat: int, base_seed: int) -> int:
    digest = hash_text(f"{benchmark}|{item_id}|{repeat}|{base_seed}")
    return int(digest[:8], 16) % (2**31 - 1)


def _messages_with_thinking(
    item: EvalItem, thinking: ThinkingConfig
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Item messages plus the extra_body knobs vLLM needs for the thinking mode."""
    messages = [dict(m) for m in item.messages]
    extra: dict[str, Any] = {}
    if thinking.mode is ThinkingMode.OFF:
        # Qwen-style templates accept this kwarg; templates without it ignore it
        # (vLLM filters unknown kwargs harmlessly), so it is safe to always send.
        extra["chat_template_kwargs"] = {"enable_thinking": False}
    elif thinking.prefill_think:
        messages.append({"role": "assistant", "content": _THINK_PREFILL})
        extra["continue_final_message"] = True
    return messages, extra


def _extract_ok(content: str | None, item: EvalItem) -> bool:
    if content is None:
        return False
    if item.retry_grammar is None:  # non-MCQA styles have nothing to re-ask
        return True
    return extract_mcqa(content, item.verify.letters).path is not ExtractionPath.FAILED


def generate_all(
    client: Any,
    model_ref: str,
    items: list[EvalItem],
    sampling: SamplingConfig,
    thinking: ThinkingConfig,
    store: CompletionStore,
    *,
    max_workers: int = 32,
) -> dict[str, int]:
    """Generate every missing (item, repeat) completion. Returns per-benchmark counts."""
    todo: list[tuple[EvalItem, int]] = []
    for item in items:
        for repeat in range(sampling.n_repeats):
            if not store.has(item.benchmark, item.item_id, repeat):
                todo.append((item, repeat))
    log.info("generation: %d completions to produce (%d items x %d repeats, minus resume hits)",
             len(todo), len(items), sampling.n_repeats)

    done: dict[str, int] = {}
    written = 0
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_one, client, model_ref, item, repeat, sampling, thinking): item
            for item, repeat in todo
        }
        for fut in as_completed(futures):
            record = fut.result()
            store.write(record)
            written += 1
            done[record.benchmark] = done.get(record.benchmark, 0) + 1
            if written % 200 == 0:
                rate = written / max(time.monotonic() - t0, 1e-9)
                log.info("generated %d/%d (%.1f/s)", written, len(todo), rate)
    return done


def _one(
    client: Any,
    model_ref: str,
    item: EvalItem,
    repeat: int,
    sampling: SamplingConfig,
    thinking: ThinkingConfig,
) -> GenRecord:
    messages, extra = _messages_with_thinking(item, thinking)
    kwargs: dict[str, Any] = {
        "model": model_ref,
        "messages": messages,
        "temperature": sampling.temperature,
        "top_p": sampling.top_p,
        "max_tokens": thinking.total_budget,
        "seed": _seed_for(item.benchmark, item.item_id, repeat, sampling.seed),
        "timeout": _REQUEST_TIMEOUT_S,
    }
    if extra:
        # vLLM-specific knobs travel in the request *body*, not as SDK parameters:
        # openai's create() has a closed signature and would raise TypeError before
        # any request is sent -- a failure the retry loop would happily swallow.
        kwargs["extra_body"] = extra
    t0 = time.monotonic()
    error: str | None = None
    response: Any = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            response = client.chat.completions.create(**kwargs)
            break
        except Exception as exc:
            error = f"{type(exc).__name__}: {str(exc)[:300]}"
            response = None
            if attempt < _MAX_ATTEMPTS - 1:
                time.sleep(_BACKOFF_S[attempt])

    content = reasoning = finish_reason = None
    prompt_tokens = completion_tokens = None
    if response is not None:
        choice = response.choices[0]
        content = choice.message.content
        reasoning = getattr(choice.message, "reasoning_content", None)
        finish_reason = choice.finish_reason
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)

    retry_content: str | None = None
    if error is None and not _extract_ok(content, item):
        retry_content = _constrained_retry(client, model_ref, item, sampling)

    return GenRecord(
        benchmark=item.benchmark,
        item_id=item.item_id,
        repeat=repeat,
        content=content,
        reasoning=reasoning,
        finish_reason=finish_reason,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        retry_content=retry_content,
        elapsed_s=round(time.monotonic() - t0, 3),
        error=error,
    )


def _constrained_retry(
    client: Any, model_ref: str, item: EvalItem, sampling: SamplingConfig
) -> str | None:
    """One last-resort re-ask with the whole output constrained to the answer contract."""
    try:
        response = client.chat.completions.create(
            model=model_ref,
            messages=[dict(m) for m in item.messages],
            temperature=max(sampling.temperature, 0.1),
            max_tokens=32,
            seed=_seed_for(item.benchmark, item.item_id, 9999, sampling.seed),
            timeout=300.0,
            extra_body={"structured_outputs": {"regex": item.retry_grammar or "Answer: [A-E]"}},
        )
        content = response.choices[0].message.content
        return str(content) if content is not None else None
    except Exception as exc:
        log.warning("constrained retry failed for %s/%s: %s", item.benchmark, item.item_id, exc)
        return None
