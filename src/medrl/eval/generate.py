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
import random
import threading
import time
from collections.abc import Callable
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
# Far above any healthy burst (a 55k-completion run observed zero errors);
# reached in minutes only when the server is dead or wedged.
_ABORT_ERROR_STREAK = 256
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


class GenerationAbortedError(RuntimeError):
    """The serving process died (or stopped answering) mid-generation.

    Raised as soon as the pool notices, so the failure is attributed to the
    server -- not misread hours later as a model that failed every item.
    """


class CompletionStore:
    """Append-only JSONL with an in-memory key index for resume.

    Invariants, shared with :meth:`read_all`:

    - An *errored* record (``error`` set, no content) is not resume-complete: a
      transient failure regenerates rather than permanently scoring 0.
    - A *torn* trailing line (killed run) is skipped, not fatal.
    - ``fresh=True`` discards the file instead of resuming it -- the semantics the
      CLI's ``--fresh`` promises.
    """

    def __init__(self, path: Path, *, fresh: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._keys: set[tuple[str, str, int]] = set()
        if self.path.exists():
            if fresh:
                n = sum(1 for line in self.path.open(encoding="utf-8") if line.strip())
                self.path.unlink()
                log.info("fresh run: discarded %d existing completions in %s", n, self.path)
            else:
                torn = 0
                with self.path.open(encoding="utf-8") as fh:
                    for line in fh:
                        if not line.strip():
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            torn += 1
                            continue
                        if rec.get("error") and rec.get("content") is None:
                            continue
                        self._keys.add((rec["benchmark"], rec["item_id"], rec["repeat"]))
                if torn:
                    log.warning("resuming: skipped %d torn lines in %s", torn, self.path)
                log.info("resuming: %d completions already in %s", len(self._keys), self.path)
        self.path.touch()

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
        """The grading view of the store: exactly one record per key.

        Resume appends regenerations after the attempts they replace, so a key can
        appear more than once (errored attempt, then its usable regeneration). The
        last *usable* record for a key wins; a key with no usable record keeps its
        last errored one -- grading must still see, score, and report the failure.
        Without this collapse, a resumed run grades both the stale zero and the
        regeneration and silently shifts repeat columns. Torn lines are skipped,
        mirroring :meth:`CompletionStore.__init__`.
        """
        by_key: dict[tuple[str, str, int], GenRecord] = {}
        # Streamed, not slurped: a full run's store is multi-GB of thinking
        # text, and read_text()+splitlines() triples it in RAM right before the
        # judge phase wants that memory for its own client pool. errors="replace"
        # extends the torn-line tolerance to non-UTF-8 bytes a hard crash can
        # leave behind.
        with Path(path).open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                try:
                    rec = GenRecord(**json.loads(line))
                except (json.JSONDecodeError, TypeError):
                    continue
                key = (rec.benchmark, rec.item_id, rec.repeat)
                if rec.content is not None or rec.error is None:
                    by_key[key] = rec  # usable: supersedes whatever came before it
                elif key not in by_key or by_key[key].content is None:
                    by_key[key] = rec  # errored: placeholder until a usable one lands
        return list(by_key.values())


def _seed_for(benchmark: str, item_id: str, repeat: int, base_seed: int) -> int:
    digest = hash_text(f"{benchmark}|{item_id}|{repeat}|{base_seed}")
    return int(digest[:8], 16) % (2**31 - 1)


def _normalize_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Normalize messages to ensure proper user/assistant alternation.

    Some chat templates are strict about requiring user/assistant alternation
    and may not handle system messages properly. This converts system messages
    to be part of the first user message for maximum compatibility.
    """
    if not messages:
        return messages

    normalized = []
    system_content = ""

    # Collect system messages
    for msg in messages:
        if msg["role"] == "system":
            system_content += msg["content"] + "\n\n"
        else:
            normalized.append(msg)

    # Prepend system content to the first user message
    if system_content and normalized and normalized[0]["role"] == "user":
        normalized[0] = {
            "role": "user",
            "content": system_content.strip() + "\n\n" + normalized[0]["content"]
        }
    elif system_content:
        # If there's no user message, add one
        normalized.insert(0, {"role": "user", "content": system_content.strip()})

    return normalized


def _messages_with_thinking(
    item: EvalItem, thinking: ThinkingConfig
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Item messages plus the extra_body knobs vLLM needs for the thinking mode."""
    messages = _normalize_messages([dict(m) for m in item.messages])
    extra: dict[str, Any] = {}
    if thinking.mode is ThinkingMode.OFF:
        # Qwen-style templates accept this kwarg; templates without it ignore it
        # (vLLM filters unknown kwargs harmlessly), so it is safe to always send.
        extra["chat_template_kwargs"] = {"enable_thinking": False}
    elif thinking.prefill_think:
        messages.append({"role": "assistant", "content": _THINK_PREFILL})
        # The pair is validated server-side: continuing the final (prefilled)
        # message is only legal with the generation-prompt header OFF -- vLLM
        # defaults add_generation_prompt to True and 400s on the combination.
        extra["add_generation_prompt"] = False
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
    max_workers: int = 192,
    alive: Callable[[], bool] | None = None,
) -> dict[str, int]:
    """Generate every missing (item, repeat) completion. Returns per-benchmark counts.

    ``alive`` is the serving process's liveness probe. When the server dies the
    pool stops immediately instead of draining every queued request into a dead
    socket (three attempts of backoff each, across tens of thousands of items);
    a wedged-but-alive engine is caught by the consecutive-error streak, which
    fires long before the flat per-request timeouts would.
    """
    todo: list[tuple[EvalItem, int]] = []
    for item in items:
        for repeat in range(sampling.n_repeats):
            if not store.has(item.benchmark, item.item_id, repeat):
                todo.append((item, repeat))
    log.info("generation: %d completions to produce (%d items x %d repeats, minus resume hits)",
             len(todo), len(items), sampling.n_repeats)

    done: dict[str, int] = {}
    written = 0
    consecutive_errors = 0
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_one, client, model_ref, item, repeat, sampling, thinking): item
            for item, repeat in todo
        }
        try:
            for fut in as_completed(futures):
                record = fut.result()
                store.write(record)
                written += 1
                done[record.benchmark] = done.get(record.benchmark, 0) + 1
                consecutive_errors = 0 if record.content is not None else consecutive_errors + 1
                if written % 200 == 0:
                    rate = written / max(time.monotonic() - t0, 1e-9)
                    log.info("generated %d/%d (%.1f/s)", written, len(todo), rate)
                if alive is not None and written % 64 == 0 and not alive():
                    raise GenerationAbortedError(
                        f"serving process died after {written}/{len(todo)} completions; "
                        "remaining items were not attempted"
                    )
                if consecutive_errors >= _ABORT_ERROR_STREAK:
                    # Before aborting, verify the server is actually unhealthy
                    # (not just a transient network hiccup or template error)
                    if alive is not None and alive():
                        raise GenerationAbortedError(
                            f"{consecutive_errors} consecutive failed completions after "
                            f"{written}/{len(todo)} -- server process is alive but not serving; "
                            "check for chat template errors, OOM, or GPU issues in serve log"
                        )
                    else:
                        raise GenerationAbortedError(
                            f"{consecutive_errors} consecutive failed completions after "
                            f"{written}/{len(todo)} -- server process has died; "
                            "see completions.jsonl error fields and the serve log"
                        )
        except BaseException:
            # Ctrl-C, an abort, anything: cancel what has not started rather than
            # executing the entire submitted queue on the way out.
            for fut in futures:
                fut.cancel()
            raise
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
        "presence_penalty": sampling.presence_penalty,
        "max_tokens": thinking.total_budget,
        "seed": _seed_for(item.benchmark, item.item_id, repeat, sampling.seed),
        "timeout": _REQUEST_TIMEOUT_S,
    }
    # top_k is a vLLM extension, not an OpenAI SDK parameter: it travels in the
    # body alongside the thinking knobs, or create() TypeErrors client-side.
    extra["top_k"] = sampling.top_k
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
            error = None  # a stale error string must not ride along on a success
            break
        except Exception as exc:
            error = f"{type(exc).__name__}: {str(exc)[:300]}"
            # Detect template-related errors for better diagnostics
            if "template" in str(exc).lower() or "chat" in str(exc).lower():
                error = f"TemplateError: {str(exc)[:300]}"
            response = None
            if attempt < _MAX_ATTEMPTS - 1:
                # Jittered so a shared hiccup does not re-synchronize 192 workers
                # into one thundering retry wave exactly when the server is slowest.
                time.sleep(_BACKOFF_S[attempt] * (0.5 + random.random()))

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
            messages=_normalize_messages([dict(m) for m in item.messages]),
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
