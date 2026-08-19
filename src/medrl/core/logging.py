"""Console + file logging."""

from __future__ import annotations

import logging
from pathlib import Path

_CONFIGURED = False


def setup_logging(level: int | str = logging.INFO, log_file: Path | None = None) -> None:
    """Install a rich console handler (plus an optional file handler). Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    handlers: list[logging.Handler] = []
    try:
        from rich.logging import RichHandler

        handlers.append(RichHandler(rich_tracebacks=True, show_path=False))
        fmt = "%(message)s"
    except ImportError:
        handlers.append(logging.StreamHandler())
        fmt = "%(asctime)s %(levelname)-8s %(name)s | %(message)s"

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(name)s | %(message)s"))
        handlers.append(fh)

    logging.basicConfig(level=level, format=fmt, datefmt="[%X]", handlers=handlers, force=True)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
