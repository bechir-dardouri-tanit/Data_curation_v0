"""Deterministic seeding.

Seeding alone does not make LLM inference reproducible: batch size changes the reduction
order inside RMSNorm/matmul/attention kernels, so the same prompt in a different batch can
decode differently. We therefore seed everything we can and additionally pin server-side
concurrency for decision-grade runs -- see ``medrl.eval.serving``.
"""

from __future__ import annotations

import os
import random


def seed_everything(seed: int, *, deterministic_torch: bool = False) -> None:
    """Seed python/numpy/torch. ``deterministic_torch`` also forces deterministic kernels."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        # Required for bitwise-repeatable training steps; costs throughput.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
