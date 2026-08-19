"""Load an EvalConfig from a preset name or a YAML file.

Eval configs reference cluster and judge configs *by name* (``cluster: local_2xh100``),
resolved here from ``configs/cluster/`` and ``configs/judge/``. That indirection is the
scale-portability mechanism: the same eval config runs on any topology by pointing the
cluster preset elsewhere, and no benchmark list ever duplicates hardware facts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml

from medrl.core.config import ClusterConfig, EvalConfig, JudgeConfig
from medrl.core.paths import repo_root
from medrl.eval.tasks.benchmarks import TASKS


def _resolve(pathlike: str | Path, *folders: Path) -> Path:
    candidate = Path(pathlike)
    if candidate.suffix in {".yaml", ".yml"}:
        return candidate if candidate.is_absolute() else repo_root() / candidate
    for folder in folders:
        named = folder / f"{pathlike}.yaml"
        if named.is_file():
            return named
    searched = ", ".join(str(f) for f in folders)
    raise FileNotFoundError(f"no config or preset named {pathlike!r} (searched {searched})")


def _load_named[CfgT: EvalConfig | ClusterConfig | JudgeConfig](
    section: Any, folder: Path, cls: type[CfgT]
) -> CfgT:
    """Expand a str preset / nested mapping / file path into a config object."""
    if section is None:
        raise ValueError(f"cannot load a {cls.__name__} from None")
    if isinstance(section, str):
        path = _resolve(section, folder, folder.parent)
        return cast(CfgT, cls.model_validate(yaml.safe_load(path.read_text())))
    if isinstance(section, dict):
        if len(section) == 1 and "preset" in section:
            path = _resolve(section["preset"], folder, folder.parent)
            return cast(CfgT, cls.model_validate(yaml.safe_load(path.read_text())))
        return cast(CfgT, cls.model_validate(section))
    if isinstance(section, Path):
        return cast(CfgT, cls.model_validate(yaml.safe_load(section.read_text())))
    raise ValueError(f"cannot interpret config section {section!r}")


def load_eval_config(spec: str | Path) -> EvalConfig:
    """Parse, expand presets, validate. Raises with the offending value on bad input."""
    path = _resolve(spec, repo_root() / "configs" / "eval", repo_root() / "configs")
    raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a mapping")

    if "cluster" not in raw:
        raise ValueError(f"{path}: cluster is required (topology-invariance is the point)")
    raw["cluster"] = _load_named(raw["cluster"], repo_root() / "configs" / "cluster", ClusterConfig)
    if raw.get("judge") is not None:
        raw["judge"] = _load_named(raw["judge"], repo_root() / "configs" / "judge", JudgeConfig)

    benchmarks: list[dict[str, Any]] = []
    for name in raw.get("benchmarks", []):
        if isinstance(name, str):
            if name not in TASKS:
                raise ValueError(
                    f"unknown benchmark {name!r}; known: {TASKS.names()}"
                )
            benchmarks.append(TASKS.get(name).as_benchmark_config().model_dump())
        else:
            benchmarks.append(dict(name))
    raw["benchmarks"] = benchmarks
    return EvalConfig.model_validate(raw)
