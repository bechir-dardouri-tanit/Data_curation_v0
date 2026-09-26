"""Configuration schema for ablation sweeps.

Defines the declarative sweep specification including:
- Parameter spaces (categorical, continuous, integer)
- Optimization strategies (grid search, Bayesian optimization)
- Multi-objective optimization (accuracy vs compute)
- Decision rules (MDE-based promotion, guardrails)
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import Field

from medrl.core.config import Frozen


class ParameterType(StrEnum):
    """Types of parameters to sweep over."""

    CATEGORICAL = "categorical"
    CONTINUOUS = "continuous"
    INTEGER = "integer"


class Parameter(Frozen):
    """One dimension of the sweep space.

    Examples:
        categorical: ["on", "off"]
        continuous: [0.0, 1.0]
        integer: [1, 10]
    """

    name: str
    type: ParameterType
    # For categorical: list of string choices
    # For continuous/integer: [min, max] inclusive range
    values: list[Any] | tuple[float, float]
    # Optional description for reports
    description: str | None = None
    # For integer types, the step size
    step: int | None = None

    @property
    def choices(self) -> list[Any]:
        """Materialized list of all values for this parameter."""
        if isinstance(self.values, tuple):
            min_val, max_val = self.values
            if self.type is ParameterType.INTEGER:
                step = self.step or 1
                return list(range(int(min_val), int(max_val) + 1, step))
            # Continuous is materialized during grid generation
            return [min_val, max_val]
        return list(self.values)


class ObjectiveType(StrEnum):
    """What we're optimizing."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class Objective(Frozen):
    """One objective in multi-objective optimization.

    metric: The metric to optimize, e.g. "accuracy", "compute_cost"
    weight: Relative importance in scalarization
    target: Optional target value for early stopping
    """

    metric: str
    type: ObjectiveType = ObjectiveType.MAXIMIZE
    weight: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0
    target: float | None = None
    # MDE threshold for decision-grade benchmarks
    mde_threshold: float | None = None


class GridStrategy(Frozen):
    """Exhaustive grid search over the parameter space.

    For continuous parameters, specifies the number of points to sample
    linearly within the range.
    """

    type: Literal["grid"] = "grid"
    # Number of points for continuous parameters
    continuous_samples: Annotated[int, Field(ge=2)] = 5
    # Maximum total arms to generate (safety limit)
    max_arms: Annotated[int, Field(ge=1)] = 1000


class BayesianStrategy(Frozen):
    """Bayesian optimization using Gaussian Process models.

    Efficiently explores high-dimensional spaces by modeling the objective
    function and balancing exploration/exploitation.
    """

    type: Literal["bayesian"] = "bayesian"
    # Number of initial random samples before optimization
    n_init: Annotated[int, Field(ge=1)] = 10
    # Number of iterations to run
    n_iter: Annotated[int, Field(ge=1)] = 50
    # Acquisition function: "ei" (expected improvement), "ucb" (upper confidence bound)
    acquisition: Literal["ei", "ucb", "poi"] = "ei"
    # Exploration parameter for UCB
    kappa: Annotated[float, Field(ge=0.0)] = 2.0
    # Expected improvement improvement parameter
    xi: Annotated[float, Field(ge=0.0)] = 0.01


OptimizationStrategy = GridStrategy | BayesianStrategy


class DecisionRules(Frozen):
    """Rules for determining promotion from sweep results.

    A sweep arm is promoted if it satisfies all criteria:
    - Wins >= min_wins on decision-grade benchmarks
    - No losses on any benchmark
    - No guardrail violations
    - Compute cost within acceptable bounds
    """

    # Minimum number of benchmark wins required
    min_wins: Annotated[int, Field(ge=1)] = 3
    # Maximum effect size (points) to consider resolved
    max_effect_points: Annotated[float, Field(ge=0.0)] = 1.0
    # Maximum regression allowed on guardrails
    guardrail_max_regression_points: Annotated[float, Field(ge=0.0)] = 0.0
    # Maximum compute overhead vs baseline (multiplier)
    max_compute_multiplier: Annotated[float, Field(ge=1.0)] = 2.0
    # Benchmarks that act as guardrails (block on regression)
    guardrails: tuple[str, ...] = ()
    # Benchmarks that only report (never gate decisions)
    reporting_only: tuple[str, ...] = ()


class SweepConfig(Frozen):
    """Configuration for the entire sweep execution."""

    # Parameters and their search spaces
    parameters: tuple[Parameter, ...] = ()
    # How to explore the space
    strategy: OptimizationStrategy = GridStrategy()
    # Objectives to optimize (multi-objective if >1)
    objectives: tuple[Objective, ...] = ()
    # Decision rules for promotion
    decision_rules: DecisionRules = DecisionRules()
    # Maximum parallel arms to execute
    max_parallel: Annotated[int, Field(ge=1)] = 4
    # Whether to resume from existing results
    resume: bool = True
    # Random seed for reproducibility
    seed: int = 0


class AblationConfig(Frozen):
    """Top-level configuration for an ablation sweep.

    A sweep is defined by:
    - A baseline eval run to compare against
    - The parameter space to explore
    - The optimization strategy
    - The decision rules

    The sweep generates a set of arms (parameter assignments), executes
    each as an eval run, and compares results against the baseline.
    """

    name: str
    baseline_id: str  # run_id of the baseline eval
    description: str | None = None
    sweep: SweepConfig = SweepConfig()
    # Optional: tags for grouping/filtering
    tags: tuple[str, ...] = ()

    @property
    def fingerprint(self) -> str:
        """Stable content hash for caching."""
        from medrl.core.hashing import hash_obj
        return hash_obj(self.model_dump(mode="json"))


# --------------------------------------------------------------------------------------
# Presets: A1-A7 ablations from the build plan
# --------------------------------------------------------------------------------------


def _make_ablation_preset(
    name: str,
    description: str,
    parameters: dict[str, Any],
    strategy: OptimizationStrategy = GridStrategy(),
    decision_rules: DecisionRules | None = None,
    objectives: tuple[Objective, ...] = (),
    tags: tuple[str, ...] = (),
) -> AblationConfig:
    """Helper to build an ablation preset."""
    param_objs = []
    for pname, pdef in parameters.items():
        if isinstance(pdef, dict):
            # Remove 'name' from pdef if present to avoid duplicate keyword
            pdef_copy = dict(pdef)
            pdef_copy.pop('name', None)
            param_objs.append(Parameter(name=pname, **pdef_copy))
        else:
            param_objs.append(pdef)

    return AblationConfig(
        name=name,
        baseline_id="",  # Filled in by the caller
        description=description,
        sweep=SweepConfig(
            parameters=tuple(param_objs),
            strategy=strategy,
            decision_rules=decision_rules or DecisionRules(),
            objectives=objectives,
        ),
        tags=tags,
    )


# A1: Thinking mode ablation
ABLATION_A1_TEMPLATE = _make_ablation_preset(
    name="A1_thinking_mode",
    description="Effect of thinking mode on reasoning benchmarks",
    parameters={
        "thinking.mode": {
            "name": "thinking.mode",
            "type": ParameterType.CATEGORICAL,
            "values": ["on", "off"],
            "description": "Whether thinking mode is enabled",
        },
        "thinking.think_budget": {
            "name": "thinking.think_budget",
            "type": ParameterType.INTEGER,
            "values": (0, 16384),
            "step": 2048,
            "description": "Thinking token budget",
        },
    },
    tags=("reasoning", "thinking"),
)


# A2: Sampling strategy ablation
ABLATION_A2_TEMPLATE = _make_ablation_preset(
    name="A2_sampling_strategy",
    description="Impact of sampling parameters on reliability",
    parameters={
        "sampling.temperature": {
            "name": "sampling.temperature",
            "type": ParameterType.CATEGORICAL,
            "values": [0.0, 0.3, 0.6, 1.0],
            "description": "Sampling temperature",
        },
        "sampling.n_repeats": {
            "name": "sampling.n_repeats",
            "type": ParameterType.CATEGORICAL,
            "values": [1, 4, 8, 16],
            "description": "Number of repeats for bootstrap CI",
        },
    },
    tags=("sampling", "reliability"),
)


# A3: Judge strength ablation
ABLATION_A3_TEMPLATE = _make_ablation_preset(
    name="A3_judge_strength",
    description="Effect of judge consistency on rubric scoring",
    parameters={
        "judge.tier": {
            "name": "judge.tier",
            "type": ParameterType.CATEGORICAL,
            "values": ["fast", "strong"],
            "description": "Judge model tier",
        },
        "judge.n_consistency": {
            "name": "judge.n_consistency",
            "type": ParameterType.INTEGER,
            "values": (1, 5),
            "step": 1,
            "description": "Number of consistency checks",
        },
    },
    tags=("judge", "rubric"),
)


# A4: Context length ablation
ABLATION_A4_TEMPLATE = _make_ablation_preset(
    name="A4_context_length",
    description="Performance vs context length trade-off",
    parameters={
        "cluster.max_model_len": {
            "name": "cluster.max_model_len",
            "type": ParameterType.CATEGORICAL,
            "values": [8192, 16384, 32768, 65536],
            "description": "Maximum model context length",
        },
    },
    objectives=(
        Objective(metric="accuracy", weight=0.7),
        Objective(metric="compute_cost", type=ObjectiveType.MINIMIZE, weight=0.3),
    ),
    tags=("context", "compute"),
)


# A5: Top-k and top-p ablation
ABLATION_A5_TEMPLATE = _make_ablation_preset(
    name="A5_nucleus_sampling",
    description="Top-k and top-p interaction on quality",
    parameters={
        "sampling.top_k": {
            "name": "sampling.top_k",
            "type": ParameterType.INTEGER,
            "values": (1, 100),
            "step": 10,
            "description": "Top-k sampling parameter",
        },
        "sampling.top_p": {
            "name": "sampling.top_p",
            "type": ParameterType.CATEGORICAL,
            "values": [0.8, 0.9, 0.95, 0.99, 1.0],
            "description": "Nucleus sampling threshold",
        },
    },
    tags=("sampling", "quality"),
)


# A6: Multi-objective ablation (accuracy vs compute)
ABLATION_A6_TEMPLATE = _make_ablation_preset(
    name="A6_accuracy_compute",
    description="Pareto frontier of accuracy vs compute cost",
    parameters={
        "sampling.temperature": {
            "name": "sampling.temperature",
            "type": ParameterType.CATEGORICAL,
            "values": [0.0, 0.3, 0.6],
        },
        "sampling.n_repeats": {
            "name": "sampling.n_repeats",
            "type": ParameterType.CATEGORICAL,
            "values": [4, 8, 16],
        },
        "thinking.think_budget": {
            "name": "thinking.think_budget",
            "type": ParameterType.INTEGER,
            "values": (4096, 8192, 16384),
            "step": 4096,
        },
    },
    objectives=(
        Objective(metric="accuracy", weight=0.6),
        Objective(metric="compute_cost", type=ObjectiveType.MINIMIZE, weight=0.4),
    ),
    strategy=BayesianStrategy(n_init=10, n_iter=50),
    tags=("multi-objective", "compute"),
)


# A7: Guardrail ablation (catastrophic forgetting)
ABLATION_A7_TEMPLATE = _make_ablation_preset(
    name="A7_guardrails",
    description="Impact of guardrails on promotion decisions",
    parameters={
        "thinking.strict_incomplete": {
            "name": "thinking.strict_incomplete",
            "type": ParameterType.CATEGORICAL,
            "values": [True, False],
            "description": "Count incomplete responses as errors",
        },
    },
    decision_rules=DecisionRules(
        min_wins=2,
        guardrail_max_regression_points=0.0,
        guardrails=("medmcqa", "healthbench_hard"),
    ),
    tags=("guardrails", "forgetting"),
)


def ABLATION_A1(baseline_id: str) -> AblationConfig:  # noqa: N802 - Intentional uppercase for preset function
    """A1: Thinking mode ablation with baseline."""
    return ABLATION_A1_TEMPLATE.model_copy(update={"baseline_id": baseline_id})


def ABLATION_A2(baseline_id: str) -> AblationConfig:  # noqa: N802 - Intentional uppercase for preset function
    """A2: Sampling strategy ablation with baseline."""
    return ABLATION_A2_TEMPLATE.model_copy(update={"baseline_id": baseline_id})


def ABLATION_A3(baseline_id: str) -> AblationConfig:  # noqa: N802 - Intentional uppercase for preset function
    """A3: Judge strength ablation with baseline."""
    return ABLATION_A3_TEMPLATE.model_copy(update={"baseline_id": baseline_id})


def ABLATION_A4(baseline_id: str) -> AblationConfig:  # noqa: N802 - Intentional uppercase for preset function
    """A4: Context length ablation with baseline."""
    return ABLATION_A4_TEMPLATE.model_copy(update={"baseline_id": baseline_id})


def ABLATION_A5(baseline_id: str) -> AblationConfig:  # noqa: N802 - Intentional uppercase for preset function
    """A5: Nucleus sampling ablation with baseline."""
    return ABLATION_A5_TEMPLATE.model_copy(update={"baseline_id": baseline_id})


def ABLATION_A6(baseline_id: str) -> AblationConfig:  # noqa: N802 - Intentional uppercase for preset function
    """A6: Accuracy vs compute Pareto sweep with baseline."""
    return ABLATION_A6_TEMPLATE.model_copy(update={"baseline_id": baseline_id})


def ABLATION_A7(baseline_id: str) -> AblationConfig:  # noqa: N802 - Intentional uppercase for preset function
    """A7: Guardrail sweep with baseline."""
    return ABLATION_A7_TEMPLATE.model_copy(update={"baseline_id": baseline_id})
