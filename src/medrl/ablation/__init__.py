"""Declarative ablation sweep framework with decision rules.

Provides:
- Config schema for sweep definitions and optimization strategies
- Grid search and Bayesian optimization execution
- Multi-objective optimization (accuracy vs compute)
- Progress tracking and resumption via content-addressed manifests
- Integration with eval harness for arm execution
- Statistical comparison (paired tests) and MDE-based promotion
- Visualization and report generation

Usage:
    from medrl.ablation import (
        AblationConfig,
        SweepConfig,
        SweepResult,
        run_sweep,
        analyze_results,
        generate_report,
        # Presets
        ABLATION_A1,
        ABLATION_A2,
        ABLATION_A3,
        ABLATION_A4,
        ABLATION_A5,
        ABLATION_A6,
        ABLATION_A7,
    )

    # Run a predefined sweep
    results = run_sweep(ABLATION_A1, baseline_id="eval-abc123")

    # Or define a custom sweep
    config = AblationConfig(
        name="my_sweep",
        baseline_id="eval-abc123",
        parameters={...},
        decision_rules=DecisionRules(...),
    )
"""

from medrl.ablation.analysis import (
    ArmComparison,
    ParetoFrontier,
    analyze_promotion,
    analyze_results,
    compare_arm_to_baseline,
    compute_pareto_frontier,
)
from medrl.ablation.config import (
    ABLATION_A1,
    ABLATION_A2,
    ABLATION_A3,
    ABLATION_A4,
    ABLATION_A5,
    ABLATION_A6,
    ABLATION_A7,
    AblationConfig,
    BayesianStrategy,
    DecisionRules,
    GridStrategy,
    Objective,
    OptimizationStrategy,
    Parameter,
    SweepConfig,
)
from medrl.ablation.report import (
    generate_html_report,
    generate_markdown_report,
    generate_report,
)
from medrl.ablation.sweep import (
    SweepArm,
    SweepExecutor,
    SweepResult,
    SweepStatus,
    run_sweep,
)

__all__ = [
    # Analysis
    "ABLATION_A1",
    "ABLATION_A2",
    "ABLATION_A3",
    "ABLATION_A4",
    "ABLATION_A5",
    "ABLATION_A6",
    "ABLATION_A7",
    "AblationConfig",
    "ArmComparison",
    "BayesianStrategy",
    "DecisionRules",
    "GridStrategy",
    "Objective",
    "OptimizationStrategy",
    "Parameter",
    "ParetoFrontier",
    "SweepArm",
    "SweepConfig",
    "SweepExecutor",
    "SweepResult",
    "SweepStatus",
    "analyze_promotion",
    "analyze_results",
    "compare_arm_to_baseline",
    "compute_pareto_frontier",
    "generate_html_report",
    "generate_markdown_report",
    "generate_report",
    "run_sweep",
]
