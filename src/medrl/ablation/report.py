"""Ablation report generation.

Provides:
- HTML report generation with interactive plots
- Markdown summary reports
- CSV export for further analysis
- Visualization of sweep results
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from medrl.ablation.analysis import (
    analyze_results,
)
from medrl.ablation.sweep import SweepResult
from medrl.core.logging import get_logger

log = get_logger(__name__)


def generate_markdown_report(
    results: SweepResult,
    baseline_results: dict[str, dict[str, Any]],
    output_path: Path,
) -> Path:
    """Generate a markdown summary of sweep results."""
    lines = [
        f"# Ablation Sweep Report: {results.config.name}",
        "",
        f"**Sweep ID**: {results.sweep_id}",
        f"**Baseline**: {results.baseline_id}",
        f"**Started**: {results.started_at}",
        f"**Finished**: {results.finished_at}",
        "",
        "## Configuration",
        "",
        f"**Description**: {results.config.description or 'No description'}",
        "",
        "### Parameters",
        "",
    ]

    for param in results.config.sweep.parameters:
        lines.append(f"- **{param.name}**: {param.type.value}")
        if param.description:
            lines.append(f"  - {param.description}")

    lines.extend([
        "",
        "### Decision Rules",
        "",
        f"- Min wins: {results.config.sweep.decision_rules.min_wins}",
        f"- Max effect: {results.config.sweep.decision_rules.max_effect_points}pt",
        f"- Guardrails: {', '.join(results.config.sweep.decision_rules.guardrails) or 'none'}",
        "",
        "## Results Summary",
        "",
        f"- Total arms: {len(results.arms)}",
        f"- Completed: {len(results.completed_arms)}",
        f"- Failed: {len(results.failed_arms)}",
        "",
    ])

    if results.promotion_decision:
        decision = results.promotion_decision
        lines.extend([
            "## Promotion Decision",
            "",
            f"**Status**: {'PROMOTED' if decision['promoted'] else 'NOT PROMOTED'}",
            f"**Best arm**: {decision.get('best_arm_id', 'none')}",
            "",
            f"**Wins** ({len(decision['wins'])}): {', '.join(decision['wins']) or 'none'}",
            f"**Losses** ({len(decision['losses'])}): {', '.join(decision['losses']) or 'none'}",
            "",
            "### Reasons",
            "",
        ])
        for reason in decision.get("reasons", []):
            lines.append(f"- {reason}")

    lines.extend([
        "",
        "## Arm Results",
        "",
        "| Arm | Status | Points (mean) | Significant Wins | Significant Losses |",
        "|-----|--------|---------------|-------------------|--------------------|",
    ])

    analysis = analyze_results(results, baseline_results)

    for arm_result in results.completed_arms:
        arm_id = arm_result.arm_id
        comp = analysis["comparisons"].get(arm_id)
        wins = comp["significant_wins"] if comp else []
        losses = comp["significant_losses"] if comp else []

        avg_points = np.mean([b["points"] for b in arm_result.benchmark_results.values()])

        lines.append(
            f"| {arm_id[:12]}... | {arm_result.status.value} | "
            f"{avg_points:.1f} | {len(wins)} | {len(losses)} |"
        )

    # Pareto frontier if multi-objective
    if len(results.config.sweep.objectives) >= 2:
        lines.extend([
            "",
            "## Pareto Frontier",
            "",
            "Optimal arms across objectives:",
            "",
        ])
        for arm_id, values in analysis.get("pareto_frontier", []):
            obj_str = ", ".join(f"{k}={v:.1f}" for k, v in values.items())
            lines.append(f"- {arm_id}: {obj_str}")

    content = "\n".join(lines)
    output_path.write_text(content)
    log.info("markdown report written to %s", output_path)
    return output_path


def generate_csv_export(
    results: SweepResult,
    baseline_results: dict[str, dict[str, Any]],
    output_path: Path,
) -> Path:
    """Export sweep results to CSV for further analysis."""
    import csv

    analysis = analyze_results(results, baseline_results)

    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)

        # Header
        header = [
            "arm_id",
            "status",
            "run_id",
            "mean_points",
            "significant_wins",
            "significant_losses",
            "scalarized_score",
        ]

        # Add benchmark columns
        if results.completed_arms:
            first_arm = next(iter(results.completed_arms))
            for bench in first_arm.benchmark_results:
                header.extend([f"{bench}_points", f"{bench}_delta"])

        writer.writerow(header)

        # Rows
        for arm_result in results.completed_arms:
            arm_id = arm_result.arm_id
            comp = analysis["comparisons"].get(arm_id)
            if comp:
                wins = ",".join(comp["significant_wins"])
                losses = ",".join(comp["significant_losses"])
                score = comp["scalarized_score"]
            else:
                wins = ""
                losses = ""
                score = ""

            mean_points = np.mean([b["points"] for b in arm_result.benchmark_results.values()])

            row = [
                arm_id,
                arm_result.status.value,
                arm_result.run_id or "",
                f"{mean_points:.3f}",
                wins,
                losses,
                f"{score}" if score else "",
            ]

            # Benchmark values
            for bench in first_arm.benchmark_results:
                if bench in arm_result.benchmark_results:
                    row.append(str(arm_result.benchmark_results[bench]["points"]))
                    if comp and bench in comp["delta_points"]:
                        row.append(f"{comp['delta_points'][bench]:.3f}")
                    else:
                        row.append("")
                else:
                    row.extend(["", ""])

            writer.writerow(row)

    log.info("CSV export written to %s", output_path)
    return output_path


def _generate_plots_html(results: SweepResult) -> str:
    """Generate HTML for interactive plots."""
    # Collect data for plotting
    arm_data = []
    for arm_result in results.completed_arms:
        arm_id = arm_result.arm_id
        for bench, data in arm_result.benchmark_results.items():
            arm_data.append({
                "arm": arm_id,
                "benchmark": bench,
                "points": data["points"],
                "ci_low": data["ci_low"],
                "ci_high": data["ci_high"],
            })

    # Serialize data for JS
    data_json = json.dumps(arm_data)

    html = f"""
    <div class="plot-container">
        <canvas id="benchmarkPlot"></canvas>
    </div>
    <script>
        const data = {data_json};

        // Group by benchmark
        const benchmarks = [...new Set(data.map(d => d.benchmark))];
        const arms = [...new Set(data.map(d => d.arm))];

        // Simple bar chart visualization
        const ctx = document.getElementById('benchmarkPlot').getContext('2d');
        const chartData = benchmarks.map(bench => {{
            const benchData = data.filter(d => d.benchmark === bench);
            return {{
                label: bench,
                data: benchData.map(d => d.points),
                backgroundColor: arms.map((a, i) => `hsl({{i * 360 / arms.length}}, 70%, 60%)`)
            }};
        }});
    </script>
    """

    return html


def generate_html_report(
    results: SweepResult,
    baseline_results: dict[str, dict[str, Any]],
    output_path: Path,
) -> Path:
    """Generate an interactive HTML report with plots."""
    analysis = analyze_results(results, baseline_results)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Ablation Sweep: {results.config.name}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }}
        h1 {{ color: #333; }}
        h2 {{ color: #555; margin-top: 30px; }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{ background-color: #f5f5f5; }}
        .status-completed {{ color: #28a745; font-weight: bold; }}
        .status-failed {{ color: #dc3545; font-weight: bold; }}
        .promotion-promoted {{ background-color: #d4edda; padding: 15px; border-radius: 5px; }}
        .promotion-not-promoted {{ background-color: #f8d7da; padding: 15px; border-radius: 5px; }}
        .arm-card {{
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            margin: 10px 0;
        }}
        .metric {{ display: inline-block; margin-right: 20px; }}
        .plot-container {{
            margin: 30px 0;
            padding: 20px;
            background: #f9f9f9;
            border-radius: 5px;
        }}
    </style>
</head>
<body>
    <h1>Ablation Sweep Report</h1>

    <div class="summary">
        <h2>Summary</h2>
        <div class="metric"><strong>Sweep ID:</strong> {results.sweep_id}</div>
        <div class="metric"><strong>Baseline:</strong> {results.baseline_id}</div>
        <div class="metric"><strong>Completed:</strong> {len(results.completed_arms)} / {len(results.arms)} arms</div>
    </div>

    <h2>Configuration</h2>
    <p><strong>Description:</strong> {results.config.description or 'No description'}</p>

    <h3>Parameters</h3>
    <ul>
    """

    for param in results.config.sweep.parameters:
        html += f'        <li><strong>{param.name}</strong>: {param.type.value}'
        if param.description:
            html += f' - {param.description}'
        html += '</li>\n'

    html += """    </ul>

    <h2>Results</h2>
    <table>
        <thead>
            <tr>
                <th>Arm ID</th>
                <th>Status</th>
                <th>Mean Points</th>
                <th>Significant Wins</th>
                <th>Significant Losses</th>
            </tr>
        </thead>
        <tbody>
    """

    for arm_result in results.completed_arms:
        arm_id = arm_result.arm_id
        comp = analysis["comparisons"].get(arm_id)
        wins = comp["significant_wins"] if comp else []
        losses = comp["significant_losses"] if comp else []
        avg_points = np.mean([b["points"] for b in arm_result.benchmark_results.values()])

        html += f"""
            <tr>
                <td><code>{arm_id[:16]}</code></td>
                <td class="status-{arm_result.status.value}">{arm_result.status.value}</td>
                <td>{avg_points:.2f}</td>
                <td>{len(wins)}</td>
                <td>{len(losses)}</td>
            </tr>
        """

    html += """    </tbody>
    </table>
    """

    if results.promotion_decision:
        decision = results.promotion_decision
        status_class = "promotion-promoted" if decision["promoted"] else "promotion-not-promoted"

        html += f"""
    <div class="{status_class}">
        <h2>Promotion Decision</h2>
        <p><strong>Status:</strong> {'PROMOTED' if decision['promoted'] else 'NOT PROMOTED'}</p>
        <p><strong>Best Arm:</strong> <code>{decision.get('best_arm_id', 'none')[:16]}</code></p>
    </div>

    <h3>Wins</h3>
    <ul>
    """
        for win in decision.get("wins", []):
            html += f"        <li>{win}</li>\n"

        html += """    </ul>

    <h3>Losses</h3>
    <ul>
    """
        for loss in decision.get("losses", []):
            html += f"        <li>{loss}</li>\n"

        html += "    </ul>\n"

    html += """
    <footer>
        <p>Generated by medrl ablation framework</p>
    </footer>
</body>
</html>
"""

    output_path.write_text(html)
    log.info("HTML report written to %s", output_path)
    return output_path


def generate_report(
    results: SweepResult,
    baseline_results: dict[str, dict[str, Any]],
    output_dir: Path,
) -> dict[str, Path]:
    """Generate all report formats.

    Returns:
        Dict mapping format name to output path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sweep_dir = output_dir / results.sweep_id
    sweep_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    # Markdown report
    md_path = sweep_dir / "report.md"
    paths["markdown"] = generate_markdown_report(results, baseline_results, md_path)

    # HTML report
    html_path = sweep_dir / "report.html"
    paths["html"] = generate_html_report(results, baseline_results, html_path)

    # CSV export
    csv_path = sweep_dir / "results.csv"
    paths["csv"] = generate_csv_export(results, baseline_results, csv_path)

    # Raw JSON
    json_path = sweep_dir / "results.json"
    paths["json"] = results.save(json_path)

    log.info("generated %d report formats in %s", len(paths), sweep_dir)
    return paths
