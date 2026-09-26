#!/bin/bash
# Wrapper script to run all baseline evaluations with the robust pipeline

set -euo pipefail

MEDRL_DIR="${MEDRL_DIR:-/root/medrl}"
PIPELINE_SCRIPT="${MEDRL_DIR}/scripts/robust_evaluation_pipeline.sh"

# All baseline configurations
BASELINES=(
    "baseline_4b"
    "baseline_9b_think_on"
    "baseline_9b_think_off"
    "baseline_biomistral"
    "baseline_huatuo_o1"
    "baseline_ii_medical_8b"
    "baseline_medgemma_27b"
    "baseline_medgemma_4b"
    "baseline_medreason"
)

echo "========================================================================"
echo "MedRL Baseline Evaluations - Robust Pipeline"
echo "========================================================================"
echo "Starting at $(date)"
echo ""
echo "Baselines to run:"
for baseline in "${BASELINES[@]}"; do
    echo "  - ${baseline}"
done
echo ""
echo "========================================================================"
echo ""

# Check if pipeline script exists
if [ ! -f "${PIPELINE_SCRIPT}" ]; then
    echo "ERROR: Pipeline script not found at ${PIPELINE_SCRIPT}"
    exit 1
fi

# Run the pipeline
"${PIPELINE_SCRIPT}" --pipeline "${BASELINES[@]}"

exit_code=$?

echo ""
echo "========================================================================"
echo "Pipeline completed at $(date) with exit code ${exit_code}"
echo "========================================================================"
echo ""
echo "Reports are available in:"
echo "  Logs: /tmp/eval_pipeline_logs/"
echo "  Reports: /tmp/eval_pipeline_reports/"
echo ""
echo "To view the latest report:"
echo "  cat /tmp/eval_pipeline_reports/final_report_*.txt"
echo ""
echo "To resume from failures (if any):"
echo "  ${PIPELINE_SCRIPT} --resume"
echo ""

exit ${exit_code}
