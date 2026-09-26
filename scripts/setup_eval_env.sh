#!/bin/bash
# Evaluation environment setup - sets HF_HOME and other critical variables

export HF_HOME=/scratch/medrl/.cache/huggingface
export HF_DATASETS_CACHE=/scratch/medrl/.cache/datasets
export HF_HUB_CACHE=/scratch/medrl/.cache/huggingface

echo "Evaluation environment configured:"
echo "  HF_HOME: $HF_HOME"
echo "  HF_DATASETS_CACHE: $HF_DATASETS_CACHE"
echo "  HF_HUB_CACHE: $HF_HUB_CACHE"
echo "  Disk space available:"
df -h / | tail -1
