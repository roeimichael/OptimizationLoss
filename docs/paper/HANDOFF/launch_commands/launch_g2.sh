#!/bin/bash
# G2 — Asymmetric (L,G) on TissueMNIST + AIDER. 960 cells, ~13h on 2× Blackwell.
# Closes Limitation 2 part 1 of main.tex §6 (asymmetric robustness off DermMNIST).
#
# This is the long one — best fired overnight.

set -euo pipefail

cd ~/OptimizationLoss
mkdir -p logs
LOG="logs/g2_$(date +%m%d_%H%M).log"
echo "LOG=$LOG"

# Generate configs:
~/anaconda3/envs/optloss/bin/python -m src.config_generators.gen_g2_asym_tissue_aider

# Launch on GPUs 2 + 3:
export CUDA_VISIBLE_DEVICES=2,3
EXPERIMENT_DIR=results/pending_runs/g2_asym_tissue_aider \
  setsid bash -c "echo all | ~/anaconda3/envs/optloss/bin/python -u main.py > $LOG 2>&1" &
disown

sleep 8
echo --- PID ---
pgrep -u michaer8 -f "main.py" | head -3
echo --- LOG HEAD ---
tail -25 "$LOG" 2>/dev/null

# Watch:
#   ssh dsisco02 'cd ~/OptimizationLoss && find results/pending_runs/g2_asym_tissue_aider -name evaluation_metrics.csv | wc -l && echo "/ 960"'
#
# Aggregate:
#   python paper/HANDOFF/aggregators/agg_g2.py
