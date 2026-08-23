#!/bin/bash
# G3 — Multi-class robustness on TissueMNIST. 360 cells, ~5h on 2× Blackwell.
# Closes Limitation 2 part 2 of main.tex §6 (multi-class robustness off DermMNIST).
#
# Alt constrained classes: cls 2 (CST), cls 5 (PTC), cls 7 (TUB).
# cls 4 (GE) is the headline class — already in Table A.

set -euo pipefail

cd ~/OptimizationLoss
mkdir -p logs
LOG="logs/g3_$(date +%m%d_%H%M).log"
echo "LOG=$LOG"

# Generate configs:
~/anaconda3/envs/optloss/bin/python -m src.config_generators.gen_g3_multiclass_tissue

# Launch on GPUs 2 + 3:
export CUDA_VISIBLE_DEVICES=2,3
EXPERIMENT_DIR=results/pending_runs/g3_multiclass_tissue \
  setsid bash -c "echo all | ~/anaconda3/envs/optloss/bin/python -u main.py > $LOG 2>&1" &
disown

sleep 8
echo --- PID ---
pgrep -u michaer8 -f "main.py" | head -3
echo --- LOG HEAD ---
tail -25 "$LOG" 2>/dev/null

# Watch:
#   ssh dsisco02 'cd ~/OptimizationLoss && find results/pending_runs/g3_multiclass_tissue -name evaluation_metrics.csv | wc -l && echo "/ 360"'
#
# Aggregate:
#   python paper/HANDOFF/aggregators/agg_g3.py
