#!/bin/bash
# G4 — 12-cell cosmetic Table B backfill. ~5 min on Blackwell single GPU.
#
# Run ON dsisco02 (Blackwell, paper-eligible). Verify GPU 2 + 3 are clear first:
#   ssh dsisco02 'nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader'
#
# Single-GPU is fine for 12 cells.

set -euo pipefail

cd ~/OptimizationLoss
mkdir -p logs
LOG="logs/g4_$(date +%m%d_%H%M).log"
echo "LOG=$LOG"

# Generate configs (if not already done):
~/anaconda3/envs/optloss/bin/python -m src.config_generators.gen_g4_table_b_missing_seeds

# Launch on GPU 2, detached:
export CUDA_VISIBLE_DEVICES=2
EXPERIMENT_DIR=results/pending_runs/g4_table_b_backfill \
  setsid bash -c "echo all | ~/anaconda3/envs/optloss/bin/python -u main.py > $LOG 2>&1" &
disown

sleep 4
echo --- PID ---
pgrep -u michaer8 -f "main.py" | head -3
echo --- LOG HEAD ---
tail -20 "$LOG" 2>/dev/null

# Watch progress:
#   ssh dsisco02 'cd ~/OptimizationLoss && find results/pending_runs/g4_table_b_backfill -name evaluation_metrics.csv | wc -l && echo "/ 12"'
#
# When done (12/12), run:
#   python paper/HANDOFF/aggregators/agg_g4.py
