#!/bin/bash
# G1 — MobileNetV2 non-saturated 2nd backbone. 240 cells, ~3h on 2× Blackwell.
# Closes Limitation 3 of main.tex §6 (F1 corroboration on a non-saturated backbone).
#
# Pre-check on dsisco02:
#   ssh dsisco02 'nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader'
# Abort if ANY GPU has a non-michaer8 process. Driver-crash risk if shared.

set -euo pipefail

cd ~/OptimizationLoss
mkdir -p logs
LOG="logs/g1_$(date +%m%d_%H%M).log"
echo "LOG=$LOG"

# Make sure the registry has MobileNetV2:
~/anaconda3/envs/optloss/bin/python -c \
  "from src.models.model_factory import MODEL_REGISTRY; \
   assert 'MobileNetV2' in MODEL_REGISTRY, 'sync src/models/imagery/__init__.py + model_factory.py first'"

# Generate configs:
~/anaconda3/envs/optloss/bin/python -m src.config_generators.gen_g1_mobilenetv2

# Launch on GPUs 2 + 3 (avoid 0+1 if davidlevin is on them):
export CUDA_VISIBLE_DEVICES=2,3
EXPERIMENT_DIR=results/pending_runs/g1_mobilenetv2 \
  setsid bash -c "echo all | ~/anaconda3/envs/optloss/bin/python -u main.py > $LOG 2>&1" &
disown

sleep 8
echo --- PID ---
pgrep -u michaer8 -f "main.py" | head -3
echo --- LOG HEAD ---
tail -25 "$LOG" 2>/dev/null
echo --- GPU ---
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

# Watch progress:
#   ssh dsisco02 'cd ~/OptimizationLoss && find results/pending_runs/g1_mobilenetv2 -name evaluation_metrics.csv | wc -l && echo "/ 240"'
#
# When done:
#   python paper/HANDOFF/aggregators/agg_g1.py
