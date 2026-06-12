#!/bin/bash
# Lean batch runner (probe already done; ResNet18 removed - not in server registry).
# Runs each pending root to completion on whichever of GPU 2,3 is free of OTHER users.
cd ~/OptimizationLoss || exit 1
PYEXE=/home/dsi/michaer8/anaconda3/envs/optloss/bin/python
LOG=logs/queue2.log
echo "queue2 start $(date)" > "$LOG"

pick_gpus() {
  local free=""
  for g in 2 3; do
    local other=""
    for p in $(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
      u=$(ps -o user= -p "$p" 2>/dev/null)
      [ -n "$u" ] && [ "$u" != "michaer8" ] && other="$u"
    done
    [ -z "$other" ] && free="${free},${g}"
  done
  echo "${free#,}"
}

for root in tissue_lowwarm_validation grid_l40_l60 octmnist_sweep; do
  dir="results/pending_runs/$root"
  [ -d "$dir" ] || { echo "skip $root (no dir yet) $(date)" >> "$LOG"; continue; }
  pend=$(find "$dir" -name config.json | wc -l)
  done=$(find "$dir" -name evaluation_metrics.csv | wc -l)
  [ "$done" -ge "$pend" ] && { echo "skip $root ($done/$pend done)" >> "$LOG"; continue; }
  for w in $(seq 1 144); do
    g=$(pick_gpus); [ -n "$g" ] && break
    echo "$(date '+%T') $root waiting: GPU 2,3 busy (other user)" >> "$LOG"; sleep 300
  done
  g=$(pick_gpus); [ -z "$g" ] && { echo "ABORT $root: no free GPU" >> "$LOG"; continue; }
  echo "=== batch $root on GPU [$g]  pending=$((pend-done))  $(date) ===" >> "$LOG"
  echo "$g" | EXPERIMENT_DIR="$dir" $PYEXE -u main.py >> "$LOG" 2>&1
  echo "=== batch $root DONE $(date) ===" >> "$LOG"
done
touch logs/QUEUE_DONE
echo "QUEUE2 COMPLETE $(date)" >> "$LOG"
