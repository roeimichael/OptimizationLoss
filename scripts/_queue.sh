#!/bin/bash
# Autonomous experiment queue on dsisco01 GPU 2,3 (only free capacity; dsisco02 full).
# 1) wait for MobileNetV3 to finish the low-warmup sweep
# 2) stop the sweep dispatcher
# 3) OCTMNIST saturation probe (1 GPU)
# 4) run each pending root to completion on whichever of GPU 2,3 is free of OTHER users
# Never stacks on davidle: pick_gpus excludes any GPU with a non-michaer8 process.
cd ~/OptimizationLoss || exit 1
PYEXE=/home/dsi/michaer8/anaconda3/envs/optloss/bin/python
LOG=logs/queue.log
echo "queue start $(date)" > "$LOG"

pick_gpus() {                      # echo comma-list of GPUs in {2,3} free of other users
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

# 1. wait for MobileNetV3 (80/80)
for i in $(seq 1 160); do
  mn=$(find results/pending_runs/tissue_lowwarm_validation/MobileNetV3 -name evaluation_metrics.csv 2>/dev/null | wc -l)
  echo "$(date '+%F %T') wait MNv3=$mn/80" >> "$LOG"
  [ "$mn" -ge 80 ] && break
  sleep 180
done

# 2. stop the low-warmup sweep (and any leftover finalizer)
echo "stopping sweep $(date)" >> "$LOG"
pkill -u michaer8 -f "_finalize_lwval" 2>/dev/null
pkill -u michaer8 -f "main.py" 2>/dev/null; sleep 3
pkill -u michaer8 -f "src.experiments.runner" 2>/dev/null; sleep 5

# 3. OCTMNIST saturation probe on one free GPU
pg=$(pick_gpus); pg1=${pg%%,*}; [ -z "$pg1" ] && pg1=2
echo "probe on GPU $pg1 $(date)" >> "$LOG"
CUDA_VISIBLE_DEVICES=$pg1 $PYEXE -u scripts/_probe_octmnist_saturation.py > logs/octmnist_probe.log 2>&1
echo "probe done rc=$? $(date)" >> "$LOG"

# 4. batch queue: each root to completion on free GPUs of {2,3}
for root in tissue_lowwarm_validation grid_l40_l60 octmnist_sweep; do
  dir="results/pending_runs/$root"
  [ -d "$dir" ] || { echo "skip $root (no dir yet) $(date)" >> "$LOG"; continue; }
  pend=$(find "$dir" -name config.json | wc -l)
  done=$(find "$dir" -name evaluation_metrics.csv | wc -l)
  [ "$done" -ge "$pend" ] && { echo "skip $root (all $done/$pend done)" >> "$LOG"; continue; }
  for w in $(seq 1 144); do                 # wait up to 12h for a free GPU
    g=$(pick_gpus); [ -n "$g" ] && break
    echo "$(date '+%T') $root waiting: GPU 2,3 both busy (other user)" >> "$LOG"; sleep 300
  done
  g=$(pick_gpus); [ -z "$g" ] && { echo "ABORT $root: no free GPU" >> "$LOG"; continue; }
  echo "=== batch $root on GPU [$g]  pending=$((pend-done))  $(date) ===" >> "$LOG"
  echo "$g" | EXPERIMENT_DIR="$dir" $PYEXE -u main.py >> "$LOG" 2>&1
  echo "=== batch $root DONE $(date) ===" >> "$LOG"
done

touch logs/QUEUE_DONE
echo "QUEUE COMPLETE $(date)" >> "$LOG"
