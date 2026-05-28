#!/bin/bash
# Wait for current main.py to exit, then launch AIDER seed-ext sweep on GPU2.
cd ~/OptimizationLoss
echo "[chain] waiting for current main.py to exit..."
while pgrep -u michaer8 -f "python -u main.py" > /dev/null; do
  sleep 30
done
echo "[chain] no main.py running, free GPU2 check..."
GPU2_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 2 | awk "{print \$1}")
echo "[chain] GPU2 mem=${GPU2_MEM}MiB"
if [ "$GPU2_MEM" -gt 1000 ]; then
  echo "[chain] GPU2 still busy (>1GB), abort"
  exit 1
fi
LOG=logs/aider_ext_$(date +%Y%m%d_%H%M).log
echo "[chain] launching AIDER seed-ext, log=$LOG"
unset CUDA_VISIBLE_DEVICES
EXPERIMENT_DIR=results/pending_runs/aider_seed_ext setsid bash -c "echo 2 | ~/anaconda3/envs/optloss/bin/python -u main.py > $LOG 2>&1" < /dev/null &
sleep 5
echo "[chain] launched, $(ps -u michaer8 -o pid,etime,cmd | grep "python -u main.py" | grep -v grep | head -1)"
