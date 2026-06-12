#!/bin/bash
# Wait for MobileNetV3 to finish (80/80) in the low-warmup sweep, then stop the
# sweep (abandon RegNet/ResNet18) and launch the OCTMNIST saturation probe on GPU 2.
cd ~/OptimizationLoss || exit 1
PYEXE=/home/dsi/michaer8/anaconda3/envs/optloss/bin/python
LOG=logs/handoff_oct.log
echo "handoff start $(date) - waiting for MobileNetV3 80/80" > "$LOG"
for i in $(seq 1 140); do          # up to 7h
  mn=$(find results/pending_runs/tissue_lowwarm_validation/MobileNetV3 \
        -name evaluation_metrics.csv 2>/dev/null | wc -l)
  echo "$(date '+%F %T') poll=$i MobileNetV3=$mn/80" >> "$LOG"
  if [ "$mn" -ge 80 ]; then echo "MNv3 COMPLETE $(date)" >> "$LOG"; break; fi
  sleep 180
done
# stop the low-warmup sweep: kill dispatcher first (stops spawning), then runners
echo "stopping low-warmup sweep $(date)" >> "$LOG"
pkill -u michaer8 -f "main.py" 2>/dev/null
sleep 3
pkill -u michaer8 -f "src.experiments.runner" 2>/dev/null
sleep 4
echo "remaining main.py: $(pgrep -u michaer8 -f main.py | wc -l)" >> "$LOG"
# launch OCTMNIST saturation probe on GPU 2 (davidle is on 0,1)
echo "launching octmnist probe on GPU2 $(date)" >> "$LOG"
CUDA_VISIBLE_DEVICES=2 $PYEXE -u scripts/_probe_octmnist_saturation.py \
    > logs/octmnist_probe.log 2>&1
echo "probe exited rc=$? $(date)" >> "$LOG"
