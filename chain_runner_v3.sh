#!/bin/bash
# v3: waits for v2 (AIDER seed-ext) to fully drain, then launches ViT probes.
# Only fires when (a) no main.py running, (b) GPU2 free, (c) aider_seed_ext is
# fully done (config count == eval_metrics count), (d) vit_probe has work.
cd ~/OptimizationLoss
DEADLINE=$(( $(date +%s) + 43200 ))
while [ $(date +%s) -lt $DEADLINE ]; do
  if pgrep -u michaer8 -f "python -u main.py" > /dev/null; then
    echo "[chain3 $(date +%H:%M)] main.py running, sleep 180"
    sleep 180
    continue
  fi
  GPU2_APPS=$(nvidia-smi -i 2 --query-compute-apps=pid --format=csv,noheader | wc -l)
  if [ "$GPU2_APPS" -gt 0 ]; then
    OWNER=$(nvidia-smi -i 2 --query-compute-apps=pid --format=csv,noheader | head -1 | xargs ps -o user= -p 2>/dev/null | xargs)
    echo "[chain3 $(date +%H:%M)] GPU2 busy (owner=$OWNER), sleep 180"
    sleep 180
    continue
  fi
  AE_CFG=$(find results/pending_runs/aider_seed_ext -name config.json 2>/dev/null | wc -l)
  AE_DONE=$(find results/pending_runs/aider_seed_ext -name evaluation_metrics.csv 2>/dev/null | wc -l)
  if [ "$AE_DONE" -lt "$AE_CFG" ]; then
    echo "[chain3 $(date +%H:%M)] aider_seed_ext not done yet ($AE_DONE/$AE_CFG), sleep 180"
    sleep 180
    continue
  fi
  VP_CFG=$(find results/pending_runs/vit_probe -name config.json 2>/dev/null | wc -l)
  VP_DONE=$(find results/pending_runs/vit_probe -name evaluation_metrics.csv 2>/dev/null | wc -l)
  if [ "$VP_DONE" -ge "$VP_CFG" ]; then
    echo "[chain3 $(date +%H:%M)] vit_probe done ($VP_DONE/$VP_CFG), exit"
    exit 0
  fi
  LOG=logs/vit_probe_$(date +%Y%m%d_%H%M).log
  echo "[chain3 $(date +%H:%M)] launching vit_probe ($VP_DONE/$VP_CFG done), log=$LOG"
  unset CUDA_VISIBLE_DEVICES
  EXPERIMENT_DIR=results/pending_runs/vit_probe setsid bash -c "echo 2 | ~/anaconda3/envs/optloss/bin/python -u main.py > $LOG 2>&1" < /dev/null &
  sleep 30
done
echo "[chain3 $(date +%H:%M)] 12h deadline, exit"
