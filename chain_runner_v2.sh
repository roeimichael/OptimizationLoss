#!/bin/bash
cd ~/OptimizationLoss
DEADLINE=$(( $(date +%s) + 43200 ))
while [ $(date +%s) -lt $DEADLINE ]; do
  if pgrep -u michaer8 -f "python -u main.py" > /dev/null; then
    echo "[chain2 $(date +%H:%M)] our main.py running, sleep 120"
    sleep 120
    continue
  fi
  GPU2_APPS=$(nvidia-smi -i 2 --query-compute-apps=pid --format=csv,noheader | wc -l)
  if [ "$GPU2_APPS" -gt 0 ]; then
    OWNER=$(nvidia-smi -i 2 --query-compute-apps=pid --format=csv,noheader | head -1 | xargs ps -o user= -p 2>/dev/null | xargs)
    echo "[chain2 $(date +%H:%M)] GPU2 busy (owner=$OWNER, apps=$GPU2_APPS), sleep 180"
    sleep 180
    continue
  fi
  PENDING=$(find results/pending_runs/aider_seed_ext -name config.json 2>/dev/null | wc -l)
  DONE=$(find results/pending_runs/aider_seed_ext -name evaluation_metrics.csv 2>/dev/null | wc -l)
  REMAIN=$(( PENDING - DONE ))
  if [ "$REMAIN" -le 0 ]; then
    echo "[chain2 $(date +%H:%M)] no AIDER work remaining ($DONE/$PENDING), exit"
    exit 0
  fi
  LOG=logs/aider_ext_$(date +%Y%m%d_%H%M).log
  echo "[chain2 $(date +%H:%M)] GPU2 free, $REMAIN AIDER cells remaining, launching $LOG"
  unset CUDA_VISIBLE_DEVICES
  EXPERIMENT_DIR=results/pending_runs/aider_seed_ext setsid bash -c "echo 2 | ~/anaconda3/envs/optloss/bin/python -u main.py > $LOG 2>&1" < /dev/null &
  sleep 30
done
echo "[chain2 $(date +%H:%M)] 12h deadline reached, exit"
