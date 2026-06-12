#!/bin/bash
# Autonomous finalizer for the tissue low-warmup validation sweep.
# Polls progress; refreshes the (partial) verdict every cycle so
# logs/lwval_VERDICT.txt is ALWAYS current; exits + writes final verdict
# when all runs complete OR the dispatcher dies.
cd ~/OptimizationLoss || exit 1
PYEXE=/home/dsi/michaer8/anaconda3/envs/optloss/bin/python
ROOT=results/pending_runs/tissue_lowwarm_validation
LOG=logs/lwval_finalize.log
echo "finalizer start $(date)" > "$LOG"
for i in $(seq 1 140); do          # 140 * 6min ~= 14h hard cap
  total=$(find "$ROOT" -name config.json | wc -l)
  done=$(find "$ROOT" -name evaluation_metrics.csv | wc -l)
  disp=$(pgrep -u michaer8 -f "main.py" | wc -l)
  echo "$(date '+%F %T') poll=$i done=$done/$total dispatcher=$disp" >> "$LOG"
  # always refresh the partial verdict (aggregator is cheap + nan-safe)
  $PYEXE -m scripts.agg_lowwarm_validation > logs/lwval_VERDICT.txt 2>>"$LOG"
  if [ "$done" -ge "$total" ]; then echo "ALL DONE $(date)" >> "$LOG"; break; fi
  if [ "$disp" -eq 0 ]; then echo "DISPATCHER GONE (partial) $(date)" >> "$LOG"; break; fi
  sleep 360
done
$PYEXE -m scripts.agg_lowwarm_validation > logs/lwval_VERDICT.txt 2>>"$LOG"
echo "finalizer end $(date)" >> "$LOG"
