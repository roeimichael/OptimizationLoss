#!/bin/bash
# claude_smallbb_launcher.sh <release-sha> -- the small-backbone blocks in Yuval's pipeline on dsisco01
# (experiments/claude_yuval_smallbb_prereg_20260927.md), after the recipe factorial. It waits until every
# factorial study job is claimed and the newest claim is 5 min old, then runs the pilots 4399 and 4499,
# their integrity gate (score_smallbb.py --gate: no score printed), and the study: 4300-4323 into
# claude-yuval-mn3 and 4400-4423 into claude-yuval-rgy, in claim queues of tools/claude_claim_queue.sh.
# A GPU takes a new queue only while it holds fewer than 3 knee processes (the factorial's, seen by
# nvidia-smi, plus this launcher's live queues) and has NEED MiB free.
set -u
SHA="$1"
REL=/home/dsi/michaer8/tralo-rebuild/releases/$SHA
PY=/home/dsi/michaer8/anaconda3/envs/optloss/bin/python
RUNS=/home/dsi/michaer8/tralo-rebuild/runs
RECIPE=$RUNS/claude-recipe
PILOT=$RUNS/claude-yuval-smallbb-pilot
MN3=$RUNS/claude-yuval-mn3
RGY=$RUNS/claude-yuval-rgy
Q=$REL/tools/claude_claim_queue.sh
GATE=/home/dsi/michaer8/tralo-rebuild/lab/smallbb/score_smallbb.py   # analysis code, next to its score_yuval.py
NEED=3000      # MiB; a ResNet18 knee_yuval process peaks near 2.4 GB, the small backbones below that
mkdir -p "$PILOT" "$MN3" "$RGY"
say() { echo "[$(date '+%F %T')] $*"; }
free_mib() { nvidia-smi -i "$1" --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F', ' '{print $1 - $2}'; }
busy() {
  local n=0 p f
  for p in $(nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader); do
    ps -o args= -p "$p" | grep -q "claude_yuval_42" && n=$((n + 1))
  done
  for f in "$PILOT"/.pid_gpu"$1"_*; do [ -e "$f" ] && kill -0 "$(cat "$f")" 2>/dev/null && n=$((n + 1)); done
  echo "$n"
}
foreign() {   # compute processes of other users on gpu $1: never share a card with them
  local p
  for p in $(nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader); do ps -o user= -p "$p"; done | grep -vc "^michaer8$"
}
slot() {
  local g
  for g in 0 1 2 3; do
    [ "$(foreign "$g")" -eq 0 ] && [ "$(busy "$g")" -lt 3 ] && [ "$(free_mib "$g")" -ge "$NEED" ] && { echo "$g"; return; }
  done
}
start_queue() {   # gpu run-dir tag seeds...
  local g="$1" out="$2" tag="$3"; shift 3
  setsid nohup bash "$Q" "$SHA" "$out" "$g" "$@" > "$out/queue_gpu${g}_${tag}.log" 2>&1 < /dev/null &
  echo "$!" > "$PILOT/.pid_gpu${g}_${tag}"
  say "started queue $tag on gpu$g for $(basename "$out") (pid $!), free was $(free_mib "$g") MiB"
}
claimed() { ls -d "$1"/.claim_* 2>/dev/null | wc -l; }
say "waiting for all 192 factorial study jobs to be claimed"
until [ "$(ls -d "$RECIPE"/.claim_42[0-2][0-9]_* 2>/dev/null | wc -l)" -ge 192 ]; do sleep 120; done
newest=$(stat -c %Y "$RECIPE"/.claim_42[0-2][0-9]_* | sort -n | tail -1)
while [ $(( $(date +%s) - newest )) -lt 300 ]; do sleep 30; done
say "factorial fully claimed; pilots 4399 and 4499 next"
for seed in 4399 4499; do
  g=$(slot)
  until [ -n "$g" ]; do sleep 60; g=$(slot); done
  start_queue "$g" "$PILOT" "p$seed" "$seed"
  sleep 30
done
until [ "$(ls "$PILOT"/seed4[34]99/summary.json 2>/dev/null | wc -l)" -ge 2 ]; do
  if cat "$PILOT"/queue_*.log 2>/dev/null | grep " END " | grep -qv "exit 0"; then say "PILOT RUN FAILED"; exit 1; fi
  sleep 60
done
if ! CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 PYTHONPATH="$REL" "$PY" "$GATE" --gate "$PILOT" > "$PILOT/pilot_gate.txt" 2>&1; then
  say "PILOT GATE FAILED (see $PILOT/pilot_gate.txt)"; exit 1
fi
say "pilot gate passed; study queues next"
until [ "$(claimed "$MN3")" -ge 24 ] && [ "$(claimed "$RGY")" -ge 24 ]; do
  g=$(slot)
  if [ -z "$g" ]; then sleep 120; continue; fi
  if [ "$(claimed "$MN3")" -lt 24 ] && { [ "$(claimed "$MN3")" -le "$(claimed "$RGY")" ] || [ "$(claimed "$RGY")" -ge 24 ]; }; then
    start_queue "$g" "$MN3" "s$(date +%H%M%S)" $(seq 4300 4323)
  else
    start_queue "$g" "$RGY" "s$(date +%H%M%S)" $(seq 4400 4423)
  fi
  sleep 30
done
say "LAUNCHER DONE: all 48 study seeds claimed"
