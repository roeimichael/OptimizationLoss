#!/bin/bash
# claude_smallbb_launcher.sh <release-sha> -- the small-backbone blocks in Yuval's pipeline on dsisco01
# (experiments/claude_yuval_smallbb_prereg_20260927.md), after the recipe factorial. It waits until every
# factorial study job is claimed and the newest claim is 5 min old, then runs the pilots 4399 and 4499,
# their integrity gate (the release's analysis/score_smallbb.py --gate: no score printed), and the study,
# interleaved: 4300-4323 into claude-yuval-mn3 and 4400-4423 into claude-yuval-rgy.
# Every job is its own one-seed queue (tools/claude_claim_queue.sh). It starts only on a GPU with no other
# user's process, fewer than 3 knee processes and NEED MiB free. A failed job, or one whose queue dies
# without an END line, stops the launcher (jobs already running continue).
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
GATE=$REL/analysis/score_smallbb.py
NEED=3000      # MiB; a ResNet18 knee_yuval process peaks near 2.4 GB, the small backbones below that
say() { echo "[$(date '+%F %T')] $*"; }
[ -f "$Q" ] && [ -f "$GATE" ] || { say "release $SHA lacks the queue or the gate"; exit 2; }
for d in "$PILOT" "$MN3" "$RGY"; do
  if [ -n "$(ls -A "$d" 2>/dev/null)" ]; then say "$d is not empty: move its old outputs to a quarantine folder first"; exit 2; fi
done
mkdir -p "$PILOT" "$MN3" "$RGY"
free_mib() { nvidia-smi -i "$1" --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F', ' '{print $1 - $2}'; }
foreign() {   # compute processes on gpu $1 not owned by michaer8; an owner ps cannot resolve counts as foreign
  local p n=0
  for p in $(nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader); do
    [ "$(ps -o user= -p "$p" 2>/dev/null | xargs)" = michaer8 ] || n=$((n + 1))
  done
  echo "$n"
}
busy() {   # every compute process on gpu $1 that is not this study's, plus this study's live queues
  local p f n=0
  for p in $(nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader); do
    ps -o args= -p "$p" 2>/dev/null | grep -q "claude_yuval_4[34][0-9][0-9]" || n=$((n + 1))
  done
  for f in "$PILOT"/.pid_gpu"$1"_*; do [ -e "$f" ] && kill -0 "$(cat "$f")" 2>/dev/null && n=$((n + 1)); done
  echo "$n"
}
slot() {
  local g
  for g in 0 1 2 3; do
    [ "$(foreign "$g")" -eq 0 ] && [ "$(busy "$g")" -lt 3 ] && [ "$(free_mib "$g")" -ge "$NEED" ] && { echo "$g"; return; }
  done
}
start_job() {   # run-dir seed: a one-seed queue on the first free slot, waiting for one
  local out="$1" seed="$2" g
  g=$(slot)
  until [ -n "$g" ]; do sleep 60; g=$(slot); done
  setsid nohup bash "$Q" "$SHA" "$out" "$g" "$seed" > "$out/queue_gpu${g}_$seed.log" 2>&1 < /dev/null &
  echo "$!" > "$PILOT/.pid_gpu${g}_$seed"
  say "started $seed on gpu$g into $(basename "$out") (pid $!), free was $(free_mib "$g") MiB"
  sleep 20
}
failed() { cat "$PILOT"/queue_*.log "$MN3"/queue_*.log "$RGY"/queue_*.log 2>/dev/null | grep " END " | grep -qv "exit 0"; }
lost() {   # a started job whose queue process is gone without an END line
  local f base g seed out
  for f in "$PILOT"/.pid_gpu*_*; do
    [ -e "$f" ] || continue
    kill -0 "$(cat "$f")" 2>/dev/null && continue
    base=${f##*/.pid_gpu}; g=${base%%_*}; seed=${base#*_}
    case "$seed" in 4399|4499) out=$PILOT ;; 43*) out=$MN3 ;; *) out=$RGY ;; esac
    grep -q " END $seed " "$out/queue_gpu${g}_$seed.log" 2>/dev/null || { echo "$seed"; return; }
  done
}
stop_if_broken() {
  if failed; then say "A JOB FAILED: launcher stopped (running jobs continue)"; exit 1; fi
  local l; l=$(lost)
  if [ -n "$l" ]; then say "JOB $l ENDED WITHOUT AN END LINE: launcher stopped"; exit 1; fi
}
say "waiting for all 192 factorial study jobs to be claimed"
until [ "$(ls -d "$RECIPE"/.claim_42[0-2][0-9]_* 2>/dev/null | wc -l)" -ge 192 ]; do sleep 120; done
newest=$(stat -c %Y "$RECIPE"/.claim_42[0-2][0-9]_* | sort -n | tail -1)
while [ $(( $(date +%s) - newest )) -lt 300 ]; do sleep 30; done
say "factorial fully claimed; pilots 4399 and 4499 next"
start_job "$PILOT" 4399
start_job "$PILOT" 4499
until [ "$(cat "$PILOT"/queue_*.log 2>/dev/null | grep -c " END 4[34]99 exit 0")" -ge 2 ]; do
  stop_if_broken
  sleep 60
done
if ! CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 PYTHONPATH="$REL" "$PY" "$GATE" --gate "$PILOT" > "$PILOT/pilot_gate.txt" 2>&1; then
  say "PILOT GATE FAILED (see $PILOT/pilot_gate.txt)"; exit 1
fi
say "pilot gate passed; study next"
for i in $(seq 0 23); do
  for job in "$MN3:$((4300 + i))" "$RGY:$((4400 + i))"; do
    stop_if_broken
    start_job "${job%%:*}" "${job##*:}"
  done
done
say "LAUNCHER DONE: all 48 study seeds started"
