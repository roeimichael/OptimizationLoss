#!/bin/bash
# claude_stepens_launcher.sh <release-sha> [rgy <pid>] -- the step-ensemble study in Yuval's pipeline on
# dsisco01 (experiments/claude_stepens_prereg_20260927.md), or with "rgy" its RegNetY replication
# (experiments/claude_stepens_rgy_prereg_20260927.md). It runs the pilot job (4000_stepens; rgy: 4400_stepens),
# its integrity gate (the release's analysis/score_stepens.py --gate against the stored run of that seed: no
# score printed), then the study seeds 4500-4571 into claude-stepens (rgy: 4600-4671 into claude-stepens-rgy,
# started only once process <pid>, the ResNet18 launcher, has exited).
# Every job is its own one-seed queue (tools/claude_claim_queue.sh). It starts only on a GPU with no other
# user's process, fewer than 3 knee processes and NEED MiB free. A failed job, or one whose queue dies
# without an END line, stops the launcher (jobs already running continue).
set -u
SHA="$1"; BLOCK="${2:-r18}"; AFTER="${3:-}"
REL=/home/dsi/michaer8/tralo-rebuild/releases/$SHA
PY=/home/dsi/michaer8/anaconda3/envs/optloss/bin/python
RUNS=/home/dsi/michaer8/tralo-rebuild/runs
case "$BLOCK" in
  r18) PILOT_JOB=4000_stepens; PILOT=$RUNS/claude-stepens-pilot; STUDY=$RUNS/claude-stepens; REF=$RUNS/claude-yuval-r18
       FIRST=4500; LAST=4571; OWN="claude_yuval_4000_stepens\|claude_yuval_45[0-9][0-9]" ;;
  rgy) PILOT_JOB=4400_stepens; PILOT=$RUNS/claude-stepens-rgy-pilot; STUDY=$RUNS/claude-stepens-rgy; REF=$RUNS/claude-yuval-rgy
       FIRST=4600; LAST=4671; OWN="claude_yuval_4400_stepens\|claude_yuval_46[0-9][0-9]" ;;
  *) echo "unknown block $BLOCK"; exit 2 ;;
esac
Q=$REL/tools/claude_claim_queue.sh
GATE=$REL/analysis/score_stepens.py
NEED=4000      # MiB; a ResNet18 knee_yuval process peaks near 2.4 GB, the side copies add a little
say() { echo "[$(date '+%F %T')] $*"; }
[ -f "$Q" ] && [ -f "$GATE" ] && [ -f "$REF/seed${PILOT_JOB%_stepens}/summary.json" ] || { say "release $SHA lacks the queue or the gate, or the reference run is missing"; exit 2; }
[ "$BLOCK" = r18 ] || [ -n "$AFTER" ] || { say "the rgy block needs the pid of the ResNet18 launcher to wait for"; exit 2; }
for d in "$PILOT" "$STUDY"; do
  if [ -n "$(ls -A "$d" 2>/dev/null)" ]; then say "$d is not empty: move its old outputs to a quarantine folder first"; exit 2; fi
done
mkdir -p "$PILOT" "$STUDY"
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
    ps -o args= -p "$p" 2>/dev/null | grep -q "$OWN" || n=$((n + 1))
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
failed() { cat "$PILOT"/queue_*.log "$STUDY"/queue_*.log 2>/dev/null | grep " END " | grep -qv "exit 0"; }
lost() {   # a started job whose queue process is gone without an END line
  local f base g seed out
  for f in "$PILOT"/.pid_gpu*_*; do
    [ -e "$f" ] || continue
    kill -0 "$(cat "$f")" 2>/dev/null && continue
    base=${f##*/.pid_gpu}; g=${base%%_*}; seed=${base#*_}
    case "$seed" in *_stepens) out=$PILOT ;; *) out=$STUDY ;; esac
    grep -q " END $seed " "$out/queue_gpu${g}_$seed.log" 2>/dev/null || { echo "$seed"; return; }
  done
}
stop_if_broken() {
  if failed; then say "A JOB FAILED: launcher stopped (running jobs continue)"; exit 1; fi
  local l; l=$(lost)
  if [ -n "$l" ]; then say "JOB $l ENDED WITHOUT AN END LINE: launcher stopped"; exit 1; fi
}
say "pilot $PILOT_JOB first"
start_job "$PILOT" "$PILOT_JOB"
until [ "$(cat "$PILOT"/queue_*.log 2>/dev/null | grep -c " END $PILOT_JOB exit 0")" -ge 1 ]; do
  stop_if_broken
  sleep 60
done
if ! CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 PYTHONPATH="$REL" "$PY" "$GATE" --gate "$PILOT" "$REF" > "$PILOT/pilot_gate.txt" 2>&1; then
  say "PILOT GATE FAILED (see $PILOT/pilot_gate.txt)"; exit 1
fi
say "pilot gate passed; study next"
if [ -n "$AFTER" ]; then
  say "waiting for launcher $AFTER to exit"
  while kill -0 "$AFTER" 2>/dev/null; do sleep 120; done
fi
for seed in $(seq "$FIRST" "$LAST"); do
  stop_if_broken
  start_job "$STUDY" "$seed"
done
say "LAUNCHER DONE: all $((LAST - FIRST + 1)) study seeds started"
