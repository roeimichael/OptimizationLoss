#!/bin/bash
# claude_fmow_launcher.sh <release-sha> <gpu,gpu,...> [<pid>] -- the fmow2 step-ensemble study in Yuval's pipeline
# (experiments/claude_fmow_stepens_prereg_20260928.md): mobilenet_v3_large, class 1 capped, seeds 5000-5047, pilot 5099.
# Run it ON the host whose gpus are listed. If <pid> is given, nothing starts until that process (the RegNetY launcher on
# dsisco01) has exited. It first runs the pilot job (5099_stepens) and its reference (5099_ref: the same config with the
# steps off) on ONE gpu, then the release's analysis/score_fmow_stepens.py --gate (integrity only, no score): the pilot's
# PTO must be byte-identical to the reference at every epoch. Then the study seeds, each a one-seed queue
# (tools/claude_fmow_queue.sh), start on a listed gpu with no other user's process, fewer than MAXPER compute processes
# of ours in total, and NEED MiB free. A failed job, or one whose queue dies without an END line, stops the launcher
# (jobs already running continue).
set -u
SHA="$1"; GPUS="${2//,/ }"; AFTER="${3:-}"
REL=/home/dsi/michaer8/tralo-rebuild/releases/$SHA
PY=/home/dsi/michaer8/anaconda3/envs/optloss/bin/python
RUNS=/home/dsi/michaer8/tralo-rebuild/runs
MAXPER=3
PID0=5099; FIRST=5000; LAST=5047; NEED=6000; OWN="claude_fmow_50[0-9][0-9]"
PILOT=$RUNS/claude-fmow-pilot; REFD=$RUNS/claude-fmow-ref; STUDY=$RUNS/claude-fmow
Q=$REL/tools/claude_fmow_queue.sh
GATE=$REL/analysis/score_fmow_stepens.py
say() { echo "[$(date '+%F %T')] $*"; }
[ -n "$GPUS" ] || { say "no gpus listed"; exit 2; }
[ -f "$Q" ] && [ -f "$GATE" ] && [ -f "$REL/experiments/configs/claude_fmow_${PID0}_ref.json" ] || { say "release $SHA lacks the queue, the gate or the configs"; exit 2; }
for d in "$PILOT" "$REFD" "$STUDY"; do
  if [ -n "$(ls -A "$d" 2>/dev/null)" ]; then say "$d is not empty: move its old outputs to a quarantine folder first"; exit 2; fi
done
mkdir -p "$PILOT" "$REFD" "$STUDY"
free_mib() { nvidia-smi -i "$1" --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F', ' '{print $1 - $2}'; }
foreign() {   # compute processes on gpu $1 not owned by michaer8; an owner ps cannot resolve counts as foreign
  local p n=0
  for p in $(nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader); do
    [ "$(ps -o user= -p "$p" 2>/dev/null | xargs)" = michaer8 ] || n=$((n + 1))
  done
  echo "$n"
}
busy() {   # every compute process on gpu $1 that is not this block's, plus this block's live queues
  local p f n=0
  for p in $(nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader); do
    ps -o args= -p "$p" 2>/dev/null | grep -q "$OWN" || n=$((n + 1))
  done
  for f in "$PILOT"/.pid_gpu"$1"_*; do [ -e "$f" ] && kill -0 "$(cat "$f")" 2>/dev/null && n=$((n + 1)); done
  echo "$n"
}
usable() { [ "$(foreign "$1")" -eq 0 ] && [ "$(busy "$1")" -lt "$MAXPER" ] && [ "$(free_mib "$1")" -ge "$NEED" ]; }
slot() {
  local g
  for g in $GPUS; do usable "$g" && { echo "$g"; return; }; done
}
launch() {   # run-dir job gpu
  setsid nohup bash "$Q" "$SHA" "$1" "$3" "$2" > "$1/queue_gpu${3}_$2.log" 2>&1 < /dev/null &
  echo "$!" > "$PILOT/.pid_gpu${3}_$2"
  say "started $2 on gpu$3 into $(basename "$1") (pid $!), free was $(free_mib "$3") MiB"
  sleep 20
}
start_job() {   # run-dir job: a one-seed queue on the first usable gpu, waiting for one
  local g
  g=$(slot)
  until [ -n "$g" ]; do sleep 60; g=$(slot); done
  launch "$1" "$2" "$g"
}
ended() { cat "$PILOT"/queue_*.log "$REFD"/queue_*.log 2>/dev/null | grep -c " END $1 exit 0"; }
failed() { cat "$PILOT"/queue_*.log "$REFD"/queue_*.log "$STUDY"/queue_*.log 2>/dev/null | grep " END " | grep -qv "exit 0"; }
lost() {   # a started job whose queue process is gone without an END line
  local f base g job out
  for f in "$PILOT"/.pid_gpu*_*; do
    [ -e "$f" ] || continue
    kill -0 "$(cat "$f")" 2>/dev/null && continue
    base=${f##*/.pid_gpu}; g=${base%%_*}; job=${base#*_}
    case "$job" in *_stepens) out=$PILOT ;; *_ref) out=$REFD ;; *) out=$STUDY ;; esac
    grep -q " END $job " "$out/queue_gpu${g}_$job.log" 2>/dev/null || { echo "$job"; return; }
  done
}
stop_if_broken() {
  if failed; then say "A JOB FAILED: launcher stopped (running jobs continue)"; exit 1; fi
  local l; l=$(lost)
  if [ -n "$l" ]; then say "JOB $l ENDED WITHOUT AN END LINE: launcher stopped"; exit 1; fi
}
if [ -n "$AFTER" ]; then
  say "waiting for process $AFTER to exit"
  while kill -0 "$AFTER" 2>/dev/null; do sleep 60; done
fi
say "fmow2 on gpus $GPUS: pilot ${PID0}_stepens and reference ${PID0}_ref first, on one gpu"
g=$(slot)
until [ -n "$g" ] && [ "$(busy "$g")" -le $((MAXPER - 2)) ]; do sleep 60; g=$(slot); done
launch "$PILOT" "${PID0}_stepens" "$g"
launch "$REFD" "${PID0}_ref" "$g"
until [ "$(ended "${PID0}_stepens")" -ge 1 ] && [ "$(ended "${PID0}_ref")" -ge 1 ]; do
  stop_if_broken
  sleep 60
done
if ! CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 PYTHONPATH="$REL" "$PY" "$GATE" --gate "$PILOT" "$REFD" > "$PILOT/pilot_gate.txt" 2>&1; then
  say "PILOT GATE FAILED (see $PILOT/pilot_gate.txt)"; exit 1
fi
say "pilot gate passed; study next"
for seed in $(seq "$FIRST" "$LAST"); do
  stop_if_broken
  start_job "$STUDY" "$seed"
done
say "LAUNCHER DONE: all $((LAST - FIRST + 1)) study seeds started"
