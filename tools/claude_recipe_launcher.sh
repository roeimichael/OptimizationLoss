#!/bin/bash
# claude_recipe_launcher.sh <release-sha> -- run the recipe factorial on dsisco01 as the B5 block frees memory.
# Queues start only after every B5 seed is claimed, so B5 memory only frees from then on and a factorial
# process never takes the gap between two B5 seeds. Then: the pilot 4299 (8 cells), its integrity gate
# (score_recipe.py --gate: no score printed), and the study 4200-4223 x 8 cells, seed-major, in claim
# queues of tools/claude_claim_queue.sh, at most 3 per GPU, each started only with NEED MiB free.
set -u
SHA="$1"
REL=/home/dsi/michaer8/tralo-rebuild/releases/$SHA
PY=/home/dsi/michaer8/anaconda3/envs/optloss/bin/python
OUT=/home/dsi/michaer8/tralo-rebuild/runs/claude-recipe
B5=/home/dsi/michaer8/tralo-rebuild/runs/claude-yuval-b5
Q=$REL/tools/claude_claim_queue.sh
GATE=/home/dsi/michaer8/tralo-rebuild/lab/recipe/score_recipe.py   # analysis code, newer than the training release
NEED=6000
mkdir -p "$OUT"
say() { echo "[$(date '+%F %T')] $*"; }
cells() { for a in 1 0; do for s in 1 0; do for e in 1 0; do echo "${1}_a${a}s${s}e${e}"; done; done; done; }
PILOT=$(cells 4299)
STUDY=$(for seed in $(seq 4200 4223); do cells "$seed"; done)
free_mib() { nvidia-smi -i "$1" --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F', ' '{print $1 - $2}'; }
live_on() { local n=0 f; for f in "$OUT"/.pid_gpu"$1"_*; do [ -e "$f" ] && kill -0 "$(cat "$f")" 2>/dev/null && n=$((n + 1)); done; echo "$n"; }
start_queue() {   # gpu tag jobs...
  local g="$1" tag="$2"; shift 2
  setsid nohup bash "$Q" "$SHA" "$OUT" "$g" "$@" > "$OUT/queue_gpu${g}_${tag}.log" 2>&1 < /dev/null &
  echo "$!" > "$OUT/.pid_gpu${g}_${tag}"
  say "started queue $tag on gpu$g (pid $!), free was $(free_mib "$g") MiB"
}
STARTED=0
fill() {   # tag-prefix max-new jobs...: start queues where memory allows, one per GPU per pass
  local prefix="$1" max="$2" g; shift 2
  STARTED=0
  for g in 0 1 2 3; do
    [ "$STARTED" -ge "$max" ] && break
    if [ "$(free_mib "$g")" -ge "$NEED" ] && [ "$(live_on "$g")" -lt 3 ]; then
      start_queue "$g" "${prefix}$(date +%H%M%S)" "$@"; STARTED=$((STARTED + 1))
    fi
  done
}
say "waiting for all 24 B5 seeds to be claimed"
until [ "$(ls -d "$B5"/.claim_41[0-2][0-9] 2>/dev/null | wc -l)" -ge 24 ]; do sleep 60; done
say "B5 fully claimed; pilot 4299 next"
n=0
until [ "$n" -ge 4 ]; do
  fill p $((4 - n)) $PILOT
  n=$((n + STARTED))
  [ "$n" -ge 1 ] && [ "$(ls -d "$OUT"/.claim_4299_* 2>/dev/null | wc -l)" -ge 8 ] && break
  [ "$n" -ge 4 ] || sleep 120
done
until [ "$(ls "$OUT"/seed4299_*/summary.json 2>/dev/null | wc -l)" -ge 8 ]; do
  if grep -h "END 4299" "$OUT"/queue_gpu*_p*.log 2>/dev/null | grep -qv "exit 0"; then say "PILOT RUN FAILED"; exit 1; fi
  sleep 60
done
if ! CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 PYTHONPATH="$REL" "$PY" "$GATE" --gate "$OUT" 4299 > "$OUT/pilot_gate.txt" 2>&1; then
  say "PILOT GATE FAILED (see $OUT/pilot_gate.txt)"; exit 1
fi
say "pilot gate passed; study queues next"
until [ "$(ls -d "$OUT"/.claim_42[0-2][0-9]_* 2>/dev/null | wc -l)" -ge 192 ]; do
  fill s 4 $STUDY
  sleep 180
done
say "LAUNCHER DONE: all 192 study jobs claimed"
