#!/usr/bin/env bash
# Autonomous repository cleanup loop.
#
#   bash run_autoclean.sh            # run until STATUS: COMPLETE or a guard trips
#   MAX_ITERS=30 bash run_autoclean.sh
#   DRY_RUN=1 bash run_autoclean.sh  # show what would happen, launch nothing
#
# Stops on: STATUS: COMPLETE, iteration cap, or three consecutive no-progress
# sessions. It never pushes and never leaves the working branch.
set -euo pipefail

STATE_FILE="CLEANUP_STATE.md"
PROMPT_FILE="CLEANUP_PROMPT.md"
LOG_FILE="autoclean.log"
BRANCH="${BRANCH:-chore/autoclean}"
MAX_ITERS="${MAX_ITERS:-20}"
STALL_LIMIT="${STALL_LIMIT:-3}"
DRY_RUN="${DRY_RUN:-0}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }
die() { log "ABORT: $*"; exit 1; }

# ---------------------------------------------------------------- guards ----
command -v claude >/dev/null 2>&1 || die "claude CLI not on PATH."
command -v git    >/dev/null 2>&1 || die "git not on PATH."

git rev-parse --show-toplevel >/dev/null 2>&1 || die "not inside a git repository."
# Anchor at the repo root. Do not compare paths: on Windows git reports C:/... while
# pwd reports /c/..., and a string compare there fails on a perfectly valid tree.
cd "$(git rev-parse --show-toplevel)"
[ -f "$PROMPT_FILE" ] || die "$PROMPT_FILE is missing (expected at the repository root)."

# The agent may only touch tracked files, and only outside the training path.
# Start from a clean training path so any dirt at the end is provably the agent's.
if [ -n "$(git status --porcelain src/ configs/ main.py tests/ 2>/dev/null)" ]; then
  die "src/ configs/ main.py tests/ are dirty. Commit or stash before starting."
fi

CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [ "$CURRENT_BRANCH" != "$BRANCH" ] && [ "$DRY_RUN" != "1" ]; then
  # Untracked files survive a branch switch, so only tracked modifications block it.
  if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    die "tracked files are modified; cannot switch to $BRANCH. Commit or stash first."
  fi
  log "switching to working branch $BRANCH (from $CURRENT_BRANCH)"
  git switch -c "$BRANCH" 2>/dev/null || git switch "$BRANCH"
fi

if [ ! -f "$STATE_FILE" ]; then
  log "seeding $STATE_FILE"
  {
    echo "STATUS: IN_PROGRESS"
    echo "CURRENT_PHASE: 1_DISCOVERY"
    echo "LAST_UPDATED: $(date '+%Y-%m-%d')"
    echo "NEXT_STEP: Run git ls-files and classify every tracked file into CLEANUP_AUDIT.md."
    echo "NOTES:"
  } > "$STATE_FILE"
fi

if [ "$DRY_RUN" = "1" ]; then
  log "DRY_RUN: branch=$BRANCH max_iters=$MAX_ITERS state=$(head -1 $STATE_FILE)"
  log "DRY_RUN: would run -> claude -p \"\$(cat $PROMPT_FILE)\" --dangerously-skip-permissions"
  exit 0
fi

# ------------------------------------------------------------------ loop ----
iter=0
stall=0
prev_fingerprint=""

while :; do
  if head -1 "$STATE_FILE" | grep -q "STATUS: COMPLETE"; then
    log "STATUS: COMPLETE -- cleanup pipeline finished."
    log "review with:  git log --oneline $CURRENT_BRANCH..$BRANCH  &&  cat CLEANUP_AUDIT.md"
    exit 0
  fi

  iter=$((iter + 1))
  if [ "$iter" -gt "$MAX_ITERS" ]; then
    die "hit MAX_ITERS=$MAX_ITERS without completing. State: $(sed -n 2p "$STATE_FILE")"
  fi

  prev_fingerprint="$(git rev-parse HEAD)-$(git hash-object "$STATE_FILE")"

  log "=== session $iter/$MAX_ITERS === $(sed -n 2p "$STATE_FILE")"
  set +e
  claude -p "$(cat "$PROMPT_FILE")" --dangerously-skip-permissions 2>&1 | tee -a "$LOG_FILE"
  rc=${PIPESTATUS[0]}
  set -e
  log "session $iter exited rc=$rc"

  # A violation of the training-path freeze is the one thing worth stopping for.
  if [ -n "$(git status --porcelain src/ configs/ main.py tests/ 2>/dev/null)" ]; then
    log "!! agent modified the frozen training path; reverting those paths"
    git checkout -- src/ configs/ main.py tests/ || true
    die "training path was touched (see $LOG_FILE). Reverted src/ configs/ main.py tests/."
  fi

  new_fingerprint="$(git rev-parse HEAD)-$(git hash-object "$STATE_FILE")"
  if [ "$new_fingerprint" = "$prev_fingerprint" ]; then
    stall=$((stall + 1))
    log "no progress (no new commit, no state change) -- stall $stall/$STALL_LIMIT"
    [ "$stall" -lt "$STALL_LIMIT" ] || die "stalled $STALL_LIMIT sessions in a row. Inspect $LOG_FILE."
  else
    stall=0
  fi

  # Escalating backoff: covers a rate limit without hammering it.
  backoff=$((60 * (stall + 1)))
  log "sleeping ${backoff}s before the next session"
  sleep "$backoff"
done
