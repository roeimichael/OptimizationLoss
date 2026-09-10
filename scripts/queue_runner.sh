#!/bin/bash
# queue_runner.sh <gpu> <label> <worktree>:<root> [<worktree>:<root> ...]
#
# Runs a LIST of campaigns back to back on ONE gpu, waiting for that gpu to be
# free before each. Written 2026-09-10 so the rig keeps working for a full day
# without a person in the loop.
#
# WHY IT IS THIS PARANOID. Every rule below is a failure this project already
# paid for:
#   * it NEVER shares a gpu with another user, and re-checks ownership right
#     before each claim rather than once at the start -- dsisco02's driver
#     crashes the HOST on gpu sharing;
#   * it ACTIVATES THE CONDA ENV ITSELF and then PROVES torch can see a cuda
#     device before dispatching -- see the block below, this one bit on the
#     first launch;
#   * it owns exactly ONE gpu, so two runners on a host is the documented
#     2-gpu ceiling and a third cannot appear by accident;
#   * it runs `data_present` before every campaign, because a fresh worktree
#     passes every launch gate and then fails 24 runs in 120 seconds on a
#     gitignored .npy;
#   * it refuses a campaign whose configs are not a SINGLE code_version --
#     a split stamp means the tree moved under a staged campaign.
#
# 🛑 THE CPU TRAP, MEASURED 2026-09-10 AND THE REASON assert_gpu_ready EXISTS.
# The first launch did `conda activate optloss` in the ssh shell and then
# `setsid nohup bash queue_runner.sh`. The activation did NOT survive into the
# detached shell: the runner's child came up as
# `~/anaconda3/bin/python` -- BASE, whose torch is CPU-only -- and trained
# fmow x MobileNetV2 on the CPU at 120 cores with GPU 2 sitting at 0% / 3 MiB
# and ZERO nvidia fds on the process. Nothing raised. The campaign reported a
# `running` status and would have produced numbers.
# Two things make that impossible now: the env is activated HERE by absolute
# path rather than inherited, and no campaign is dispatched until `python`
# resolves inside that env AND `torch.cuda.is_available()` is true under the
# very CUDA_VISIBLE_DEVICES the run will use. `rig_status` documents this
# class ("a launch that ran 40 runs on CPU"); this is the same check moved to
# the point where it can actually stop the launch.
#
# It does NOT resume, retry or reset anything. A campaign that ends with
# pending runs is left exactly as it is for a person to read.

set -u

GPU="$1"; shift
LABEL="$1"; shift
QUEUE=("$@")

ENV_NAME="optloss"
CONDA_SH="$HOME/anaconda3/etc/profile.d/conda.sh"

LOG_DIR="$HOME/queue_logs"
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/${LABEL}.log"

say() { echo "[$(date '+%F %T')] $*" | tee -a "$MAIN_LOG"; }

me="$(whoami)"

if [ ! -f "$CONDA_SH" ]; then
    say "ABORT: no conda profile at $CONDA_SH"
    exit 4
fi
# shellcheck disable=SC1090
. "$CONDA_SH"
conda activate "$ENV_NAME" || { say "ABORT: cannot activate $ENV_NAME"; exit 4; }

# Refuse to dispatch unless torch really sees a device on THIS gpu. Checked
# with CUDA_VISIBLE_DEVICES already exported, so it tests what the run gets.
assert_gpu_ready() {
    local py rc name
    py="$(command -v python || true)"
    case "$py" in
        *"/envs/$ENV_NAME/bin/python") : ;;
        *)  say "ABORT: python is '$py', not the $ENV_NAME env -- base torch is CPU-only"
            exit 4 ;;
    esac
    CUDA_VISIBLE_DEVICES="$GPU" python -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() and torch.cuda.device_count() >= 1 else 1)'
    rc=$?
    if [ "$rc" != "0" ]; then
        say "ABORT: torch.cuda unavailable under $py (rc=$rc) -- this would train on CPU"
        exit 4
    fi
    name="$(CUDA_VISIBLE_DEVICES=$GPU python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null)"
    say "gpu ready: $py sees '$name' as cuda:0 (host gpu $GPU)"
}

# Every distinct user with a compute process on our gpu, us included.
gpu_users() {
    local pids u out=""
    pids="$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null)"
    for p in $pids; do
        u="$(ps -o user= -p "$p" 2>/dev/null | xargs)"
        [ -n "$u" ] && out="$out $u"
    done
    echo "$out" | tr ' ' '\n' | grep -v '^$' | sort -u | tr '\n' ' '
}

# Block until the gpu has no compute process at all. Abort the queue the moment
# somebody else appears on it -- we do not queue behind another user.
wait_for_gpu() {
    local users
    while true; do
        users="$(gpu_users)"
        if [ -z "$users" ]; then return 0; fi
        for u in $users; do
            if [ "$u" != "$me" ]; then
                say "ABORT: gpu $GPU has a foreign user ($u). Never sharing a gpu."
                exit 3
            fi
        done
        sleep 120
    done
}

say "queue $LABEL starting on gpu $GPU with ${#QUEUE[@]} campaign(s)"
for entry in "${QUEUE[@]}"; do say "   queued: $entry"; done
assert_gpu_ready

for entry in "${QUEUE[@]}"; do
    WT="${entry%%:*}"
    ROOT="${entry#*:}"
    NAME="$(basename "$ROOT")"
    RUN_LOG="$LOG_DIR/${LABEL}_${NAME}.log"

    if [ ! -d "$WT/$ROOT" ]; then
        say "SKIP $NAME -- $WT/$ROOT does not exist"
        continue
    fi

    pending="$(grep -l '"status": "pending"' "$WT/$ROOT"/*/*/*/*/*/config.json 2>/dev/null | wc -l)"
    if [ "$pending" -eq 0 ]; then
        say "SKIP $NAME -- 0 pending runs, nothing to dispatch"
        continue
    fi

    # One stamp per campaign, or the tree moved under it.
    stamps="$(cd "$WT" && python -c "
import glob, json, sys
v = {json.load(open(f))['code_version']
     for f in glob.glob('$ROOT/*/*/*/*/*/config.json')}
print(len(v)); print(sorted(v)[0] if v else 'none')
" 2>/dev/null | head -2 | tr '\n' ' ')"
    n_stamp="$(echo "$stamps" | awk '{print $1}')"
    if [ "$n_stamp" != "1" ]; then
        say "SKIP $NAME -- $n_stamp distinct code_version stamps, not 1 ($stamps)"
        continue
    fi

    if ! (cd "$WT" && python -m scripts.data_present "$ROOT" >>"$RUN_LOG" 2>&1); then
        say "SKIP $NAME -- data_present RED, see $RUN_LOG"
        continue
    fi

    say "waiting for gpu $GPU to free, for $NAME ($pending pending, stamp $stamps)"
    wait_for_gpu
    assert_gpu_ready
    say "claiming gpu $GPU for $NAME  (worktree $WT)"

    (
        cd "$WT" || exit 1
        export EXPERIMENT_DIR="$ROOT"
        export CUDA_VISIBLE_DEVICES="$GPU"
        exec python -u main.py
    ) >>"$RUN_LOG" 2>&1

    rc=$?
    done_n="$(grep -l '"status": "completed"' "$WT/$ROOT"/*/*/*/*/*/config.json 2>/dev/null | wc -l)"
    tot_n="$(ls -d "$WT/$ROOT"/*/*/*/*/*/ 2>/dev/null | wc -l)"
    say "$NAME finished rc=$rc  completed $done_n/$tot_n  log $RUN_LOG"
done

say "queue $LABEL DONE -- no campaigns left"
