#!/bin/bash
# One status digest for the ranking campaigns. Prints a few lines and exits;
# the caller emits them only when they CHANGE.
#
# It reports every TERMINAL state, not just the happy one. A heartbeat that only
# watches progress is silent through a crashloop, a dead dispatcher or a foreign
# user taking the card -- and silence is indistinguishable from "still running".
#
# v2, after a real miss: a run that FAILS writes config.json but never writes
# final_predictions_raw.csv, so `done < total` stays true forever and a campaign
# that finished with failures was reported as STALLED. That is the wrong verdict
# and it points at the wrong fix (relaunch the dispatcher, rather than fix the
# bug that killed the runs). Completion is now read from the dispatcher's own
# "ALL DONE" line, and failures are counted and named.
ROOT=/home/dsi/michaer8/optloss-rank/results
LOGS=$HOME/queue_logs
ME=michaer8
ALLDONE=1

log_for() {
    case "$1" in
        *MobileNetV3)  ls -t "$LOGS"/k3_mn3_*.log 2>/dev/null | head -1 ;;
        *MobileNetV2)  ls -t "$LOGS"/k3_mn2_*.log 2>/dev/null | head -1 ;;
        *RegNetY400MF) ls -t "$LOGS"/k3_rgn_*.log 2>/dev/null | head -1 ;;
    esac
}

for c in rank3_MobileNetV3 rank3_MobileNetV2 rank3_RegNetY400MF; do
    d=$(ls -d "$ROOT/$c"/*/*/*/*/seed_*/final_predictions_raw.csv 2>/dev/null | wc -l)
    t=$(ls -d "$ROOT/$c"/*/*/*/*/seed_*/config.json 2>/dev/null | wc -l)
    f=$(log_for "$c")

    # A dispatcher owns its campaign for the whole run. Read it off the PROCESS
    # environment, never off nvidia-smi: main.py releases the cuda context
    # between runs, so the card reads empty for a few seconds every time one
    # run ends.
    alive=0
    for p in $(pgrep -u "$ME" -f 'main\.py' 2>/dev/null); do
        e=$(tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | sed -n 's/^EXPERIMENT_DIR=//p')
        case "$e" in *"$c"*) alive=1 ;; esac
    done

    # The dispatcher's own verdict outranks a file count.
    finished=0
    [ -n "$f" ] && grep -q "ALL DONE" "$f" 2>/dev/null && finished=1
    nfail=0
    [ -n "$f" ] && nfail=$(grep -oE '[0-9]+ failed' "$f" 2>/dev/null | tail -1 | grep -oE '^[0-9]+')
    : "${nfail:=0}"

    if [ "$finished" = "1" ] && [ "$nfail" -gt 0 ]; then
        echo "FAILED   $c finished with $nfail FAILED of $t -- $d usable"
    elif [ "$finished" = "1" ]; then
        echo "DONE     $c $d/$t all runs usable"
    elif [ "$alive" = "1" ]; then
        echo "running  $c $d/$t"
        ALLDONE=0
    else
        # No dispatcher and no ALL DONE: it really did die mid-campaign.
        echo "ALERT    $c STALLED at $d/$t -- no dispatcher, no ALL DONE line"
        ALLDONE=0
    fi
done

# Never share a card. If a foreign user appears on one of ours, that is a
# decision the user has to make, not something to wait out.
for g in 1 2 3; do
    for p in $(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
        u=$(ps -o user= -p "$p" 2>/dev/null | xargs)
        [ -n "$u" ] && [ "$u" != "$ME" ] && echo "ALERT    GPU $g has foreign user $u -- we must not share it"
    done
done

# The CPU-fallback signature. dsisco01 prints float16 + GradScaler; enabled=False
# means it is training on the CPU with a perfectly healthy-looking log.
for l in k3_mn3 k3_mn2 k3_rgn; do
    f=$(ls -t "$LOGS/${l}"_*.log 2>/dev/null | head -1)
    [ -z "$f" ] && continue
    grep -q "AMP: enabled=False" "$f" 2>/dev/null && \
        echo "ALERT    $l shows AMP enabled=False -- CPU fallback, kill it"
done

# Distinct crash signatures, so a NEW failure mode is not hidden behind a known
# one. The unpack defect produced 88 identical lines; a second bug underneath it
# would have been invisible in a bare count.
for l in k3_mn3 k3_mn2 k3_rgn; do
    f=$(ls -t "$LOGS/${l}"_*.log 2>/dev/null | head -1)
    [ -z "$f" ] && continue
    grep -hoE "(ValueError|RuntimeError|TypeError|KeyError|AttributeError): [^)]*" "$f" 2>/dev/null \
        | grep -v "invalid literal for int" | sort -u | while read -r e; do
        echo "ALERT    $l error: $e"
    done
done

# The decisive trigger: the byte-identity check cannot run until a rank arm
# lands, and everything downstream waits on it.
for c in rank3_MobileNetV3 rank3_MobileNetV2 rank3_RegNetY400MF; do
    n=0
    for p in $(ls -d "$ROOT/$c"/*/*/*/*/seed_* 2>/dev/null); do
        [ -f "$p/final_predictions_raw.csv" ] || continue
        case "$p" in *"/rank_clip/"*|*"/aug_rank_clip/"*) n=$((n+1)) ;; esac
    done
    [ "$n" -gt 0 ] && echo "RANKARM  $c has $n usable rank-arm run(s)"
done

[ "$ALLDONE" = "1" ] && echo "ALLDONE  no ranking campaign is still running"

# LIVENESS, independent of the run counter. A run takes ~10 minutes, so the
# completed-run count is flat for long stretches and cannot distinguish mid-run
# from hung -- which cost a false stall reading once. The queue log gets an
# epoch line every few epochs, so its mtime is the real pulse: a live dispatcher
# whose log has been silent for >8 minutes is wedged, not working.
now=$(date +%s)
for l in k3_mn3 k3_mn2 k3_rgn; do
    f=$(ls -t "$LOGS/${l}"_*.log 2>/dev/null | head -1)
    [ -z "$f" ] && continue
    age=$(( (now - $(stat -c %Y "$f")) / 60 ))
    [ "$age" -gt 8 ] && echo "ALERT    $l log silent for ${age} min -- dispatcher alive but not training"
done
exit 0
