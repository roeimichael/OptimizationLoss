#!/usr/bin/env bash
# =============================================================================
# 🛑🛑 SUPERSEDED 2026-09-01 -- 1 OF ITS 2 (backbone x cap) CELLS POSE
#
#   ARCHIVED 2026-09-02 -- AND THE RECIPE IS THE REASON, not the cap.
#   This campaign ran grad_mode: clip, so it is a DIFFERENT METHOD from the
#   current recipe (iwildcam + constraint_fp32:True +
#   constraint_grad_mode:normalize). It is moved out of results/ to
#   `~/optloss-archive-stale-2026-09-02/` and must not be pooled with
#   the corpus. Its caps are fine; its method is not the one we score.
#
#    NO QUESTION. This campaign has already RUN, so the banner is about how to
#    READ it as much as about re-running it.
#
#   A cap poses a question only where it evicts >= 10 predictions, leaves
#   ERRORS inside K, and cuts at p@K < 0.99 (FRAMEWORK 2(z16), 2(z17)).
#   Classified against `configs/task_windows.yml` on 2026-09-01:
#
#     ViTB16        L80_G95     NON-TASK   class 7 K/n=0.798 vs 0.90-1.00
#     ViTB16        L90_G95     ✅ TASK
#
#   ⚠️ AND READ 2(z24) BEFORE QUOTING ANY NUMBER FROM THIS CAMPAIGN. The
#   window above is a MEAN over seeds whose unconstrained counts spread 105
#   items, so a cell marked TASK here can still be one whose cap binds in only
#   some seeds. `scripts.task_window` now reports `binds n/N` per fraction; run
#   it on THIS campaign's own `tralo_null` runs rather than trusting the row.
#
#   `configs/gen_campaign.py` REFUSES the dead caps now, so re-running this
#   script as written exits non-zero. That refusal is correct, not a broken
#   generator. Re-issue with per-class tags (`L<c2>-<c7>_G<g>`) inside both
#   classes' windows -- the two capped classes' windows differ on every
#   backbone, so one fraction cannot sit inside both.
# =============================================================================
# =============================================================================
#  results/loosevit1  --  THE LOOSE-CAP REGIME ON ViTB16, THE WORST-HIT BACKBONE
# =============================================================================
#
#   why         Every campaign this project has run sweeps L20-L50, and 2(v)
#               now prices what that costs: at those caps the WHOLE gap to a
#               perfect ranking is 0.42 to 4.08 items against a paired seed sd
#               of 7.6 to 16.7, so **a method capturing 100% of the prize
#               would still not be detectable**. Converted to seeds per cell
#               at 80% power that is 2607 at L20 and 546 at L30/L50.
#
#               The same table says 20 seeds at K/n = 0.8 and SEVEN at 0.9,
#               against the four the protocol already runs. So the reason
#               nothing has ever been measurable here is the CAP LEVEL, and
#               that choice was never priced.
#
#   🔑 AND THE WORK-TO-PRIZE RATIO SAYS THE SAME THING MECHANISTICALLY.
#   Measured on iwc4's 36 unconstrained runs: the model's raw argmax count is
#   368.2 for class 2 (n=370) and 465.2 for class 7 (n=456), i.e. raw/n ~ 1.0.
#   So the items the constraint must EVICT, against the prize on offer:
#
#       K/n     evict    prize    evictions per item of prize
#       20%       294     0.42     700x     <- protocol
#       30%       257     1.17     220x     <- protocol
#       50%       183     4.08      45x
#       70%       109    11.50     9.5x
#       80%        72    18.00     4.0x
#       90%        35    29.83     1.2x
#
#   At L20 the constraint does ~700 items of work for every item it could
#   possibly gain, so it CANNOT avoid evicting correct ones -- which is
#   exactly what 2(t) measured. At L90 the work and the prize are matched.
#
#   🛑 THE OBVIOUS OBJECTION, CHECKED BEFORE SPENDING THE GPU. "A loose cap
#   does not bind, so the treatment is vacuous." **It binds.** At K/n = 0.9
#   the budget is 333 against a raw count of 368, still 35 items over, on both
#   capped classes and at every level in the table. The cap binds WEAKLY
#   (8.4x less pressure than L20), and weakly is the point: that is the regime
#   where the constraint is not forced to spend the prize backwards.
#
#   ⚠️ FALSIFIABLE, stated before the data, and it is the whole reason this is
#   worth running: **a null here is INFORMATIVE and a null at L20-L50 is not.**
#     * If `tralo_uniform` beats `clip` on ccF1 in at least 5 of 6 cells, this
#       is the first measurable win the method has ever produced.
#     * If it ties, the method is CLOSED on iwildcam -- because this is the
#       one regime with the resolution to have seen a win, so a tie here is a
#       measurement rather than the absence of one.
#   Either outcome is a result. That is not true of any campaign run so far.
#
#   what        the uniform1 design at two LOOSE caps. `G95 > L`, so the local
#               scope binds and the global is inert by construction -- the
#               same scope uniform1 and iwc4 used, changed in level only.
#
#   arms        tralo          the damaged reference, in-campaign
#               tralo_uniform  the fix (2(w)): free at protocol caps, and the
#                              only arm that could plausibly win here
#               tralo_null     the lambda=0 twin, one serves both
#               tralo_reseed   the per-backbone RNG floor
#               clip           the equal-compute quality bar, and the STRONGER
#               focal_clip     of the two. Both auto-added by gen_campaign.
#
#   size        2 cells x 6 arms x 4 seeds = 48 runs
#               2 cells is ONE backbone x 2 caps, so the sign-test floor here
#               is 0.25 and this campaign CANNOT call anything alone. It is
#               run to be scored WITH vitu1 --
#                   full_panel --campaign results/vitu1 results/loosevit1
#               which gives ViTB16 five cells (floor 1/32 = 0.031) spanning
#               K/n from 0.2 to 0.9, and it is pinned to vitu1's own commit
#               so the two are one code_version. Say the cell count out loud
#               when quoting it; 2 cells alone is not a result.
#
#               ViTB16 earns this because it is where the damage is LARGEST:
#               vitu1 measured `tralo` at AP -0.0933 against a -0.0142 RNG
#               floor, the biggest effect anywhere in the project.
#
#   host        dsisco01 (Quadro RTX 6000, FP16 + GradScaler), GPU 3,
#               with `--constraint-fp32` -- which iwc4 verified removes
#               the FP16 dose loss entirely, 1044/1044 over 36 runs.
#
#   read it     python -m scripts.rig_status
#               python -m scripts.dose_landed results/loosevit1
#               python -m scripts.paired_noise --campaign results/loosevit1
#                 ^ 🛑 the seed count for THIS regime, measured rather than
#                   extrapolated off iwc3's curve.
#               python -m scripts.full_panel --campaign results/loosevit1 --control tralo_null
#               python -m scripts.full_panel --campaign results/loosevit1 --control clip
#                 ^ the second one carries the falsification condition above.
#
# =============================================================================
set -euo pipefail

PIN=74f85865                 # the same training path as uniform1 and vitu1:
                             # clamp_probability live, clamp_denominator
                             # reachable only from soft_count_mode == margin,
                             # which no arm here uses.
TREE=~/optloss-loosevit         # its OWN worktree. Worktrees share one object
                             # store, so this fetches and checks out its own
                             # tree and runs NO git maintenance: never gc,
                             # prune, repack or worktree prune while any
                             # campaign is running.
ROOT=results/loosevit1

# 🛑 REFUSE TO RUN FROM INSIDE THE TREE THIS SCRIPT IS ABOUT TO CHECK OUT.
# Bash reads a script incrementally, by byte offset, so a checkout of $TREE
# would rewrite this file underneath the interpreter at an offset it has not
# reached yet. Copy it out first:
#
#     git show origin/headroom/small-cnn:docs/launch_loosevit1.sh > ~/launch_loosevit1.sh
#     GPU=0 bash ~/launch_loosevit1.sh
#
SELF=$(cd "$(dirname "$0")" && pwd -P)
TREEP=$(cd "$(eval echo $TREE)" 2>/dev/null && pwd -P || true)
# `[ -n "$TREEP" ]`, NOT `${TREEP:-__none__}`: on a FIRST launch $TREE does not
# exist yet, so TREEP is empty -- and an empty prefix turns the glob below into
# `/*`, which matches every absolute path and refuses unconditionally.
if [ -n "$TREEP" ]; then
    case "$SELF/" in
      "$TREEP"/*) echo "REFUSING: this script lives inside \$TREE ($TREEP),"
                  echo "  and it is about to git-checkout that tree. Copy it"
                  echo "  out and run the copy -- see the block above."
                  exit 1 ;;
    esac
fi

# THE DISPATCHER IS NOT THE ONLY PROCESS. main.py spawns each run as
# `python -u -m src.experiments.runner <config>`, whose command line contains
# NO "main.py", so a killed dispatcher leaves runners a main.py-only pgrep
# cannot see. Check for BOTH. `|| true` because pgrep exits 1 on no match.
BUSY=$( { pgrep -u "$(whoami)" -f "envs/optloss/bin/python .*main.py" || true
          pgrep -u "$(whoami)" -f "src.experiments.runner"           || true
        } | sort -u | wc -l)
if [ "$BUSY" -gt 0 ]; then
    echo "REFUSING: $BUSY dispatcher/runner process(es) already alive as $(whoami)"
    echo "  ON THIS HOST. Stop them by explicit PID -- never pkill -- then:"
    echo "      python -m scripts.rig_status"
    exit 1
fi

MINE=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
       | while read -r p; do ps -o user= -p "${p// /}" 2>/dev/null; done \
       | grep -c "$(whoami)" || true)
if [ "${MINE:-0}" -ge 2 ]; then
    echo "REFUSING: $MINE GPUs on this host already carry my processes."
    exit 1
fi
echo "the house limit is 2 GPUs ACROSS THE CLUSTER and this guard is per-host."
echo "loose1 takes the other one on dsisco02: this is the second and last."

if [ ! -d "$TREE" ]; then
    git -C ~/OptimizationLoss worktree add "$TREE" --detach "$PIN"
fi
cd "$TREE"
git -c gc.auto=0 fetch -q origin
git checkout -q --detach "$PIN"
test -z "$(git status --porcelain src/ configs/ main.py)" || {
    echo "REFUSING: src/ configs/ main.py are dirty -- code_version would be"
    echo "  stamped -dirty and the campaign would read as split."; exit 1; }

# 🛑 DO NOT TEST FOR THE DIRECTORY. `data/iwildcam/oodslice/train_meta.csv`
# and `test_meta.csv` are TRACKED IN GIT, so checking out ANY commit creates
# that directory holding those two CSVs and nothing else. A `[ -e data/... ]`
# guard then sees the directory, skips the link, and every run dies on
# `train_images.npy`. Link the ARRAYS, and verify the file the runner opens.
IWSRC=~/optloss-audit/data/iwildcam/oodslice
IWDST=data/iwildcam/oodslice
test -d "$IWSRC" || { echo "REFUSING: $IWSRC does not exist on this host."; exit 1; }
mkdir -p "$IWDST"
for f in "$IWSRC"/*.npy; do
    b=$(basename "$f")
    [ -e "$IWDST/$b" ] || ln -s "$f" "$IWDST/$b"
done
for need in train_images train_labels test_images test_labels; do
    test -s "$IWDST/$need.npy" || {
        echo "REFUSING: $IWDST/$need.npy is missing or empty after linking."
        echo "  Every run would die on it, instantly, and reset to pending."
        exit 1; }
done
echo "data OK: $(ls "$IWDST"/*.npy | wc -l) arrays linked from $IWSRC"

PY=$HOME/anaconda3/envs/optloss/bin/python
"$PY" -m configs.gen_campaign \
    --root "$ROOT" \
    --datasets iwildcam \
    --models ViTB16 \
    --caps L80_G95 L90_G95 \
    --arms tralo tralo_uniform tralo_null tralo_reseed \
    --constraint-grad-mode clip \
    --constraint-fp32

# THE THREE GATES. Each refuses a different way to waste a week.
"$PY" -m scripts.audit_config
"$PY" -m scripts.check_parity "$ROOT"
"$PY" -m scripts.smoke_arms

GPU=${GPU:-3}
# main.py prompts for a GPU and reads the answer from stdin. With
# CUDA_VISIBLE_DEVICES pinned to one device it sees exactly one, so the answer
# is always index 0 -- NOT $GPU, which is the physical id.
echo 0 > /tmp/gpuchoice_loosevit1
setsid env CUDA_VISIBLE_DEVICES="$GPU" EXPERIMENT_DIR="$ROOT" \
    PYTHONIOENCODING=utf-8 nohup "$PY" main.py \
    > /tmp/loosevit1.log 2>&1 < /tmp/gpuchoice_loosevit1 &
sleep 25
echo "launched $ROOT on GPU $GPU from $TREE at $PIN"
tail -5 /tmp/loosevit1.log || true
