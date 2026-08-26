#!/usr/bin/env bash
# =============================================================================
#  results/vitu1  --  DOES THE UNIFORM COUNT STILL FREE THE CONSTRAINT ON ViTB16?
# =============================================================================
#
#   why         `results/uniform1` is the first positive result this project
#               has had. `tralo_uniform` replaces the `sum` count (per-item
#               gradient `p(1-p)`, largest exactly where the model is most
#               confident) with a straight-through log-odds count whose
#               per-item weight is FLAT, and the ranking damage disappears:
#
#                   AP vs its own lambda=0 twin
#                     tralo          -0.0754   0/9 cells   *** LOSS  q=0.0072
#                     tralo_uniform  +0.0030   6/3 cells   tie
#
#               and it is NOT a cheap tie -- it still pulls the raw count
#               4.2x the RNG floor against `tralo`'s 7.5x, i.e. it keeps 56%
#               of the enforcement at zero ranking cost. FRAMEWORK 2(w).
#
#   🛑 WHAT uniform1 COULD NOT TEST. It ran MobileNetV2, MobileNetV3 and
#   RegNetY400MF. **ViTB16 is the backbone where the damage was LARGEST** --
#   2(t) measured the constraint evicting -30.8 items per cell there against
#   -3.4 on MobileNetV3, a 9x spread. A fix demonstrated only where the
#   defect is mild is not demonstrated. This runs the same design on the
#   backbone with the most to lose.
#
#   ⚠️ FALSIFIABLE, stated before the data. **`tralo_uniform` on ViTB16 loses
#   AP against its own `tralo_null` in at least 2 of 3 cells, by a margin
#   exceeding `tralo_reseed`'s on the SAME backbone.** That is the outcome
#   that would confine 2(w) to light backbones. The reseed term is not
#   optional: on RegNetY400MF in uniform1 the RNG floor is itself -0.0255, so
#   two of three cells that read as losses are smaller than reseeding alone.
#   Score against the floor of the backbone the arm ran on, never a campaign
#   mean.
#
#   what        the SAME design as uniform1, one factor changed: the backbone.
#               Same three caps, same seeds, same grad mode, so the ViTB16
#               rows drop straight into 2(w)'s per-backbone table.
#
#   ⛔ WHAT THIS DELIBERATELY IS NOT. The tempting follow-up -- "it is free,
#   so buy more of it" -- is void twice over and must not be launched:
#     * magnitude is not a lever. Under `constraint_grad_mode: clip` the step
#       delivered is exactly `lr*clip` whatever lambda says, so a lambda
#       sweep would be a FIFTH inert flag (house rule 3, four already).
#     * step COUNT cannot rise without breaking equal compute against the
#       clippers, which rule 2 forbids, and warm-up 50 closes it from the
#       other side.
#   The live axis is the backbone, which is why this campaign is a backbone.
#
#   arms        tralo          the damaged reference, so the contrast is
#                              WITHIN this campaign and not against uniform1
#               tralo_uniform  the treatment
#               tralo_null     the lambda=0 twin -- one null serves BOTH
#                              trained arms, because at lambda=0 the count
#                              function is never evaluated and `sum` and
#                              `uniform` coincide exactly
#               tralo_reseed   the per-backbone RNG floor. It is the term the
#                              falsification condition above is stated in.
#               clip           the equal-compute quality bar, and the stronger
#               focal_clip     of the two. Both auto-added by gen_campaign.
#
#               `tralo_head` is deliberately DROPPED. uniform1 showed it ties
#               everything because it barely constrains -- 1.7x the RNG floor
#               against tralo_uniform's 4.2x -- so its tie carries no
#               information and it is not worth a seventh of the campaign.
#
#   size        3 cells x 6 arms x 4 seeds = 72 runs
#               3 cells is one backbone x 3 caps. Note the sign-test floor at
#               3 cells is 0.25, so a 3-of-3 sweep is NOT callable on its own
#               -- this campaign is powered to move the per-backbone table in
#               2(w), not to stand alone. Say that when quoting it.
#
#   host        dsisco02 (RTX PRO 6000 Blackwell, BF16 AMP), GPU 0. BF16
#               already lands 100% of the dose; `--constraint-fp32` is kept
#               anyway so the knob set matches uniform1 exactly.
#
#   read it     python -m scripts.rig_status
#               python -m scripts.dose_landed results/vitu1
#                 ^ 🛑 FIRST, AND ON THE RUNNING CAMPAIGN. One arm low = the
#                   loss shape; every arm low = the host.
#               python -m scripts.full_panel --campaign results/vitu1 --control tralo_null
#               python -m scripts.full_panel --campaign results/vitu1 --control tralo_null --percell
#                 ^ the per-cell block is what feeds 2(w)'s table, and it is
#                   the only way to see the span across cap levels.
#               python -m scripts.log_health results/vitu1
#               python -m scripts.full_panel --campaign results/vitu1 --control clip
#
# =============================================================================
set -euo pipefail

PIN=74f85865                 # src/ carrying the dtype-safe probability clamp
                             # AND clamp_denominator. The only training-path
                             # delta from uniform1's 1e7829c7 is the latter,
                             # which is reached solely from
                             # `if SOFT_COUNT_MODE == "margin"`
                             # (tralo/train.py:279) -- no arm here uses margin,
                             # so the two campaigns are behaviourally identical
                             # for these arms. Verified by AST, not by grep.
TREE=~/optloss-vitu          # its OWN worktree. NOT ~/optloss-iwc4 (iwc4 is
                             # LIVE on dsisco01), NOT ~/optloss-audit,
                             # ~/optloss-select or ~/optloss-uniform.
                             # Worktrees share one object store, so this
                             # fetches and checks out its own tree and runs NO
                             # git maintenance: never gc, prune, repack or
                             # worktree prune while any campaign is running.
ROOT=results/vitu1

# 🛑 REFUSE TO RUN FROM INSIDE THE TREE THIS SCRIPT IS ABOUT TO CHECK OUT.
# Bash reads a script incrementally, by byte offset, so a checkout of $TREE
# would rewrite this file underneath the interpreter at an offset it has not
# reached yet. Copy it out first:
#
#     git show origin/headroom/small-cnn:docs/launch_vitu1.sh > ~/launch_vitu1.sh
#     GPU=0 bash ~/launch_vitu1.sh
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
echo "iwc4 holds one on dsisco01, so this campaign is the second and last."

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
    --caps L20_G50 L30_G50 L50_G30 \
    --arms tralo tralo_uniform tralo_null tralo_reseed \
    --constraint-grad-mode clip \
    --constraint-fp32

# THE THREE GATES. Each refuses a different way to waste a week.
"$PY" -m scripts.audit_config
"$PY" -m scripts.check_parity "$ROOT"
"$PY" -m scripts.smoke_arms

GPU=${GPU:-0}
# main.py prompts for a GPU and reads the answer from stdin. With
# CUDA_VISIBLE_DEVICES pinned to one device it sees exactly one, so the answer
# is always index 0 -- NOT $GPU, which is the physical id.
echo 0 > /tmp/gpuchoice_vitu1
setsid env CUDA_VISIBLE_DEVICES="$GPU" EXPERIMENT_DIR="$ROOT" \
    PYTHONIOENCODING=utf-8 nohup "$PY" main.py \
    > /tmp/vitu1.log 2>&1 < /tmp/gpuchoice_vitu1 &
sleep 25
echo "launched $ROOT on GPU $GPU from $TREE at $PIN"
tail -5 /tmp/vitu1.log || true
