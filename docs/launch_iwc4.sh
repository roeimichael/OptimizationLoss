#!/usr/bin/env bash
# =============================================================================
#  results/iwc4  --  iwc3 RE-RUN AT FULL CONSTRAINT DOSE
# =============================================================================
#
#   why         `results/iwc3` is this project's strongest CALLABLE result:
#               `tralo` against its own lambda=0 twin loses ALL SIX
#               allocation-free metrics at 0 of 9 cells, BH q <= 0.0143,
#               while `tralo_reseed` ties on four of them -- so the
#               representation damage is attributable and the macro-F1
#               damage is NOT (a pure reseed costs MORE macro-F1 than the
#               constraint does). See FRAMEWORK 2(p-post).
#
#               🛑 IT WAS MEASURED AT 68.6% OF ITS DOSE. `iwc3` landed 716 of
#               1044 constraint steps and lost at least one in 36 of 36 runs.
#               The cause is not the arm: dsisco01 runs FP16 + GradScaler,
#               which SKIPS an optimizer step whose gradient overflows. Every
#               FP16 campaign here shows it and no BF16 one does:
#
#                   iwc1   float16    68.8%     24/32 runs
#                   iwc2   float16    74.6%      8/8
#                   iwc3   float16    68.6%     36/36
#                   xfam1  bfloat16  100.0%      0/108
#
#               FRAMEWORK 2(u) has the audit.
#
#   what        the SAME design, one knob different: `--constraint-fp32`,
#               which makes `finish_constraint_step` skip `scaler.unscale_`
#               entirely (constraint_step.py:221), so an FP16 loss scale can
#               no longer cancel a constraint step.
#
#   it answers TWO questions with one campaign, and both matter:
#     (1) does the 0-of-9 sweep SURVIVE at 100% dose? Less dose means less
#         constraint, and the finding is that the constraint DAMAGES the
#         representation -- so iwc3 is a LOWER bound and this should come
#         back at least as negative. If it comes back POSITIVE, the damage
#         was a dosing artefact and 2(p-post) is retracted.
#     (2) does `constraint_fp32` actually remove the FP16 loss? That decides
#         how to read every archived FP16 number, and it is read straight off
#         full_panel's CONSTRAINT DOSE block on the FIRST completed run --
#         not at the end.
#
#   ⚠️ FALSIFIABLE, stated before the data. `tralo` at 100% dose loses AP in
#   at least 8 of 9 cells, by a margin no smaller than iwc3's -0.0394.
#   Anything else is a result about DOSE, not about the constraint, and must
#   be reported as one.
#
#   arms        tralo         the treatment
#               tralo_null    its lambda=0 twin -- SAME warm-up, allocator
#                             and seed. Without it nothing is attributable
#                             (CLAUDE.md rule 1).
#               tralo_reseed  the noise floor: that null with the RNG stream
#                             perturbed and nothing else. It is what split
#                             iwc3's table in two.
#               clip          the equal-compute quality bar, and the STRONGER
#               focal_clip    of the two clippers. Both auto-added by
#                             gen_campaign's `mandatory_arms`.
#
#   size        9 cells x 5 arms x 4 seeds = 180 runs
#               9 cells is deliberate: the exact sign-test floor there is
#               0.00391 against BH 0.00455, so a 0-of-9 sweep is CALLABLE.
#               iwc1's 2 cells never could be.
#
#   host        dsisco01 (Quadro RTX 6000, FP16 + GradScaler) -- ON PURPOSE.
#               Running this on the BF16 host would answer question (1) and
#               silently skip question (2), because BF16 already lands 100%.
#
#   read it     python -m scripts.rig_status
#               python -m scripts.full_panel --campaign results/iwc4 --control tralo_null
#                 ^ 🛑 CONSTRAINT DOSE BLOCK FIRST, then RAW-PREDICTION
#                   IDENTITY, and only then a metric. If `tralo` is not at
#                   100%, STOP: the campaign is measuring the scaler.
#               python -m scripts.log_health results/iwc4
#               python -m scripts.full_panel --campaign results/iwc4 --control clip
#
# =============================================================================
set -euo pipefail

PIN=85233f4c                 # src/ carrying the dtype-safe probability clamp
                             # and the scoped gitver. Pinned, not "latest":
                             # a campaign generated on one commit and run on
                             # another is a split code_version.
TREE=~/optloss-iwc4          # its OWN worktree. NOT ~/optloss-audit (iwc3
                             # lives there), NOT ~/optloss-select (xfam1),
                             # NOT ~/optloss-uniform (uniform1 is LIVE).
                             # Five worktrees share one object store, so this
                             # fetches and checks out its own tree and runs
                             # NO git maintenance: never gc, prune, repack or
                             # worktree prune while any campaign is running.
ROOT=results/iwc4

# 🛑 REFUSE TO RUN FROM INSIDE THE TREE THIS SCRIPT IS ABOUT TO CHECK OUT.
# Bash reads a script incrementally, by byte offset, so a checkout of $TREE
# would rewrite this file underneath the interpreter at an offset it has not
# reached yet. Copy it out first:
#
#     git show origin/headroom/small-cnn:docs/launch_iwc4.sh > ~/launch_iwc4.sh
#     GPU=3 bash ~/launch_iwc4.sh
#
SELF=$(cd "$(dirname "$0")" && pwd -P)
TREEP=$(cd "$(eval echo $TREE)" 2>/dev/null && pwd -P || true)
case "${TREEP:-__none__}" in
  "") ;;
  *) case "$SELF/" in
       "$TREEP"/*) echo "REFUSING: this script lives inside \$TREE ($TREEP),"
                   echo "  and it is about to git-checkout that tree. Copy it"
                   echo "  out and run the copy -- see the block above."
                   exit 1 ;;
     esac ;;
esac

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
echo "Check the other host by hand before launching."

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
# `train_images.npy` -- which cost a launch on 2026-08-25. Link the ARRAYS,
# and verify the file the runner actually opens.
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
    --models MobileNetV2 MobileNetV3 RegNetY400MF \
    --caps L20_G50 L30_G50 L50_G30 \
    --arms tralo tralo_null tralo_reseed \
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
echo 0 > /tmp/gpuchoice_iwc4
setsid env CUDA_VISIBLE_DEVICES="$GPU" EXPERIMENT_DIR="$ROOT" \
    PYTHONIOENCODING=utf-8 nohup "$PY" main.py \
    > /tmp/iwc4.log 2>&1 < /tmp/gpuchoice_iwc4 &
sleep 25
echo "launched $ROOT on GPU $GPU from $TREE at $PIN"
tail -5 /tmp/iwc4.log || true
