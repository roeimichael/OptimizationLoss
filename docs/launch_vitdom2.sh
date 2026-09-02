#!/usr/bin/env bash
# =============================================================================
# 🛑🛑 SUPERSEDED 2026-09-01 -- DO NOT RUN THIS AS WRITTEN.
#
#   ARCHIVED 2026-09-02 -- AND THE RECIPE IS THE REASON, not the cap.
#   This campaign ran caps the task-window gate rejects, so it is a DIFFERENT METHOD from the
#   current recipe (iwildcam + constraint_fp32:True +
#   constraint_grad_mode:normalize). It is moved out of results/ to
#   `~/optloss-archive-stale-2026-09-02/` and must not be pooled with
#   the corpus. Its caps are fine; its method is not the one we score.
#
#
#   108 staged runs, 72 of them at a dead cap. 2 of its 3 cap
#   tags pose no question.
#
#   The task window (FRAMEWORK 2(z16)/2(z17), measured on all four backbones)
#   says a cap poses a question only where it evicts >= 10 predictions, leaves
#   ERRORS inside K, and cuts at p@K < 0.99. Classified against
#   `configs/task_windows.yml` on 2026-09-01:
#
#     ViTB16  L30_G50   NON-TASK   both classes K/n=0.300
#     ViTB16  L60_G95   NON-TASK   c7 K/n=0.599, window 0.90-1.00
#     ViTB16  L90_G95   ✅ TASK    c2 0.900, c7 0.901
#
#   🔑 ITS STATED PURPOSE WAS THE MID CAPS, AND THAT PURPOSE SURVIVES.
#      "L55-L70 is 0 runs anywhere" is still true and still worth closing.
#      What is wrong is that a bare `L60_G95` puts class 7 at K/n=0.599
#      against a 0.90-1.00 window, so the mid cell measures nothing. The
#      per-class form fixes exactly this.
#
#   `configs/gen_campaign.py` now REFUSES these caps, so this script exits
#   non-zero as written. That refusal is the correct outcome and this banner
#   exists so the refusal is not mistaken for a broken generator.
#
#   TO REVIVE IT: --caps L60-90_G95 L70-95_G95 L80-100_G95, which walks
#   class 2 through the mid range it was after while holding class 7
#   inside its own window.
#   Everything below the banner is preserved as the ORIGINAL reasoning, which
#   was sound on every axis EXCEPT the cap placement.
# =============================================================================
# =============================================================================
#  results/vitdom2  --  DOES THE CONSOLIDATED TraLO BEAT THE CLIPPER AND THE
#                       RIVAL DUALS, ON THE HEADLINE BACKBONE, IN EVERY REGIME?
#
#   why        Measured 2026-08-31 over ALL 372 live cells (1,340 runs, 4
#              backbones, `scripts/cell_table.py`), and it overturns the
#              assumption the project has been running on:
#
#                `tralo` (the shipped `sum` count) vs `clip`, 44 paired cells
#                  ccF1   22/44  +0.0027  p=1.000   TIE
#                  AP     21/44  -0.0182  p=0.880   LOSS
#                  AUROC  20/44  -0.0048  p=0.652   LOSS
#
#              Pooled across regimes the shipped arm beats the post-hoc bar on
#              NOTHING. The split is entirely by cap regime, and the tight half
#              is a rout: AP 0/21 and AUROC 0/21 against `clip` at tight caps,
#              21/23 and 20/23 FOR it at loose. Same code, opposite verdicts,
#              both three-star.
#
#              🟢 `tralo_uniform` is the arm that wins, 29 paired cells:
#                  AP     25/29  +0.0163  p=0.0001
#                  AUROC  25/29  +0.0047  p=0.0001
#                  macroF1 22/29 +0.0059  p=0.0081
#                  uncF1  22/29  +0.0071  p=0.0081
#                  ccF1   17/29  +0.0021  p=0.458  (tie)
#              It never loses a metric, and per backbone it takes AP and AUROC
#              in 4 of 4 -- ViTB16 4/5, RegNetY400MF 6/8 and 7/8, MobileNetV3
#              7/8 and 6/8, MobileNetV2 8/8 and 8/8 (p=0.008).
#
#              🔑 AND IT IS THE ONLY ARM THAT CLEARS ITS OWN NOISE FLOOR. A
#              pure RNG reseed beats the lambda=0 twin on ccF1 in 29/44 cells,
#              MORE than `tralo`'s 28/44 -- so TraLO's ccF1 gain is not
#              attributable. The floor does NOT move AUROC (20/44, p=0.652)
#              while `tralo_uniform` does (23/29, p=0.0023). That AUROC gain is
#              the single attributable effect in the corpus, and AUROC is one
#              of only two metrics that can change a top-K set at all.
#
#   ⛔ WHAT THIS CAMPAIGN IS NOT. It is not another count-function sweep. The
#   count function is DECIDED: `tralo_uniform` is TraLO (FRAMEWORK 2(z7)).
#   `tralo` is carried ONLY as the ablation that prices the decision, and no
#   other variant is present. `tralo_st`, `tralo_ortho`, `tralo_head` and
#   `tralo_coin` answered their questions and are retired from the rotation.
#
#   🛑 THE GAP IT CLOSES. ViTB16 x any rival dual is **0 runs anywhere**, and
#   ViTB16 was fixed a priori as the headline backbone (FRAMEWORK 1-pre)
#   precisely so a win found elsewhere could not be promoted onto it after the
#   fact. Every dominance number this project has is CNN-only. So is every
#   tie: `tralo` vs `alm` reads 9/15 on every metric, p=0.61, on CNNs alone.
#
#   🛑 AND THE MID CAPS. L55-L70 is **0 runs anywhere**, so the tight/loose
#   reversal that decides which count function to use is measured only at its
#   two endpoints. `L60_G95` is the first interior point ever run.
#
#   arms       tralo_uniform  THE MODEL. Uniform-gradient count: value is the
#                             exact p_ic so the K comparison is unchanged, but
#                             dS/du is constant across items, which is a pure
#                             bias shift in the class logit and therefore
#                             CANNOT reorder the class.
#              tralo          the ablation -- identical but for the `sum`
#                             count, whose p(1-p) per-item derivative reorders.
#                             `tralo_uniform` - `tralo` IS the value of the
#                             design decision, on the headline backbone.
#              alm            \ the three rivals. `alm` is the strongest: it
#              fioretto        ) beats `clip` on ccF1 12/15 and AP 12/15 on
#              hounie         /  CNNs. None has ever met ViTB16.
#              tralo_null     the lambda=0 twin, SHARED -- the four family
#                             nulls are byte-identical, verified 24/24
#              tralo_reseed   the RNG floor -- gen_campaign REFUSES without it
#              clip           \ auto-added mandatory clippers. `clip` is the
#              focal_clip     /  stronger bar and the control for the primary.
#
#   caps       L30_G50  TIGHT  -- where `tralo` loses AP 0/21
#              L60_G95  MID    -- the first interior cap level ever run
#              L90_G95  LOOSE  -- where `tralo` wins AP 21/23
#              Chosen to span the reversal, not to sample it densely. All
#              three bind the LOCAL scope; this campaign does not test the
#              global scope and must not be reported as if it did.
#
#   models     ViTB16 MobileNetV3. ViTB16 is the point of the campaign;
#              MobileNetV3 is present because ONE backbone gives 4 warm-up
#              units and an exact sign floor of 0.125, which cannot reach 0.05
#              at any effect size. It is the cheapest way to buy significance,
#              and it is also the CNN on which `tralo_uniform` already wins
#              AP 7/8, so it is a replication target rather than a new question.
#
#   ⚠️ FALSIFIABLE, FIXED BEFORE THE DATA.
#
#   🛑 THE UNIT IS THE WARM-UP MODEL. 2 backbones x 4 seeds = 8 independent
#   units; the three cap tags within a (model, seed) SHARE one warm-up, so
#   they are correlated replicates and NOT independent cells. Exact sign floor
#   at n=8 is 2/2^8 = 0.0078. FRAMEWORK 2(z) is the receipt: 8 of 9 dom1
#   sweeps evaporated once this was applied.
#
#   PRIMARY, exactly one:
#     `tralo_uniform` - `clip` on **AUROC**, seed-paired, sign test over the
#     8 (model, seed) units. AUROC and not ccF1 because ccF1's gain does not
#     clear the reseed floor and AUROC's does; AUROC and not AP because at 4
#     seeds its minimum detectable effect is 0.013 against AP's 0.035.
#     PASS = `tralo_uniform` beats `clip` in >= 7 of 8 units (p = 0.0703,
#     ⚠️ a DIRECTION, not significance). Only 8/8 (p = 0.0078) is significant.
#     The report must say which was met.
#
#   SECONDARY, pre-specified:
#     * DOMINANCE: `tralo_uniform` - `alm`, same metric and units. This is the
#       first ViTB16 dual comparison in the project's history.
#     * THE VALUE OF THE COUNT FUNCTION: `tralo_uniform` - `tralo`, per cap
#       tag SEPARATELY, never pooled. The prediction from 2(z7) is a SIGN
#       REVERSAL across the three caps: uniform ahead at L30_G50, behind at
#       L90_G95. If L60_G95 does not sit between them the account is wrong.
#     * 🛑 macroF1 AND uncF1 beside ccF1 in EVERY table, in items as well as
#       F1. The uncapped classes are where `tralo` does its damage (uncF1
#       12/44 against its own twin, p=0.0037) and where `tralo_uniform`
#       repairs it (14/29, a clean tie).
#     * FLOOR: every contrast quoted beside `tralo_reseed` - `tralo_null` on
#       the same metric and units. A contrast that does not exceed the floor
#       is reported as "not attributable", not as a small win.
#
#   FAIL, stated so it can happen: `tralo_uniform` fails to beat `clip` on
#   AUROC on ViTB16, or loses to `alm`. The first would mean the one
#   attributable effect in the corpus is CNN-only and does not survive onto
#   the pre-registered headline backbone. The second would mean TraLO is not
#   the best method even after consolidation. Either outcome is reportable and
#   neither is a reason to add another arm.
#
#   size       SPLIT ACROSS TWO ROOTS, one per GPU, because two dispatchers on
#              a shared NFS home must be partitioned by root path and never by
#              a filter on one root. Each root is
#                3 cells x 9 arms x 4 seeds = 108 runs   (1 backbone x 3 caps)
#              so 216 runs across the two, and 6 cells when they are scored
#              together. `vitu1` did 72 ViTB16 runs in 10.1 h, so budget
#              roughly 20-30 h wall clock with both roots running.
#
#   dose       `--constraint-fp32` mandatory and passed. `iwc1` is the receipt:
#              fp16 without it gave an ARM-DEPENDENT spread, alm 51.7% against
#              hounie 100%, which makes a cross-arm ordering a measurement of
#              the GradScaler. That campaign is quarantined for it, as are
#              `iwc2` and `iwc3`.
#
#   grad mode  `normalize`, matching dom1 and equaldose1 exactly.
#
#   host       dsisco02 (Blackwell, BF16 AMP, no GradScaler) if it has two
#              free GPUs. 🛑 NEVER share a GPU with another user. Record which
#              host produced this: dom1 (Blackwell bf16) and dom1b (Quadro
#              fp16) changed backbone and numeric regime together and are
#              therefore not poolable, which cost a whole comparison.
#
#   ⛔ SUPERSEDES docs/launch_vitdom1.sh, which was written before the
#   consolidation decision: it carried 8 arms including three retired count
#   variants and ran LOOSE caps only, so it could not have seen the reversal.
#   Do not launch both.
#
#   read it in this order
#              python -m scripts.rig_status
#              python -m scripts.quarantine --list
#              python -m scripts.dose_landed results/vitdom2
#                ^ read the CROSS-ARM ATTEMPTS block: tralo/alm attempt 29,
#                  fioretto/hounie 28, and that gap is a property of the
#                  published algorithms rather than a defect (FRAMEWORK 2(z3))
#              python -m scripts.flag_live tralo tralo_uniform
#                ^ 🛑 if the md5s match, `soft_count_mode` never reached the
#                  loss and every number below is `tralo` wearing a label.
#                  Four inert flags have shipped here already.
#              python -m scripts.cell_table --campaign results/vitdom2 --out cells_vitdom2.csv
#              python -m scripts.full_panel --campaign results/vitdom2 --control tralo_null
#              python -m scripts.full_panel --campaign results/vitdom2 --control clip
#
# =============================================================================
set -euo pipefail

PIN=1d4e179e
TREE=~/optloss-vitdom2

# TWO ROOTS, ONE PER GPU, SPLIT BY BACKBONE. The NFS home is shared between
# dsisco01 and dsisco02 and two dispatchers must be partitioned by separate
# ROOT PATHS, never by a filter on one root -- they would race for the same
# `pending` configs. Splitting on the backbone is the natural cut because the
# backbone is already part of the cell key, and ViTB16 is roughly 3x the CNN,
# so this also stops the slow half from gating the fast one.
# ONE WORKTREE, so both roots share the warm-up cache.
ROOT_VIT=results/vitdom2_vit
ROOT_CNN=results/vitdom2_cnn

# 🛑 RUN A COPY, NEVER THE FILE IN $TREE. Bash reads a script by BYTE OFFSET
# as it executes, so the `git checkout --detach` below would rewrite this file
# underneath the interpreter and the rest would execute as garbage. Fetch it
# out of git and run the copy:
#
#     git show origin/headroom/small-cnn:docs/launch_vitdom2.sh > ~/launch_vitdom2.sh
#     GPU_VIT=0 GPU_CNN=1 bash ~/launch_vitdom2.sh
#
SELF=$(cd "$(dirname "$0")" && pwd -P)
TREEP=$(cd "$(eval echo $TREE)" 2>/dev/null && pwd -P || true)
if [ -n "$TREEP" ]; then
    case "$SELF/" in
      "$TREEP"/*) echo "REFUSING: this script lives inside \$TREE ($TREEP),"
                  echo "  and it is about to git-checkout that tree. Copy it"
                  echo "  out and run the copy -- see the block above."
                  exit 1 ;;
    esac
fi

if [ ! -d "$TREE" ]; then
    git -C ~/OptimizationLoss worktree add "$TREE" --detach "$PIN"
fi
cd "$TREE"
git -c gc.auto=0 fetch -q origin
git checkout -q --detach "$PIN"
test -z "$(git status --porcelain src/ configs/ main.py)" || {
    echo "REFUSING: src/ configs/ main.py are dirty -- code_version would be"
    echo "  stamped -dirty and the campaign would read as split."; exit 1; }

# 🛑 DO NOT TEST FOR THE DIRECTORY. `train_meta.csv` and `test_meta.csv` are
# TRACKED IN GIT, so checking out any commit creates that directory holding
# those two CSVs and nothing else. A `[ -e data/... ]` guard then sees the
# directory, skips the link, and every run dies on `train_images.npy`.
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
    --root "$ROOT_VIT" \
    --datasets iwildcam \
    --models ViTB16 \
    --caps L30_G50 L60_G95 L90_G95 \
    --arms tralo_uniform tralo alm fioretto hounie tralo_null tralo_reseed \
    --constraint-grad-mode normalize \
    --constraint-fp32

"$PY" -m configs.gen_campaign \
    --root "$ROOT_CNN" \
    --datasets iwildcam \
    --models MobileNetV3 \
    --caps L30_G50 L60_G95 L90_G95 \
    --arms tralo_uniform tralo alm fioretto hounie tralo_null tralo_reseed \
    --constraint-grad-mode normalize \
    --constraint-fp32

# THE THREE GATES. Each refuses a different way to waste a week, and the
# config gates are structurally blind to a runtime crash: three arms once
# shipped with an undefined name in `train()`, burned all 29 constraint
# epochs, died, were reset to `pending`, and the campaign came back looking
# merely unfinished with `audit_config` and `check_parity` both green.
"$PY" -m scripts.audit_config
"$PY" -m scripts.check_parity "$ROOT_VIT"
"$PY" -m scripts.check_parity "$ROOT_CNN"
"$PY" -m scripts.smoke_arms
"$PY" -m scripts.smoke_arms --matrix

# 🛑 TWO GPUS, AND NEVER SHARED WITH ANOTHER USER. Check owners first:
#   python -m scripts.rig_status
GPU_VIT=${GPU_VIT:?set GPU_VIT to a CONFIRMED-FREE gpu index}
GPU_CNN=${GPU_CNN:?set GPU_CNN to a CONFIRMED-FREE gpu index}

# main.py prompts for a GPU and reads the answer from stdin. With
# CUDA_VISIBLE_DEVICES pinned to one device it sees exactly one, so the answer
# is always index 0 -- NOT $GPU_*, which is the physical id.
for pair in "vit:$ROOT_VIT:$GPU_VIT" "cnn:$ROOT_CNN:$GPU_CNN"; do
    tag=${pair%%:*}; rest=${pair#*:}; root=${rest%%:*}; gpu=${rest##*:}
    echo 0 > "/tmp/gpuchoice_vitdom2_$tag"
    setsid env CUDA_VISIBLE_DEVICES="$gpu" EXPERIMENT_DIR="$root" \
        PYTHONIOENCODING=utf-8 nohup "$PY" main.py \
        > "/tmp/vitdom2_$tag.log" 2>&1 < "/tmp/gpuchoice_vitdom2_$tag" &
    echo "launched $root on GPU $gpu"
done
sleep 25
tail -5 /tmp/vitdom2_vit.log || true
tail -5 /tmp/vitdom2_cnn.log || true
echo "vitdom2 launched from $TREE at $PIN"
