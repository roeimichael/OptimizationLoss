#!/usr/bin/env bash
# =============================================================================
# 🛑🛑 SUPERSEDED 2026-09-01 -- DO NOT RUN THIS AS WRITTEN.
#
#   240 staged runs, and 200 of them sit at a cap that cannot
#   distinguish any two methods. 5 of its 6 cap tags are dead.
#
#   The task window (FRAMEWORK 2(z16)/2(z17), measured on all four backbones)
#   says a cap poses a question only where it evicts >= 10 predictions, leaves
#   ERRORS inside K, and cuts at p@K < 0.99. Classified against
#   `configs/task_windows.yml` on 2026-09-01:
#
#     ViTB16  L75_G95   NON-TASK   c7 K/n=0.750, window 0.90-1.00
#     ViTB16  L80_G95   NON-TASK   c7 K/n=0.798, window 0.90-1.00
#     ViTB16  L85_G95   NON-TASK   c7 K/n=0.851, window 0.90-1.00
#     ViTB16  L90_G95   ✅ TASK    c2 0.900, c7 0.901, both inside
#     ViTB16  L95_G95   NON-TASK   c2 K/n=0.951, window 0.60-0.90
#     ViTB16  L95_G80   NON-TASK   c7 K/n=0.800, window 0.90-1.00
#
#   🔑 THE TWO CAPPED CLASSES' WINDOWS DO NOT OVERLAP ON ViTB16 (c2
#      0.60-0.90, c7 0.90-1.00), so NO single fraction can sit inside
#      both. Only `L90_G95` lands in the 0.005 snap zone of each. That is
#      not a coincidence to route around, it is why the per-class tag form
#      `L<c2>-<c7>_G<g>` exists.
#
#   `configs/gen_campaign.py` now REFUSES these caps, so this script exits
#   non-zero as written. That refusal is the correct outcome and this banner
#   exists so the refusal is not mistaken for a broken generator.
#
#   TO REVIVE IT: re-issue with per-class tags, e.g.
#     --caps L70-90_G95 L80-95_G95 L85-100_G95
#   which gives three DISTINCT cap levels with both classes in window.
#   Everything below the banner is preserved as the ORIGINAL reasoning, which
#   was sound on every axis EXCEPT the cap placement.
# =============================================================================
# =============================================================================
#  results/vitdom1  --  THE DOMINANCE CLAIM, ON THE BACKBONE IT WAS PROMISED ON
#
#   why        FRAMEWORK 1-pre fixed **ViTB16** as the headline backbone a
#              priori, on 2026-08-20, precisely so a win found elsewhere could
#              not be promoted after the fact. Every dominance result since is
#              on MobileNetV2/V3 (`dom1`) and RegNetY400MF (`dom1b`).
#
#              🛑 **ViTB16 HAS NEVER RUN A SINGLE RIVAL DUAL ON iwildcam.**
#              Audited 2026-08-30 across all 11 server worktrees: `hounie_rcl`,
#              `fioretto_ldf`, `fioretto_alm` and `danits_lp` appear on ViTB16
#              ONLY in `vit_diag` / `vit_ceskip`, which are **dermmnist** and
#              86 of 97 pending -- on a dataset that is leaked and deleted from
#              disk. So ViTB16 currently supports TraLO-vs-clipper and
#              TraLO-vs-its-own-null, and NOTHING about dominance.
#              ⇒ the paper's headline backbone cannot support the paper's
#              headline claim. That is what this campaign is for.
#
#   the second job, and it is nearly free
#              `loosevit1` already shows `tralo` POSITIVE on every metric on
#              ViTB16 at loose caps -- AP +0.0064, ccF1 +0.0017, **macroF1
#              +0.0021**, all 2/0 -- against a `tralo_reseed` floor that is
#              NEGATIVE (AP -0.0113, macroF1 -0.0366). That is the best-looking
#              result in the project, and **it sits on 2 cells, where the
#              minimum attainable p is 0.500 and `full_panel` prints NOT
#              CALLABLE.** Six cap tags take it to 6 cells.
#
#   caps       SIX LOOSE TAGS, verified with `scripts.verify_caps` on the real
#              slice. Five DISTINCT budgets plus one matched scope pair:
#
#                tag        class 2 K   class 7 K   K/n     binding scope
#                L75_G95        277         342     0.75    local (pinned)
#                L80_G95        296         364     0.80    local (pinned)
#                L85_G95        315         388     0.85    local (pinned)
#                L90_G95        333         411     0.90    local (pinned)
#                L95_G95        352         433     0.95    local (redundant G)
#                L95_G80        296         365     0.80    GLOBAL (free)
#
#              🔑 `L80_G95` and `L95_G80` give class 2 the SAME K=296 through
#              DIFFERENT scopes -- `verify_caps` says so explicitly and warns
#              that counting both as cap LEVELS double-counts one budget. They
#              are kept deliberately, as the scope contrast at a fixed budget,
#              and they are NOT counted as two levels anywhere below.
#              ⚠️ Every G>=L tag has an INERT global (11 inert scopes in all):
#              a global cap only adds a constraint strictly below the local
#              sum. Those five tags therefore test the LOCAL scope, and
#              `L95_G80` is the only one testing the global. Do not describe
#              this grid as a local-vs-global sweep; it is a BUDGET sweep with
#              one scope contrast bolted on.
#              ⚠️ 7 of 14 per-group ceilings are K=0. A zero ceiling binds
#              regardless of sum slack, so the local scope is never fully off.
#
#   why LOOSE only, and this is a deliberate exclusion
#              `vitu1` already ran ViTB16 at L20/L30/L50, complete, 100% dose,
#              and reports `tralo` at **AP -0.0933 (0/3)** against a reseed
#              floor of -0.0142 -- i.e. **6.6x WORSE than doing nothing**.
#              FRAMEWORK 2(y) explains why and predicts no count function fixes
#              it: at K/n=0.20 the decision boundary sits 235-442 items from
#              the cut, so the gradient is aimed where the metric never looks.
#              ⇒ re-running ViTB16 tight caps would buy a known negative.
#              `margin2` tests the tight regime with the arm built for it.
#
#   arms       tralo          the method under test
#              hounie         \ the three rival duals (`hounie_rcl`,
#              fioretto        ) `fioretto_ldf`, `fioretto_alm`). NEVER RUN on
#              alm            /  ViTB16 + iwildcam. This is the whole point.
#              lp             the LP allocator (`danits_lp`, Shifman), post-hoc
#              tralo_uniform  the tight-cap count, carried as the regime
#                             control: 2(y) predicts it LOSES here, and an
#                             arm predicted to lose is worth running
#              tralo_null     the lambda=0 twin -- without it nothing is
#                             attributable to the constraint rather than to
#                             the 29 extra epochs
#              tralo_reseed   the RNG floor. `gen_campaign` REFUSES without it,
#                             and on `loosevit1` it is NEGATIVE, which is
#                             exactly why `tralo`'s small positive matters
#              clip           \ auto-added mandatory clippers; `clip` is the
#              focal_clip     / stronger quality bar
#
#   ⚠️ FALSIFIABLE, FIXED BEFORE THE DATA.
#
#   🛑 THE UNIT IS (model, seed), NOT THE CELL. One backbone x 4 seeds = 4
#   independent units, because each seed trains its own warm-up with its own
#   `base_model_id`, while all six cap tags within a seed SHARE that warm-up
#   and two of them share a budget. 4 units gives an exact sign floor of
#   2/2^4 = 0.125, so **no contrast here can reach p<0.05 on the unit**, and
#   this campaign is powered to compare ARMS WITHIN a cell, not to establish
#   generalization. Say that in every report of it. FRAMEWORK 2(z) is the
#   receipt for why the 6-cell reading would be the flattering one.
#
#   PRIMARY, exactly one so BH has a multiplier of 1:
#     `tralo` beats EACH of the four rival duals on **ccF1**, per cap tag,
#     seed-paired. PASS = `tralo` ranks first of the five trained arms in at
#     least 5 of the 6 cap tags, AND its margin over the runner-up exceeds the
#     `tralo_reseed` - `tralo_null` floor in those tags. Ranking first while
#     inside the RNG floor is not a win and must not be reported as one.
#
#   SECONDARY, pre-specified:
#     * 🛑 macroF1 AND uncF1 beside ccF1 in EVERY table. On `dom1` `tralo`
#       is ccF1 +0.0141 (6/6) and macroF1 -0.0022 (2/6) -- it buys capped-class
#       accuracy with uncapped damage, and a ccF1-only report hides it. On
#       ViTB16 macroF1 came out POSITIVE (+0.0021 loose, +0.0196 tight), so
#       whether the damage is backbone-specific is an open question this
#       campaign answers.
#     * REGIME CONTROL -- 2(y) predicts `tralo_uniform` LOSES to `tralo` here,
#       inverting `vitu1`, where it was the only arm above the floor. If it
#       wins at loose caps too, the mechanism is wrong.
#     * BUDGET TREND -- the gap between `tralo` and the clippers should GROW
#       with K/n across the five distinct budgets, because 2(y) says the cut
#       moves toward the boundary as the cap loosens. Five budgets is enough
#       to see a trend and not enough to fit one; report the ordering, not a
#       regression.
#     * SCOPE -- `L80_G95` vs `L95_G80` at the identical K=296. `dom1` found
#       `tralo` beats fioretto by +0.0439 AP under the global scope and loses
#       by -0.0084 under the local one; that was 2 cells and post-hoc. This is
#       the out-of-sample test on the headline backbone.
#
#   FAIL, stated so it can happen: `tralo` is not first in at least 5 of 6
#   tags, or its margin sits inside the reseed floor, or a rival dual beats it
#   on macroF1 while it wins ccF1 -- which would mean it is trading uncapped
#   accuracy for capped accuracy and calling the trade a win.
#
#   ⚠️ NO PER-FAMILY NULLS, AND THAT IS A DELIBERATE NARROWING. `dom1`
#   carried `hounie_null` / `fioretto_null` / `alm_null` so each family's
#   CONSTRAINT TERM could be separated from its 29 extra epochs, and it found
#   all four terms negative. That question is answered; repeating it would cost
#   three more arms x 6 caps x 4 seeds = 72 ViTB16 runs to re-derive a known
#   result. This campaign asks the DOMINANCE question -- which arm is best at
#   equal compute -- for which the arms themselves suffice, plus `tralo_null`
#   and `tralo_reseed` to keep TraLO's own claim attributable and floored.
#   ⇒ do NOT report a per-family attribution off this campaign. It cannot
#   support one, by construction.
#
#   size       6 cells x 10 arms x 4 seeds = 240 runs (1 backbone x 6 caps).
#              ⚠️ ViTB16 is the slowest backbone here. Budget accordingly and
#              read `dose_landed` on the FIRST completed runs, not at the end.
#
#   dose       `--constraint-fp32` is mandatory and is passed. `iwc2` is the
#              receipt: ViTB16 under fp16+GradScaler without it landed
#              **173/232 steps, 74.6%**, while `loosevit1` on the same host
#              with it landed 232/232. `check_parity` is GREEN on iwc2 anyway
#              -- only `dose_landed` sees this.
#
#   grad mode  `normalize`, matching dom1, loose1, uniform1, vitu1 and
#              loosevit1 so all six are commensurable. Under `clip` the
#              delivered step is exactly `lr*clip` regardless of lambda
#              (FRAMEWORK 2(e)), which erases every dual's knob.
#
#   pin        1d921173 -- the same training path as uniform1, vitu1, loose1,
#              dom1 and margin2.
#              ⚠️ `loosevit1` is at 74f858657154, a DIFFERENT commit, so its
#              runs are NOT poolable with these. This campaign re-runs
#              L80_G95 and L90_G95 deliberately rather than reusing them.
#
#   read it in this order, and the first two before it is half done
#              python -m scripts.dose_landed results/vitdom1
#                ^ FIRST, on the RUNNING campaign. Four dual families write
#                  four different log schemas and each doses differently;
#                  `hounie_rcl` once ran at 1% of its dose while writing
#                  `status: completed`.
#              python -m scripts.flag_live tralo hounie_rcl
#                ^ md5 the raw predictions. Four inert flags have shipped here.
#              python -m scripts.full_panel --campaign results/vitdom1 --control clip
#              python -m scripts.full_panel --campaign results/vitdom1 --control tralo_null
#              python -m scripts.family_split --campaign results/vitdom1 \
#                     --families tralo fioretto hounie alm
#              python -m scripts.cut_gap results/vitdom1
#                ^ the 2(y) geometry across five budgets on one backbone --
#                  the cleanest test of the mechanism that will exist.
#
# =============================================================================
set -euo pipefail

PIN=1d921173                 # same training path as uniform1/vitu1/loose1/dom1
TREE=~/optloss-vitdom        # its OWN worktree. Worktrees share one object
                             # store, so this fetches and checks out its own
                             # tree and runs NO git maintenance: never gc,
                             # prune, repack or worktree prune while any
                             # campaign is running.
ROOT=results/vitdom1

# 🛑 REFUSE TO RUN FROM INSIDE THE TREE THIS SCRIPT IS ABOUT TO CHECK OUT.
# Bash reads a script incrementally, by byte offset, so a checkout of $TREE
# would rewrite this file underneath the interpreter at an offset it has not
# reached yet. Copy it out first:
#
#     git show origin/headroom/small-cnn:docs/launch_vitdom1.sh > ~/launch_vitdom1.sh
#     GPU=0 bash ~/launch_vitdom1.sh
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
echo "check the OTHER host before trusting it: dom1 and dom1b both count."

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
    --caps L75_G95 L80_G95 L85_G95 L90_G95 L95_G95 L95_G80 \
    --arms tralo hounie fioretto alm lp \
           tralo_uniform tralo_null tralo_reseed \
    --constraint-grad-mode normalize \
    --constraint-fp32

# THE THREE GATES. Each refuses a different way to waste a week.
# smoke_arms matters MORE than usual here: tralo_margin and tralo_st have
# never executed, and the config gates are structurally blind to a runtime
# crash -- three arms once shipped with an undefined name in train(), burned
# all 29 constraint epochs, died, reset to pending, and the campaign came back
# looking merely unfinished with audit_config and check_parity both green.
"$PY" -m scripts.audit_config
"$PY" -m scripts.check_parity "$ROOT"
"$PY" -m scripts.smoke_arms
"$PY" -m scripts.smoke_arms --matrix

GPU=${GPU:-0}
# main.py prompts for a GPU and reads the answer from stdin. With
# CUDA_VISIBLE_DEVICES pinned to one device it sees exactly one, so the answer
# is always index 0 -- NOT $GPU, which is the physical id.
echo 0 > /tmp/gpuchoice_vitdom1
setsid env CUDA_VISIBLE_DEVICES="$GPU" EXPERIMENT_DIR="$ROOT" \
    PYTHONIOENCODING=utf-8 nohup "$PY" main.py \
    > /tmp/vitdom1.log 2>&1 < /tmp/gpuchoice_vitdom1 &
sleep 25
echo "launched $ROOT on GPU $GPU from $TREE at $PIN"
tail -5 /tmp/vitdom1.log || true
