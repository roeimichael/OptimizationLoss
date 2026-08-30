#!/usr/bin/env bash
# =============================================================================
#  results/margin2  --  THE TWO ARMS THAT WERE BUILT AND NEVER RUN
#
#   why        `tralo_margin` and `tralo_st` are fully implemented,
#              protocol-registered, null-sibling-tested and gated in
#              `gen_campaign` -- and have NEVER EXECUTED. Zero run directories
#              across all nine server worktrees, checked 2026-08-28. They are
#              the only untested points in the count-function 2x2.
#
#   the 2x2    Two independent defects in the shipped count, and the arms
#              separate them (`configs/protocol.yml` states this at the
#              definition, and the 4th corner is deliberately NOT an arm --
#              margin placement on a soft value over-counts 56.6 against a
#              hard 45 and pushes past feasibility):
#
#                              gradient PLACEMENT
#                        p(1-p)              margin window
#              value   +-------------------+-------------------+
#              soft    | `tralo`           | not an arm        |
#                      | (the manuscript)  | (over-counts)     |
#              hard    | `tralo_st`        | `tralo_margin`    |
#                      | (VALUE fix only)  | (both fixed)      |
#                      +-------------------+-------------------+
#
#              `tralo` -> `tralo_st` isolates the count's VALUE.
#              `tralo_st` -> `tralo_margin` isolates the gradient's PLACEMENT.
#              Without both, a margin result is unattributable between them.
#
#   🛑 READ THIS BEFORE QUOTING A MECHANISM. FRAMEWORK 2(w3) used to say the
#   margin window "windows the gradient around the CUT (rank K)". IT DOES NOT,
#   and that sentence is why this arm was queued. `margins()` is
#   `m_ic = p_ic - max_{c'!=c} p_ic'`, i.e. distance to the DECISION BOUNDARY.
#   Three distinct points, and the old text collapsed two of them:
#       p(1-p) peak         p_c = 0.5
#       decision boundary   wherever p_c equals the runner-up (p_c = 0.20 in
#                           the 8-class case measured 2026-08-28)
#       the cut, rank K     a RANKING POSITION -- nothing implements it
#   So `cut_window_items` is MISNAMED: it is a boundary window. Corrected in
#   FRAMEWORK 2(w3) 2026-08-28.
#
#   ✅ THE REASON THAT SURVIVES, and it is stronger. Every penalty this
#   project ships has the form `f(sum_i p_ic)`, whose per-item logit gradient
#   is `f'(S) * p_ic(1 - p_ic)` -- a function of `p_ic` ALONE, hence a
#   monotone map on the logit channel, and a monotone map CANNOT move an item
#   across another. It moves the cut; it cannot re-rank. `margin` reads the
#   whole row, so two items with equal `p_ic` but different runners-up get
#   different gradients. It is the ONLY arm in the family that can reorder
#   through the direct channel.
#   ⚠️ That bounds the DIRECT channel only. FRAMEWORK 2(w4) measures BOTH
#   shipped arms reordering more than a reseed via the shared weights. So
#   `margin` makes reordering TARGETED, not possible. Do not sell it as the
#   thing that makes reordering happen.
#
#   arms       tralo          the manuscript count -- the thing to beat
#              tralo_st       value fixed only     \ the decomposition,
#              tralo_margin   value + placement    / never run before
#              tralo_coin     RANDOM direction, same norm -- THE CONTROL FOR
#                             A PLACEMENT CLAIM, and it is the whole reason
#                             this campaign can conclude anything about
#                             placement. If a random direction of the same
#                             norm moves the metric as much as `margin` does,
#                             then WHERE the gradient lands is not what is
#                             doing the work. Carried over from the
#                             superseded `launch_margin1.sh`.
#              tralo_uniform  the current best at tight caps (2(w))
#              tralo_null     lambda=0 twin, SHARED by all four (at lambda=0
#                             there is no constraint gradient, so a dedicated
#                             null per arm would be a bit-identical run
#                             costing a GPU slot; the null-sibling gate in
#                             tests/test_pipeline.py pins this)
#              tralo_reseed   the RNG floor -- gen_campaign REFUSES without it
#              clip           \ auto-added mandatory clippers. `clip` is the
#              focal_clip     / STRONGER quality bar.
#
#   caps       A 2x2 OF {TIGHT, LOOSE} x {LOCAL-BINDING, GLOBAL-BINDING},
#              AT MATCHED BUDGETS. Verified with `scripts.verify_caps`:
#
#                tag        class 2 K   class 7 K   binding scope   budget
#                L30_G50        111         137     LOCAL           K/n=0.30
#                L50_G30        111         137     GLOBAL          K/n=0.30
#                L80_G95        296         364     LOCAL           K/n=0.80
#                L95_G80        296         365     GLOBAL          K/n=0.80
#
#              🔑 EACH ROW-PAIR IMPOSES THE SAME TOTAL BUDGET THROUGH A
#              DIFFERENT SCOPE. `L30_G50` pins the DISTRIBUTION across groups
#              (per-group ceilings [0,0,0,0,31,32,48]); `L50_G30` pins only the
#              TOTAL (111) and leaves the split free under looser local
#              ceilings [0,0,0,0,51,54,80]. So scope is isolated with the
#              budget held FIXED -- the confound between "tighter" and
#              "differently scoped" is removed by construction.
#
#              This answers the two things the 3-cap version could not:
#                (a) REGIME CONSISTENCY. `tralo` is +0.0253..+0.0371 AP at
#                    loose caps and -0.0572..-0.0933 at tight ones. A method
#                    that only wins where the cap barely binds is not the
#                    thesis. Both regimes are now inside ONE campaign at ONE
#                    code_version.
#                (b) SCOPE. dom1 found `tralo` beats fioretto by +0.0439 AP at
#                    L95_G80 and LOSES by -0.0084 at L80_G95 -- same budget.
#                    That was 2 cells and post-hoc. This tests it at a second
#                    budget, on 3 backbones, pre-registered.
#
#              ⛔ DO NOT ADD `L30_G30`. verify_caps: at L30_G50 the global
#              K=185 sits ABOVE the local sum 111, so it is INERT and the tag
#              runs the same experiment as L30_G30. They are ONE cap level and
#              counting both double-counts a single measurement.
#              🛑 THE SCOPE CONTRAST IS NARROWER THAN THE TAG SUGGESTS,
#              and `gen_campaign` says so on all four tags. 7 of 14 per-group
#              ceilings are K=0 (no true instance of that species at that
#              camera), and a ZERO ceiling binds regardless of how much slack
#              the sum has. So "GLOBAL-binding" NEVER means the local scope is
#              off -- it means the three NON-ZERO groups' ceilings are slack.
#              What the pair actually contrasts is a PINNED distribution over
#              those three groups against a FREE one at the same total. State
#              it that way; do not write "local vs global".
#
#   ⚠️ FALSIFIABLE, FIXED BEFORE THE DATA. Restated 2026-08-30 when the
#   cap grid changed from 3 tags to the matched 2x2. That edit is legitimate
#   ONLY because no run of this campaign exists yet -- a pre-registration may
#   be rewritten before the data and never after. The grid now gives 6 TIGHT
#   cells (3 backbones x {L30_G50, L50_G30}) and 6 LOOSE cells (3 backbones x
#   {L80_G95, L95_G80}).
#
#   PRIMARY, and there is exactly ONE so BH has a multiplier of 1:
#     `tralo_margin` - `tralo`, on **AP**, exact two-sided sign test over all
#     12 cells. PASS = positive in >= 10 of 12 (p = 2*79/4096 = 0.0386).
#     9 of 12 is p = 0.1460 and does NOT pass. Every other number below is
#     SECONDARY and must be reported with that word attached.
#
#   SECONDARY, pre-specified so they cannot be fished for afterwards:
#     * REGIME CONSISTENCY -- the reason the 2x2 exists. `tralo_margin` -
#       `tralo` positive in >= 4 of the 6 TIGHT cells AND >= 4 of the 6 LOOSE
#       cells. A method that buys one regime with the other has fixed nothing,
#       and that is exactly the failure mode of the two shipped counts.
#     * FLOOR -- both must clear `tralo_reseed` - `tralo_null` on the same
#       metric and cells. An arm at or below the RNG floor produced nothing.
#     * SCOPE -- `tralo` - `fioretto` larger in the GLOBAL-binding cells than
#       in the LOCAL-binding ones, at BOTH budgets. The out-of-sample test of
#       the dom1 L95_G80 finding, which was 2 cells and post-hoc.
#     * DECOMPOSITION -- `tralo` -> `tralo_st` sizes the count's VALUE fix
#       alone. Never measured; worth having even if `tralo_margin` fails.
#     * CONTROL -- if `tralo_coin` (random direction, same norm) moves AP as
#       much as `tralo_margin`, then PLACEMENT is not what does the work and
#       the margin direction is dead regardless of its sign.
#     * 🛑 macroF1 AND uncF1 -- `tralo_margin` - `tralo`, same 12 cells.
#       This is the OTHER central defect, not a nice-to-have. On dom1 `tralo`
#       buys ccF1 +0.0141 (6/6) with uncF1 -0.0077 (1/6) and lands macroF1
#       **-0.0022 (2/6)**, below the reseed floor. The mechanism predicts
#       `margin` should help here: the penalty's cross-term redistributes the
#       mass it removes from class c across the other classes in proportion to
#       `p_ij`, so every item pushed off a capped class becomes a FALSE
#       POSITIVE somewhere uncapped. Windowing the push onto items already at
#       their decision boundary is exactly the subset whose reassignment costs
#       least. If macroF1 does not improve, the window is not finding those
#       items and the mechanism story is wrong.
#
#   FAIL, stated so it can actually happen: `tralo_margin` at or below
#   `tralo_reseed` on AP in either regime, or `tralo_coin` matching it, or
#   fewer than 10 of 12 cells positive on the primary.
#
#   🛑 macroF1 IS REPORTED BESIDE ccF1 IN EVERY TABLE. On dom1 `tralo`
#   is ccF1 +0.0141 (6/6) and macroF1 **-0.0022 (2/6)** -- it buys capped-class
#   accuracy with uncapped-class damage, and a ccF1-only report hides that.
#
#   size       12 cells x 9 arms x 4 seeds = 432 runs (3 backbones x 4 caps).
#              12 cells gives an exact sign floor of 2/2^12 = 0.0005, so a
#              pre-registered primary contrast clears BH with room to spare --
#              which 6 cells provably cannot (see docs/MISSION.md rule 3).
#
#   supersedes `docs/launch_margin1.sh`, which was staged, never fired, and
#              is DELETED rather than left to be picked up by mistake. It was
#              MobileNetV3 only at {L50_G30, L40_G30} = 2 cells, which the
#              generator's own power line calls unable to reach a `***` at any
#              effect size. Its `tralo_coin` control is kept; its caps are not,
#              because the regime axis of 2(w3) did not exist when it was
#              written.
#              🔪 CUT IT AT THE ROOT. Read the FIRST completed tight cell
#              before it has run a day. Only a POSITIVE `tralo_margin` signal
#              earns the rest -- a negative one ends it now.
#
#   grad mode  `normalize`, matching dom1, xfam1, loose1 and uniform1 so the
#              four are commensurable. Under `clip` the delivered step is
#              exactly `lr*clip` regardless of lambda (FRAMEWORK 2(e)), so
#              magnitude is already void; `normalize` makes that explicit and
#              equal across arms.
#
#   pin        1d921173 -- the SAME training path as uniform1, vitu1, loose1
#              and dom1. VERIFIED 2026-08-28 that this commit already contains
#              `tralo_margin`, `tralo_st` and `window_temp`, so those four
#              campaigns' `tralo` / `tralo_uniform` arms are directly
#              comparable with this one's.
#
#   host       whichever frees first. --constraint-fp32 is passed either way:
#              on dsisco01 (FP16 + GradScaler) it removes the skipped-step
#              dose loss that cost iwc3 328 of 1044 steps, and on dsisco02
#              (BF16) it is a no-op.
#
#   read it    python -m scripts.rig_status
#              python -m scripts.dose_landed results/margin2
#                ^ 🛑 FIRST, AND ON THE RUNNING CAMPAIGN. `tralo_uniform` once
#                  landed 1 of 29 steps beside `tralo` at 29/29 in the SAME
#                  campaign while both wrote `status: completed`. These two
#                  arms have NEVER RUN, so that risk is at its maximum here.
#              python -m scripts.flag_live tralo tralo_margin
#                ^ 🛑 md5 the raw predictions across the arms. A brand-new arm
#                  is the highest-risk case for a FIFTH inert flag, and this
#                  project has four already. If the md5s match, the arm is
#                  inert and every number below it is `tralo` wearing a label.
#              python -m scripts.flag_live tralo tralo_st
#              python -m scripts.full_panel --campaign results/margin2 --control clip
#              python -m scripts.full_panel --campaign results/margin2 --control tralo_null
#              python -m scripts.order_probe --campaign results/margin2 --arm tralo_margin
#                ^ the direct-channel question this arm exists for: does it
#                  reorder more than a reseed, and DOES IT DO SO IN THE BAND?
#                  2(w4) shows global and band dissociate.
#
# =============================================================================
set -euo pipefail

PIN=1d921173                 # same training path as uniform1/vitu1/loose1/dom1
TREE=~/optloss-margin2        # its OWN worktree. Worktrees share one object
                             # store, so this fetches and checks out its own
                             # tree and runs NO git maintenance: never gc,
                             # prune, repack or worktree prune while any
                             # campaign is running.
ROOT=results/margin2

# 🛑 REFUSE TO RUN FROM INSIDE THE TREE THIS SCRIPT IS ABOUT TO CHECK OUT.
# Bash reads a script incrementally, by byte offset, so a checkout of $TREE
# would rewrite this file underneath the interpreter at an offset it has not
# reached yet. Copy it out first:
#
#     git show origin/headroom/small-cnn:docs/launch_margin2.sh > ~/launch_margin2.sh
#     GPU=0 bash ~/launch_margin2.sh
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
    --models MobileNetV2 MobileNetV3 RegNetY400MF \
    --caps L30_G50 L50_G30 L80_G95 L95_G80 \
    --arms tralo tralo_st tralo_margin tralo_coin tralo_uniform \
           tralo_null tralo_reseed \
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
echo 0 > /tmp/gpuchoice_margin2
setsid env CUDA_VISIBLE_DEVICES="$GPU" EXPERIMENT_DIR="$ROOT" \
    PYTHONIOENCODING=utf-8 nohup "$PY" main.py \
    > /tmp/margin2.log 2>&1 < /tmp/gpuchoice_margin2 &
sleep 25
echo "launched $ROOT on GPU $GPU from $TREE at $PIN"
tail -5 /tmp/margin2.log || true
