#!/usr/bin/env bash
# =============================================================================
# ⚠️ THE 2026-09-01 "SUPERSEDED" BANNER ON THIS FILE IS WITHDRAWN (2026-09-02).
#    It said 2 of 6 (backbone x cap) cells pose NO QUESTION, on the strength of
#    class 7's window being 0.90-1.00. That window was measured with the PRIZE
#    counted over a GLOBAL top-K, and every allocator here is per-group: on
#    iwildcam 7 of 14 per-group ceilings are ZERO, so a global top-K counts
#    items the allocator can never emit and reads the cut at a probability no
#    group actually cuts at. Re-measured per group, class 7's window is
#    0.60-0.80 (MNv2) / 0.70-0.90 (MNv3), and NONE of this campaign's cells is
#    a non-task. Classified against the rebuilt `configs/task_windows.yml`:
#
#      MobileNetV2   L80_G95     TASK       c2 0.800 strict | c7 0.798 strict
#      MobileNetV2   L90_G95     partial    c2 0.900        | c7 0.901
#      MobileNetV2   L95_G80     TASK       c2 0.800 strict | c7 0.800 strict
#      MobileNetV3   L80_G95     partial    c2 0.800        | c7 0.798 strict
#      MobileNetV3   L90_G95     partial    c2 0.900        | c7 0.901 strict
#      MobileNetV3   L95_G80     partial    c2 0.800        | c7 0.800 strict
#
#    PARTIAL means the cap binds in SOME seeds only, so the effective n is
#    below the seed count: a positive there is CONSERVATIVE, a null is NOT
#    evidence of no effect. Say PARTIAL wherever those four cells are quoted.
#
#    This campaign has already RUN and it is the corpus's strongest: as
#    deployed it supplies two of the five independent units, and `tralo` beats
#    `clip` in both.
#
#   `configs/gen_campaign.py` REFUSES the dead caps now, so re-running this
#   script as written exits non-zero. That refusal is correct, not a broken
#   generator. Re-issue with per-class tags (`L<c2>-<c7>_G<g>`) inside both
#   classes' windows -- the two capped classes' windows differ on every
#   backbone, so one fraction cannot sit inside both.
# =============================================================================
# =============================================================================
#  results/dom1  --  TraLO AGAINST EVERY OTHER METHODOLOGY, BOTH SCOPES
# =============================================================================
#
#   why         Every campaign so far measured TraLO against its OWN lambda=0
#               twin -- an ATTRIBUTION question ("is the constraint doing
#               this?"). The thesis claim is a DOMINANCE question: does TraLO
#               beat the other eight methodologies? Those are different
#               contrasts and the second has never been run with nulls, at
#               enough cells to be callable, in the regime where TraLO works.
#
#   🟢 AND THERE NOW IS SUCH A REGIME. `results/loose1` (144 runs, 100% dose,
#   L80/L90) is the first campaign in which the constraint HELPS:
#
#       PAIRED vs its own tralo_null      AP        AUROC        ccF1
#       tralo                        +0.0253 5/1  +0.0075 6/0  +0.0120 6/0
#       tralo_reseed  (RNG floor)    -0.0016 tie  +0.0016 tie  +0.0088 6/0
#       tralo_uniform                +0.0005 tie  +0.0038 tie  +0.0077 6/0
#
#   🛑 READ THE RESEED ROW. A pure RNG reseed also produces a 6/0 ccF1 -- win --
#   of +0.0088, so `tralo`'s +0.0120 is only **1.36x the floor** and
#   `tralo_uniform`'s +0.0077 is BELOW it. **The ccF1 gain at loose caps is
#   mostly the seed.** What survives its control is the RANKING: AP +0.0253
#   and AUROC +0.0075 against a floor that TIES on both. That is the first
#   attributable positive constraint effect in the project, and it is a
#   ranking effect, not an allocation one.
#
#   The sign is the opposite of every tight-cap campaign, where `tralo` costs
#   AP 0.057 to 0.093 against its own null. FRAMEWORK 2(w3) has the mechanism:
#   at L20 the constraint must evict 294 items to win a 0.42-item prize, 700x
#   more work than gain, so it cannot avoid evicting correct ones. At L80-L90
#   it evicts 35-72 for an 18-30 item prize, a ratio of 1.2-4x.
#
#   🛑 BUT loose1 IS 6 CELLS AND CANNOT REACH `***`. Its wins all read
#   "not after BH": the exact Wilcoxon floor at 6 cells is p=0.031 and BH over
#   11 metrics needs p<0.0045. `gen_campaign` says it plainly -- **9 cells is
#   the minimum at which any single metric can reach a *** verdict**. This
#   campaign is 9 cells for exactly that reason, and it is the whole design
#   constraint.
#
#   what        3 backbones x 3 caps = 9 cells, all nine methodologies, every
#               trained arm carrying its own lambda=0 twin.
#
#   caps        BOTH SCOPES, which is half the thesis claim:
#                 L80_G95   local sum 0.80 < global 0.95  -> LOCAL binds
#                 L90_G95   local sum 0.90 < global 0.95  -> LOCAL binds
#                 L95_G80   global 0.80 < local sum 0.95  -> GLOBAL binds
#               L80_G95 and L95_G80 are a MATCHED PAIR: identical binding
#               budget (0.80) and only the SCOPE differs, so any gap between
#               them is a scope effect and not a dose one. vitu1 measured 1.8x
#               on exactly that comparison at tight caps.
#
#   arms        the nine the paper claims, in one campaign:
#                 tralo          the method
#                 tralo_uniform  its fixed count (2(w)) -- free at tight caps,
#                                and the arm most likely to win here
#                 fioretto       fioretto_ldf   \
#                 hounie         hounie_rcl      > the three competing duals
#                 alm            fioretto_alm   /
#                 clip           greedy allocator, the STRONGER quality bar
#                 lp             danits_lp / Shifman-LP
#                 focal_lp       \
#                 cb_lp           > the three imbalanced recipes, LP-clipped.
#                 la_lp          /  ⚠️ these are where the claim is most at
#                                risk: the archive records NO ccF1 edge for
#                                TraLO over them.
#               plus tralo_null / fioretto_null / hounie_null / alm_null so
#               every dual is read against its own compute, and tralo_reseed
#               as the RNG floor (gen_campaign REFUSES without it).
#
#   ⚠️ **THE PRE-REGISTRATION BELOW IS STATED ON ccF1 AND THAT IS NOW KNOWN
#   TO BE THE WEAK METRIC HERE** -- loose1's reseed floor eats 73% of the ccF1
#   effect. It is left EXACTLY as written because it was fixed before the data
#   and changing it after seeing loose1 would be the whole disease. Read AP and
#   AUROC beside it, and read every arm against `tralo_reseed` before calling
#   any ccF1 ordering a result.
#
#   ⚠️ FALSIFIABLE, stated before the data. The claim is DOMINANCE, not a
#   clean sweep -- "some positivity", not 100%. So:
#     * PASS: `tralo` or `tralo_uniform` beats each of fioretto, hounie and
#       alm on ccF1 in at least 6 of 9 cells, AND is not worse than `clip` on
#       ccF1, AND the same ordering holds in the LOCAL-binding and
#       GLOBAL-binding cells read separately.
#     * FAIL: any competing dual beats both TraLO arms on ccF1 in 6+ of 9, or
#       TraLO loses to `clip`.
#   The scope split is a DIRECTION claim only: 6 local cells and 3 global
#   cells cannot each reach ***, and saying otherwise is the 4-cell error the
#   generator refuses to let pass silently.
#
#   size        6 cells x 16 arms x 4 seeds = 384 runs on THIS host, and
#               dom1b adds 3 cells x 16 arms x 4 seeds = 192 runs on dsisco01
#               -- NINE cells when the two roots are scored together, which is
#               the minimum at which any metric can reach ***.
#               SPLIT ACROSS TWO HOSTS by backbone, into two roots, scored
#               together -- `full_panel` takes `--campaign a b`:
#                 dom1  dsisco02  MobileNetV2 + MobileNetV3  6 cells  384 runs
#                 dom1b dsisco01  RegNetY400MF               3 cells  192 runs
#               Both pin the SAME commit or the join is a split code_version.
#               Partition by ROOT, never by --filter (reference: multi-GPU).
#
#   grad mode   **`normalize`, and this is forced, not preferred.**
#               `check_parity` REFUSES four trained methodologies under
#               `clip`, and it is right: the clip delivers `min(raw, 1.0)`
#               while the arms' natural gradient norms are hounie 0.005-0.11,
#               tralo 0.64-1826, fioretto 17,667-80,827. So fioretto and alm
#               saturate the clip and hounie NEVER reaches it -- a ~20x dose
#               spread with every config file saying `constraint_grad_clip:
#               1.0`. A dual-vs-dual delta across that gap measures the dose.
#
#               The objection to `normalize` is that it erases the dual knobs.
#               It mostly does not erase anything that was LIVE: FRAMEWORK
#               2(e) measured that under `clip` the delivered step is exactly
#               `lr*clip` regardless of lambda, so **magnitude is already void**
#               and only DIRECTION and step COUNT are live levers. `normalize`
#               makes that explicit and, unlike `clip`, makes it EQUAL across
#               arms. lambda still sets the relative weight of the global and
#               local terms, i.e. the direction, which is what survives.
#
#               ⚠️ SAY IT WHEN QUOTING THE RESULT: this compares the four
#               duals at a MATCHED step size, not at each method's own
#               natural one. `xfam1` made the same choice (252 runs,
#               `normalize`), so the two are at least commensurable.
#
#   host        dsisco02 (RTX PRO 6000 Blackwell, BF16 AMP), GPU 0.
#
#   read it     python -m scripts.rig_status
#               python -m scripts.dose_landed results/dom1
#                 ^ 🛑 FIRST, and on the RUNNING campaign. The four duals have
#                   a KNOWN dose asymmetry: under `clip` the gradient norms are
#                   hounie 0.005-0.11 against fioretto 17,667-80,827, so
#                   fioretto and alm saturate the clip and hounie never reaches
#                   it. A dual-vs-dual delta across that gap is confounded.
#               python -m scripts.full_panel --campaign results/dom1 results/dom1b --control clip
#               python -m scripts.full_panel --campaign results/dom1 results/dom1b --control tralo_null
#               python -m scripts.family_split --families tralo tralo_uniform fioretto hounie alm
#
# =============================================================================
set -euo pipefail

PIN=1d921173                 # the training path used by uniform1, vitu1 and
                             # loose1. dom1b MUST pin this same commit.
TREE=~/optloss-dom           # its OWN worktree. Worktrees share one object
                             # store, so this fetches and checks out its own
                             # tree and runs NO git maintenance: never gc,
                             # prune, repack or worktree prune while any
                             # campaign is running.
ROOT=results/dom1

# 🛑 REFUSE TO RUN FROM INSIDE THE TREE THIS SCRIPT IS ABOUT TO CHECK OUT.
# Bash reads a script incrementally, by byte offset, so a checkout of $TREE
# would rewrite this file underneath the interpreter at an offset it has not
# reached yet. Copy it out first:
#
#     git show origin/headroom/small-cnn:docs/launch_dom1.sh > ~/launch_dom1.sh
#     GPU=0 bash ~/launch_dom1.sh
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
echo "loosevit1 holds the other on dsisco01; dom1b waits for it to finish."

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
    --models MobileNetV2 MobileNetV3 \
    --caps L80_G95 L90_G95 L95_G80 \
    --arms tralo tralo_uniform fioretto hounie alm \
           tralo_null fioretto_null hounie_null alm_null tralo_reseed \
           lp focal_lp cb_lp la_lp \
    --constraint-grad-mode normalize \
    --constraint-fp32

# THE THREE GATES. Each refuses a different way to waste a week.
"$PY" -m scripts.audit_config
"$PY" -m scripts.check_parity "$ROOT"
"$PY" -m scripts.smoke_arms

GPU=${GPU:-0}
# main.py prompts for a GPU and reads the answer from stdin. With
# CUDA_VISIBLE_DEVICES pinned to one device it sees exactly one, so the answer
# is always index 0 -- NOT $GPU, which is the physical id.
echo 0 > /tmp/gpuchoice_dom1
setsid env CUDA_VISIBLE_DEVICES="$GPU" EXPERIMENT_DIR="$ROOT" \
    PYTHONIOENCODING=utf-8 nohup "$PY" main.py \
    > /tmp/dom1.log 2>&1 < /tmp/gpuchoice_dom1 &
sleep 25
echo "launched $ROOT on GPU $GPU from $TREE at $PIN"
tail -5 /tmp/dom1.log || true
