#!/usr/bin/env bash
# =============================================================================
# 🛑🛑 SUPERSEDED 2026-09-01 -- 2 OF ITS 6 (backbone x cap) CELLS POSE
#    NO QUESTION. This campaign has already RUN, so the banner is about how to
#    READ it as much as about re-running it.
#
#   A cap poses a question only where it evicts >= 10 predictions, leaves
#   ERRORS inside K, and cuts at p@K < 0.99 (FRAMEWORK 2(z16), 2(z17)).
#   Classified against `configs/task_windows.yml` on 2026-09-01:
#
#     MobileNetV2   L80_G95     ✅ TASK
#     MobileNetV2   L90_G95     ✅ TASK
#     MobileNetV2   L95_G80     ✅ TASK
#     MobileNetV3   L80_G95     NON-TASK   class 7 K/n=0.798 vs 0.90-1.00
#     MobileNetV3   L90_G95     ✅ TASK
#     MobileNetV3   L95_G80     NON-TASK   class 7 K/n=0.800 vs 0.90-1.00
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
#  results/equaldose1  --  IS THE DOMINANCE CLAIM A 3.4% HEAD START?
#
#   why        Measured 2026-08-30 on `dom1` AND `dom1b`, 24 and 12 runs per
#              arm: `tralo` and `alm` attempt **29** constraint steps per run
#              while `fioretto` and `hounie` attempt **28**, at an identical
#              `constraint_epochs: 29` -- and `dose_landed` reported
#              **100.0% for all four**, because that figure is
#              applied/attempted WITHIN an arm and is structurally blind to a
#              cross-arm gap. (It now prints a CROSS-ARM block.)
#
#              🛑 VERIFIED AT THE GRADIENT LEVEL, so it is dose and not
#              accounting. dom1 / MobileNetV3 / L80_G95 / seed 1, epoch 1:
#
#                arm         lambda at epoch 1   logged grad norm
#                tralo             0.01               3.09    a real step
#                alm               mu0 > 0          6426.97    a real step
#                fioretto          0                   0.00    NO step
#                hounie            0                   0.00    NO step
#
#              The subgradient duals guard their step on `has_work` (is any
#              lambda > 0) and perform the dual update at the END of the
#              epoch, so their first constraint epoch does nothing. TraLO
#              guards on `has_constraint`, which is structural, and starts at
#              `lambda_global: 0.01`.
#
#   ✅ THIS IS NOT A BUG IN THE BASELINES, and it must not be "fixed" by
#   editing them. `lambda^0 = 0` is what subgradient dual ascent specifies;
#   changing it would make `fioretto` and `hounie` unfaithful to their papers
#   and every number they have produced here incomparable. The asymmetry is
#   fixed from OUR side, by giving TraLO the same start and asking whether the
#   win survives.
#
#   ⚠️ AND IT IS A HYPERPARAMETER WE CHOSE. `lambda_global: 0.01` is in
#   `configs/protocol.yml`, not in any paper. Every head-to-head this project
#   has ever run gave TraLO one extra effective gradient step, 1 in 29 = 3.4%.
#   A reviewer will ask whether the win is the method or the head start. This
#   campaign is the answer, and it is cheap.
#
#   arms       tralo          the shipped arm -- lambda starts at 0.01, 29 steps
#              tralo_lam0     THE CONTROL -- lambda starts at 0, ratchet
#                             UNCHANGED, so epoch 1 carries a zero gradient
#                             exactly as the duals' does and epochs 2..29 run
#                             normally: 28 effective steps, matching them
#              fioretto       \ the three rivals whose margin is under test
#              hounie          ) alm also takes 29, so it is carried to show
#              alm            /  the asymmetry is not a TraLO peculiarity
#              tralo_null     the lambda=0 twin, SHARED (at lambda 0 with
#                             lambda_step 0 there is no constraint gradient)
#              tralo_reseed   the RNG floor -- gen_campaign REFUSES without it
#              clip           \ auto-added mandatory clippers
#              focal_clip     /
#
#   caps       L80_G95 L90_G95 L95_G80 -- EXACTLY dom1's three, so this is the
#              same contrast with one variable changed. `L80_G95` and
#              `L95_G80` give class 2 the same K=296 through different scopes.
#
#   models     MobileNetV2 MobileNetV3 -- exactly dom1's two. This campaign
#              exists to re-run dom1's head-to-head with the head start
#              removed; changing the backbone too would confound it.
#
#   ⚠️ FALSIFIABLE, FIXED BEFORE THE DATA.
#
#   🛑 THE UNIT IS THE WARM-UP MODEL. 2 backbones x 4 seeds = 8 independent
#   units; the three cap tags within a (model, seed) SHARE one warm-up and two
#   of them share a budget, so they are correlated replicates, not cells.
#   Exact sign floor at n=8 is 2/2^8 = 0.0078. FRAMEWORK 2(z) is the receipt:
#   8 of 9 dom1 sweeps evaporated when this was applied.
#   ⚠️ AND CHECK FOR DUPLICATION BEFORE POOLING WITH dom1. dom1's L80_G95 and
#   L90_G95 cells are byte-identical to loose1's in 80/80 files. md5 this
#   campaign against both before combining anything.
#
#   PRIMARY, exactly one:
#     `tralo_lam0` - `fioretto` and `tralo_lam0` - `hounie`, on **ccF1**,
#     seed-paired, sign test over the 8 (model, seed) units.
#     PASS = `tralo_lam0` still beats BOTH in >= 7 of 8 units (p = 0.0703 --
#     ⚠️ which does NOT reach 0.05; at 8 units only 8/8, p = 0.0078, does).
#     So the honest bar is **8 of 8 for a significant claim, 7 of 8 for a
#     direction**, and the report must say which was met.
#
#   SECONDARY, pre-specified:
#     * THE SIZE OF THE HEAD START: `tralo` - `tralo_lam0`, same metric and
#       units. This is what one extra constraint step at lambda 0.01 is worth,
#       and it has never been measured. If it is LARGER than
#       `tralo` - `fioretto`, then the published margin is the head start.
#     * 🛑 macroF1 AND uncF1 beside ccF1 in EVERY table, in items as well as
#       F1. On dom1 `tralo` is ccF1 +0.0141 and macroF1 -0.0022.
#     * FLOOR: every contrast quoted beside `tralo_reseed` - `tralo_null`.
#     * DOSE: `dose_landed`'s CROSS-ARM block must show `tralo_lam0` at 29.00
#       attempted/run -- the counter increments on the ATTEMPT, and the
#       epoch-1 step is attempted with a zero gradient. The 28-vs-29 question
#       is therefore settled from the LOG, not the counter: `Grad_Norm` at
#       epoch 1 must be 0.0 for `tralo_lam0` and 3.09-ish for `tralo`.
#       🛑 CHECK THAT ON THE FIRST COMPLETED RUN. If `tralo_lam0`'s epoch-1
#       grad norm is non-zero the arm has not done what it claims and the
#       whole campaign is void.
#
#   FAIL, stated so it can happen: `tralo_lam0` loses to `fioretto` or
#   `hounie` where `tralo` beat them. That would mean the dominance result
#   this project has been building on is a 3.4% dose advantage, and it would
#   be the most important negative the project has produced.
#
#   size       6 cells x 9 arms x 4 seeds = 216 runs (2 backbones x 3 caps).
#              MobileNets, so this is the cheapest campaign in the queue.
#
#   dose       `--constraint-fp32` mandatory and passed. `iwc1` is the receipt:
#              fp16 without it gave an ARM-DEPENDENT spread, alm 51.7% against
#              hounie 100%, which makes a cross-arm ordering a measurement of
#              the GradScaler. That campaign is now quarantined.
#
#   grad mode  `normalize`, matching dom1 exactly.
#
#   pin        10d37518 -- the commit that adds `tralo_lam0`. dom1's pin is
#              older, so these runs are NOT poolable with dom1's on
#              `code_version`; this campaign carries its own `tralo`,
#              `fioretto`, `hounie` and `alm` arms for exactly that reason.
#
#   read it in this order
#              python -m scripts.quarantine --check results/equaldose1
#              python -m scripts.dose_landed results/equaldose1
#                ^ read the CROSS-ARM ATTEMPTS block
#              python -m scripts.flag_live tralo tralo_lam0
#                ^ 🛑 a brand-new arm is the highest-risk case for a FIFTH
#                  inert flag. If the md5s match, lambda_global never reached
#                  the loss and every number below is `tralo` wearing a label.
#              python -m scripts.full_panel --campaign results/equaldose1 --control tralo_null
#              python -m scripts.full_panel --campaign results/equaldose1 --control clip
#
# =============================================================================
set -euo pipefail

PIN=10d37518                  # the commit that adds tralo_lam0
TREE=~/optloss-equaldose      # its OWN worktree. Worktrees share one object
                              # store, so this fetches and checks out its own
                              # tree and runs NO git maintenance: never gc,
                              # prune, repack or worktree prune while any
                              # campaign is running.
ROOT=results/equaldose1

# 🛑 REFUSE TO RUN FROM INSIDE THE TREE THIS SCRIPT IS ABOUT TO CHECK OUT.
# Bash reads a script incrementally, by byte offset, so a checkout of $TREE
# would rewrite this file underneath the interpreter at an offset it has not
# reached yet. Copy it out first:
#
#     git show origin/headroom/small-cnn:docs/launch_equaldose1.sh > ~/launch_equaldose1.sh
#     GPU=3 bash ~/launch_equaldose1.sh
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
echo "check the OTHER host before trusting it."

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
    --arms tralo tralo_lam0 fioretto hounie alm tralo_null tralo_reseed \
    --constraint-grad-mode normalize \
    --constraint-fp32

# THE THREE GATES. Each refuses a different way to waste a week. smoke_arms
# matters more than usual: `tralo_lam0` has never executed, and the config
# gates are structurally blind to a runtime crash -- three arms once shipped
# with an undefined name in train(), burned all 29 constraint epochs, died,
# reset to pending, and the campaign came back looking merely unfinished with
# audit_config and check_parity both green.
"$PY" -m scripts.audit_config
"$PY" -m scripts.check_parity "$ROOT"
"$PY" -m scripts.smoke_arms
"$PY" -m scripts.smoke_arms --matrix

GPU=${GPU:-3}
# main.py prompts for a GPU and reads the answer from stdin. With
# CUDA_VISIBLE_DEVICES pinned to one device it sees exactly one, so the answer
# is always index 0 -- NOT $GPU, which is the physical id.
echo 0 > /tmp/gpuchoice_equaldose1
setsid env CUDA_VISIBLE_DEVICES="$GPU" EXPERIMENT_DIR="$ROOT" \
    PYTHONIOENCODING=utf-8 nohup "$PY" main.py \
    > /tmp/equaldose1.log 2>&1 < /tmp/gpuchoice_equaldose1 &
sleep 25
echo "launched $ROOT on GPU $GPU from $TREE at $PIN"
tail -5 /tmp/equaldose1.log || true
