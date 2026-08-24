#!/usr/bin/env bash
# THE ORDER-PRESERVING COUNT CAMPAIGN. Fire the moment a GPU frees.
#
# WHY, in one measurement (`scripts/order_probe --evictions`, results/iwc2,
# 16 cell-class-seed points, 2026-08-24). Against its own lambda=0 twin the
# shipped constraint moves 73 items per cell, and:
#
#       precision of what it EVICTED    0.6880
#       precision of what it ADMITTED   0.3007
#       NET                            -30.44 items per cell, 16/16 negative
#
# The control is the whole measurement: `tralo_reseed` moves a comparable 63
# items and nets +0.38, evicted and admitted precision equal to three decimals.
# So a perturbation of no consequence swaps items of equal quality, and this one
# does not. -30.8 items is attributable to the constraint, against a total
# headroom from `clip` to a PERFECT allocator of 1.9-9.9 items.
#
# And it is not a boundary effect: the cut sits at p=0.536 while evicted items
# average p=0.788 and admitted ones p=0.251.
#
# REGIME, stated out loud before launching (FRAMEWORK rule):
#   dataset      iwildcam           -- the only one that can carry a constraint
#   backbones    MobileNetV2, MobileNetV3, RegNetY400MF
#   caps         L20_G50, L30_G50, L50_G30
#   cells        3 x 3 = 9. DELIBERATE: the exact sign-test floor at 9 cells is
#                p=0.00391 against a BH threshold of 0.00455, so this campaign
#                can return a CALLABLE verdict. margin1's 2 cells cannot, and
#                iwc1/iwc2 could not either -- every read off them is a
#                direction. This is the smallest size that stops being one.
#   capped       classes 2 and 7 (impala, cattle)
#   warm-up      1 / constraint 29 for trained arms; 30 / 0 for post-hoc
#   arms         tralo          sum_i p_ic,   gradient p(1-p)   the manuscript
#                tralo_uniform  sum_i p_ic,   gradient CONSTANT the fix
#                tralo_null     lambda = 0                      the twin
#                tralo_reseed   the twin, one RNG draw          the noise floor
#                clip, focal_clip                               in-campaign bars
#   dose         constraint_grad_mode normalize -- ESSENTIAL HERE. It fixes the
#                delivered step to a protocol constant, so `tralo` and
#                `tralo_uniform` differ ONLY in the DIRECTION of the step and
#                not in its size. Under `clip` the two modes could deliver
#                different norms and any difference would be a dose effect,
#                which is unattributable and is exactly the trap that made the
#                hounie baseline meaningless.
#   size         9 cells x 6 arms x 4 seeds = 216 runs
#
# PRE-REGISTERED, before any run (and duplicated in the source docstring of
# `uniform_grad_count` so it cannot be quietly rewritten):
#
#   PREDICTED   `tralo_uniform` recovers the ~30 items and lands ON `tralo_null`
#               -- i.e. net items vs the twin goes to ~0 and `order_probe`'s
#               rho_arm goes to ~1. Read it with:
#                   python -m scripts.order_probe --campaign results/uniform1 \
#                          --arm tralo_uniform
#   NOT PREDICTED  that it BEATS the twin. A uniform shift is a prior shift and
#               top-K is invariant to prior shifts (FRAMEWORK 2(j)). The claim
#               is "the constraint becomes free", not "the constraint wins".
#   FALSIFIED IF  net items vs the twin stays materially negative. Then the
#               damage is coming through the SHARED BACKBONE, not through the
#               per-item output term, and the next lever is the parameter set
#               the constraint is allowed to touch -- not the count.
#
# HOW TO READ IT, in this order, and stop at the first one that fails:
#   python -m scripts.rig_status --campaign results/uniform1
#   python -m scripts.order_probe --campaign results/uniform1 --arm tralo_uniform
#   python -m scripts.order_probe --campaign results/uniform1 --arm tralo_uniform --evictions
#   python -m scripts.full_panel  --campaign results/uniform1 --control tralo_null
#   python -m scripts.full_panel  --campaign results/uniform1 --control clip
set -euo pipefail

ROOT=results/uniform1
PIN=c5e65623                 # the commit that introduced soft_count_mode:
                             # uniform. Pinned, not "latest": a campaign
                             # generated on one commit and run on another is
                             # the split this project just spent a morning
                             # cleaning out of results/iwc3.
TREE=~/optloss-uniform       # its OWN worktree. NOT ~/optloss-audit (iwc3 is
                             # live there) and NOT ~/optloss-select (xfam1 is
                             # live there). Four worktrees share one object
                             # store, so this fetches and checks out its own
                             # tree and runs NO git maintenance: never gc,
                             # prune, repack or worktree prune while any
                             # campaign in the family is running.

if pgrep -u "$(whoami)" -f "envs/optloss/bin/python main.py" >/dev/null 2>&1; then
    echo "REFUSING: a dispatcher is already running on this host."
    echo "  Deploy after the last run, never during. One dispatcher per host,"
    echo "  and the house limit is 2 GPUs across the cluster."
    exit 1
fi

MINE=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
       | while read -r p; do ps -o user= -p "${p// /}" 2>/dev/null; done \
       | grep -c "$(whoami)" || true)
if [ "${MINE:-0}" -ge 2 ]; then
    echo "REFUSING: $MINE GPUs already carry my processes; the house limit is 2."
    exit 1
fi

if [ ! -d "$TREE" ]; then
    git -C ~/OptimizationLoss worktree add "$TREE" --detach "$PIN"
fi
cd "$TREE"
git fetch -q origin
git checkout -q --detach "$PIN"
test -z "$(git status --porcelain src/ configs/ main.py)" || {
    echo "REFUSING: src/ configs/ main.py are dirty -- code_version would be"
    echo "  stamped -dirty and the campaign would read as split."; exit 1; }

mkdir -p data
[ -e data/iwildcam ] || ln -s ~/optloss-audit/data/iwildcam data/iwildcam

PY=$HOME/anaconda3/envs/optloss/bin/python
"$PY" -m configs.gen_campaign \
    --root "$ROOT" \
    --datasets iwildcam \
    --models MobileNetV2 MobileNetV3 RegNetY400MF \
    --caps L20_G50 L30_G50 L50_G30 \
    --arms tralo tralo_uniform tralo_null tralo_reseed clip focal_clip \
    --constraint-grad-mode normalize

# THE THREE GATES. Each refuses a different way to waste a week, and a campaign
# that launches past a red one is how this project loses nights.
"$PY" -m scripts.audit_config
"$PY" -m scripts.check_parity "$ROOT"
"$PY" -m scripts.smoke_arms

# Is the new flag LIVE, or a fifth inert one? (CLAUDE.md rule 3.) This is the
# md5 check across arms, and it is the difference between an arm and a rename.
"$PY" -m scripts.flag_live tralo tralo_uniform

GPU=${GPU:-0}
# main.py prompts for a GPU and reads the answer from stdin. With
# CUDA_VISIBLE_DEVICES pinned to one device it sees exactly one, so the answer
# is always index 0 -- NOT $GPU, which is the physical id and would be an
# out-of-range choice the moment GPU != 0.
echo 0 > /tmp/gpuchoice_uniform1
setsid env CUDA_VISIBLE_DEVICES="$GPU" EXPERIMENT_DIR="$ROOT" \
    PYTHONIOENCODING=utf-8 nohup "$PY" main.py \
    > /tmp/uniform1.log 2>&1 < /tmp/gpuchoice_uniform1 &

sleep 45
grep -m1 -E "Device:|GPU:" /tmp/uniform1.log || {
    echo "no device line yet -- check /tmp/uniform1.log before walking away."; }
echo "launched $ROOT on GPU $GPU from $TREE at $PIN"
