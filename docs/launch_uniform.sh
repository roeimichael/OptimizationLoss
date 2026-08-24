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
# does not. -30.8 items is attributable to the constraint on ViTB16, and -3.4 on
# MobileNetV3 (results/iwc1), where a pointless reseed already costs 7.4 items.
#
# For scale: the headroom from `clip` to a PERFECT allocator on iwildcam is
# 0.2-2.0 items per class (scripts.headroom, corrected 2026-08-24 -- the tool
# had been using the INERT global cap instead of the binding local sum and
# reported 59-114). So the constraint is spending 2-150x the entire prize
# backwards. Removing that is the largest available win on this dataset.
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
#                tralo_uniform  sum_i p_ic,   gradient CONSTANT fix 1: PLACEMENT
#                tralo_ortho    step projected off the CE grad  fix 2: DIRECTION
#                tralo_null     lambda = 0                      the twin
#                tralo_reseed   the twin, one RNG draw          the noise floor
#                clip, focal_clip                               in-campaign bars
#
#                📌 AMENDMENT, 2026-08-24, BEFORE ANY RUN OF THIS CAMPAIGN
#                EXISTS. `tralo_ortho` was added after the campaign was first
#                generated and before it was launched, on the strength of
#                FRAMEWORK 2(s): the constraint's damage to the six UNCAPPED
#                classes does not arrive through the output layer at all --
#                the softmax cross-term perturbs those logits and provably
#                cannot reorder them (zero flips across a 50x dose range) --
#                so it arrives through the shared backbone, which is what the
#                projection acts on. 2(t) then found that direction was filed
#                as rejected without ever being tested.
#
#                THE TWO FIXES ARE ORTHOGONAL AND SHARE EVERY CONTROL, so one
#                campaign answers both: 7 arms x 9 cells x 4 seeds = 252 runs
#                against 216 + 216 = 432 for two campaigns, and it makes them
#                a HEAD-TO-HEAD under identical controls instead of two
#                readings that can only be compared across campaigns.
#   dose         constraint_grad_mode normalize -- ESSENTIAL HERE. It fixes the
#                delivered step to a protocol constant, so `tralo` and
#                `tralo_uniform` differ ONLY in the DIRECTION of the step and
#                not in its NORM. Under `clip` the two modes could deliver
#                different norms and any difference would be a dose effect,
#                which is unattributable and is exactly the trap that made the
#                hounie baseline meaningless.
#
#                ⚠️ EQUAL NORM IS NOT EQUAL EFFECT, measured 2026-08-24 AFTER
#                this campaign was generated (`scripts.collateral_probe`, 16
#                stored runs, effect matched). To remove the same 20 capped
#                predictions, `sum` needs eta 7.5 and `uniform` needs 51.9 --
#                **`uniform` does ~7x less cap enforcement per unit step** --
#                and `uniform` failed to reach 100 removals in several cells at
#                any eta <= 4096 while `sum` reached it. `normalize` equalises
#                the NORM, which is what makes the contrast legal, but it
#                cannot equalise the WORK, so `tralo_uniform` is underdosed
#                relative to `tralo` by construction.
#
#                This does not invalidate the contrast -- direction is the
#                thing under test and the norm is held fixed -- but it makes
#                ONE failure mode much more likely than the pre-registration
#                assumed: `tralo_uniform` taking a near-zero effective step,
#                moving no count, and writing `completed`. That reads as "the
#                fix is harmless" when it is really "the fix never acted".
#
#                🛑 SO THE FIRST READ IS A LIVENESS READ, NOT A VERDICT:
#                    python -m scripts.log_health results/uniform1
#                `tralo_uniform`'s capped-count trajectory must MOVE relative
#                to `tralo_null`. If it does not, the arm is a silent null,
#                the comparison is void, and the answer is a dose sweep on
#                `lr_constraint` for that arm alone -- NOT a verdict.
#   size         9 cells x 7 arms x 4 seeds = 252 runs
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
#               ⚠️ 2(s) ALREADY MAKES THIS THE LIKELIER OUTCOME. The output
#               layer was measured and cleared; `tralo_uniform` is being run
#               because 2(r)'s eviction finding is independent of that and
#               still stands, not because the output-space story survived.
#               ✅ AND 2(s) GAVE IT A MECHANISM. Per-item push, unit-normalised,
#               by how confident the item is on the capped class:
#                   p          sum         uniform
#                   0.5-0.9    3.83e-02    7.29e-03    <- sum's 100x peak
#                   0.99-1.0   3.74e-04    8.85e-03    <- 23.7x the other way
#               `sum` concentrates on the boundary band (the cut sits at
#               p=0.536; 2(r)'s evicted items average p=0.788), which is
#               exactly the eviction 2(r) measured. `uniform` is flat, and it
#               is GAUGE-INVARIANT to machine precision where the one-vs-rest
#               alternative was not.
#               🛑 IT WILL ENFORCE LESS, AND THAT IS PREDICTED, NOT A FAILURE.
#               At matched effect `uniform` reaches feasibility in 11 of 56
#               stored runs against `sum`'s 25. Flattening the gradient does
#               not make the constraint stronger, it aims it differently. So a
#               LOWER native satisfaction for this arm is expected; judge it on
#               d capF1 in ITEMS against `tralo_null`, never on enforcement.
#
#   AND FOR `tralo_ortho`, stated separately because the two can fail
#   differently:
#   PREDICTED   uncF1 vs the twin recovers toward 0 while ccF1 is unchanged.
#               2(s) puts TraLO's constraint at -0.0144 uncF1 and -0.0020 ccF1;
#               the projection is aimed at the first and should not touch the
#               second. Read `full_panel`'s uncF1 line BESIDE ccF1, never
#               macroF1 alone -- macroF1 is their sum and hides which moved.
#   NOT PREDICTED  a ccF1 gain. Nothing about projecting the step off the CE
#               direction makes the allocator better, and 2(s) measured the
#               capped-class term as near-identical across all three dual
#               families, i.e. not the thing that varies.
#   SIZE IT     the one prior measurement is AP +0.0041 against 2(s)'s -0.0609
#               AP constraint term: ~7% recovery, from a cell where the cap
#               barely bound. A nibble at that size. It is run because it is
#               the only intervention aimed at the mechanism 2(s) found and
#               the only one whose recorded sign is positive -- not because
#               +0.0041 is expected to reappear.
#   FALSIFIED IF  uncF1 vs the twin is unchanged. Then the backbone story is
#               wrong too, and what remains is that the damage is intrinsic to
#               training under a count penalty at all -- which would make the
#               post-hoc clipper the honest recommendation.
#
# HOW TO READ IT, in this order, and stop at the first one that fails:
#   python -m scripts.rig_status --campaign results/uniform1
#   python -m scripts.order_probe --campaign results/uniform1 --arm tralo_uniform
#   python -m scripts.order_probe --campaign results/uniform1 --arm tralo_uniform --evictions
#   python -m scripts.order_probe --campaign results/uniform1 --arm tralo_ortho
#   python -m scripts.family_split --campaign results/uniform1
#   python -m scripts.full_panel  --campaign results/uniform1 --control tralo_null
#   python -m scripts.full_panel  --campaign results/uniform1 --control clip
set -euo pipefail

ROOT=results/uniform1
PIN=eb453a57                 # the commit carrying BOTH candidate fixes:
                             # soft_count_mode uniform and ortho_project.
                             # Pinned, not "latest": a campaign
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
    --arms tralo tralo_uniform tralo_ortho tralo_null tralo_reseed clip focal_clip \
    --constraint-grad-mode normalize

# THE THREE GATES. Each refuses a different way to waste a week, and a campaign
# that launches past a red one is how this project loses nights.
"$PY" -m scripts.audit_config
"$PY" -m scripts.check_parity "$ROOT"
"$PY" -m scripts.smoke_arms

# Is the new flag LIVE, or a fifth inert one? (CLAUDE.md rule 3.) This is the
# md5 check across arms, and it is the difference between an arm and a rename.
"$PY" -m scripts.flag_live tralo tralo_uniform
"$PY" -m scripts.flag_live tralo tralo_ortho

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
