#!/usr/bin/env bash
# THE MARGIN-COUNT DECOMPOSITION CAMPAIGN. Fire when dsisco is reachable.
#
# REGIME, stated out loud before launching (FRAMEWORK rule):
#   dataset      dermmnist  (test set is LEAKED 38.7% -- PAIRED arm-vs-arm
#                            survives the shared confound; NO absolute number
#                            from this campaign may be quoted)
#   backbone     MobileNetV3
#   caps         L50_G30, L40_G30   -- two levels, and G < L on both so the
#                                      GLOBAL scope actually binds
#   capped       classes 2 and 4    -- coupled multi-class, the one real opening,
#                                     AND nearly equal in size (n = 220 vs 223),
#                                     so the penalty's 1/K gradient asymmetry is
#                                     ~1.01x between them. The penalty is f(E/K),
#                                     so a bigger budget gets a SMALLER gradient
#                                     per unit excess -- capping 1 and 2 instead
#                                     would put a 2.1x dose difference between the
#                                     two constrained classes and confound the
#                                     see-saw with the shape.
#   warm-up      1 / constraint 29 for trained arms
#                30 / 0 for post-hoc arms          => 30 CE epochs each side
#   arms         tralo       soft value, p(1-p) placement   (the manuscript)
#                tralo_st    HARD value, p(1-p) placement   (value fix alone)
#                tralo_margin HARD value, margin placement  (both)
#                tralo_coin  RANDOM direction, same norm     (the coin control)
#                tralo_null  lambda = 0                     (the regime control)
#                clip, focal_clip                           (in-campaign bars)
#   dose         cut_window_items 5
#   size         2 cells x 7 arms x 4 seeds = 56 runs
#   COST         1648 epochs = 2 cells x 4 seeds x 206. The five trained arms
#                share ONE cached warm-up (same base_model_id, verified), so
#                only clip and focal_clip pay a full 30-epoch warm-up each.
#                At ~1 min/epoch that is 27.5 GPU-hours => ~13.7 h wall on the
#                2 GPUs the house rule allows; at 1.3 min/epoch, ~18 h.
#                Budget a night, not an afternoon.
#   ⚠️ NO RUN RECORDS ITS WALL TIME -- config.json has no duration key, so this
#      is an estimate from "a 29-epoch run is ~30 minutes", not a measurement.
#
# UNDERPOWERED BY CONSTRUCTION: 2 cells cannot reach significance on any single
# metric. This reports DIRECTION and per-cell consistency only. If it moves,
# extend with octmnist -- a DATASET adds independence, a backbone only
# resolution.
set -euo pipefail
ROOT=results/margin1
cd ~/OptimizationLoss
git pull --ff-only
python -m pytest tests -q
python -m scripts.audit_config
python -m scripts.smoke_arms --matrix
python -m configs.gen_campaign --root "$ROOT" \
    --datasets dermmnist --models MobileNetV3 \
    --caps L50_G30 L40_G30 \
    --arms tralo tralo_st tralo_margin tralo_coin tralo_null \
    --constrained-class 2 4
python -m scripts.verify_caps "$ROOT"
python -m scripts.check_parity "$ROOT"
python -m scripts.flag_live tralo tralo_st tralo_margin tralo_coin tralo_null
echo "ALL GATES PASSED -- launch with main.py against $ROOT"
cat <<'READ'

HOW TO READ IT -- fixed in advance, so it cannot drift after the numbers land.

  python -m scripts.full_panel --campaign RESULTS/margin1 --control clip
  python -m scripts.full_panel --campaign RESULTS/margin1 --control tralo_null
  python -m scripts.full_panel --campaign RESULTS/margin1 --control tralo_coin
  python -m scripts.full_panel --campaign RESULTS/margin1 --control tralo

`clip` is the headline bar and the stronger clipper. `tralo_null` separates the
REGIME from the treatment. `tralo_coin` is the pre-registered kill condition.
`tralo` as control gives the decomposition directly.

THE READING, in order:

1. RAW-PREDICTION IDENTITY first, before any metric. If two arms share an md5
   on a cell-seed, that comparison is void whatever the metrics say.

2. ccP is the metric that decides this. ccF1 is precision@K rescaled and
   ccP/ccR/ccF1 are one metric in three costumes -- quote ONE. An arm that
   moves AUROC and not ccP has been run twice already (`budget_margin`, and
   the shipped penalty); that is not a win.

3. tralo_margin must beat tralo_coin. Same step norm, no information. If it
   does not, the direction contributed nothing a coin could not have and no
   dose or width changes that.

4. Decomposition: tralo -> tralo_st isolates the count VALUE, tralo_st ->
   tralo_margin isolates the WINDOW. Report both, never the bundle.

5. CHECK THE CONSTRAINT ACTUALLY BOUND, per seed, before averaging. With
   straight_through the penalty is relu(hard - K), so a seed whose count is
   already under budget at warm-up 1 takes NO step for the whole run and is
   bit-identical to tralo_null. That is correct behaviour and a real zero, but
   it is not a treated seed, and averaging it in dilutes the effect toward
   zero while looking like a null. `clip` binds on 63-84% of runs, so expect
   one or two untreated seeds. Read Global_Satisfied / Local_Satisfied in each
   training_log.csv and report how many epochs each arm actually stepped.
   flag_live already skips seeds where nothing exceeded the budget.

6. `flips`, raw count over K, and "proximity to feasibility" are NOT metrics.
   Post-hoc filling is free. When quality ties the honest report is "this arm
   produced nothing".

7. 2 cells CANNOT reach significance on any single metric. This reports
   DIRECTION and per-cell consistency. Do not quote a p-value as a verdict.

POWER, STATED SO THE RESULT IS NOT OVER-READ. Paired seed sd is ~2.7 items:

    4 seeds  -> SE 1.35 items; a 3-item effect is 2.2 sigma, a 5-item 3.7
    6 seeds  -> SE 1.10 items; 2.7 sigma / 4.5
   10 seeds  -> SE 0.85 items; 3.5 sigma / 5.9

At 4 seeds this campaign resolves a LARGE effect and is marginal on a small
one, and the whole headroom is only 2-10 items. So an ambiguous result is a
likely outcome and is NOT a null -- it means add seeds. Considered running 5
arms first and adding the attribution arms later; rejected, because it saves
only ~4 h and gen_campaign skipping completed runs makes the extension cheap
either way, while a partial first stage cannot attribute anything it finds.

IF THE DIRECTION IS POSITIVE, ADD SEEDS -- NOT CELLS. The entire headroom is
2 to 10 items (F1 = 2TP/(K+n), so 0.67-1.65 items per 0.01 capF1), and the
paired seed sd is ~0.04, about 2.7 items. At 4 seeds the SE of the mean
difference is ~1.35 items against an effect that cannot exceed ~10. Going to
10 seeds cuts that to ~0.85; adding a cell does not cut it at all, it only
adds another underpowered cell. Precedent exists (`r2_seeds10`).

A TIE IS THE PRE-REGISTERED EXPECTATION (FRAMEWORK 1b): post-hoc is optimal
given the probabilities, the cap adds no information the training set lacks,
and under K << n_true top-K by probability IS the clipper. If it ties, the
next move is proposal 1c -- optimise precision@K with LABELS -- not a third
count.
READ
