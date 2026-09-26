# Snapshot-ensemble cut: confirmation on fresh seeds -- PREREGISTERED 2026-09-26

Written before any seed of the confirmation sets below was scored for this question. Author: Claude.

## Why

This was found post hoc on the two finished targeted-step studies (seeds 1801-1824 at cap 76, 1901-1924 at cap 50; `~/tralo-rebuild/lab/labelfree_gate/`). A label-free gate showed that an item's epoch history predicts wrong top-cap occupants beyond final-epoch p3. The obvious control, the top-cap by the MEAN p3 over the epoch 6-10 snapshots, gained more than the history re-rank:

| arm | cap 76 | cap 50 |
|---|---|---|
| tralo_null | +3.17 [+1.92, +4.46] slots | +2.46 [+1.46, +3.58] |
| clipper | +3.42 [+2.54, +4.29] | +2.04 [+1.08, +3.04] |

Figures are the change in correct capped slots relative to capped_first on the final probabilities, with 95% seed-bootstrap CIs, n = 24. The rule was chosen after seeing P2b, so it must be confirmed on seeds it has never seen.

## Rule (fixed)

- **ENS:** capped_first on the element-wise mean of the five dev probability snapshots `epochNN_after_constraint.pt`, NN = 06..10.
- **BASE:** capped_first on `final_probabilities.pt`, which equals the epoch-10 snapshot.
- No weights, no tuning, no labels.

## Confirmation sets

These sets are fixed. Each is scored only after its own preregistered primary analysis has been written up.
1. `claude-bandcons-a1cap50`, seeds 2401-2424, cap 50. Four seeds had finished when this was written; they have not been scored for this question.
2. The CUTPAIR study, seeds 2701-2724, cap 76, if it launches.

## Endpoint and test

- **Primary:** correct capped slots ENS - BASE for `clipper` and for `tralo_null`, paired t over seeds, two-sided alpha 0.05, Holm over the 2 arms within each set.
- **Secondary:** the same quantity for every other arm; grade-3 F1 = 2TP/(106+cap).

## Readings (fixed)

- **ENS - BASE > 0, Holm-significant for both arms in set 1:** confirmed. The post-hoc bar rises to "clipper on the snapshot ensemble", and every TraLO comparison must be made at equal ensembling (ENS vs ENS). The practical lever at the cut is variance of p3, not the constraint direction.
- **Null or negative:** the post-hoc finding does not replicate and is recorded as not confirmed.

## RESULT, set 1 (BANDCONS 2401-2424, cap 50), scored 2026-09-26 after the BANDCONS primary write-up

Scorer: `analysis/score_snapshot_ensemble.py`, which reproduced the exploratory numbers exactly
before use. Raw output: `analysis/ens_bandcons_a1cap50.txt`.

| arm | ENS - BASE, correct slots | p | Holm | W/T/L |
|---|---|---|---|---|
| clipper (primary) | **+1.62 [+0.53, +2.72]** | 0.006 | **0.011** | 18/1/5 |
| tralo_null (primary) | **+0.96 [+0.02, +1.89]** | 0.045 | **0.045** | 14/5/5 |
| aug_clip | +1.58 [+0.60, +2.56] | 0.003 | | 16/3/5 |
| bandcons | +2.71 [+1.25, +4.17] | 0.0008 | | 18/1/5 |
| bandcons_unc | +2.96 [+1.42, +4.50] | 0.0006 | | 18/3/3 |
| bandcons_rand | +2.79 [+1.84, +3.74] | <1e-4 | | 19/3/2 |

**Reading: CONFIRMED.** Both primaries are positive and Holm-significant on seeds the rule had never seen.

- **The effect shrinks** from the exploratory +2.0 to +2.5 slots at cap 50 to +1.0 to +1.6 slots,
  which is 1.2 to 2.1 F1 points at cap 50.
- **The post-hoc bar is now "clipper on the snapshot ensemble".** Every TraLO comparison is to be made
  ENS vs ENS.

Set 2 (CUTPAIR 2701-2724) is scored after the CUTPAIR primary write-up.

## RESULT, set 2 (CUTPAIR 2701-2724, cap 76), scored 2026-09-26 after the CUTPAIR primary write-up

Raw output: `analysis/ens_cutpair_cap76.txt`.

| arm | ENS - BASE, correct slots | p | Holm | W/T/L |
|---|---|---|---|---|
| clipper (primary) | **+2.58 [+1.22, +3.95]** | 0.0007 | **0.0007** | 15/3/6 |
| tralo_null (primary) | **+2.88 [+1.65, +4.10]** | 0.0001 | **0.0001** | 17/4/3 |
| aug_clip | +2.00 [+0.88, +3.12] | 0.001 | | 15/6/3 |
| cutpair_aug | +2.21 [+1.34, +3.08] | <1e-4 | | 19/4/1 |
| cutpair_aug_shift | +2.25 [+1.38, +3.12] | <1e-4 | | 18/3/3 |

**CONFIRMED again, at cap 76.** Two independent fresh confirmation sets, both caps, both primaries Holm-significant in each.
