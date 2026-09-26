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
