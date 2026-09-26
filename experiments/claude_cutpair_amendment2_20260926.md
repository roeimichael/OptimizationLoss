# CUTPAIR amendment 2: ensembled secondaries -- 2026-09-26

Written while study seeds 2701-2724 were running (14 of 24 complete). No study seed has been scored or
inspected for any endpoint. It amends `claude_cutpair_protocol_20260926.md`. It adds secondaries only;
the primary family, endpoint and readings are unchanged.

## Why

The snapshot ensemble was confirmed on fresh seeds (`claude_snapshot_ensemble_prereg_20260926.md`,
set 1). On the target studies it also roughly halves the width of paired arm-vs-arm CIs:

| cap | target - clipper, BASE | target - clipper, ENS |
|---|---|---|
| 76 | [-0.31, +2.48] | [-0.69, +1.19] |
| 50 | [-1.35, +2.02] | [+0.08, +1.51] |

Source: `analysis/ens_vs_ens_target_studies.txt`. The post-hoc bar is now the clipper on the ensemble.

## Added secondaries (fixed now)

- **P1-ENS, P2-ENS, P3-ENS:** the three primary contrasts with every arm's probabilities replaced by
  the mean of its epoch 06-10 `after_constraint` snapshots. Paired t over seeds, Holm over the three,
  reported beside the primary.
- **Reading:** if P3-ENS (`cutpair_aug` - `clipper`, both ensembled) is positive and Holm-significant,
  CUTPAIR beats the strongest post-hoc bar measured. If P3 is positive but P3-ENS is not, CUTPAIR's
  gain is a variance effect that ensembling already delivers.
