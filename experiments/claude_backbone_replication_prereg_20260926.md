# Targeted-step replication on two more backbones -- PREREGISTERED 2026-09-26

Written before any seed of this study ran. Author: Claude, branch `claude/bandcons-20260926`.

Disclosure: the author has seen every knee result so far:
- LEDGER #9 and #10;
- the depth probe;
- BANDCONS and CUTPAIR;
- the snapshot-ensemble confirmations.

The seeds are fresh.

## Why

Every targeted-step result (LEDGER #9) rests on one cell, knee grade 3 with ResNet18. A thesis claim that
"a correctly dosed count constraint equals the post-hoc cut" needs a second cell. This study keeps the
data and the design and changes the backbone. Both backbones are ImageNet-pretrained and appeared in the
earlier fmow2 corpus.

## Design (fixed)

Runner: `tralo/knee_e2e_v3.py`, unchanged except for a `backbone` config key. The ResNet18 default is
bit-identical.

| block | backbone | pilot | study | cap |
|---|---|---|---|---|
| `mn3` | MobileNetV3-large | 3000 | 3001-3024 | 76 |
| `rgy` | RegNetY-400MF | 3100 | 3101-3124 | 76 |

Weights: MobileNetV3-large and RegNetY-400MF use torchvision IMAGENET1K_V2, because the offline server caches only the V2 files. ResNet18 stays on V1. MobileNetV3 has classifier dropout (p = 0.2), which ResNet18 did not; all arms of a seed still share the seed, the start and the batch order.

Everything else is as in `claude_targeted_step_protocol_20260925.md` (rebuild branch):
- 5 CE warm-up epochs + 5 epochs, batch 32, task Adam 1e-4, FP32, dsisco01, OMP_NUM_THREADS=8;
- arms `clipper`, `tralo_null`, `tralo_adam`, `tralo_target`, `sham_target`;
- the targeted step bisected to the smallest radius meeting the hard cap;
- the sham matches that radius and the per-tensor norms.

## Endpoint and contrasts (fixed, per block)

Primary endpoint: capped_first grade-3 F1 (76 slots), which is 2TP/182.

**Primary family per block (Holm over 2, paired over 24 seeds, two-sided alpha 0.05):**
- **C1** `tralo_target` - `tralo_null`
- **C2** `tralo_target` - `sham_target`

**Secondary:**
- C1 and C2 on the snapshot ensemble: the mean of the epoch 06-10 `after_constraint` dev probabilities.
- `tralo_adam` - `tralo_null`.
- `clipper` - `tralo_null`.
- Eviction overlap of the targeted step with the post-hoc cut (as in `analysis/eviction_precision.py`).
- **Snapshot-ensemble confirmation sets 3 (mn3) and 4 (rgy):** ENS - BASE for `clipper` and
  `tralo_null`, Holm over the two, with the same rule and readings as
  `claude_snapshot_ensemble_prereg_20260926.md`.

## Readings, fixed before the numbers

1. **Both blocks give C1 and C2 non-significant:** the ResNet18 result replicates on two further
   backbones, and the claim generalises across architectures on this data.
2. **C1 > 0 and C2 > 0, Holm-significant, in either block:** the direction carries WHO information on
   that backbone. The ResNet18 negative does not generalise, and that backbone becomes the lead.
3. **C2 < 0 significant in either block:** on that backbone the direction evicts worse than a random move.
4. **C1 < 0 with C2 non-significant:** harm from the perturbation, not the information.

## Pilot gates (3000 and 3100; excluded from the study)

Any failure stops that block and is recorded.
- All five arms complete.
- Warm-up and batch hashes identical across arms.
- 1810 task updates in every arm.
- Every applied targeted step has hard_before > 76, and for `tralo_target` hard_after <= 76.
- `tralo_target` applies at least one step.
- The natural grade-3 count at the first post-warm-up check exceeds 76. If the cap does not bind, the
  study cannot test the constraint on that backbone.
- The seed finishes in at most 60 minutes.

## Amendment 1 (2026-09-26 17:0x, both pilots still running, no pilot output read)

The gate item "natural grade-3 count at the first post-warm-up check exceeds 76" is withdrawn. Tested
on a valid seed of the original ResNet18 study (1801), it fails: the count there is exactly 76 at the
first check. Yet the cap bound at 3 of 5 checks in that seed, and the step applied 3 times. The count
moves between epochs, so the first check is the wrong test of whether the cap binds.

**Replacement:**
- `tralo_target` must apply at least one step, which is the original protocol's binding requirement
  and is already a gate item.
- The number of checks with hard count > 76 is reported for each pilot.

Everything else is unchanged. Checker: `analysis/repl_pilot_gate.py`.
