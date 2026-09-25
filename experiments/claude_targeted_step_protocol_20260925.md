# TraLO's direction at the dose that lands on the hard cap, with a sham -- PREREGISTERED 2026-09-25

**Written before any seed of this study ran.** Author: Claude, branch
`claude/rebuild-validation-20260925`. Disclosure: the author has seen the n = 1 pilot of the
predecessor study (`claude_controller_sham_protocol_20260925_result.md`, seed 1701), which
motivated this design. Seeds 1801-1824 are fresh, and none have run.

## Why

The predecessor's pilot showed the published constraint step overshoots about 10x. One step
moved the grade-3 soft count from 82.6 to 8.2 against a cap of 76. A random direction of the
same norm moved it by only +4.7. With a single capped class, lambda and rho only rescale the
gradient, so the controller cannot fix the size. This study keeps TraLO's direction and fixes
the size to exactly what the cap asks for.

## Design (fixed)

Identical to the predecessor, apart from the arms and seeds:
- Chen OAI knee, 5778 train / 826 development. The test split is not scored and its labels
  are not read.
- ImageNet ResNet18 with a trainable backbone.
- 5 CE warm-up epochs + 5 constraint epochs, batch 32, task Adam 1e-4.
- One constraint check per post-warm-up epoch.
- Grade-3 cap 76.
- dsisco01 (FP32 Quadro), OMP_NUM_THREADS=8 (shown byte-neutral).
- Runner `tralo/knee_e2e_v3.py`; seeds **1801-1824**, all 24 reported.
- Seed 1801 is the integrity pilot.

| arm | constraint step |
|---|---|
| `clipper` | none; task Adam kept across the boundary |
| `tralo_null` | none; task Adam reset at the boundary |
| `tralo_adam` | the published separate-Adam step, lr 3e-5 (replication of the overshoot) |
| `tralo_target` | `tralo/targeted_step.py`. **Trigger:** the hard grade-3 count exceeds 76. **Direction:** steepest descent of the grade-3 soft count (verified streamed gradient). **Size:** the smallest radius that brings the hard count to at most 76 (doubling bracket, then 20 bisection steps). |
| `sham_target` | Same trigger and same radius (found along the real direction). The move itself is in a seeded random direction with the real step's per-tensor norms. |

Development images only; no label enters any step.

## Endpoint and contrasts (fixed)

Primary endpoint: development grade-3 F1 under `capped_first` (exactly 76 slots). F1 = 2TP/182,
so every contrast is also reported in correct slots.

Primary family (Holm, paired over seeds, two-sided alpha 0.05):
- **C1** `tralo_target` - `tralo_null`: does the constraint's direction, at the right size, help?
- **C2** `tralo_target` - `sham_target`: is any effect the INFORMATION, not the perturbation?

Secondary, with t intervals, not in the family:
- raw (unallocated) grade-3 F1 for every contrast; `tralo_target` is feasible by construction;
- `tralo_adam` - `tralo_null` (the overshoot, replicated);
- `sham_target` - `tralo_null`;
- `clipper` - `tralo_null`;
- slot turnover against the null's reseed floor;
- per-step radius and hard counts before and after.

MDE, stated now: the paired SD of 7.87 F1 points gives an 80%-power MDE of ~4.5 F1 points
(~4 correct slots) at n = 24. Smaller effects must be worded as underpowered, not absent.

## Readings, fixed before the numbers

1. **C1 > 0 and C2 > 0, both Holm-significant:** at the right size, the constraint's information
   improves which patients get the slots. This would be the first positive result.
2. **Both non-significant:** fixing the dose does not rescue the method at this power; the
   negative result survives its strongest objection.
3. **C2 < 0 significant:** the constraint direction evicts the wrong items; it is worse than
   a random move of the same size.
4. **C1 < 0 with C2 non-significant:** the harm comes from the perturbation, not the
   information.

## Integrity gate for the pilot (seed 1801)

Any failure stops the study and is recorded, not patched.
- All five arms complete.
- Warm-up and batch hashes identical across arms.
- 1810 task updates in every arm.
- Every applied targeted step has hard_before > 76 and, for `tralo_target`, hard_after <= 76.
- `tralo_target` applies at least one step.
- If both targeted arms apply a step at the first check (same state), their radii are equal.
- The seed finishes in at most 60 minutes.
