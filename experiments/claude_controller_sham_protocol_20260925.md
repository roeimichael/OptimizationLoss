# Live controller and sham control, end-to-end knee -- PREREGISTERED 2026-09-25

**Written before any seed of this study ran.** Author: Claude, branch
`claude/rebuild-validation-20260925`. Motivated by the code audit
(`analysis/CODE_AUDIT_20260925.md`) and the log analysis (`analysis/FINDINGS_20260925.md`).

## Why

The audit found the arithmetic correct and the design unable to answer the question:
- **D1** the separate Adam makes the multiplier/rho controller inert (every step ~ lr*sign(g));
- **D3** knee 82/16 caps never bind on hard counts;
- **D4** a frozen head can only re-rank by feature L1 norm.
The log analysis found that end-to-end training amplifies ANY perturbation (fmow2: 33%
slot turnover from ~10 constraint steps, reseed floor 50%), so TraLO minus Null cannot
separate the constraint's information from a perturbation of the same size.

This study fixes D1, D3 and D4 at once and adds the missing control.

## Design (fixed)

Chen OAI knee, 5778 train / 826 development; the 1656-image test split is NOT scored and
its labels are no longer read (D7 fixed). ImageNet ResNet18, **trainable backbone**, 5 CE
warm-up + 5 constraint epochs, batch 32, task Adam 1e-4, `constraint_lr` 3e-5,
lambda 0.01 / step 0.05, rho 0.5, one constraint step per post-warm-up epoch -- identical to
`knee_end_to_end_protocol_20260924.md` except for the arms. Grade-3 cap **76**; the runner
records whether it binds on the hard count at the first constraint epoch (it did in every
seed of the 24 Sept run: null raw calls 95.75 on average).

| arm | constraint step |
|---|---|
| `clipper` | none; task Adam kept across the boundary |
| `tralo_null` | none; task Adam reset at the boundary |
| `tralo_adam` | separate Adam (the published arm; replicates the -2.20) |
| `tralo_sgd` | `CalibratedSGD`: first step's norm = the exact Adam first-step norm (epsilon included), then proportional to multiplier x violation -- the controller acts |
| `sham_sgd` | `ShamOptimizer`: `tralo_sgd`'s per-tensor step norm at every step, seeded random direction |

Seeds **1701-1724**, all 24 reported, five arms per seed, one host (dsisco01, FP32 Quadro).
Seed 1701 is an integrity pilot; the rest launch only if it passes the gate below.

## Endpoint and contrasts (fixed)

Primary endpoint: development grade-3 F1 under `capped_first` (exactly 76 slots). With 106
true grade-3 images this is a monotone function of the number of correct slots:
F1 = 2*TP/182, so every contrast is also reported in correct slots.

Primary family (Holm, 2 contrasts, paired over seeds, two-sided alpha 0.05):
- **C1** `tralo_sgd` - `tralo_null`: does a constraint whose controller acts help?
- **C2** `tralo_sgd` - `sham_sgd`: is any effect the constraint's INFORMATION, not the perturbation?

Secondary, reported with t intervals, not in the family: `tralo_adam` - `tralo_null`
(replication), `sham_sgd` - `tralo_null` (does perturbation alone move the endpoint?),
`clipper` - `tralo_null` (the schedule), and slot turnover of every arm against the
null's own reseed floor.

Minimum detectable effect, stated now: the 24 Sept paired SD was 7.87 F1 points, so at
n=24 the 80%-power MDE is ~4.5 F1 points (~4 correct slots of 76). Smaller true effects
will read as null and must be worded as underpowered, not absent.

## Readings, fixed before the numbers

1. **C1 > 0 and C2 > 0, both Holm-significant**: with an acting controller the constraint's
   information improves which patients get the slots. The project's first positive result.
2. **C1 and C2 both non-significant**: fixing D1 does not rescue the method at this dose; the
   negative result survives its strongest objection.
3. **C2 < 0 significant**: the constraint direction is WORSE than a random step of equal
   size -- it actively moves the wrong items (consistent with mechanism M1).
4. **C1 < 0 with C2 non-significant**: the harm is the perturbation, not the information.

## Integrity gate for the pilot (seed 1701)

All five arms complete; warm-up and batch hashes identical across arms; the 1810 task
updates equal; constrained arms apply their constraint steps with finite parameters;
`sham_sgd` and `tralo_sgd` first displacement equal to 1e-4; `tralo_sgd` first displacement
equal to `tralo_adam`'s to 1e-3; the cap binds on the hard count. Any failure stops the
study and is recorded, not patched silently.
