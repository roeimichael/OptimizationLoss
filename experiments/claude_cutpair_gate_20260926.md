# CUTPAIR liveness gate -- PREREGISTERED 2026-09-26

Written before the gate ran. It decides whether CUTPAIR is worth implementing and running at all.

## Where CUTPAIR comes from

- Proposed by the reshape research workflow (wf_e8af9e08-2de), where it ranked first.
- It survived adversarial critique as "run with fixes".
- Idea: a supervised hinge on the grade-3 log-odds `s = z3 - logsumexp(z_other)` of TRAIN items,
  anchored at `tau`, the logit of the cap-th largest development p3. `tau` is exactly where
  capped_first cuts.
- The WHO information comes from the train labels. The unlabeled development pool only places
  the anchor.

## Risk

The knee training set is memorised by epoch 5: training CE is ~0.15 at epoch 5 and ~0.06 at
epoch 10, in stored runs 1901, 1905 and 1910. The earlier train-top-K ranking loss went silent
because its training top-K was pure grade 3 (`knee_hard_pair_protocol_20260924.md`).

## Gate (fixed now)

- `tralo/cutpair_gate.py`, seed **2000**. This seed is non-study and is excluded from every
  analysis block.
- Schedule: tralo_null (5 CE warm-up epochs, then a task-Adam reset, then 5 CE epochs).
- After each epoch 6-10, it measures on an eval-mode training bank:
  - `N_act`: non-grade-3 items with `s > tau - 1`;
  - `P_act`: grade-3 items with `tau - 3 < s < tau + 1`.
- It does this at caps 50 and 76. m = 1 and W = 3 are fixed.

**Kill rule, per cap:** CUTPAIR is dead on arrival if the mean over epochs 6-10 of `N_act` < 20
OR of `P_act` < 20.

- If it is dead at both caps: record "train-label information at the development cut is
  exhausted by memorisation". Do not implement CUTPAIR or PREC@K-RAMP, which draws on the same
  information source.
- If it is alive: implement CUTPAIR with all critic fixes and preregister its study
  (seeds 2201-2224 at cap 50 as primary, 2301-2324 at cap 76 as replication) before launch.

Development labels are not read by the gate.

## RESULT (2026-09-26): DEAD AT BOTH CAPS -- CUTPAIR and PREC@K-RAMP are not implemented

Release 5193051f, dsisco01 GPU 0, seed 2000. Full output: `analysis/cutpair_gate_20260926.json`.

| epoch | train acc proxy | cap 50: tau / N_act / P_act | cap 76: tau / N_act / P_act |
|---|---|---|---|
| 6 | 0.994 | 4.40 / 2 / 176 | 2.10 / 15 / 63 |
| 7 | 0.998 | 4.86 / 0 / 122 | 2.35 / 5 / 17 |
| 8 | 0.997 | 4.37 / 1 / 128 | 1.84 / 5 / 23 |
| 9 | 1.000 | 3.15 / 0 / 93 | 0.27 / 2 / 6 |
| 10 | 0.990 | 1.91 / 3 / 127 | -1.12 / 40 / 33 |
| **mean** | | **N_act 1.2**, P_act 129 | **N_act 13.4**, P_act 28 |

- **Kill rule met at both caps.** `N_act` < 20: of ~5,040 training non-grade-3 images, ~5,020
  sit more than one log-odds unit below the score where the development cut falls.
- **Why.** The development set's ranking errors, the wrong-grade items inside the 50 or 76
  slots, are generalisation errors. They have no counterpart in the memorised training set, so
  a supervised loss anchored at the cut has nothing to push down.
- **What P_act adds.** It is non-trivial only at cap 50, and there it can only push positives
  up. That is a one-sided shift of the grade-3 score: a count effect, closed by LEDGER #9.
- **Conclusion:** train-label information at the development cut is exhausted by memorisation.
  This closes CUTPAIR and PREC@K-RAMP on this cell without spending a study. The remaining
  information source is the unlabeled development IMAGES (BANDCONS).
