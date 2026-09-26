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
