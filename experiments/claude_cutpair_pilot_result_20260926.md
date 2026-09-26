# CUTPAIR pilot (seed 2700): gate PASS -- 2026-09-26

Release c0ab1344. Checked with `analysis/cutpair_pilot_gate.py`, which is label-free. No development score was read.

| check | result |
|---|---|
| all five arms complete | PASS |
| warm-up, batch and TTA hashes identical | PASS |
| 1810 updates in every arm | PASS |
| dose ratio 0.1 where active (both cutpair arms) | PASS (worst deviation 6e-9) |
| cutpair_aug mean N_act >= 20 | PASS (66.4) |
| cutpair_aug mean P_act >= 20 | PASS (221.8) |
| cutpair_aug active-batch fraction >= 0.5 | PASS (0.756) |
| seed <= 90 min | PASS (30.6) |

## Per-epoch active sets

| arm | epoch 6 | epochs 7-10 |
|---|---|---|
| `cutpair_aug` | tau +2.03, N_act 22, P_act 61, 60/181 batches | tau -0.5 to -1.4, N_act 61-113, P_act 237-303, 150-170/181 |
| `cutpair_aug_shift` | drew rank 228 in all 5 epochs: N_act 670-1474, P_act 0-22, 176-181/181 batches | |

Notes:
- **Epoch 6** is the first augmented epoch, which is why the model is still near memorised there.
- **The five-of-five draw is chance, not a defect.** Simulating the seeded draws shows seeds 2701-2724
  mix both ranks (the number of 228 draws per seed ranges 0-5, mode 2).
- **At rank 228 the shift hinge is almost purely a negative push**, a different active-set composition.
  This is the confound that amendment 1 already reads P1 against. Per-rank reporting is required.

## Launch

The study seeds 2701-2724 launched 2026-09-26 12:42 IDT on release 6fb21e48:
- same trained path as the pilot (log-only changes, bit-identity tested);
- dsisco01, 4 queues x 6 seeds, all GPUs ours only.
