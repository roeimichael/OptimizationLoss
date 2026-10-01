# CUTPAIR cap-76 study: result -- 2026-09-26

- **Protocol:** `claude_cutpair_protocol_20260926.md` plus amendment 1 (dose-frequency reading of P1) and amendment 2 (ensembled secondaries).
- **Run:** release 6fb21e48, seeds 2701-2724, dsisco01. The pilot 2700 is excluded.
- **Scorer:** `analysis/score_cutpair.py`. Raw output: `analysis/cutpair_cap76.txt` / `.json`.

**Integrity: 24/24 seeds pass.** Hashes matched, 1810 updates, dose ratio 0.1 wherever a batch was active, no missing seed.

**Active sets, mean per epoch:**

| arm | anchor rank | N_act | P_act | active batches | dosed batches |
|---|---|---|---|---|---|
| `cutpair_aug` | 76 | 72.0 | 186.2 | 0.708 | 0.555 |
| `cutpair_aug_shift` | 25 (65 epochs) | 2.9 | 449.5 | 0.916 | 0.841 |
| `cutpair_aug_shift` | 228 (55 epochs) | 1234.5 | 7.5 | 0.994 | 0.982 |

**Amendment 1:** D_shift/D_cut is about 0.91/0.555 = 1.64, inside [0.5, 2], so P1 reads as preregistered. The shift arm is nonetheless one-sided at each rank: almost only positives at rank 25, almost only negatives at rank 228.

## Primary (clean capped_first grade-3 F1, paired over 24 seeds, Holm over P1-P3)

| | contrast | mean [95% CI] | Holm p | W/T/L |
|---|---|---|---|---|
| P1 | cutpair_aug - cutpair_aug_shift | +0.37 [-0.35, +1.09] | 0.61 | 13/3/8 |
| P2 | cutpair_aug - aug_clip | **-0.00 [-0.92, +0.92]** | 1.00 | 9/4/11 |
| P3 | cutpair_aug - clipper | **+3.02 [+1.33, +4.71]** | **0.0035** | 17/1/6 |

- **TTA:** P3 +3.66 [+2.42, +4.91].
- **Amendment 2, ensembled:** P1-ENS +0.32, P2-ENS +0.23 [-0.40, +0.86], P3-ENS **+2.61 [+1.08, +4.14]** (Holm 0.006).
- **Arm means (F1):** clipper 65.52, tralo_null 65.06, aug_clip 68.54, cutpair_aug 68.54, cutpair_aug_shift 68.18.

## Reading

The preregistered readings did not enumerate "P3 > 0 with P1 and P2 null". It is stated here plainly, not reinterpreted.

- **For the CUTPAIR question, this is reading 3 (all null).**
  - Anchoring at the cap's cut adds nothing over a shifted anchor (P1).
  - The hinge adds nothing over its own augmentation-trained null (P2).
  - P2 is a tight bound: under 0.92 F1, about 0.8 of 76 slots, at 95%.
- **The hinge barely changes who fills the slots.** Its slot turnover vs aug_clip is 0.100 (0.42x the reseed floor). Its swaps split 50.0% correct-direction, net 0.00 slots.
- **P3's +3.0 is entirely the augmentation schedule.** aug_clip - tralo_null is +3.48 [+1.71, +5.25], and P3 = P2 + (aug_clip - clipper). Strong train-time augmentation in epochs 6-10 is a baseline improvement, not a TraLO component. The fair post-hoc bar at cap 76 is therefore the clipper trained with the same augmentation (and ensembled). CUTPAIR ties that bar.
- **Training labels at the cut are closed as a who-source on this cell, like augmentation stability (BANDCONS).** Every information source tried at training time now ties or loses to the post-hoc cut on the same score.
