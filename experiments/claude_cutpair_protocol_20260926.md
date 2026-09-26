# CUTPAIR on augmentation: training labels spent at the deployment cut -- PREREGISTERED 2026-09-26

**Written before any seed of this study ran.** Author: Claude, branch `claude/bandcons-20260926`.

Disclosure: the author has seen the targeted-step results (LEDGER #9), the CUTPAIR gates (dead
without augmentation; alive at cap 76 with it) and the BANDCONS pilots. Seeds 2700-2724 are fresh.

## Why

- **A count loss carries no WHO information** (LEDGER #9, and the property tests in
  `tests/test_count_gradient_properties.py` on the rebuild branch).
- **Training labels do carry it, but only where training items still sit near the deployment cut.**
  - Without augmentation the training set is memorised: 1.2 / 13.4 training negatives near the cut.
  - With the aug_clip schedule, 73.8 training negatives and 225 training positives sit near the
    cap-76 cut (`analysis/cutpair_gate_aug_20260926.json`).
- **CUTPAIR spends a supervised hinge exactly there.** The anchor tau is the score at which
  capped_first cuts the unlabeled development pool, taken from images only, with no development labels.

## Design (fixed)

Runner `tralo/knee_e2e_v5.py`; cap **76** only; seeds pilot **2700**, study **2701-2724**; dsisco01;
OMP_NUM_THREADS=8. Unchanged from the earlier studies:
- Chen OAI knee;
- ImageNet ResNet18, trainable, FP32;
- 5 CE warm-up epochs + 5 epochs, batch 32, task Adam 1e-4.

| arm | epochs 6-10 |
|---|---|
| `clipper` | CE, Adam kept |
| `tralo_null` | CE, Adam reset |
| `aug_clip` | CE on strongly augmented train images (**the null for CUTPAIR**) |
| `cutpair_aug` | aug_clip + the cut hinge, with tau = logit of the 76th-largest development p3 (the capped_first boundary) |
| `cutpair_aug_shift` | Identical, except the anchor rank is drawn per epoch from {25, 228} (a two-sided shift, seeded) |

**The cut hinge:**
- At each epoch start, an eval-mode pass over the clean training images gives each item's grade-3
  log-odds s.
- Active sets: N_act = {non-grade-3, s > tau - 1} and P_act = {grade 3, tau - 3 < s < tau + 1}.
- Per batch: the mean of relu(s - tau + 1) over active negatives and relu(tau + 1 - s) over active
  positives.
- It is added under the gradient-norm dose rule: the hinge gradient always has 10% of the CE
  gradient's norm (r = 0.1), the same rule as BANDCONS.

## Endpoint and contrasts (fixed)

Primary endpoint: capped_first grade-3 F1 (76 slots) on CLEAN probabilities.

**Primary Holm family (3, paired, two-sided alpha 0.05):**
- **P1** `cutpair_aug` - `cutpair_aug_shift`: attributable. Does anchoring at the cap's cut matter?
- **P2** `cutpair_aug` - `aug_clip`: over its own null.
- **P3** `cutpair_aug` - `clipper`: the thesis bar.

**Secondary:**
- TTA versions of P1-P3;
- each arm minus tralo_null;
- slot swaps against aug_clip, with the correct-direction fraction;
- turnover against aug_clip's reseed floor;
- per-epoch N_act, P_act and active batches.

**MDE:** paired SD ~3-4.5 points gives ~2-2.6 F1 points (~2 of 76 slots) at n = 24 and 80% power.
A null is read as a bound.

## Readings, fixed before the numbers

1. **P1 > 0 and P2 > 0, Holm-significant:** training labels spent at the deployment cut improve
   which patients fill the slots. This is the first attributable training-time win. If P3 > 0 as
   well, it clears the thesis bar.
2. **P2 > 0 with P1 null:** a margin loss helps, but the cap's location adds nothing. That is a
   score result.
3. **All null:** training-label information at the cut, even when alive under augmentation, does not
   convert at this power.
4. **P2 < 0 significant:** the hinge damages the score.

## Pilot gate (seed 2700)

Any failure stops the study and is recorded.
- All five arms complete.
- warm-up, batch and TTA hashes identical.
- 1810 updates in every arm.
- Realised dose ratio 0.1 (to 1e-6) wherever a batch is active.
- Mean N_act >= 20 AND mean P_act >= 20 over epochs 6-10 in `cutpair_aug`. This repeats the gate
  inside the real arm.
- In at least 50% of batches of epochs 6-10, at least one item is active.
- The seed finishes in at most 90 minutes.
