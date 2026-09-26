# BANDCONS: view-consistency on the unlabeled development items at the cut -- PREREGISTERED 2026-09-26

**Written before any seed of this study ran.** Author: Claude, branch `claude/bandcons-20260926`.

Disclosure:
- The author has seen the cap-76 and cap-50 targeted-step results (LEDGER #9) and the CUTPAIR
  gate (dead: training-label information at the cut is exhausted by memorisation).
- The band half-width w = round(0.25 x cap) is a generic prior. The reshape research motivated it
  with a reachable-prize figure (swaps within ~20 ranks of the cut), which was computed from
  development labels of EARLIER seeds.
- Seeds 2000-2124 are fresh.

## Why

Every count loss on development probabilities reproduces the allocator (LEDGER #9). Training
labels carry no information at the development cut (CUTPAIR gate). The remaining information
the allocator lacks is in the unlabeled development IMAGES themselves: whether the grade-3
evidence of an item near the cut is stable under the augmentations a radiologist would ignore.
Clean probabilities do not determine that disagreement, so M1 does not bind.

## Design (fixed)

Unchanged from the targeted-step studies:
- Chen OAI knee, 5778 train / 826 development; test split never scored.
- ImageNet ResNet18, trainable, FP32.
- 5 CE warm-up epochs + 5 epochs, batch 32, task Adam 1e-4.
- dsisco hosts, OMP_NUM_THREADS=8.

Runner: `tralo/knee_e2e_v4.py`. Six arms share init, warm-up and batch order (hash-checked):

| arm | epochs 6-10 |
|---|---|
| `clipper` | CE, task Adam kept |
| `tralo_null` | CE, task Adam reset |
| `aug_clip` | CE on strongly augmented TRAIN images (Adam reset) |
| `bandcons` | CE + 1.0 x SmoothL1(s(strong), stopgrad s(weak)) on 8 band items per labeled batch. The band is the development items at p3 ranks cap-w+1 .. cap+w, from start-of-epoch probabilities. |
| `bandcons_unc` | Same, with the band centred on the natural argmax grade-3 count (cap-free control). |
| `bandcons_rand` | Same, on 2w random development items outside the cap band (placement control). |

Fixed parameters:
- s is the grade-3 log-odds.
- w = 12 at cap 50 and 19 at cap 76.
- The view forward runs in eval mode, so BatchNorm statistics are untouched.
- Weak view: flip. Strong view: flip + resized crop (0.8-1.0) + rotation of +/-10 degrees +
  brightness and contrast of 0.8-1.2.

Every arm is scored by capped_first twice: on clean final probabilities, and on
test-time-augmentation (TTA) probabilities (8 fixed strong draws, identical across arms).

Blocks, each analysed as its own paired study:
- **cap 50: seeds 2001-2024 (PRIMARY)**, on dsisco01;
- cap 76: seeds 2101-2124 (replication).

The cap-76 block may run on dsisco02 if dsisco01 is full. Host is then confounded with the
block, not with any arm contrast.

## Endpoint and contrasts (fixed)

Primary endpoint: capped_first grade-3 F1 on CLEAN probabilities (points).

**Primary Holm family per block (4 contrasts, paired, two-sided alpha 0.05):**
- **B1** `bandcons` - `bandcons_rand`: does placing the consistency at the cut matter?
- **B2** `bandcons` - `bandcons_unc`: does the CAP's location matter, beyond the model's own boundary?
- **B3** `bandcons` - `tralo_null`
- **B4** `bandcons` - `aug_clip`: the post-hoc clipper on an augmentation-trained model, which is the bar.

An attributable win for the cap requires B1 > 0 AND B2 > 0, both Holm-significant, in the primary
block. If B3 or B4 is positive while B1 or B2 is null, the reading is "transductive consistency
helps; the cap adds nothing". That is a score result, reported as such, and it is not a TraLO win.

**Secondary (t intervals):**
- the same four contrasts on TTA probabilities;
- bandcons_tta - clipper_tta (the post-hoc use of the same augmentations);
- offline swap analysis with development labels: of the band items that change side of the cut
  relative to tralo_null, the fraction that moves in the correct direction;
- slot turnover against the null's reseed floor;
- per-arm GPU time.

**MDE, stated now:** the cap-50 paired SD in the targeted-step study was 3.2-4.5 points, giving
~2-2.6 F1 points (~1.5-2 of 50 slots) at n = 24 and 80% power. A null is read as a bound, not as
an absence.

## Readings, fixed before the numbers

1. **B1 and B2 positive and significant:** the unlabeled images near the cut carry WHO information
   that a training-time objective can use and the allocator cannot. This is the first attributable
   positive result.
2. **B3 or B4 positive, with B1 or B2 null:** a score improvement, not a cap effect.
3. **All null:** this information source is also exhausted at this power. Combined with LEDGER #9
   and the CUTPAIR gate, no information source available to a training-time constraint on this cell
   beats the post-hoc clipper.
4. **B3 or B4 significantly negative:** the consistency term damages the score.

## Pilot gate (seed 2000, non-study)

Any failure stops the study and is recorded.
- All six arms complete.
- warm-up, batch and TTA-draw hashes identical.
- task updates equal in every arm.
- bandcons band size = 2w at every post-warm-up epoch.
- Mean |s_strong - s_weak| on the cap band at epoch 6 >= 0.05 log-odds (else there is nothing to
  propagate).
- The seed finishes in <= 75 minutes.
