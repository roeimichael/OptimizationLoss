# BANDCONS amendment 1: gradient-norm dose rule -- PREREGISTERED 2026-09-26, before any study seed

Amends `claude_bandcons_protocol_20260926.md`. No seed of blocks 2001-2024 or 2101-2124 ever ran;
those blocks are retired unused.

## What the pilot showed (seed 2000, release f72b53bc, 26 min)

- Every preregistered gate item passed.
- Label-free diagnostics showed the fixed beta = 1.0 lets the consistency term dominate
  cross-entropy and destabilise training:

| arm | band disagreement at epoch start, epochs 6-10 (log-odds) | natural argmax grade-3 count |
|---|---|---|
| bandcons | 2.19, 1.36, 0.71, 1.27, 1.47 | (band centre fixed at cap) |
| bandcons_unc | 2.72, 0.78, 1.73, 0.76, 0.75 | 104, **805**, 105, 19, 75 |
| bandcons_rand | 2.55, 2.01, **91.1**, 1.33, 0.74 | -- |

At that point training CE is ~0.1, so a SmoothL1 term of order 1-2 per item at beta = 1 dominates
the supervised signal by roughly 10-20x. This is the Phase 3 overshoot again, in a new loss.

**Disclosure.** The author also saw the pilot's capped grade-3 F1 (development labels, n = 1):

| arm | F1 |
|---|---|
| clipper | 56.4 |
| null | 57.7 |
| aug_clip | 57.7 |
| bandcons | 44.9 |
| bandcons_unc | 34.6 |
| bandcons_rand | 48.7 |

The amendment below is chosen from the LABEL-FREE diagnostics above and the critic's pre-pilot
recommendation (research workflow wf_e8af9e08-2de, BANDCONS critique fix 7). It is not tuned on F1:
there is one fixed ratio, chosen a priori, with no search.

## The amendment (fixed)

1. **Dose rule for bandcons, bandcons_unc and bandcons_rand.** On every labeled batch the update
   direction is `g = g_CE + r * (||g_CE|| / ||g_cons||) * g_cons`, with global L2 norms and
   **r = 0.1**. The consistency term always contributes exactly 10% of the supervised gradient's
   magnitude, in every arm, at every step. Dose is therefore identical across the three bandcons arms,
   so B1 and B2 compare placement only. There is one optimizer step per batch, as before.
2. **Arms, endpoint, Holm family B1-B4, readings, TTA and swap analysis:** unchanged.
3. **Fresh seeds:**
   - pilot **2400**;
   - cap-50 block **2401-2424 (PRIMARY)**;
   - cap-76 block **2501-2524** (replication).
4. **Pilot gate:** all items of the original gate, plus two label-free kill bounds.
   - The realised dose ratio equals 0.1 to 1e-6 in every bandcons arm.
   - No arm's natural argmax grade-3 count leaves [cap/3, 400] at any post-warm-up epoch start, and
     no band disagreement exceeds 10 log-odds. The same bounds applied to the null's own trajectory
     define normal. If the NULL itself leaves them, the bound is uninformative and is reported, not
     applied.

Any failure stops the study and is recorded.
