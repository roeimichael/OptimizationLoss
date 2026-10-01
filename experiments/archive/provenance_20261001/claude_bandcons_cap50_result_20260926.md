# BANDCONS cap-50 study: result -- 2026-09-26

**Flag: run under deviation D-A1.** The natural-count gate item failed in the pilot. Per-seed natural counts are in `analysis/bandcons_a1cap50.txt`.

- **Protocol:** `claude_bandcons_protocol_20260926.md` plus amendment 1 (dose rule, D-A1).
- **Run:** release c5634488, seeds 2401-2424, cap 50, dsisco01.
- **Scorer:** `analysis/score_bandcons.py`. Raw output: `analysis/bandcons_a1cap50.txt` / `.json`.

**Integrity: all 24 seeds pass.** Hashes matched, 1810 updates, band sizes 2w, and dose ratio 0.1 to 1e-6 in every bandcons arm. The maximum band disagreement is 5.7 log-odds, within amendment 1's bound of 10.

## Primary (clean capped_first grade-3 F1, paired over 24 seeds, Holm over B1-B4)

| | contrast | mean [95% CI] | Holm p | W/T/L |
|---|---|---|---|---|
| B1 | bandcons - bandcons_rand | -0.53 [-2.59, +1.52] | 1.00 | 10/3/11 |
| B2 | bandcons - bandcons_unc | -0.69 [-3.44, +2.05] | 1.00 | 10/0/14 |
| B3 | bandcons - tralo_null | **-3.47 [-5.38, -1.56]** | **0.004** | 3/3/18 |
| B4 | bandcons - aug_clip | **-2.88 [-5.03, -0.74]** | **0.032** | 7/2/15 |

Arm means (F1): clipper 55.72, tralo_null 57.21, aug_clip 56.62, bandcons 53.74, bandcons_unc 54.43, bandcons_rand 54.27.

## Reading (fixed before the numbers): 4. The consistency term damages the score.

Placement carries nothing (B1, B2 null), and the term itself costs about 3 F1 points. Every variant loses to the null:
- the random band, -2.94 [-5.09, -0.78];
- the uncertainty band, -2.78 [-5.32, -0.24].

The loss therefore comes from the term, not from where it is placed.

## Supporting (secondary, not tested for the reading)

- **On TTA probabilities:** B4 is -3.47 [-5.26, -1.68]. bandcons_tta - clipper_tta is -1.92 [-3.65, -0.20].
- **Swaps vs tralo_null:** bandcons moves 41.5% of its swaps in the correct direction (net -2.71 slots). The comparable figures are clipper 45.9% and aug_clip 48.6%.
- **Turnover:** every arm sits at the reseed floor (0.86-1.00x).
- **Why:** under bandcons the natural grade-3 count swings between 0 and 792 across epochs, against 51-148 for the null. The term shakes the whole grade-3 logit. Near the cut that noise mostly moves correct items out, as the depth probe and LEDGER #10 predict for a push that carries nothing beyond p3.

## Consequence

- **Augmentation stability is closed as a training-time information source on this cell.** It adds no who-information over a random band, and it harms.
- **The cap-76 replication block (2501-2524) is not run.** Its only purpose was to replicate a positive result. A significant harm at cap 50 with null placement contrasts leaves nothing to replicate, and its GPU time is better spent elsewhere.
