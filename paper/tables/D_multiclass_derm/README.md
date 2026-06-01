# D — Multi-class robustness (choice of constrained class)

Checks that the win does not depend on MEL being the constrained class, by
re-running the headline setup with three other DermMNIST classes spanning a range
of test prevalences.

## Backs in `main.tex`
- **Table `tab:multiclass_summary`** (overall mean across the 15 cells).

## Configuration
- **Dataset/backbone:** DermMNIST / MobileNetV3 · **Group:** `loc_group`
- **Constrained class:** AKIEC (`0`, 3.2%), BCC (`1`, 5.1%), BKL (`2`, 11.0%)
  — the two rarest classes (DF 1.1%, VASC 1.4%) are skipped: their absolute cap K
  collapses below the resolution of a 2,005-image test set.
- **Tightness:** 5 symmetric cells per class → 15 (class, tightness) cells
- **Methods:** all 6 · **Seeds:** 4.

## Headline result
- **Flips win holds in 5/5 tightness cells per class** (TraLO overall ≈ 2.0 flips
  vs 7–66 for the post-hoc baselines).
- The F1 gap to the post-hoc baselines **closes as the class becomes more
  prevalent**, reversing into a TraLO win on the most prevalent of the three:
  TraLO's per-class mean F1 vs the heuristic is −0.005 (AKIEC), −0.008 (BCC),
  **+0.006 (BKL)**. This is the graded form of the AIDER saturated-regime crossover.

## Files
- `table_D_phase4_multiclass_derm.csv` — 90 rows = 3 `constrained_class` ×
  5 `constraint_tag` × 6 methods, 4 seeds each. The `constrained_class` column is
  the DermMNIST class index: 0 = AKIEC, 1 = BCC, 2 = BKL.

## Provenance
Phase-4 multi-class sweep; aggregated per (constrained_class, constraint_tag,
method) over 4 seeds.

> Note: a separate **TissueMNIST** multi-class sweep (G3) is in progress; its raw
> output lives in `paper/HANDOFF/tables/g3_multiclass_tissue_*.csv` and is not yet
> a paper table.
