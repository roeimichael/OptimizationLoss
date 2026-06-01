# B — Asymmetric-tightness sweep (L ≠ G)

Tests whether the TraLO story survives when the global and local budgets are set
to **different** tightness levels (the headline uses L = G).

## Backs in `main.tex`
- **Table `tab:asym_summary`** — overall mean across the 20 off-diagonal cells.
- **Figure `fig_asymmetric_summary_v2`** — the 2×2 four-metric panel (quality row
  tied, deployability row won).

## Configuration
- **Dataset/backbone:** DermMNIST / MobileNetV3 · **Constrained class:** MEL ·
  **Group:** `loc_group` (3 anatomical sites)
- **Tightness:** 20 off-diagonal (L, G) configurations with L ≠ G (the 5 symmetric
  cells belong to experiment A) · **Methods:** all 6 · **Seeds:** 4.

## Headline result (overall mean across the 20 cells)
| Method | Macro F1 ↑ | Flips ↓ | Sat% ↑ |
|---|---|---|---|
| **TraLO** | 0.733 | **1.9** | **100%** |
| TraLO-bounded | 0.733 | 5.7 | <100% |
| Fioretto LDF | 0.733 | 5.7 | <100% |
| Hounie RCL | 0.731 | 8.8 | 100% |
| Danits LP / Heuristic | 0.721 | 129 | 0% |

A **no-free-lunch** result: quality (F1, accuracy ≈ 0.845) is statistically
indistinguishable across all methods (below the σ ≈ 0.011 seed-noise floor),
while TraLO buys a large flip reduction and full in-training feasibility at no
measurable accuracy cost.

## Files
- `table_B_phase2_asymmetric_derm.csv` — 120 rows = 20 `constraint_tag` × 6 methods,
  4 seeds each. Columns: `macro_f1_mean/std, flips_mean/std, accuracy_mean/std,
  satisfied_pct`.

## Provenance
Phase-2 asymmetric sweep, aggregated per (constraint_tag, method) over 4 seeds.
