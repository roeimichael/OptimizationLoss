# A — Headline sweep

The central result of the paper: the symmetric-tightness sweep across all three
active datasets.

## Backs in `main.tex`
- **Table `tab:headline_f1_flips`** (A1) — Macro F1 + post-hoc flips per cell.
- **Table `tab:headline_sat_ece`** (A2) — in-training Sat% + ECE per cell.
- **Figures** `fig_convergence_v2`, `fig_f1_tightness_v2`, `fig_flips_tightness_v2`,
  `fig_satisfaction_v2` (all read `docs/table_a_summary.csv`, which is a copy of
  `table_a_summary.csv` here).

## Configuration
- **Backbone:** MobileNetV3 · **Datasets:** TissueMNIST, DermMNIST, AIDER
- **Tightness:** 5 symmetric regimes L=G ∈ {20, 30, 50, 70, 80}
- **Methods:** TraLO, TraLO-bounded, Fioretto LDF, Hounie RCL, Danits LP, Heuristic
- **Seeds:** 4 per cell → 15 (dataset, tightness) cells × 6 methods.

## Headline result (overall mean across the 15 cells)
| Method | Macro F1 ↑ | Flips ↓ | Sat% ↑ | ECE ↓ |
|---|---|---|---|---|
| **TraLO (ours)** | 0.669 | **4** | **1.00** | 0.185 |
| TraLO-bounded | 0.667 | 11 | 0.83 | 0.187 |
| Fioretto LDF | 0.667 | 11 | 0.93 | 0.187 |
| Hounie RCL | 0.661 | 20 | 1.00 | 0.195 |
| Danits LP | 0.659 | 72 | 0.07 | 0.174 |
| Heuristic | 0.660 | 74 | 0.07 | 0.174 |

F1 is a **tie** (all within 0.01); TraLO separates on post-hoc flips
(order of magnitude) and in-training satisfaction. Post-hoc baselines win ECE.

## Files
- `table_a_summary.csv` — per-(ds, tight, method) means/stds for every metric
  (F1m, Acc, ECE, Brier, Flips, Sat%, SatEp, Time). The figures' data source.
- `table_a_per_seed.csv` — per-seed rows (used for the **paired** significance tests).
- `scoreboard.csv` / `win_matrix.csv` — TraLO win/tie/loss vs the 5 baselines,
  per dataset × metric, paired bootstrap over matched seeds.
- `stats_headline_f1.md`, `stats_flips_dominance.md`, `stats_scoreboard.md` —
  paired-bootstrap write-ups behind the scoreboard.

## Provenance
Aggregated from the full per-seed census (`paper/HANDOFF/tables/all_cells_raw.csv`)
by the aggregators in `paper/HANDOFF/aggregators/`. Scoreboard/stats built from
the same census filtered to the canonical TraLO recipe. Honest by construction:
the AIDER F1 concession and all ties are included, not filtered out.
