# Claim-to-Table mapping

For every empirical claim made in `paper/main.tex` §5 Results, this maps the
sentence to the CSV that backs it. Useful when (a) updating a number after
new data lands or (b) defending the claim to a reviewer.

All file paths are relative to repo root.

## §5.1 Headline (Table A — `paper/HANDOFF/tables/table_a_summary.csv`)

| Claim sentence (paraphrased)                                | Table / column                                       |
|--------------------------------------------------------------|------------------------------------------------------|
| "TraLO wins or ties Flips in every 1 of 15 cells"           | `table_a_summary.csv` → `flips_mean` column, group by (ds, tight) |
| "Overall mean: TraLO 4, Hounie 20, Danits/Heur 72–74 flips" | `table_a_summary.csv` → `flips_mean` mean across all rows where method=tralo etc. |
| "TraLO satisfies in 100% of seeds (Sat%=1.00)"              | `table_a_summary.csv` → `satisfied_pct` column      |
| "Fioretto LDF Sat%=0.93 overall, TraLO-bounded 0.83"        | same column, aggregated across all cells per method |
| "TissueMNIST: TraLO wins F1 5/5 cells"                      | `table_a_summary.csv` → filter ds=tissue, compare `f1m_mean` |
| "DermMNIST: TraLO wins 3/5, ties 1, loses 1 (L30)"          | same, filter ds=derm                                |
| "AIDER: post-hoc baselines win F1 by 0.003–0.010"           | same, filter ds=aider                               |
| Binomial p=3.1e-5 (Flips 15/15 wins-or-ties)                | computed in `paper/scripts/significance_tests.py`   |
| Binomial p=4.9e-4 (F1 14/15 vs trained baselines)           | same                                                |
| Figure: convergence dynamics                                | `paper/figures/fig_convergence_v2.png` (from `paper/scripts/fig_regen_v2.py`) |

## §5.2 Asymmetric tightness (Table B — `paper/HANDOFF/tables/table_B_phase2_asymmetric_derm.csv`)

| Claim                                                       | Backing                                              |
|--------------------------------------------------------------|------------------------------------------------------|
| "TraLO holds the flip win on every off-diagonal cell"       | `table_B_*.csv` → `flips_mean`, compare TraLO row vs others per `constraint_tag` |
| "F1 picture mirrors the symmetric diagonal"                  | same, `f1m_mean` column                              |
| Figure: asymmetric 4-metric summary                          | `paper/figures/fig_asymmetric_summary_v2.png` (from `paper/scripts/fig_asymmetric_summary.py`) |

**Note:** rows L20_G50 and L50_G20 currently report `n_seeds=1` for the
two post-hoc methods (cosmetic, see G4). Run `agg_g4.py` after the G4
sweep finishes for the repaired table.

## §5.3 Backbone robustness saturated regime (Table C — `paper/HANDOFF/tables/table_C_backbone_saturated.csv`)

| Claim                                                       | Backing                                              |
|--------------------------------------------------------------|------------------------------------------------------|
| "ResNet18 + EfficientNetB0 saturate warmup at ep1"          | training logs in `paper/data_cache/` (warmup acc)    |
| "TraLO keeps the lowest Flips on both saturated backbones"  | `table_C_*.csv` → `flips_mean` column                |
| "Sat%=1.00 for trained methods on both backbones"            | `table_C_*.csv` → `satisfied_pct` column             |
| "F1 advantage does NOT extend to saturated backbones"        | `table_C_*.csv` → `macro_f1_mean`, TraLO vs Fior/Hou |

**Pending G1 supplement:** once `g1_mobilenetv2_summary.csv` lands, the
paragraph "F1 advantage on a non-saturated 2nd backbone remains future
work" can be tightened to a positive claim if MobileNetV2 wins on tissue
+ derm (closing Limitation 3).

## §5.4 Multi-class robustness (Table D — `paper/HANDOFF/tables/table_D_phase4_multiclass_derm.csv`)

| Claim                                                       | Backing                                              |
|--------------------------------------------------------------|------------------------------------------------------|
| "Win holds on AKIEC, BCC, BKL (3 alt classes)"              | `table_D_*.csv` filter on cls column                 |
| "DermMNIST only" — limitation                                | (acknowledged in §6)                                 |

**Pending G3 supplement:** `g3_multiclass_tissue_summary.csv` will let
the paragraph extend to TissueMNIST alt classes (CST, PTC, TUB) — closes
Limitation 2 part 2.

## §5.5 Group-column ablation (Table E — `paper/HANDOFF/tables/table_E_phase5_sexgroup_derm.csv`)

| Claim                                                       | Backing                                              |
|--------------------------------------------------------------|------------------------------------------------------|
| "Win persists on sex group (binary, balanced)"               | `table_E_*.csv`                                      |

No pending supplement — derm `sex` is the only non-`loc_group` real
attribute available; would need a different dataset (out of scope).

## §6 Discussion — Limitations paragraph mapping

The four limitations in `paper/main.tex` §6 map exactly to:

| Limitation # | Paragraph theme                                      | Closes via             |
|--------------|------------------------------------------------------|------------------------|
| 1            | Synthetic groups on tissue + aider                   | (out of scope — needs new dataset) |
| 2            | Asymmetric/multi-class/group-col are derm-only       | G2 (asym) + G3 (multi-class) |
| 3            | Backbone robustness = saturated backbones only       | G1 (MobileNetV2)       |
| 4            | Compute cost — already analyzed                      | (no work; already discussed) |

## §7 Conclusion — last paragraph

> "Extending the asymmetric-tightness, multi-class, and group-column
> robustness analyses…to the other two benchmarks, and confirming the
> F1 advantage on a non-saturated alternative backbone, would close the
> two generality gaps that the present study leaves open."

When G1+G2+G3 land, this sentence becomes obsolete — the gaps are closed,
not future work. Update the conclusion accordingly.
