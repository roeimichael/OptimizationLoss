# paper/tables — finalized results showcased in the paper

This folder is the **canonical, documented home of every result that appears in
`paper/main.tex`**. One subdirectory per showcased experiment; each holds the
actual result CSV(s) plus a `README.md` describing what it is, the experimental
configuration, and which table/figure in the paper it backs.

All headline experiments are **MobileNetV3, 4 seeds/cell, 6 methods**
(TraLO, TraLO-bounded, Fioretto LDF, Hounie RCL, Danits LP, Heuristic).

| Subdir | Experiment | Backs in `main.tex` | One-line result |
|---|---|---|---|
| `A_headline/` | Headline sweep — 3 datasets × 5 symmetric tightness | Tables `tab:headline_f1_flips`, `tab:headline_sat_ece`; Figs convergence / f1_tightness / flips_tightness / satisfaction | F1 a tie; TraLO wins flips (4 vs 11–74) and in-training Sat% (100%) |
| `B_asymmetric_tightness/` | Off-diagonal L≠G sweep on DermMNIST (20 cells) | Table `tab:asym_summary`; Fig `fig_asymmetric_summary_v2` | Quality tied (F1 0.733); TraLO wins deployability (1.9 vs 5.7–129 flips) |
| `C_backbone_robustness/` | ResNet18 + EfficientNetB0 (saturated warmup), tissue+derm | Table `tab:backbone_summary` | F1 tie by construction; TraLO still separates on flips/Sat% |
| `D_multiclass_derm/` | Constrained class ∈ {AKIEC, BCC, BKL} on DermMNIST (15 cells) | Table `tab:multiclass_summary` | Flips win 5/5 cells per class; F1 gap closes as class gets more prevalent |
| `E_group_column_sex/` | Group column = `sex` instead of `loc_group` (5 cells) | Table `tab:sexgroup_summary` | TraLO flips 3.1 vs 8–92; win robust to group axis |
| `F_component_ablation/` | Leave-one-out over TraLO components | Table `tab:component_ablation` | **PENDING** (G5 running) — skeleton only |

## Conventions
- CSV granularity is **per-cell** (each row = one `constraint_tag × method`, mean
  over 4 seeds). The paper tables report the **overall mean across cells**; the
  per-cell rows here are the source those means aggregate from.
- Provenance / regeneration for each table is in its subdir README.
- The build-side working caches (`docs/table_a_*.csv`, `paper/HANDOFF/`) are
  regenerable and **not** canonical — this folder is.
