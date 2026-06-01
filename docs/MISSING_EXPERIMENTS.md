# Missing Experiments — run-list to complete the paper

**Created:** 2026-05-31 (during the harsh-review pass on `paper/main.tex`)
**Purpose:** the paper as currently written is *fully backed by existing data* (Tables A–E + the 4 figures all map onto CSVs in `docs/` and `paper/tables/`). The experiments below are **not yet run** and would either (a) close a limitation the paper currently states as future work, or (b) strengthen a generality claim. Priority order is by paper impact.

## What the paper already presents (data exists — do NOT re-run)

| Table / Fig | Scope | Backing data | Cells |
|---|---|---|---|
| Table A (headline) | MobileNetV3 × {tissue,derm,aider} × 5 sym tight × 6 meth × 4 seed | `docs/table_a_summary.csv` | 360 ✅ |
| Table B (asymmetric) | MobileNetV3 / derm / MEL / loc_group, 20 off-diag (L,G) | `paper/tables/B_asymmetric_tightness/table_B_phase2_asymmetric_derm.csv` | 480 ✅ (see gap G4) |
| Table C (backbone, saturated) | {ResNet18, EfficientNetB0} × {tissue,derm} × 5 sym × 6 meth × 4 seed | `docs/all_cells_raw.csv` → `paper/tables/C_backbone_robustness/table_C_backbone_saturated.csv` | 480 ✅ |
| Table D (multi-class) | MobileNetV3 / derm / {AKIEC,BCC,BKL} | `paper/tables/D_multiclass_derm/table_D_phase4_multiclass_derm.csv` | 360 ✅ |
| Table E (group-column) | MobileNetV3 / derm / sex | `paper/tables/E_group_column_sex/table_E_phase5_sexgroup_derm.csv` | 120 ✅ |
| Figs 1–4 | convergence, F1-vs-tightness, satisfaction, asym heatmap | regenerated 2026-05-31 from above | ✅ |

## Gaps — experiments to run (priority order)

### G1 — MobileNetV2 non-saturated backbone (HIGHEST priority)
**Why:** the paper's third stated limitation is "confirming the F1 advantage on a second *non-saturated* backbone remains future work." ResNet18/EfficientNetB0 (Table C) both saturate at warmup ep1, so they can only corroborate the *deployability* (flips/Sat%) win, not the *F1* win. MobileNetV2 (~3.5M params) is the natural non-saturated co-backbone (CLAUDE.md lists it as a Blackwell-validated co-winner, but **no MobileNetV2 cells exist anywhere in the repo**).
**Run:** MobileNetV2 on the headline setup — `{tissue,derm,aider} × 5 sym tight × 6 meth × 4 seed = 360`. Minimum viable: `{tissue,derm}` (the two non-saturated datasets) `= 240`.
**Closes:** Limitation 3; lets the backbone section claim an F1 win on a 2nd backbone, not just a deployability win.

### G2 — Asymmetric tightness on TissueMNIST + AIDER
**Why:** Table B (asymmetric) is DermMNIST-only; Limitation 2 flags that we don't know if the off-diagonal picture holds on the other two datasets.
**Run:** `{tissue,aider} × 20 off-diag (L,G) × 6 meth × 4 seed = 960` (mirror `gen_paperv2_phase2`/`gen_aider_asymmetric.py`).
**Closes:** the asymmetric half of Limitation 2.

### G3 — Multi-class + group-column on a 2nd dataset
**Why:** Tables D and E are DermMNIST-only; Limitation 2 also covers constrained-class and group-column generality.
**Run:** multi-class on tissue (alt constrained classes) and/or aider; group-column has no natural 2nd real attribute outside derm, so this is lower value.
**Closes:** the multi-class half of Limitation 2.

### G4 — Asymmetric data completeness (2 under-seeded cells)
**Why:** in `table_B_phase2_asymmetric_derm.csv`, the post-hoc baselines at `L20_G50` (and check `L50_G20`) have `n_seeds=1` instead of 4 (heuristic/danits_lp). Trained methods are full 4-seed; the post-hoc rows are LP/greedy so deterministic, but the table reports `±std` over seeds.
**Run:** re-dispatch the 3 missing seeds for the affected post-hoc cells (cheap, <1 min each).
**Closes:** a cosmetic std/`n` inconsistency in Table B; not a headline risk.

### G5 — RegNetY400MF + ShuffleNetV2 corroboration (OPTIONAL)
**Why:** CLAUDE.md / PAPER_PLAN list these as corroboration backbones, but **no data exists**. Not referenced in the paper, so optional. Only run if a reviewer asks for breadth beyond MobileNetV2.
**Run:** `{RegNetY400MF, ShuffleNetV2} × {derm,aider} × 5 sym × 6 meth × 4 seed`.

## Notes for whoever runs these
- Use the `gen_paperv2_*` generator pattern and `EXPERIMENT_DIR` pinning from `docs/PAPER_PLAN.md` §7.
- All runs on dsisco02 Blackwell; `conda activate optloss`; never share a GPU (see `feedback_gpu_sharing`).
- After a slice completes, re-run the table aggregator and re-render figures with `paper/scripts/fig_regen_v2.py` and `fig_asymmetric_summary.py`.
- G1 (MobileNetV2) is the only gap that changes a *paper claim*; G2–G5 broaden generality and can be reported as additional robustness blocks or left as stated future work.
