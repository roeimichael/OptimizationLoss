# Paper handoff package

**Built:** 2026-06-01 by the experiment-orchestration session.
**Audience:** the next session, focused on paper writing.
**Source of truth:** `docs/PAPER_PLAN.md`, `docs/MISSING_EXPERIMENTS.md`, `paper/main.tex`.

This folder contains everything needed to write the empirical sections of
`paper/main.tex` without touching the LaTeX. It is structured as:

```
HANDOFF/
├── README.md                        ← you are here
├── tables/                          ← every CSV cited by §5 Results
├── launch_commands/                 ← copy-paste shell to start each pending sweep
├── aggregators/                     ← post-sweep aggregation scripts (per run)
└── status.md                        ← live tracker: what's done, what's running, what's left
```

## What the paper already presents (data EXISTS — no work required)

| § / Table     | Scope                                                                | Backing CSV                                | Cells |
|---------------|----------------------------------------------------------------------|--------------------------------------------|------:|
| §5.1 Table A  | MobileNetV3 × {tissue, derm, aider} × 5 sym × 6 meth × 4 seeds       | `tables/table_a_summary.csv` + `_raw.csv`  | 360 ✅ |
| §5.2 Table B  | MobileNetV3 / derm / MEL / loc_group × 20 off-diag (L,G)             | `tables/table_B_phase2_asymmetric_derm.csv`| 480 ✅ |
| §5.3 Table C  | {ResNet18, EfficientNetB0} × {tissue, derm} × 5 sym × 6 meth × 4 sd  | `tables/table_C_backbone_saturated.csv`    | 480 ✅ |
| §5.4 Table D  | MobileNetV3 / derm / {AKIEC, BCC, BKL}                               | `tables/table_D_phase4_multiclass_derm.csv`| 360 ✅ |
| §5.5 Table E  | MobileNetV3 / derm / sex                                             | `tables/table_E_phase5_sexgroup_derm.csv`  | 120 ✅ |
| Figs 1–4      | convergence, F1-vs-tightness, satisfaction, asymmetric heatmap       | `paper/figures/fig_*_v2.png`               | ✅    |

The flat `tables/all_cells_raw.csv` is the union of every cell ever run
that lives anywhere under `results/pending_runs/` — useful for ad-hoc
slicing.

## Gaps the paper currently flags as future work (data being collected)

These map 1-to-1 onto the four "Limitations" paragraphs in §6 Discussion of
`paper/main.tex`:

| Gap | Closes which limitation | Status | Target cells |
|-----|------------------------|--------|-------------:|
| G1 — MobileNetV2 non-saturated 2nd backbone | Limit 3 (F1 corroboration on a 2nd backbone) | Queued, awaiting Blackwell | 240 |
| G2 — Asymmetric (L,G) on tissue + aider     | Limit 2 part 1 (asymmetric off-derm)         | Queued, awaiting Blackwell | 960 |
| G3 — Multi-class robustness on tissue       | Limit 2 part 2 (multi-class off-derm)        | Queued, awaiting Blackwell | 360 |
| G4 — Cosmetic Table B post-hoc seed fill    | Footnote-level only                          | Queued, ~2 min            | 12  |

The text in §6 already accommodates all four as future work. Writing can
proceed immediately on the current 1,800 cells of data; G1–G4 land as
optional additions that strengthen the robustness claims when they finish.

## Where the new data will land

All four new sweeps drop their `evaluation_metrics.csv` files under
shared NFS so they appear on both servers simultaneously:

```
results/pending_runs/g1_mobilenetv2/{tissue,derm}/<tight>/<method>/seed_*/
results/pending_runs/g2_asym_tissue_aider/{tissue,aider}/<tight>/<method>/seed_*/
results/pending_runs/g3_multiclass_tissue/cls_<c>/<tight>/<method>/seed_*/
results/pending_runs/g4_table_b_backfill/<tight>/<method>/seed_*/
```

The aggregator scripts in `aggregators/` collapse those into a single
`paper/tables/table_<X>_<name>.csv` keyed identically to the existing
tables.

## Background context — read once before editing main.tex

1. **`paper/main.tex` is the only paper artifact you should touch.** All
   tables in `tables/` are derived; never edit them by hand.
2. **The headline framing is two-claim, not one** (see §5.1, §5.3, §6):
   - Deployability win (post-hoc flips + Sat%): universal across every
     dataset, every backbone, every cell.
   - F1 win: regime-dependent — present in the headroom regime
     (TissueMNIST + DermMNIST tight cells on MobileNetV3), absent in
     saturated regimes (AIDER + ResNet18/EfficientNetB0).
3. **Do NOT add a 3rd backbone family** (RegNetY400MF / ShuffleNetV2)
   unless G1 lands AND a reviewer asks for it. The Turing-side screening
   sweep includes those for safety but they are not in the paper.
4. **Blackwell-only policy.** Per `docs/PAPER_PLAN.md` every cell cited
   in the paper must be run on dsisco02 (RTX PRO 6000 Blackwell, 96 GB).
   Turing runs on dsisco01 are for screening only and never enter a
   paper table.

## Where to go next

1. Open `status.md` to see live progress on the four pending sweeps.
2. Open `launch_commands/` to find the exact shell to start each one
   (already-written, copy-paste, dsisco02 GPU-paired).
3. Open `aggregators/` for the post-sweep collation scripts.
4. Anything missing or unclear → `paper/CORE_DRAFT.md` is the
   higher-level narrative draft and `docs/PAPER_PLAN.md` is the run plan.
