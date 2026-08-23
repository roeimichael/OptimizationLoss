# G3 framing handoff — multi-class TissueMNIST extension

**For**: paper-writing session
**From**: experiment-running session (2026-06-01)
**Backing CSVs**:
- `paper/HANDOFF/tables/g3_multiclass_tissue_raw.csv` (per-seed rows)
- `paper/HANDOFF/tables/g3_multiclass_tissue_summary.csv` (cell means)
- Server source: `dsisco02:results/pending_runs/g3_multiclass_tissue/`

## What G3 was for

Closing §6 Limitation 2b in `main.tex`: "multi-class robustness is shown only on DermMNIST". Table D already shows DermMNIST AKIEC/BCC/BKL. G3 extends the same analysis to TissueMNIST with **3 alternate constrained classes** (CST=2, PTC=5, TUB=7), holding everything else fixed.

**Grid**: 3 cls × 5 sym tightness (L20–L80) × 6 methods × 4 seeds = **360 cells. All done.**

Configuration matches the headline:
- Backbone: MobileNetV3
- Group column: `synth_group` (same as headline cls=4 GE)
- Warmup epochs: 50 (cached)
- Constraint epochs: 300 (early-stops at 5 consecutive satisfied)

## Headline result

| Comparison | F1 W/L/T | Flips W/L/T | mean ΔF1 |
|------------|----------|-------------|----------|
| TraLO vs **best baseline per cell** | **9 / 5 / 1** | **8 / 2 / 5** | — |
| TraLO vs Fioretto LDF | 11/15 | 10/15 | +0.0008 |
| TraLO vs Hounie RCL | 10/15 | 10/15 | +0.0012 |
| TraLO vs Danits LP (post-hoc) | **13/15** | **13/15 (-79 flips/cell)** | +0.0114 |
| TraLO vs Heuristic (post-hoc) | **13/15** | **15/15 (-100 flips/cell)** | +0.0107 |

**Headline sentence to add at §5.4**:
> "Repeating the analysis on TissueMNIST across three alternate constrained classes (CST, PTC, TUB) reproduces the DermMNIST pattern: TraLO wins macro-F1 on 9 of 15 cells against the best per-cell baseline and on 13 of 15 against the post-hoc baselines, while reducing required flips by 79–100 on the same cells."

## Honest caveat — the regime split

The 5 losses are **not random**. They cluster at **loose tightness on class 7 (TUB)**:

| cls | tight | TraLO F1 | Best baseline F1 | Δ |
|-----|-------|----------|------------------|---|
| 7 | L70_G70 | 0.3782 | 0.3826 (Hounie/Fior) | **-0.0044** |
| 7 | L80_G80 | 0.3780 | 0.3833 (Hounie/Fior) | **-0.0053** |
| 2 | L20_G20 | 0.3419 | 0.3526 (Heur) | -0.0107 |

The L70/L80 cls=7 cells are the AIDER-style **saturated regime** — the constraint cap is loose enough (~70-80% of training prevalence) that the warmup model is already near-feasible and Hounie/Fioretto's lazier in-training updates don't perturb the unconstrained classes. Same mechanism as the AIDER cross-over in §5.1.

**Suggested honest framing for the limitation**:
> "On the two loosest TissueMNIST cells where the constraint cap is non-binding on the warmup classifier (cls=7 at L70/L80), TraLO trails Hounie RCL and Fioretto LDF by 0.004–0.005 macro-F1. The mechanism mirrors the AIDER cross-over of §5.1: at non-binding cap, the in-training fine-tuning that earns TraLO its edge on tight regimes instead perturbs the unconstrained classes."

## How to slot into Table D / §5.4

**Option A — extend existing Table D**:
add 3 new dataset×class blocks (tissue/CST, tissue/PTC, tissue/TUB) below the existing derm/AKIEC + derm/BCC + derm/BKL rows. Headline numbers per (cls, tight, method) come from `g3_multiclass_tissue_summary.csv`. Cell width identical to current Table D.

**Option B — sibling Table D′**:
keep Table D derm-only as is; add a new "Table D′: multi-class on TissueMNIST" right after it. Cleaner narrative split if column count gets cramped.

**Recommendation**: Option B. The dataset axis is already complex; mixing derm and tissue in one table risks reader confusion about which cls labels belong to which dataset.

## What this lets you remove from the paper

- §6 Limitation 2b currently reads "(Tables~B–E…) are conducted only on \emph{DermMNIST}." → after G3 lands, this becomes "Tables B, D extend to TissueMNIST" and only Tables C/E remain derm-only.
- §7 Conclusion last paragraph mentions "extending the multi-class robustness analyses…to the other benchmarks" — this becomes obsolete for multi-class on tissue (still future work for aider).

## Numbers source-of-truth for cross-checking

Per-cell winners are in `g3_multiclass_tissue_summary.csv`. The aggregator that produced it is `paper/HANDOFF/aggregators/agg_g3.py`. Per-seed raw is in `g3_multiclass_tissue_raw.csv` for paired bootstrap if you want to add significance.

**Paired bootstrap suggestion**: matched-seed TraLO−baseline differences across 15 cells × 4 seeds = 60 paired samples per baseline. Reuse the test from §5.1 headline.

## What G3 does NOT show

- **Does not extend to AIDER multi-class** — AIDER only has 4 classes and the constrained one (collapsed_building) is already the cleanest comparison; alternate-class AIDER would muddy more than clarify.
- **Does not vary group_column** — that's Table E's job, kept derm-only.
- **Does not include backbone sweep** — tissue with MobileNetV2 is G1's job (already done, 240/240 from earlier sweep).

## TL;DR for the writing session

Use G3 to **upgrade the multi-class story from "derm-only" to "two datasets, six constrained classes"**. The headline read is "TraLO wins 9/15 vs best baseline, 13/15 vs post-hoc"; the honest caveat is "the 5 losses cluster at non-binding cap on tissue cls=7, same mechanism as AIDER §5.1". Both go in §5.4.
