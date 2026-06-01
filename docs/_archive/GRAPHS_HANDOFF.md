# Graphs Handoff — for the paper-writing session

**Created:** 2026-05-25
**From:** experiments session
**To:** paper-writing session (the one working through `docs/PAPER_WRITING_PLAN.md`)

This is a task brief for producing the results figures from Phases 1–5. Treat it like Block 8 of the paper plan (figure regeneration), but with the specific figures and the honest framing below.

> **Session boundary reminder:** you own `paper/`. Read data from `docs/`. Do not modify `results/`, `src/`, `main.py`, or `docs/PAPER_PLAN.md`. You MAY add scripts under `paper/scripts/` and figures under `paper/figures/`.

---

## 0. The honest framing (read first — it drives every figure)

After rigorous, seed-noise-aware analysis across all completed phases, the result is **not** "TraLO wins F1." It is:

- **F1-macro: TraLO is statistically TIED with the best baseline on essentially every cell, every dataset.** Gaps are within seed std (~0.006–0.011). Not clearly ahead (even tissue is +0.0065 ± 0.011), not clearly behind (derm: 3 real losses / 55 cells).
- **Flips (post-hoc corrections needed): TraLO wins decisively everywhere** — saves 8.6 (tissue) → 17 (derm) → 53 (aider) flips vs the best baseline. Tradeoff ≈ 57–273 flips saved per 0.01 F1m conceded.
- **In-training satisfaction:** TraLO ~99%, post-hoc baselines (heuristic/danits) ~7% (they satisfy only via flips).
- **Regime effect:** TraLO's tiny F1 edge scales with task difficulty / warmup headroom — positive on tissue (warmup ~50% acc), ~zero on derm (~85%), slightly negative on aider (~99%).

**Every figure must support the real claim: "TraLO matches baseline accuracy while needing far fewer post-hoc corrections and satisfying constraints in-training."** Do NOT make figures that imply a clean F1 win — the data doesn't support it and a reviewer will catch it.

---

## 1. Data source (single file)

`docs/all_cells_raw.csv` — one row per completed cell, 1723 rows. Columns:

```
ds, model, cls, grp, tight, L, G, method, seed,
f1m, f1w, acc, ece, brier, flips, sat, sat_epoch, phase
```

- `method` ∈ {tralo, tralo_bounded, fioretto_ldf, hounie_rcl, danits_lp, heuristic}. `tralo` rows are already filtered to the canonical breakthrough recipe.
- `sat` = in-training Raw All Satisfied (1.0 = satisfied before any flips).
- `flips` = Flips Required (post-hoc corrections).
- Datasets present: tissuemnist, dermmnist, aider, and a few leftover eurosat (DROP eurosat — `ds == "eurosat"` — it's not in the paper).
- For aggregate stats by dataset, also see the script `src/evaluation/tradeoff_analysis.py` (run it; it prints the pooled W/T/L and flip savings).

Aggregate Table A numbers are also in `docs/table_a_summary.md` / `.csv`. Aider-only analysis is in `docs/aider_results/`.

---

## 2. Figures to produce

Put a generation script at `paper/scripts/fig_results_v2.py` (matplotlib, Agg backend, 300 dpi). Output PNGs to `paper/figures/`. Each figure below is one panel or small multiple.

### FIG 1 — F1-vs-flips tradeoff scatter  (THE headline figure)
- One subplot per dataset (tissuemnist, dermmnist, aider).
- x-axis = post-hoc flips required (log scale, +1 offset so 0 is plottable), y-axis = F1-macro.
- One point per (method) = mean over all that dataset's cells+seeds, with error bars = std.
- Color/marker per method; highlight TraLO.
- **Story it tells:** TraLO sits at the same F1 height as baselines but far to the left (few flips). danits/heuristic sit at same height but far right (many flips).
- File: `fig_tradeoff_scatter.png`.

### FIG 2 — F1m gap distribution (are we tied?)
- For each dataset, box/strip plot of per-cell (TraLO F1m − best-baseline F1m).
- Shade the "noise band" ±(mean seed std ≈ 0.01) in grey.
- **Story:** the gap distribution straddles zero inside the noise band → statistical tie. Honest and disarming.
- File: `fig_f1_gap.png`.

### FIG 3 — Flips saved bar chart
- Grouped bars: per dataset, mean flips by method.
- **Story:** TraLO + the trained methods low; danits/heuristic tall. TraLO usually lowest.
- File: `fig_flips_bar.png`.

### FIG 4 — Regime effect (warmup headroom vs TraLO F1 edge)
- x-axis = warmup test accuracy per dataset (tissue ≈ 0.50, derm ≈ 0.85, aider ≈ 0.99 — pull exact from any warmup log or the per-dataset acc in the CSV as a proxy), y-axis = mean TraLO F1m gap.
- 3 points (tissue/derm/aider) + trend line.
- **Story:** TraLO's F1 advantage shrinks as the base task gets easier — explains why aider loses and tissue (slightly) wins. This is a genuine scientific insight, frame it as such.
- File: `fig_regime.png`.

### FIG 5 — In-training satisfaction rate
- Grouped bars: per dataset (or per tightness), mean `sat` by method.
- **Story:** TraLO ~1.0, Fioretto high, Hounie ~1.0 (but at high flip cost), danits/heuristic ~0 (they never satisfy during training).
- File: `fig_satisfaction_v2.png`.

### FIG 6 — Asymmetric tightness heatmap (Phase 2 / Table B, derm only)
- 5×5 grid over (L, G) ∈ {20,30,50,70,80}². Cell color = TraLO F1m − best-baseline F1m (diverging colormap centered at 0).
- **Story:** no asymmetric corner where TraLO collapses; mostly within noise band.
- Data: filter CSV to `ds==dermmnist, model==MobileNetV3, cls==4, grp==loc_group`.
- File: `fig_asym_heatmap.png`.

### FIG 7 (optional) — Backbone + multi-class robustness small-multiples
- Phase 3 (backbones: MobileNetV3/ResNet18/EfficientNetB0) and Phase 4 (cls 0/1/2/4) as grouped bars of F1m gap, to show the tie holds across both axes.
- File: `fig_robustness.png`.

---

## 3. Convergence figure (already in your Block 8)
Your Block 8 already covers `fig_convergence_v2.png` from training logs. Keep it — it supports the "satisfies in-training" claim. Pull TraLO/Fioretto/Hounie excess-vs-epoch from `results/pending_runs/paperv2_phase1/dermmnist/...` or `paper400_tralofix/...` training_log.csv. (Read-only; don't modify those dirs.)

---

## 4. How to wire into the paper
After generating, reference the figures from the Results section of `main.tex`:
- FIG 1 (tradeoff scatter) → headline results figure, goes with the §Headline subsection.
- FIG 2 (gap) + FIG 4 (regime) → Discussion section (the honest "it's a tie on F1, and here's why" narrative).
- FIG 3 + FIG 5 → §Headline / efficiency claim.
- FIG 6 → §Asymmetric tightness subsection (replaces that `\todo{}`).
- FIG 7 → §Backbone / §Multi-class subsections (replaces those `\todo{}`).

Update the prose in those subsections to match the honest framing in §0 — the placeholders currently assume data is pending; now it's in.

---

## 5. What's still running (do not wait on it)
Phase 6 (tissue backbones: ResNet18 + EfficientNetB0 on tissuemnist) is still running on the experiments side. When it finishes, `docs/all_cells_raw.csv` will be regenerated to include it and FIG 7 can add the tissue-backbone arm. For now, build all figures from what's in the CSV; they'll refresh when the data updates.
