# TraLO — AAAI 7-page paper package (`final_AAAI_PAPER/`)

Self-contained Overleaf-ready package for the TraLO AAAI submission. TraLO trains a
network to satisfy **transductive prediction-count constraints** (a constrained class
has a hard cap on how many held-out items may receive it; optionally per-group local
caps) and is compared against constraint-trained dual-ascent baselines and post-hoc
clipping baselines.

## What's in the folder

```
final_AAAI_PAPER/
  main.tex            7-page AAAI draft (\input's the tables, \includegraphics the figures)
  references.bib      minimal bibliography (see cite keys below)
  paper.html          two-column readable preview (no LaTeX toolchain needed)
  README.md           this file
  figures/            6 figures, each as BOTH .pdf (vector; pdflatex uses this) and .png
                      (>=300 dpi; the HTML preview uses this). main.tex \includegraphics
                      uses the EXTENSIONLESS name so pdflatex prefers the vector PDF.
    fig_loss_shape    bounded transductive penalty vs a naive unbounded penalty (single col)
    fig_mechanism     why dual-ascent fails: CE -> NaN + escalating multiplier vs TraLO (figure*)
    fig_deployment    native satisfaction by method x backbone (Pillar 1 / deployment win) (figure*)
    fig_octmnist      OctMNIST per-level cc-F1 hard win vs trained baselines, +/-1 SD (Pillar 2)
    fig_ablation      component ablation: reset + hinge load-bearing, min-max whiskers (Pillar 3)
    fig_convergence   the core convergence claim: best-so-far excess -> 0 (figure*)
  tables/             3 \input-ready LaTeX floats (NO preamble — float environment only)
    tab_regime.tex            anti-cherry-pick master regime table (every canonical cell, all ties/losses)
    tab_results.tex           headline cc-F1 (Table 1) + deployment cost (Table 2), MobileNetV3
    tab_ablation_complete.tex complete 5-component ablation (reset + hinge load-bearing), OctMNIST L30/L40
  scripts/
    fig_style.py             shared AAAI figure style (Type-42 fonts, Okabe-Ito palette,
                             Times+STIX, dual PDF+PNG export) — imported by every make_*.py
    make_figs.py             fig_loss_shape + fig_mechanism
    make_deployment_fig.py   fig_deployment
    make_octmnist_fig.py     fig_octmnist
    make_ablation_fig.py     fig_ablation
    make_convergence_fig.py  fig_convergence (reads paper/data_cache/training_logs/;
                             replaces the legacy archive/ generator — now regenerable in-pipeline)
```

## Figure standards (AAAI production pass)

All six figures were brought to AAAI camera-ready standards via `scripts/fig_style.py`:
vector **PDF** export (plus a >=300 dpi PNG sibling), **embedded TrueType fonts** (Type 42 —
verified zero Type-3), **Times Roman + STIX** to match the body, the **Okabe-Ito**
colorblind-safe palette with redundant marker/line-style/hatch encoding (grayscale-safe),
figures authored at their **final printed width** (single-column ~3.5in `figure`; the three
wide ones promoted to full-width `figure*`), light grid / no top-right spines, and visible
**seed/dataset variability** on every result figure (per-dataset dots on deployment, +/-1 SD
bands on octmnist, min-max whiskers on the ablation). Each caption states `n` and the error
definition. Remaining camera-ready nicety: convert to CMYK if the proceedings require it
(matplotlib emits RGB; submission accepts RGB). Regenerate everything with
`for s in make_figs make_deployment_fig make_octmnist_fig make_ablation_fig make_convergence_fig; do python final_AAAI_PAPER/scripts/$s.py; done`.

## Three robust pillars (the only claims the paper makes)

1. **Universal deployment win vs post-hoc clipping** — native satisfaction 1.00 vs ~0,
   far fewer label flips, macro-F1 preserved (clipping erodes it). Reported via
   native-satisfaction + macro-F1, not raw flip counts. See `tab_results.tex` (Table 2)
   and `fig_deployment.png`.
2. **OctMNIST tight-binding hard win vs the constraint-trained baselines** (cc-F1, L=G in
   {30,40}, all backbones, 4/4 winrate; backbone-general by leave-one-backbone-out CV).
   It is an inverted-U in tightness and dataset-specific to OctMNIST; the bulk of the
   symmetric tissue+derm grid is an honest tie vs trained. See `fig_octmnist.png`,
   the bold Oct/tight-sym cell in `tab_regime.tex`, and OctMNIST rows in `tab_results.tex`.
3. **Component ablation** — **two load-bearing components**: the **optimizer reset** at
   satisfaction (+0.079 cc-F1, p=3e-7) and the **undershoot hinge** (+0.036, p=3e-6) on
   OctMNIST L30/L40 across MNV3 / RegNet / ViT (n=24 paired); the rho schedule, freeze, and
   KL anchor are cc-F1-neutral. See `tab_ablation_complete.tex` + `fig_ablation.png`.

**Mechanism (stated qualitatively):** under a hard cap the dual-ascent baselines escalate
an unbounded multiplier and their cross-entropy goes NaN at constraint-epoch ~2 (structural:
Fioretto 80/80, Hounie 80/80, across dual-step 0.001–0.1). This is a *training-stability*
event, not collapse — the baselines recover from a best-satisfied checkpoint and stay
competitive (cc-F1 0.23–0.62, satisfied, macro within ~0.005). TraLO's bounded, frozen
multiplier ($\le$0.09) keeps CE finite and preserves constrained-class recall. The
*magnitude* of instability does not predict the win — never claim "more escalation = bigger
win". See `fig_mechanism.png`.

## Cite keys available in `references.bib`

Core: `fioretto2020lagrangian` (Fioretto LDF baseline), `hounie2023resilient` (Hounie
RCL baseline), `shifman2025classification` (Shifman-LP baseline),
`howard2019searching` (MobileNetV3), `radosavovic2020designing` (RegNetY),
`dosovitskiy2021image` (ViT-B/16), `sandler2018mobilenetv2` (MobileNetV2),
`yang2023medmnist` (MedMNIST), `tschandl2018ham10000` (DermMNIST/HAM10000).
Related work: `chamon2020probably`, `chamon2023constrained`, `stooke2020responsive`,
`ramirez2025position`, `bertsekas2014constrained`, `sangalli2021constrained`,
`lin2017focal`, `vapnik1998statistical`, `joachims1999transductive`, `he2016deep`,
`cortes2016learning`, `shifman2025classification`, `vanderschueren2024perspective`.

## How to compile (Overleaf)

1. Create a new Overleaf project and upload this entire folder (`main.tex`,
   `references.bib`, `figures/`, `tables/`).
2. Set the compiler to **pdfLaTeX** and the main document to `main.tex`.
3. Compile twice (pdfLaTeX → BibTeX → pdfLaTeX → pdfLaTeX) so citations and
   cross-references resolve.

The tables in `tables/` are float-only fragments (no `\documentclass` / `\begin{document}`
wrapper); they are pulled in by `main.tex` via `\input{tables/...}` and rely on the
`booktabs` and `multirow` packages loaded in the preamble.

## Preview without LaTeX

Open `paper.html` in any browser for a two-column readable rendering of the paper.

## Provenance

All numbers trace to the frozen, audited corpus
`paper/aaai_tables/_corpus_with_final.csv` (`sweep=='paper_final'`, the 1944-cell grid)
and the ground-truth checkpoint `paper/STRONG_RESULTS_CHECKPOINT.md`. The headline /
deployment tables derive from `paper/aaai_tables/tableshowing.tex`; the regime master
from `paper/aaai_tables/regime_master.tex`; the complete component ablation
(`tab_ablation_complete.tex` / `fig_ablation.png`) comes from
`extra_experiments/ablation_complete.csv` — a separate verified 480-run campaign
(`extra_experiments/CAMPAIGN_RESULTS.md`) isolated from the paper grid, with the hinge leg
also present in `_corpus_with_final.csv`.
