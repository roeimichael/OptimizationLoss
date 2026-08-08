# `docs/` — index

Reorganised 2026-08-08. Three tiers: **active** at the root, **track_b/** for the Track-B
record, **archive/** for superseded material. Nothing was deleted except regenerable LaTeX
build artifacts (`.aux`, `.log`, `.out`).

## Active (root)

| File | What it is |
|---|---|
| `main.tex` | **The manuscript.** The professor's TMLR conversion *"Two Portable Components for Meeting Hard Prediction Quotas"* with the Track-B additions marked in blue. ⚠️ Does **not** compile in place — tables/figures/bib live in `paper/`. Build in Overleaf, or via the `paper/main_rev.tex` working copy. The pre-blue original is `paper/main.tex` and must not be edited. |
| `BLUE_REVISION_BRIEFING.tex` / `.pdf` | Advisor-meeting briefing: every blue change, B1–B8 with the experiment and reasoning behind each, what was omitted and why, and the open to-dos. 14 pp. Rebuild with `pdflatex` ×2. |
| `PAPER_PLAN.md` | Source of truth for paper scope. **Referenced by `CLAUDE.md`.** |
| `REJECTED.md` | Datasets/models tried and dropped — read before re-introducing any. **Referenced by `CLAUDE.md` and `src/models/imagery/vit.py`.** |
| `NATIVE_RES_CAMPAIGN.md` | Pre-registered native-resolution campaign design (the B2 work). **Referenced by `scripts/prep_medmnist224.py`, `src/config_generators/gen_native_res.py`, `src/utils/data_loader.py`.** |
| `MISSING_EXPERIMENTS.md` | Experiment gap list. **Referenced by `src/config_generators/gen_paper_backbones.py`.** |
| `THESIS_CONTEXT.md` | Project orientation / background. |
| `all_cells_raw.csv` | Per-cell corpus export. **10 inbound references from scripts** — gitignored on purpose, do not move. |
| `table_a_summary.csv` | Summary export. 2 inbound references — gitignored on purpose, do not move. |

## `track_b/` — the Track-B record

Six files, three documents. No inbound code references; these are read by humans.

| File | What it is |
|---|---|
| `TRACK_B_RESULTS_HANDOFF.md` | **Source of truth for Track-B results.** Full record, B1–B8 in order, each mapped to the original spec in `paper/HANDOFF_TRACK_B.tex`. |
| `TRACK_B_RESULTS_HANDOFF.tex` / `.pdf` | Compact 2-page professor-facing version of the same. |
| `TRACK_B_DETAILED.tex` / `.pdf` | 7-page expanded methods + results, written to be pasted into the Overleaf appendix (standard packages only, no custom macros). |
| `TRACK_B_REPORT.html` | Standalone styled HTML report of the same results. |

The original Track-B *plan* (the B1–B8 spec these answer) lives at `paper/HANDOFF_TRACK_B.tex`.

## `archive/` — superseded

Kept for provenance; nothing here is current.

| File | Why archived |
|---|---|
| `BLUE_REVISION_BRIEFING.md` | Markdown twin of the briefing. The `.tex`/`.pdf` pair is now the live version; keeping both invited drift. |
| `AUDIT_2026-07-31.md` | Quota-fill audit, superseded by the Track-B adjudication. |
| `PROFESSOR_REVIEW.md` | Earlier review round. |
| `RESULTS_SUMMARY.md` | Pre-Track-B results summary. Linked from `PROFESSOR_REVIEW.md` (both moved together, so the link still resolves). |
| `WARMUP_ABLATION.md` | Warmup ablation notes, folded into the paper. |
