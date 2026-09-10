# PAPER_INDEX

**Current paper: `paper/` = the TMLR submission** (conversion of the AAAI manuscript, started by the professor).
**Source of truth is the shared Overleaf project** ("TMLR Paper (with Roei)"); this local copy was unpacked from
`TMLR_Paper__with_Roei_.zip` on 2026-07-22 — sync Overleaf <-> local manually before/after editing.

## paper/ contents

| item | what |
|---|---|
| `main.tex` | TMLR manuscript (professor's conversion in progress) |
| `main_old.tex` | AAAI manuscript carried over for reference during conversion |
| `HANDOFF_TRACK_B.tex` | professor's handoff notes: additional experiments to run |
| `math_commands.tex`, `references.bib`, `fancyhdr.sty` | shared inputs |
| `tmlr.sty`, `tmlr.bst` | TMLR class + bibliography style |
| `figures/` | fig_convergence, fig_datasets, fig_deployment, fig_loss_shape, fig_mechanism, fig_octmnist (pdf + png) |
| `tables/` | 11 result tables (.tex) |

## Retired AAAI-2027 submission (moved here 2026-07-22; `archive/` is gitignored)

- `archive/legacy/final_AAAI_PAPER/` — the full self-contained AAAI submission (text, figures, tables, data, scripts; includes the final uncommitted fixes)
- `archive/legacy/PAPER_INDEX_AAAI.md` — the old index of that tree
- `archive/legacy/AAAI2027_paper_final/` + `.zip` — the Desktop submission bundle
- Committed history of the AAAI paper remains in git (through `c8a08631`)

## Experiment data

- `results/pending_runs/` — the only experiment root (runs live on the GPU servers)
- `archive/` — consolidated old data (raw_runs/, by_axis/, tables/, MASTER_INDEX.csv)
