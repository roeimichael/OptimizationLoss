# E — Group-column ablation (sex instead of loc_group)

Checks that the local-constraint win does not depend on the specific subgroup
axis, by swapping the group column from `loc_group` (three anatomical sites, the
headline) to `sex` (binary, balanced).

## Backs in `main.tex`
- **Table `tab:sexgroup_summary`** (overall mean across the 5 symmetric cells).

## Configuration
- **Dataset/backbone:** DermMNIST / MobileNetV3 · **Constrained class:** MEL
- **Group column:** `sex` (binary, balanced)
- **Tightness:** 5 symmetric cells · **Methods:** all 6 · **Seeds:** 4.

## Headline result
The win is robust to the group axis: TraLO needs ≈ 3.1 post-hoc flips versus
8–92 for the other methods, with Macro F1 a tie. Swapping a heterogeneous
3-site group for a balanced binary one does not change the deployability story.

## Files
- `table_E_phase5_sexgroup_derm.csv` — 30 rows = 5 `constraint_tag` × 6 methods,
  4 seeds each. Columns: `macro_f1_mean/std, flips_mean/std, accuracy_mean/std,
  satisfied_pct`.

## Provenance
Phase-5 group-column ablation; aggregated per (constraint_tag, method) over 4 seeds.
