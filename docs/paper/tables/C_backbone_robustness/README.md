# C — Backbone robustness (saturated-warmup regime)

Checks that the deployability win is not an artifact of the MobileNetV3 backbone,
using two backbones whose warmup **saturates** in the first epoch — the regime
where TraLO concedes its F1 edge but should still win on flips.

## Backs in `main.tex`
- **Table `tab:backbone_summary`**.

## Configuration
- **Backbones:** ResNet18, EfficientNetB0 · **Datasets:** TissueMNIST, DermMNIST
- **Tightness:** 5 symmetric cells per (backbone, dataset) → 10 cells/backbone
- **Methods:** all 6 · **Seeds:** 4.

## Headline result
Both backbones reach their constrained-class accuracy ceiling in warmup epoch 1,
so **Macro F1 is a tie among the trained methods by construction**. The separation
is entirely in the deployability columns: TraLO needs far fewer post-hoc flips
(e.g. ResNet18 ≈ 8.3, EfficientNetB0 ≈ 6.2) than the post-hoc baselines (≈ 11–50)
while holding Sat% = 100%. This is the robustness role of the table: the win holds
even when there is no F1 headroom.

## Files
- `table_C_backbone_saturated.csv` — 120 rows = 2 backbones × 2 datasets ×
  5 `constraint_tag` × 6 methods, 4 seeds each. Columns: `backbone, dataset,
  constraint_tag, method, macro_f1_mean/std, flips_mean/std, accuracy_mean,
  satisfied_pct`.

## Provenance
Saturated-backbone robustness runs; aggregated per (backbone, dataset,
constraint_tag, method) over 4 seeds.
