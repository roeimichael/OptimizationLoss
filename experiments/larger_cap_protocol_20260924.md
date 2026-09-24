# Larger-cap matched comparison, 24 September 2026

The user asked whether the earlier small synthetic caps prevent TraLO from
showing a useful selection effect. This is an exploratory intervention on the
**training budget and final allocation budget together**, not a rescore of old
models. It uses the already audited knee and CIFAR feature caches. No test set
is scored or used to choose settings.

One fixed increase is tested: 1.5 times the original caps, rounded to integers.
Knee grade 3/4 caps change 82/16 to 123/24 among 826 development images. CIFAR
class 0–9 caps change from 10 to 15 each among 2,000 development images; other
classes stay uncapped. These numbers were chosen as round multipliers before
new training outcomes, not from development labels. Knee training contains
757 grade-3 and 173 grade-4 images among 5,778. CIFAR's selected training
subset has 83–107 images for each capped class among 10,000. The larger caps
create more available output slots while leaving meaningful count pressure.
The knee development cohort is still small, especially for grade 4; increasing
the cap cannot create more genuine cases.

Use the current sample-aware recipe unchanged: frozen ImageNet ResNet18
features, 20 head epochs with 5 warm-up epochs, batch 256, task Adam 0.001,
separate constraint Adam 0.0001, far-error weight 0.1, the same original
soft-count penalty, and seeds 1001–1004. Within each dataset and seed compare
Clipper, sample-aware TraLO Null and sample-aware TraLO. Caps are identical
for all arms. A seed-1001 pilot checks actual finite update dose, cache/source
identity, and output quota/metric recomputation. Expand the other three seeds
only after those operational checks pass; a favorable pilot score is **not**
a gate. Both `capped_first` (exact slot counts) and
`upper_bound_correction` (upper bounds) are reported separately, with raw
predictions, confusion counts, constrained-class F1, accuracy and paired seed
differences. The primary scientific contrast is TraLO minus its matched Null
under capped-first; Clipper is a separate baseline. No selection among caps,
policies, seeds or epochs after seeing outcomes.

The saved-probability offline check at 1×, 1.5× and 2× caps is retained at
`C:/Users/roeym/.codex/rebuild-audit-20260922/cap_sensitivity_20260924.json`.
It did not retrain any model and cannot answer this new training question.
The larger-cap study is also development-set exploratory; comparisons with
earlier cap settings are not independent confirmation.
