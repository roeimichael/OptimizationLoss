# F — Component ablation  ⚠️ PENDING

Leave-one-out ablation that fixes the problem to the headline configuration and
removes **one TraLO component at a time**, isolating each component's marginal
contribution to feasibility.

## Backs in `main.tex`
- **Table `tab:component_ablation`** — currently a **skeleton** (`\pending{}` box;
  cells marked `\textit{pending}`).

## Status
**No result CSV yet.** The leave-one-out grid is being gathered by **G5**
(7 variants × 3 datasets × 3 seeds at L30_G30, plus the existing 16-cell KL sweep).
When G5 completes, drop the aggregated CSV here and fill the table cells.

## Planned configuration
- **Fixed problem:** headline tightness L30_G30, the 3 active datasets, MobileNetV3.
- **Variants (disable one at a time):**
  1. undershoot hinge → recovers **TraLO-bounded** (this row is *already*
     supported by the TraLO-bounded column throughout Tables A–E: hinge off
     multiplies flips, e.g. 1.9 → 5.7 on the asymmetric sweep).
  2. optimizer reset at first satisfaction
  3. ratchet-freeze multiplier schedule
  4. CE-saturation skip
  5. linear ρ ramp (replaced by fixed ρ)
  6. (separately) **re-enable** the KL anchor (α > 0) to measure its drift-control
     effect — the only place α·L_KL is nonzero; it is α = 0 in every other experiment.

A component is "essential" when disabling it degrades feasibility (more flips,
lower Sat%) without a compensating F1 gain.

## Provenance (when available)
Generator: `src/config_generators/gen_g5_component_ablation.py`.
KL-sweep generator: `src/config_generators/gen_kl_ablation.py`.
