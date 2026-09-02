# COVERAGE -- the corpus that counts, and what it says

Rebuilt 2026-09-02 **after clearing the stale corpus**. Every number here comes
from runs produced by ONE version of TraLO.

> ## THE RECIPE
>
> **`iwildcam` + `constraint_fp32: True` + `constraint_grad_mode: normalize`.**
>
> Anything else is a different method and is not in this document.

---

## 0. WHAT WAS CLEARED, AND WHY

Walking every `config.json` in all 14 worktrees found **five distinct TraLO
configurations** across 277 completed `tralo` runs. Only one is current.

| cfg | runs | `fp32` | `grad_mode` | campaigns | |
|---|---|---|---|---|---|
| **1** | **106** | **True** | **normalize** | `dom1` `dom1b` `equaldose1` `taskwin2` `uniform1` `vittask1` | ✅ **THE recipe** |
| 2 | 80 | True | clip | `iwc4` `loose1` `loosevit1` `vitu1` | arms differ in DOSE as well as direction |
| 3 | 52 | False | clip | `iwc1` `iwc2` `iwc3` | 68.6-74.6% of the dose lands |
| 4 | 36 | False | normalize | `xfam1` | `run_code_version` splits by SEED |
| 5 | 3 | False | normalize | `taskwin1` `uniform1_VOID` | 69.0% and 3.4% of dose |

**18 campaigns / 1,326 configs moved to `~/optloss-archive-stale-2026-09-02/`**
(a move, not an unlink: same filesystem, reversible, and `results/` no longer
contains them so no tool can glob them). Ten stale-recipe iwildcam campaigns,
seven dermmnist campaigns on the leaked test set, and `vitdom2_cnn` /
`vitdom2_vit`, which were on the current recipe but staged entirely on caps the
task-window gate rejects.

🔑 **THE ONE THING THE CLEARING CHANGED IN THE RESULT.** The old 5-unit reading
carried `B2 = loose1 / RegNetY400MF`, the single unit that DISSENTED on all
three contrasts. `loose1` is cfg2, a different method. With it gone,
`tralo` vs `clip` goes from 4/5 to **4/4**.

⚠️ **And removing `loose1` loses no MobileNetV2 data at all.** Its `tralo` there
is **byte-identical to `dom1`'s in 4/4 seeds**, despite a different
`grad_mode` AND a different `code_version` -- because `clip` scales the step by
`min(raw_norm, 1.0)`, which **is** `normalize` wherever the raw norm is >= 1.
The two modes coincide exactly in that regime. Measured, not assumed.

---

## 1. THE GATE: is TraLO good enough to build a grid on?

**NOT YET, and it is close.** Four independent units, strict task cells only,
nothing averaged across cells (`scripts/paper_rows.py`).

| contrast | units | sign p | |
|---|---|---|---|
| `tralo` vs **`clip`** | **4/4** | 0.0625 | ✅ beats the quality bar in every unit |
| `tralo` vs its own **`_null`** | **4/4** | 0.0625 | ✅ attributable to the constraint |
| `tralo` vs **`tralo_reseed`** | **3/4** | 0.3125 | ⛔ **does not clear its own RNG floor** |
| **#1 of the four duals** | **3/6 cells** | 0.66 | ⛔ **dominance not shown** |

Per unit, in items:

| unit | backbone x host | vs `clip` | vs `_null` | vs `reseed` |
|---|---|---|---|---|
| A1 | MobileNetV2 x dsisco02 (`dom1`) | +5.77, +9.85 | +11.61, +13.23 | +3.89, +5.98 |
| A2 | MobileNetV2 x dsisco01 (`equaldose1`) | +2.84, +4.48 | +1.71, +3.80 | +6.54, +7.32 |
| B1 | RegNetY400MF x dsisco01 (`dom1b`) | +6.40, +7.98 | +4.38, +4.60 | +1.62, +2.42 |
| C1 | MobileNetV3 x dsisco01 (`taskwin2`) | +7.32 | +0.75 | **-0.27** |

🛑 **C1 IS THE FAILURE, AND IT IS SPECIFIC.** On MobileNetV3 `tralo` beats
`clip` by a healthy +7.32 items but beats its own `_null` by only **+0.75** --
below the one-item quantum -- and **loses to a pure RNG reseed by 0.27**. So on
that backbone the +7.32 is the REGIME (30 trained epochs), not the constraint.

**PASS CONDITION:** `vs_reseed` positive in every unit, AND `tralo` #1 of the
four duals in a clear majority of cells. Today 3/4 and 3/6.

## 2. THE HEAD-TO-HEAD, per cell, vs `clip`, in items

| unit | campaign | backbone | cap | `tralo` | `alm` | `fioretto` | `hounie` | #1 |
|---|---|---|---|---|---|---|---|---|
| A1 | `dom1` | MobileNetV2 | `L80_G95` | **+5.77** | +5.49 | -0.85 | -4.48 | tralo |
| A1 | `dom1` | MobileNetV2 | `L95_G80` | +9.85 | **+10.87** | +4.61 | +8.47 | alm |
| A2 | `equaldose1` | MobileNetV2 | `L80_G95` | +2.84 | +1.67 | **+3.45** | -8.31 | fioretto |
| A2 | `equaldose1` | MobileNetV2 | `L95_G80` | **+4.48** | +1.67 | +2.00 | +4.41 | tralo |
| B1 | `dom1b` | RegNetY400MF | `L80_G95` | **+6.40** | +0.78 | +2.14 | -1.83 | tralo |
| B1 | `dom1b` | RegNetY400MF | `L95_G80` | +7.98 | +9.41 | **+10.37** | +9.15 | fioretto |
| C1 | `taskwin2` | MobileNetV3 | `L70-90_G95` | +7.32 | -- | -- | -- | (no rivals run) |

🔑 **THE PATTERN IS THE CAP, NOT THE BACKBONE.** TraLO is #1 in **2 of 3**
cells at the TIGHTER `L80_G95` and **1 of 3** at the looser `L95_G80`. Where
the cap binds hard, TraLO leads; where it is slack, `alm` and `fioretto`
overtake it. That is a lead worth chasing and it is the direction to work on.

⚠️ **`hounie` is erratic**: -8.31 to +9.15 across four cells. Do not read its
mean.

## 3. WHAT THE CLEAN CORPUS ACTUALLY CONTAINS

**1,228 configs in `results/`, all one recipe.**

| campaign | runs | backbone | caps | arms |
|---|---|---|---|---|
| `dom1` | 384 | MNv2, MNv3 | `L80_G95` `L90_G95` `L95_G80` | all four duals + nulls + clippers |
| `dom1b` | 192 | RegNetY400MF | `L80_G95` `L90_G95` `L95_G80` | all four duals + nulls + clippers |
| `equaldose1` | 216 | MNv2, MNv3 | `L80_G95` `L90_G95` `L95_G80` | all four duals + nulls + clippers |
| `uniform1` | 252 | MNv2, MNv3, RegNet | `L20_G50` `L30_G50` `L50_G30` | `tralo` `tralo_uniform` + controls |
| `taskwin2` | 48 | MobileNetV3 | `L70-90_G95` `L80-100_G95` | `tralo` `tralo_cut` + controls |
| `vittask1` | 48 | **ViTB16** | `L60-90_G95` `L70-90_G95` | `tralo` `tralo_cut` + controls |
| `vitdual1` | 88 | **ViTB16** | `L60-90_G95` `L70-90_G95` | **all four duals** + nulls + clippers |

⚠️ `uniform1`'s caps are all measured NON-TASK (2(z17)), so it contributes no
task cells. It stays because it is on the current recipe and is the `_uniform`
count-function evidence.

**Backbone coverage of the four-dual comparison, on the current recipe:**

| backbone | four duals? |
|---|---|
| MobileNetV2 | ✅ `dom1`, `equaldose1` |
| RegNetY400MF | ✅ `dom1b` |
| MobileNetV3 | ✅ `dom1`, `equaldose1` (non-task caps); `taskwin2` is tralo-only |
| **ViTB16** (headline) | 🔵 **`vitdual1` STAGED, 88 runs, not yet run** |

## 4. THE ONLY QUESTION ON THE TABLE

**Not** "cover more datasets/backbones/class-counts". That grid is real and it
is written down in section 5, but it is gated behind this:

> **Make `tralo` clear its own reseed floor in every unit, and lead the four
> duals in a clear majority of cells.**

Two experiments answer it, both already staged:

1. **`vittask1`** (running) -- `tralo` on ViTB16, the headline backbone.
2. **`vitdual1`** (staged, 88 runs) -- the four duals head-to-head on ViTB16
   at both strict task caps. **This is the paper's core comparison on the
   paper's chosen backbone, and it has never been run.**

Then diagnose C1: why is `tralo - null` only +0.75 items on MobileNetV3 when it
is +11.6 to +13.2 on MobileNetV2?

## 5. THE GRID, FOR LATER (do not start these)

Written down so it is not re-derived, and explicitly **not** queued.

- [ ] **3 datasets** -- have 1. `fmow` is the clean second; measure its `p@K`
      on CPU from labels first, it needs `<= 0.92` at L30
- [ ] **symmetric (`L == G`) caps** -- never run, and at `L == G` the global
      cap is exactly redundant, so it is a local-only control, not a scope
- [ ] **varying the constrained classes** -- **400 of 400 configs cap exactly
      `[2, 7]`**. One class, three, four, different groupings: never run
- [ ] a global cap that binds -- only `L50_G30` and `L95_G80` do
