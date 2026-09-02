# COVERAGE -- the corpus that counts, and what it says

Rebuilt 2026-09-02 **after clearing the stale corpus**, then **re-scored
2026-09-02 on the AS-DEPLOYED predictions** after the panel was found to be
answering a different question. Every number here comes from runs produced by
ONE version of TraLO.

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
contains them so no tool can glob them).

⚠️ **And removing `loose1` loses no MobileNetV2 data at all.** Its `tralo` there
is **byte-identical to `dom1`'s in 4/4 seeds**, despite a different `grad_mode`
AND a different `code_version` -- because `clip` scales the step by
`min(raw_norm, 1.0)`, which **is** `normalize` wherever the raw norm is >= 1.

---

## 1. THE INSTRUMENT CHANGED, AND IT CHANGED THE ANSWER

The previous version of this file scored everything through `full_panel`.
**That was the wrong instrument for this question**, in a way that inverted
rankings rather than merely rescaling them.

`full_panel` scores the budget-equalized family on `eq` -- its OWN re-derived
equal-budget allocation, rebuilt from the raw probabilities. It is
*deliberately* allocator-blind, which makes it right for "whose learned RANKING
is better" and wrong for "which arm would you deploy". Measured on
`dom1` / MobileNetV2 / `L80_G95`:

| arm | panel `d capF1` -> items | **as-deployed captured TP, 4 seeds** |
|---|---|---|
| `tralo` | +0.00582 -> **+5.77** | **2602** |
| `alm` | +0.00617 -> **+5.49** | **2602** |

**Identical items captured.** The panel orders them anyway, entirely because
cc-F1 is macro-averaged over two classes whose `(K+n)` differ (666 vs 820):
`alm` trades 5 items out of class 7 into class 2, and class 2's smaller
denominator makes those 5 worth more F1. Nothing was won.

`scripts/deployed_h2h.py` is the instrument that reads the deployed file, in
exact captured items, and it carries the RNG floor beside every ranking.

---

## 2. THE GATE: is TraLO good enough to build a grid on?

**TWO OF THREE, YES. THE THIRD, NO.** Five independent units, as-deployed, in
exact TP items. A unit is ONE warm-up model -- `(backbone, host)`.

| contrast | units | sign p | |
|---|---|---|---|
| `tralo` vs **`clip`** | **5/5** | **0.031** | ✅ beats the quality bar in every unit |
| `tralo` vs its own **`_null`** | **5/5** | **0.031** | ✅ attributable to the constraint |
| `tralo` vs **`tralo_reseed`** | 4/5 | 0.19 | ⚠️ the 5th is exactly 0.00 |
| **`tralo` #1 of the four duals** | **2 of 6** namable cells | -- | ❌ **dominance refuted** |

🔑 **THE FIRST TWO ROWS NOW CLEAR p < 0.05, AND THE OLD READING COULD NOT
HAVE.** A sign test over n unanimous units floors at `0.5^n`, so the previous
4-unit count could not go below 0.0625 **at any effect size**. Two corrections
found the fifth unit:

* `dom1` carries **MobileNetV3 as well as MobileNetV2**, on dsisco02. It was
  never counted. It is a separate warm-up model and a separate unit.
* `taskwin2` and `equaldose1` MobileNetV3 are **md5-identical in 4/4 seeds** --
  one model, therefore ONE unit. Counting them separately was double-counting.

Per unit, mean over its cells, in items:

| unit | backbone x host | cells | vs `clip` | vs `_null` | vs `reseed` |
|---|---|---|---|---|---|
| A1 | MobileNetV2 x dsisco02 (`dom1`) | 3 | +4.00 | +8.00 | +3.00 |
| A2 | MobileNetV2 x dsisco01 (`equaldose1`) | 3 | +2.00 | +1.00 | +6.00 |
| B1 | RegNetY400MF x dsisco01 (`dom1b`) | 3 | +4.00 | +2.00 | +1.00 |
| C1 | MobileNetV3 x dsisco01 (`equaldose1`+`taskwin2`) | 5 | +7.00 | +3.00 | +3.00 |
| C2 | MobileNetV3 x dsisco02 (`dom1`) | 3 | +2.00 | +9.00 | **+0.00** |

⛔ **THE OLD "C1 IS THE FAILURE, AND IT IS THE BACKBONE" READING IS WITHDRAWN.**
It was built on `taskwin2`'s +0.75 items against its null and blamed
MobileNetV3. MobileNetV3 is fine: the same backbone delivers +5 to +13 items in
`dom1`. **The failure was the CELL**, and section 4 says exactly why.

## 3. THE HEAD-TO-HEAD, AND WHY IT NAMES ALMOST NOTHING

`scripts/deployed_h2h.py` over 19 cells, `clip` as control:

```
19 cells: #1 NAMED in 6, REFUSED in 13 (spread under the RNG floor)
10 cells are JACKKNIFE-UNSTABLE (one dropped seed changes #1)
5 cells have items and ccF1 disagreeing on the order
of the 6 named: alm 2  tralo 2  fioretto 2
```

🛑 **THE ARM-VS-ARM GAP IS EXACTLY THE RNG.** Paired per seed over the clean
corpus:

| | median | mean | n |
|---|---|---|---|
| \|`tralo` - a rival dual\| | **4.0 items** | 5.0 | 180 |
| \|`tralo` - `tralo_reseed`\| | **4.0 items** | 6.0 | 70 |

**Ratio 1.00x.** `tralo_reseed` is the same arm with the RNG stream perturbed
and nothing else. So the distance from TraLO to ALM is the distance from TraLO
to itself, and a #1 named off a 4-item lead is naming the seed. **10 of 19
cells duly change their #1 when a single seed is dropped.**

⛔ **AND THE "TIGHT CAP vs SLACK CAP" STORY IS REFUTED AT ITS PREMISE.** The
previous version claimed TraLO led where the cap bound hard and lost where it
was slack. `L95_G80` is **not** looser than `L80_G95`: measured on
`dom1`/MobileNetV2, they emit **660 and 661** predictions per seed
(K/n = 0.800 vs 0.799), because the global cap at 80% pulls the total back to
the same budget. The only genuinely looser cap in the corpus is `L90_G95`
at 744. The two tags differ in SCOPE (local-bound vs global-bound at equal
total), which is a real and unexploited axis, but not in tightness.

**So: TraLO beats the post-hoc clippers, and it does not beat the other duals.
The four duals are indistinguishable at four seeds.**

## 4. THE CAP SCREEN WAS WRONG, AND IT CHOSE THESE CAMPAIGNS' CAPS

`scripts/task_window.py` decides which caps `gen_campaign` will accept. It was
wrong in three ways that pushed in opposite directions, which is why none of
them showed up on its own:

1. **The PRIZE was counted on a GLOBAL top-K.** Every allocator here is
   per-group, and **7 of 14 per-group ceilings on iwildcam are ZERO**, so a
   global top-K counts high-scoring items the allocator can never emit.
   Measured on MobileNetV3 class 2 at K/n=0.70: **8.5 errors global vs 2.0
   local**, a 4.25x overstatement, and it overstates by >2x at 8 of 11
   fractions. => the windows were too GENEROUS about the prize.
2. **The WIGGLE was read at that same global cut**, which sits far above any
   group's own cut. Class 2 at K/n=0.30 reads p@K **0.99998 globally**
   (rejected as saturated) and **0.98258 locally** (fine). => too STRICT about
   saturation.
3. **PRIZE passed on `errors > 0`**, so a cell whose entire available gain was
   0.8 items counted as a task. The RNG floor is **3.0 TP items per capped
   class** (median \|`tralo` - `tralo_reseed`\|, n=70 per class). Nothing below
   that is measurable at 4 seeds by any method. `MIN_PRIZE = 3.0` now.

The windows MOVED rather than merely shrinking, and **one of them emptied**:

| backbone | host | class 2 | class 7 |
|---|---|---|---|
| MobileNetV2 | dsisco02 (`dom1`) | 0.60-0.90 | 0.60-0.90 |
| MobileNetV2 | dsisco01 (`equaldose1`) | 0.70-0.80 | 0.60-0.80 |
| MobileNetV3 | dsisco02 (`dom1`) | 0.60-0.70 | 0.50-0.90 |
| MobileNetV3 | dsisco01 (`equaldose1`) | **NONE** | 0.70-0.90 |
| RegNetY400MF | dsisco01 (`dom1b`) | 0.70-0.80 | 0.60-0.90 |
| ViTB16 | dsisco01 (`vittask1`) | *1 seed -- not measured* | *1 seed* |

🔑 **THIS IS THE WHOLE EXPLANATION FOR `taskwin2`.** `L70-90_G95` puts class 2
at K/n = 0.70 on the dsisco01 MobileNetV3 model -- the one whose strict band is
**empty**. At every fraction where the cap still binds in 4/4 seeds the local
prize is under 3.0 items (0.2 / 0.8 / 1.0 / 1.0 / 1.8 / 2.2 at K/n 0.20-0.70),
and at every fraction where the prize clears the floor the cap has gone slack
in some seed. **`tralo` beat its own null there by +0.75 items. The cell could
not have produced more.** It was staged as THE strict task cap, on the
4.25x-inflated number.

🛑 **AND THE WINDOW IS A PROPERTY OF THE REFERENCE MODEL, NOT THE BACKBONE.**
Same backbone, two hosts, two windows: MobileNetV3 class 2 is [0.60, 0.70] on
dsisco02 and EMPTY on dsisco01. `configs/task_windows.yml` now stores the
INTERSECTION over measured models and keeps each reading in `per_model`.

## 5. WHAT THE CLEAN CORPUS CONTAINS

**1,105 completed runs on the current recipe.**

| campaign | runs | backbone | caps | arms |
|---|---|---|---|---|
| `dom1` | 384 | MNv2, MNv3 | `L80_G95` `L90_G95` `L95_G80` | 4 duals + nulls + clippers |
| `dom1b` | 192 | RegNetY400MF | same | same |
| `equaldose1` | 216 | MNv2, MNv3 | same | same |
| `uniform1` | 252 | MNv2, MNv3, RegNet | `L20_G50` `L30_G50` `L50_G30` | `tralo` `tralo_uniform` + controls |
| `taskwin2` | 48 | MobileNetV3 | `L70-90_G95` `L80-100_G95` | `tralo` `tralo_cut` + controls |
| `vittask1` | 13/48 | **ViTB16** | `L60-90_G95` `L70-90_G95` | `tralo` `tralo_cut` + controls |
| `vitdual1` | **RUNNING** 0/88 | **ViTB16** | `L60-90_G95` `L70-90_G95` | **4 duals** + per-family nulls + clippers |

⚠️ `uniform1`'s caps are all measured NON-TASK, so it contributes no task
cells. It stays because it is on the current recipe and is the `_uniform`
count-function evidence.

## 6. THE ONLY QUESTIONS ON THE TABLE

**Not** "cover more datasets/backbones/class-counts". That grid is real, it is
in section 7, and it is gated behind these:

1. **Does anything separate the four duals at all?** Today nothing does: the
   arm-vs-arm gap IS the RNG floor. Two ways out, and they are different
   experiments:
   - **More seeds.** The per-cell price is in `deployed_h2h`'s `seeds@80%`
     column and ranges from 1 to 597. Cheap only in the handful of cells
     already showing a large gap.
   - **A cell with a bigger prize.** The corrected screen says where those are
     -- and says MobileNetV3/dsisco01 has none for class 2.
2. **`vitdual1`** (running, 88 runs) -- the four duals on the **headline**
   backbone, which has never been run. Its `tralo_null` arms also give ViTB16
   its first real task window, replacing a 1-seed placeholder.
3. **The host term.** dsisco02/bf16 and dsisco01/fp16 do not overlap on
   `tralo - null` (+8/+9 vs +1/+2/+3), and no experiment separates host from
   backbone. One A/B -- `dom1`'s exact cells re-run on dsisco01 -- closes it.

## 7. THE GRID, FOR LATER (do not start these)

Written down so it is not re-derived, and explicitly **not** queued.

- [ ] **3 datasets** -- have 1. `fmow` is the clean second; measure its `p@K`
      on CPU from labels first, it needs `<= 0.92` at L30
- [ ] **symmetric (`L == G`) caps** -- never run, and at `L == G` the global
      cap is exactly redundant, so it is a local-only control, not a scope
- [ ] **varying the constrained classes** -- **400 of 400 configs cap exactly
      `[2, 7]`**. One class, three, four, different groupings: never run
- [ ] a global cap that binds -- only `L50_G30` and `L95_G80` do
