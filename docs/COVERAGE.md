# COVERAGE -- what we actually have, against what the paper needs

Built 2026-09-02 from **every `config.json` under every one of the 14 worktrees**
(2,671 configs, 2,367 completed). Not from memory, not from a campaign list.

> **THE ORDER OF OPERATIONS.** Nothing below the line "THE GATE" gets run until
> the gate is passed. Grid coverage is worthless if the method it is covering
> does not work, and this project has already produced 2,367 runs of a method
> that has not cleared its own noise floor.

---

## THE GATE: does TraLO work on ONE model?

**NOT YET.** Measured 2026-09-02 over the five independent units that exist
(`scripts/paper_rows.py`, strict task cells only, nothing averaged over cells):

| contrast | units positive | sign p | what it means |
|---|---|---|---|
| `tralo` vs its own **lambda=0 twin** | **5/5** | **0.031** | the constraint changes the model, attributably |
| `tralo` vs **`clip`** | 4/5 | 0.188 | does not beat the quality bar |
| `tralo` vs **`tralo_reseed`** | **3/5** | 0.500 | **does not clear its own RNG floor** |

**The third row is the gate**, and it is the one that got WORSE when the fifth
unit landed (3/4 -> 3/5). An effect below the reseed floor is not an effect: a
reseed changes nothing but the RNG stream, so anything TraLO does that a reseed
also does is not attributable to the constraint.

Per-unit `vs_reseed`, in items:

| unit | backbone x host | items |
|---|---|---|
| A1 | MobileNetV2 x dsisco02 | +3.89, +5.98 |
| A2 | MobileNetV2 x dsisco01 | +6.54, +7.32 |
| B1 | RegNetY400MF x dsisco01 | +1.62, +2.42 |
| B2 | RegNetY400MF x dsisco02 | **-1.74** |
| C1 | MobileNetV3 x dsisco01 | **-0.27** |

⚠️ And **1 of 158 strict-task rows separates from its own seed noise at 2 sd**
(FRAMEWORK 2(z26)). Everything above is a SIGN, not a measurement.

**PASSING THE GATE MEANS:** `vs_reseed` positive in every unit measured, with at
least 5 units, and at least one cell whose effect clears 2 sd on its own. Today:
3/5, and one cell.

---

## 1. DATASETS -- 1 of 3

| dataset | runs | status |
|---|---|---|
| **iwildcam** | **2,176** | 🟢 the only runnable one; images on the server |
| dermmnist | 191 | ⛔ test set leaks 38.7%; removed from disk |
| octmnist | 0 | ⛔ `synth_group = index % 3`; dead by construction |
| tissuemnist | 0 | ⛔ same |
| **fmow** | 0 | 🟡 passes the screen (NET +2969, z=79.7), **META ONLY, no images** |
| **terra** | 0 | 🟡 passes the screen (NET +2546, z=75.8), **META ONLY, no images** |

**Gap: 2 more datasets.** `fmow` is the clean second (a country is an ATOMIC
group, so `factorial_control` does not apply). Its prize is unmeasured -- it
needs `p@K <= 0.92` at L30 to clear twice the noise, where iwildcam measures
0.9948-0.9972. **Get fmow's real p@K before downloading images.**

## 2. BACKBONES -- 4 claimed, but the comparison is on 3

Runs per backbone on iwildcam, and which of the paper's four duals each has:

| backbone | runs | `tralo` | `fioretto` | `hounie` | `alm` | |
|---|---|---|---|---|---|---|
| **ViTB16** | 166 | 30 | **0** | **0** | **0** | 🛑 **HEADLINE, and the comparison has NEVER been run on it** |
| MobileNetV3 | 783 | 97 | 44 | 44 | 32 | ✅ all four |
| MobileNetV2 | 670 | 82 | 36 | 36 | 24 | ✅ all four |
| RegNetY400MF | 557 | 68 | 24 | 24 | 12 | ✅ all four |

🛑 **THIS IS THE BIGGEST SINGLE HOLE.** `ViTB16` was fixed as the headline a
priori on 2026-08-20 precisely so a win could not be promoted after the fact --
and the paper's core comparison (TraLO vs Fioretto vs Hounie vs the clippers)
has never been run on it. `vittask1` (in flight) runs `tralo` + `tralo_cut` +
both clippers on ViTB16, but **not** `fioretto` / `hounie` / `alm`.

## 3. CONSTRAINT PAIRS -- asymmetric is covered, symmetric is NOT

| cap tag | scope | MNv2 | MNv3 | RegNet | ViT |
|---|---|---|---|---|---|
| `L20_G50` | asym L<G, global INERT | 111 | 140 | 104 | 44 |
| `L30_G50` | asym L<G, global INERT | 107 | 140 | 104 | 44 |
| `L80_G95` | asym L<G, global INERT | 124 | 124 | 88 | 24 |
| `L90_G95` | asym L<G, global INERT | 124 | 124 | 88 | 24 |
| `L50_G30` | **asym L>G, global BINDS** | 104 | 104 | 104 | 24 |
| `L95_G80` | **asym L>G, global BINDS** | 100 | 100 | 64 | **0** |
| `L60-90_G95` | per-class L | 0 | 0 | 0 | 6 |
| `L70-90_G95` | per-class L | 0 | 27 | 0 | 5 |
| `L80-100_G95` | per-class L | 0 | 24 | 0 | 0 |

⛔ **NOT ONE SYMMETRIC (`L == G`) CAP HAS EVER BEEN RUN ON iwildcam.** And that
is deliberate, not an oversight: at `L == G` the global cap is **redundant** --
local caps are per-group ceilings, so their sum already bounds the count
(FRAMEWORK 1). A symmetric pair therefore tests the local scope only. If the
paper wants to claim a symmetric/asymmetric contrast, it must say that the
symmetric arm is a local-only control, not a second scope.

⚠️ **Four of the nine tags have an INERT global cap** (`L < G`). Only `L50_G30`
and `L95_G80` make the global scope bind.

## 4. CONSTRAINED CLASSES -- **one configuration, ever**

**Every single iwildcam run constrains exactly `[2, 7]`** (impala, cattle).
Checked across 400 configs spanning every campaign: 400 of 400.

**Never run: 1 class. 3 classes. 4 classes. A different pair. A different
grouping.** This axis is completely unexplored, and it is the one most likely to
matter -- the local scope's teeth come from the 7 of 14 per-group ceilings that
are `K=0`, and that count is a function of WHICH classes are capped.

## 5. ARMS -- the methodology panel

| covered on iwildcam | |
|---|---|
| duals | `tralo` `fioretto` `hounie` `alm` (3 backbones, not ViTB16) |
| allocators | `clip` `focal_clip` `lp` (`danits_lp`) |
| imbalanced | `focal_lp` ✅ · `cb_lp` ⛔ inert · `la_lp` ⛔ inert (2(x1), 2(x2)) |
| controls | `tralo_null` `tralo_reseed` `alm_null` `fioretto_null` `hounie_null` `tralo_lam0` |
| variants | `tralo_uniform` (rejected) `tralo_head` `tralo_cut` (10 runs, negative so far) |

⛔ `cb_lp` and `la_lp` are **not baselines on iwildcam** -- both reduce to plain
CE on its balanced train set. `gen_campaign` now refuses them there.

---

## THE CHECKLIST

### Gate (do this first, nothing else counts until it passes)

- [ ] `tralo` beats `tralo_reseed` in **every** unit measured (today 3/5)
- [ ] at least **one cell** whose effect clears 2 sd on its own (today 1 of 158)
- [ ] `tralo` beats `clip` in every unit (today 4/5)

### Then, and only then, the grid

- [ ] **3 datasets** -- have 1 (iwildcam). fmow next, and measure its p@K FIRST
- [ ] **3+ backbones with the full dual panel** -- have 3; **ViTB16 has none**
- [ ] **symmetric AND asymmetric pairs** -- have asymmetric only, and symmetric
      is a local-only control by construction, so say so
- [ ] **a global cap that BINDS** -- only `L50_G30` and `L95_G80` do
- [ ] **varying the number of constrained classes** -- 1, 2, 3, 4 and different
      groupings. Currently **2, always the same 2**
- [ ] every cap inside its measured task window (`configs/task_windows.yml`)
- [ ] every trained arm with its `_null` twin and the `tralo_reseed` floor
- [ ] `--constraint-fp32` on every trained arm (it is the dose)
- [ ] 4 seeds minimum; more where `paper_rows` says the cell needs them

### Cheapest next moves, in order

1. **Finish `vittask1`** (in flight). Unit 6, and the first ViTB16 task cells.
2. **`vitdual1`: `fioretto` + `hounie` + `alm` on ViTB16** at `L70-90_G95`.
   Closes the headline hole. ~48 runs.
3. **Re-run `taskwin2` + `vittask1` on dsisco02** -- units 7 and 8 for free,
   because the unit is `(backbone, host)` (FRAMEWORK 2(z27)).
4. **A one-class and a three-class cap on iwildcam**, same backbone, same host.
   The first probe of an axis with zero coverage.
5. **fmow p@K**, on CPU from labels, before any download.
