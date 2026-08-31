# MISSION -- make TraLO mathematically the best, and prove it

**This file is the resume point.** A fresh session reads it first, then
`docs/FRAMEWORK.md` section 3(0) (the status board). It is updated at the end of
every working session. If it is stale, that is a defect -- fix it before doing
anything else.

Last updated: **2026-08-30** (second pass: the mechanism, and the independent-n correction).

---

## ⚠️ 0-PRE. THE DECISION THAT BLOCKS EVERYTHING (2026-08-30)

**`grep -ril iwildcam docs/paper/` returns ZERO files.** The manuscript is a
DermMNIST / OctMNIST / TissueMNIST / HAM10000 paper at **warm-up 50 with no
lambda=0 twins**. Every campaign run since 2026-08-21 is **iwildcam at warm-up
1 with twins**, and all three MedMNIST datasets are removed from disk.

⇒ **the paper and the evidence base share no dataset, no warm-up regime and
no controls.** Every iwildcam result currently has no destination, and every
paper claim rests on data that is quarantined or deleted.

**This is Roei's call and it should be made before the next campaign:**

| option | cost | consequence |
|---|---|---|
| **A. Rewrite the empirical section around iwildcam** | the 8 corpus-generated tables go, and the corpus **cannot be rebuilt** | the paper matches the evidence, and inherits warm-up 1 + real controls |
| **B. Restore a MedMNIST dataset** | re-download + re-run; `octmnist` was CLEAN in the leakage audit (keeps MedMNIST's official split), `dermmnist` leaked 38.7% | the corpus tables survive, but `octmnist`'s groups are `index % 3` -- **dead by construction** for the local scope (FRAMEWORK 2(n)) |
| **C. Both** | most expensive | iwildcam as the headline, MedMNIST as the legacy comparison |

⚠️ Option B has a trap already measured: `octmnist` passes the leakage audit
but **fails the information screen** (NET -7 items, z=-0.4), so a count
constraint carries nothing there. B restores the tables at the cost of a
dataset that cannot test the thesis.

---

## 🔴 0-NOW. THE MATHEMATICAL DEFECT, FOUND 2026-08-31/09-01

Three measurements, in the order they were made. FRAMEWORK 2(z11), 2(z12).

**(1) At the item level the constraint is at the RNG floor.** `boundary_probe
--control tralo_null`, every arm against its OWN lambda=0 twin. A pure reseed
moves **3357** items where the constraint arms move 3362-3647, and nets **+89**
at tight / **+167** at loose against `tralo_uniform`'s **-43** / **+148**. So
`tralo_uniform` does not clear its own noise floor in either regime; only
`tralo` at loose (+221) and `alm` (+255) do.

**(2) `tralo_uniform`'s founding claim is false.** Its docstring argues a
uniform step in log-odds is "a pure bias shift, which cannot reorder". The step
is taken in PARAMETERS, not logits: `dz_i = -lr*g*n*(fbar.f_i + 1)`, which
varies with `fbar.f_i`. It reorders, with the backbone FROZEN -- the leak is in
the linear head. `scripts/bias_shift_probe.py`. The only provably harmless
update is one confined to `b_c`, and that one is useless: a constant added to
`z_c` leaves the within-class order untouched, so the emitted top-K is
bit-identical.

**(3) AND THE ROOT CAUSE. The shipped count puts 0.00% of its gradient at the
cut.** `p(1-p)` is maximal at p=0.5 and vanishing at p=1; the tight-cap cut sits
at **p = 0.99984 to 1.00000**. Fraction of gradient mass on the 40 items
straddling rank K, measured on real stored features over 24 (run, class) pairs:

| weighting | mass at the cut |
|---|---|
| `cut_window` | **0.3486** |
| `p` | 0.1039 |
| `uniform` | 0.0136 |
| **`sum` -- THE SHIPPED COUNT** | **0.0001** |
| `margin` -- the BOUNDARY window | **0.0000** |

⇒ The penalty spends its whole budget where movement cannot change the
emitted set, and nothing where the metric reads. **That is (1) explained**: the
reordering is at the RNG floor because it is arbitrary with respect to the
metric. It also **derives the regime reversal with no new assumption** -- at
loose caps the cut falls to p=0.59-0.99 where `p(1-p)` finally has mass, which
is exactly where `sum` wins.

🛑 **AND IT PRICES `margin2` BEFORE IT RUNS.** `tralo_margin` windows the
DECISION BOUNDARY, puts **exactly 0.0000** at the cut, and sits at cosine
**0.989** from `tralo` -- so its 432 staged runs would mostly reproduce `tralo`.
That is CLAUDE.md rule 3's conflation costing a campaign. Run the cut window
first.

⚠️ **WHAT IS NOT CLAIMED: that aiming at the cut WINS.** Necessary, not
sufficient. At tight caps the clipper's set is already 99.6% correct and
`headroom` reads 0.0-1.0 items, so a correctly-aimed gradient can still find
nothing to take -- it may fix the aim and still lose in the very regime it was
built for. What IS predicted is that the count trajectory responds where `sum`
measurably cannot.

---

## 0. THE GOAL, stated so it can be failed

Make **TraLO** the best of the constrained-optimization methodologies, on the
mathematics, and show it. Not "not worse". Not "wins on one metric on one
backbone at one cap". The bar the work is held to:

| axis | required | have now | gap |
|---|---|---|---|
| **datasets** | **3** | **1** (iwildcam) | `fmow` screened + passes the factorial gate, needs ~21k images. Third TBD |
| **backbones** | **3** | **4 exist**: MobileNetV2/V3 (`dom1`), RegNetY400MF (`dom1b`), **ViTB16 (`vitu1` tight + `loosevit1` loose, already complete)** | not coverage -- CELLS. ViTB16 has 3 tight + 2 loose contrast cells and **no rival-dual arms at all** |
| **constraint pairs** | **varied**, both **equal and unequal** local:global ratios | 3, **all loose**, only 1 unequal-binding | `margin2`'s matched 2x2 (4 tags, 2 budgets x 2 scopes) closes this the moment a GPU frees |
| **consistency** | wins across **regimes**, not one | wins at L80-L95; **loses at L20-L50, and now we know why** | the mechanism is found (2(y)); the question is whether ANY count function can fix it |
| **metrics** | ccF1 **and** macroF1 both defensible | backbone-dependent: macroF1 **-0.0022 on MobileNet** (`dom1`) but **+0.0196 tight / +0.0021 loose on ViTB16** | the damage is a REPRESENTATION effect (2(z1)), not allocation -- and it is not universal |

🛑 **Winning only at L80/L90 is not a result.** If TraLO loses at every other
constraint pair, the claim is "TraLO helps when the constraint barely binds",
which is not the thesis.

---

## 1. WHERE WE ACTUALLY ARE (read the numbers, not the vibe)

### What is established

- **`dom1` (384 runs, complete, LOOSE caps, MobileNetV2+V3).** TraLO is #1 of
  five on ccF1 / AP / AUROC, 6/6 cells each.
- **First campaign at equal dose.** All five trained arms 100.0%; `hounie`
  672/672, which previously ran at **1%**. No earlier dual-vs-dual number is safe.
- **The four lambda=0 nulls are byte-identical 24/24**, so the compute term is
  shared exactly and arm differences are the method.

### What is NOT established, and must be said every time

| claim | reality |
|---|---|
| TraLO > fioretto | **AP 3/6 cells, p=1.00. A coin flip.** |
| TraLO > alm | 4/6 on everything, p=0.69. Not shown. |
| TraLO > hounie | 6/6 AP+AUROC on dom1, p=0.031, **fails BH** -- and on dom1b (RegNet) `tralo` is **4th of 5 on AP and 3rd on AUROC**, both **below its own reseed floor**. The ranking lead does not reproduce. FRAMEWORK 2(z5) |
| Anything survives correction | **0 of 20 contrasts** -- and it is worse than that: the independent unit is **(model, seed) = 8**, not 6 cells, because a lambda=0 twin is byte-identical across cap tags. **8 of 9 dom1 sweeps evaporate at n=8**; only class 4's allocated damage survives (0/8, p=0.0078). FRAMEWORK 2(z) |
| macroF1 | **-0.0022, 2/6 on MobileNet** -- but **+0.0196 (3/3) tight and +0.0021 loose on ViTB16**, against a reseed floor of -0.0366 loose. Backbone-dependent, and the damage is REPRESENTATION drift, not allocation (RAW -0.0107 is 44% LARGER than deployed -0.0074) |
| TraLO enforces better | **REFUTED.** Pulls +6.2 items vs hounie +23.4. The WEAKEST of the four |
| Constraints ever satisfied in training | **0 of 696 epochs.** The post-hoc allocator does all of it |
| dom1 is the headline | **No.** FRAMEWORK 1-pre fixed **ViTB16** a priori; dom1 has none |

### The regime step is REAL. The explanation for it is NOT. (corrected 2026-08-30)

✅ **The solid part.** The CNN warm-ups are shared across campaigns, which
gives a within-model tight-vs-loose contrast with no confound. Paired on the
**12 warm-up models present in both regimes**: `tralo` moves **+6.24 items
from tight to loose, 12/12, sign p = 0.00049** (the exact floor at n=12), while
**the reseed floor does not move at all** (5/12, p = 0.774). Floor-corrected,
+5.30 items, 12/12. This is the cleanest attributable result the project has.

⛔ **The part that failed.** The geometric account -- that the penalty aims at
the decision boundary while the metric reads the cut, `gap = hard - K` -- is
**consistent but not discriminated, and its sharp prediction failed**:

* `gap`, `slope_K` and `K/n` are one variable in three costumes within a model
  (`rho(gap, K) = -1.0000` exactly, hard count constant in 40/40 groups).
* Both `gap` and `slope_K` **reverse sign** once the cap is held fixed.
* `tralo_uniform` was predicted to order oppositely. It does not -- same sign
  at every level.

⇒ quote the geometry as an unrefuted account, never as a measured cause.
Testing it needs `gap` varied at FIXED `K/n`. FRAMEWORK 2(y).

🛑 **And the absolute loose-cap win does not survive honest units.** At the
cell level `tralo - null` reads 15/20 (p=0.041); at the **16 distinct warm-up
models it is 11/16, p = 0.21**, and it beats the reseed floor 11/16, p = 0.21.
**macroF1 and uncapped F1 are NEGATIVE in 11 of 16 units.** The relative
(loose-minus-tight) statement survives; the absolute one does not.

⚠️ **`dom1` is not independent of `loose1`.** Its L80_G95 and L90_G95 cells
are byte-identical to `loose1`'s in 80/80 files; `dom1` contributes only
L95_G80. Only **20 distinct warm-up models** exist across all five campaigns.

## 2. THE KNOB LEDGER -- what has been tried on TraLO itself

✅ = keep · 🟡 = live, unresolved · ⛔ = rejected, **do not retry**

| knob | verdict | evidence |
|---|---|---|
| `soft_count_mode: sum` (shipped) | 🟡 wins LOOSE, loses TIGHT -- **and the reason is now known**, 2(y) | AP +0.0253..+0.0064 loose / -0.0572..-0.0933 tight. The gradient sits at the boundary, 200-440 ranks from the cut when the cap is tight |
| `soft_count_mode: uniform` | ✅ **the tight-cap tool**, and 2(y) says why: a flat slope declines to aim | ViTB16 AP **+0.0087 tight** (only arm above the floor) / **-0.0091 loose**. `uniform1` -0.0754 -> +0.0030 |
| `soft_count_mode: margin` | ⛔ **NEVER RUN, AND NOW REPRICED DOWNWARD** (2026-09-01). Still staged as `margin2` (432 runs) but it windows the BOUNDARY, and the boundary is measured to carry **exactly 0.0000** of the gradient at the cut. 🛑 **Run `cutwin1` first** | cosine **0.989** to `tralo` on real features, so 432 runs would mostly reproduce `tralo`. FRAMEWORK 2(z12) |
| `soft_count_mode: cut` (**NEW**, `tralo_cut`) | 🟢 **BUILT, GATED, STAGED as `cutwin1`** -- the fix 0-NOW derives. Value stays exactly `sum_i p_ic`, only the gradient weight moves to a window on rank K, width in ITEMS. ⚠️ **NOT predicted to win** -- aiming is necessary, not sufficient | mass at the cut **0.0001 -> 0.3486** (pooled, 24 run-class pairs, real features). Chunked gradient == full-N exactly (maxdiff 0.00e+00). `flag_live` md5-distinct on every binding seed. 423 tests, `audit_config`, `smoke_arms --matrix` all green |
| `tralo_st` (hard-count value fix) | ❓ **NEVER RUN** -- same campaign | isolates VALUE from PLACEMENT |
| `straight_through` | ✅ keeps count value exact | -- |
| `constraint_grad_mode: normalize` | ✅ **required** -- `clip` gives a ~20x dose spread across duals | `check_parity` refuses `clip` |
| `--constraint-fp32` | ✅ removes the FP16 skipped-step dose loss | iwc3 lost 328/1044 without it |
| `tralo_head` (head-only) | ⛔ 1.7x floor, tie uninformative; masking does not freeze the backbone (90.4% step) | |
| `tralo_ortho` (CE-orthogonal) | ⛔ delivers **0.0%** of its guarantee in 16/16 conditions | `ortho_survival` |
| `tralo_coin` (random direction) | ❓ never run -- **the control** for any placement claim | in `launch_margin2.sh` |
| penalty-shape variants | ⛔ FRAMEWORK 2 | all measured worse |
| more constraint steps | ⛔ **worse** | 2(c) |
| dedicated constraint optimizer | ⛔ | 2 |
| joint objective | ⛔ overfits, -0.067 AP | |
| undershoot hinge | ⛔ not budget-equalized; +16.3% free fill | |
| finer granularity (LLP) | ⛔ refuted | |
| KL anchor | ⛔ deleted from the pipeline | |
| `select` arm | ⛔ worst measured, -22 items | |
| `rank` / `beta` arms | ⛔ null / rejected | |
| cut-centred count `sigma((p-tau_K)/T)` | ⛔ **CLOSED BY ALGEBRA** -- counts items above the K-th largest = K-1 for ANY model. Detaching `tau` gives a gradient but was not shown to measure the violation | `margin_window` docstring; re-checked 2026-08-30 |

🎯 **The next knob is `margin` + `st` + `coin`** -- `docs/launch_margin2.sh`,
**432 runs, 12 cells**, re-validated 2026-08-30 (`gen_campaign` emits 432,
`check_parity` PASSES), never fired. `margin` is the only untested corner of
the count-function 2x2 and the only arm whose per-item gradient is not a
function of `p_ic` alone -- every other penalty this project ships has the form
`f(sum_i p_ic)`, whose logit gradient `f'(S) p_ic(1-p_ic)` is a monotone map
and therefore **cannot move an item across another on the direct channel**.

Its cap grid is a **matched 2x2**, which is what makes it answer the regime
question rather than just adding cells:

| tag | K (cls 2 / cls 7) | budget | what is pinned |
|---|---|---|---|
| `L30_G50` | 111 / 137 | K/n=0.30 | the DISTRIBUTION across groups |
| `L50_G30` | 111 / 137 | K/n=0.30 | only the TOTAL |
| `L80_G95` | 296 / 364 | K/n=0.80 | the DISTRIBUTION across groups |
| `L95_G80` | 296 / 365 | K/n=0.80 | only the TOTAL |

Each row-pair imposes the **same total budget through a different scope**, so
scope is isolated with tightness held fixed. ⚠️ 7 of 14 per-group ceilings are
K=0, and a zero ceiling binds however much slack the sum has -- so
"global-binding" never means the local scope is off. Say "pinned vs free
distribution", not "local vs global".
⛔ Do NOT add `L30_G30`: at `L30_G50` the global K=185 sits above the local sum
111, so the global term is INERT and the two tags are ONE cap level.

---

## 3. THE STANDING RULES THIS WORK IS HELD TO

1. **Never idle.** A campaign running is not a reason to stop; it is a reason to
   do the cheap offline work beside it.
2. **Cells, not seeds.** 4 seeds cannot resolve any of these effects (46-91
   needed). Everything rests on sign consistency across cells. **>= 9 cells** for
   a `***`, and **>= 10** if more than a couple of contrasts are tested.
3. **Pre-register ONE primary contrast** before scoring. 20 contrasts at 9 cells
   still cannot survive BH. This is the cheapest fix in the project.
4. **Always quote the `tralo_reseed` floor beside any win.** A 6/6 sweep is not
   evidence when the RNG floor also sweeps 6/6.
5. **Always quote macroF1 beside ccF1.** ccF1 alone hides the uncapped damage.
6. **Read the logs, but never compare counts across arms from
   `training_log.csv`** -- the schemas differ (76/16/15/14 cols) and trained arms'
   logged counts disagree with their predictions. Use
   `final_predictions_raw.csv`. FRAMEWORK 3(0c).
7. **md5 the raw predictions** before reading any metric (`_raw` = model, plain
   = allocator).
8. **Update this file and FRAMEWORK 3(0) at the end of every session.** A knob
   that failed goes in the ledger so it is never retried.

---

## 4. THE QUEUE -- in priority order

Work top-down. When one finishes, score it, update sections 1-2 of this file
and FRAMEWORK 3(0), then start the next.

0. 🟢 **RUNNING: `equaldose1`** (216 runs, dsisco01 GPU 3, pin `10d37518`).
   Is the dominance claim a 3.4% head start? `tralo` and `alm` attempt 29
   constraint steps per run, `fioretto` and `hounie` 28, at identical
   `constraint_epochs: 29` -- verified at the gradient level (epoch-1 grad norm
   3.09 vs **0.0**). `tralo_lam0` starts lambda at 0 so its first step carries a
   zero gradient exactly as theirs does. **The first thing to check on the
   first completed run is `Grad_Norm` at epoch 1: 0.0 for `tralo_lam0`, ~3.09
   for `tralo`. If not, the arm is inert and the campaign is void.**
   ✅ **VOID CHECK PASSED 2026-08-30 on the first completed run**
   (`MobileNetV2/L80_G95/tralo_lam0/seed_1`). Epoch-1 `Grad_Norm`:
   `tralo` **2.16** (steps), `tralo_lam0` **0.0**, `fioretto` **0.0**,
   `hounie` **0.0** -- the control now matches the duals exactly. And it is
   NOT the null: its `Lambda_Global` rises 0.025 -> 0.05 -> 0.075 and it steps
   from epoch 2 (grad 18.69), where `tralo_null` stays 0.0 forever.
   ⚠️ 8 independent (model, seed) units, so only 8/8 (p=0.0078) is significant;
   7/8 is p=0.0703 and is a DIRECTION. Say which was met.
1. 🟢 **STAGED, LAUNCHES ON THE FIRST FREE GPU: `cutwin1`** (56 runs,
   `~/optloss-cutwin`, pin `63dd8c6b`, grabber armed). The SMALL validation of
   the 0-NOW fix, deliberately not a grid. MobileNetV3 x {L30_G50 (K/n=30%),
   L90_G95 (K/n=90%)} x {`tralo_cut`, `tralo`, `tralo_uniform`, `tralo_null`,
   `tralo_reseed`, `clip`, `focal_clip`} x 4 seeds, `normalize` so the arms
   differ in DIRECTION and not in dose.
   ⚠️ **2 cells, so it CANNOT reach significance on any metric** -- the
   generator says so itself. It is a mechanism check, not a verdict, and must
   be reported as direction + per-cell consistency only.
   **What it must show before anything is built on it**, in this order:
   `dose_landed` 100% and dose-matched; `flag_live` md5-distinct from `tralo`
   on the real runs; `log_health` for collapse/divergence and the count
   trajectory vs K; then `full_panel --control clip` reading its CONSTRAINT
   DOSE block on the FIRST completed runs; then `boundary_probe --control
   tralo_null` against the `tralo_reseed` floor.
   🔑 **THE PRE-REGISTERED PREDICTION -- CORRECTED 2026-09-01, and the
   first version was WRONG.** It said `tralo_cut` would move the capped COUNT
   more at tight caps. Measured (2(z13)): `sum` already moves the count **4.02x
   / 2.94x the reseed floor** at tight caps, because the hard count is a
   BOUNDARY readout and the boundary is exactly where `p(1-p)` peaks. That
   prediction tested the wrong quantity -- CLAUDE.md rule 3, caught by running
   the check before the GPU. The correct one:
   > **TIGHT (L30_G50)**: `tralo_cut` changes the **EMITTED top-K set** by
   > more, relative to the `tralo_reseed` floor, than `tralo` does -- read from
   > `boundary_probe --control tralo_null` on `final_predictions.csv`, in net
   > items. Its HARD COUNT should move LESS than `tralo`'s, not more, because
   > the cut window deliberately takes gradient off the boundary.
   > **LOOSE (L90_G95)**: `tralo_cut` and `tralo` are approximately THE SAME
   > ARM (cosine 0.90 on MobileNetV3; `sum` already carries 0.15 of its
   > gradient at the cut there) and must behave alike.
   🛑 **THE LOOSE HALF IS A BUILT-IN NEGATIVE CONTROL** (2(z14)), which
   makes this two-sided and hard to satisfy by chance: **a `tralo_cut` that
   "wins" in BOTH regimes is reading noise, not the mechanism.** If the emitted
   set does not move at TIGHT relative to the floor, the aim fix is dead and
   cluster C closes with it.
   ⚠️ **MobileNetV3 is the WEAK backbone for this on purpose**: its tight
   `cut_window` mass is 0.380 against ViTB16's 0.765, so this is the
   conservative test -- and a null here must NOT be promoted to ViTB16 without
   running ViTB16.
2. 🔴 **ViTB16 LOOSE, from 2 cells to >= 6.** `loosevit1` already exists, is
   100% dose, md5-clean, single `code_version`, carries `tralo_null` +
   `tralo_reseed` + both clippers -- and on it **`tralo` is positive on every
   metric including macroF1** (AP +0.0064, ccF1 +0.0017, macroF1 +0.0021, all
   2/0) against a reseed floor that is NEGATIVE (AP -0.0113, macroF1 -0.0366).
   That is the best-looking result in the project and it sits on **2 cells,
   min attainable p = 0.500, NOT CALLABLE**. More loose cap tags on ViTB16
   (L85, L95, plus the matched `L95_G80`) is the cheapest route to a callable
   headline. **Highest value per GPU-hour available.**
3. ⏸️ **`vitdom1` -- HELD pending the 0-PRE decision.** 240 ViTB16 runs
   pointed at a paper section that may not exist. Ready and validated; do not
   launch until A/B/C is chosen.
   🔴 **`vitdom1`** (`docs/launch_vitdom1.sh`, 240 runs, 6 cells, validated
   2026-08-30). **ViTB16 has never run a single rival dual on iwildcam**, so
   the dominance claim cannot be reproduced on the pre-registered headline
   backbone. Six LOOSE cap tags, five distinct budgets plus the `L80_G95` /
   `L95_G80` scope pair at an identical K=296. Also takes `loosevit1`'s
   NOT-CALLABLE 2-cell positive to 6 cells. Deliberately loose-only, and
   deliberately without per-family nulls -- the header says why.
4. 🔴 **`margin2`** (`docs/launch_margin2.sh`, 432 runs, 12 cells, validated
   2026-08-30, pre-registration fixed). Now carries a falsifiable prediction
   from 2(y): gain in the 6 LOOSE cells, none in the 6 TIGHT ones.
   ⛔ Blocked on a GPU, not on readiness.
5. ✅ ~~ViTB16 needs rival duals~~ -- **this is now queued as `vitdom1`, item 2.**
   Kept here only so the hole is not re-discovered: `hounie`, `fioretto`, `alm`, `danits_lp` have
   **never run on ViTB16 on iwildcam** -- they exist there only in the dermmnist
   `vit_diag`/`vit_ceskip` campaigns, which are 86/97 pending on a dataset that
   is removed from disk. So the `dom1` dominance claim **cannot be reproduced on
   the pre-registered headline backbone** without new GPU time. This is the
   single biggest hole in the dominance story.
6. ✅ **`dom1b` -- DONE and scored.** 192/192, all gates green. The ccF1 lead reproduces on RegNetY400MF (2.49x the floor) but the **AP and AUROC lead does NOT** -- `tralo` is 4th and 3rd, both below its own reseed floor, with `alm` first. Confounded with the numeric regime (Blackwell bf16 vs Quadro fp16), so it is scored standalone. Nothing in it is significant and nothing could be: 4 warm-up units, sign floor p=0.125. FRAMEWORK 2(z5).
7. 🟡 **Unequal L:G ratios beyond L95_G80** -- `margin2`'s matched 2x2 covers two
   budgets; `L50_G20` / `L70_G40` would extend it.
8. 🟢 **`fmow` images (~21k)** -- the only route to dataset #2. Needs the user's
   go-ahead for the download.

⛔ **Do NOT re-run ViTB16 tight caps.** `vitu1` is complete, 100% dose, and says
`tralo` is 6.6x WORSE than the RNG floor there (AP -0.0933 vs -0.0142). 2(y)
explains why and predicts no count function fixes it. `iwc2` is also ViTB16 tight
but ran at **74.6% dose** (fp16 without `--constraint-fp32`) -- drop it.

---

## 5. RESUME PROTOCOL -- what to read, in order

A fresh session with no context should do exactly this:

```bash
# 1. state of the world -- 60 seconds
cat docs/MISSION.md                      # this file: goal, ledger, queue
sed -n '/^### 3(0)/,/^### 3(1)/p' docs/FRAMEWORK.md   # the live status board

# 2. what is running RIGHT NOW
for h in dsisco01 dsisco02; do ssh $h 'nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | while IFS=, read -r u p; do echo "$(ps -o user= -p ${p// /} 2>/dev/null)"; done | sort | uniq -c'; done
ssh dsisco01 'cd ~/optloss-domb && ~/anaconda3/envs/optloss/bin/python -m scripts.rig_status'

# 3. progress of every campaign
ssh dsisco02 '~/anaconda3/envs/optloss/bin/python - <<PY
import glob,json,os,collections
seen=collections.defaultdict(collections.Counter)
for t in sorted(glob.glob(os.path.expanduser("~/optloss-*"))):
    for c in glob.glob(os.path.join(t,"results","*","*","*","*","*","seed_*","config.json")):
        p=c.split(os.sep); seen[p[p.index("results")+1]][json.load(open(c)).get("status","?")]+=1
for k,v in sorted(seen.items(), key=lambda kv:-sum(kv[1].values())):
    print("%-14s %s"%(k,dict(v)))
PY'

# 4. gates, before ANY launch
python -m pytest tests -q          # must be 402 (bump when you add one)
python -m scripts.audit_config
python -m scripts.smoke_arms
```

**Then pick up item 1 of the queue that is not already running.**

### Reading a landed campaign, in this order and no other

```bash
python -m scripts.dose_landed <root>                        # FIRST. always.
python -m scripts.full_panel --campaign <root> --control clip
python -m scripts.full_panel --campaign <root> --control tralo_null
python -m scripts.family_split --campaign <root> --families tralo fioretto hounie alm
python -m scripts.log_health <root>                         # read 3(0c) first
python -m scripts.order_probe --campaign <root> --arm tralo
```
Then: **per-cell breakdown, never a pooled digit**; the **reseed row** beside
every win; **macroF1 beside ccF1**; and an **exact sign test** with the cell
count stated.

---

## 6. WORKING IN PARALLEL

The GPU is the scarce resource; context is the other one. While a campaign runs,
delegate independent read-only analysis to subagents (the user has standing
approval for this) and keep only the conclusions:

- one agent per landed campaign that has never been scored
- one agent per offline probe that prices a direction (`ceiling_screen`,
  `paired_noise`, `dataset_screen`, `factorial_control`, `straddle_probe`)
- one agent to re-audit a defect class already found once (inert flags,
  incommensurable logs, unequal dose, pooled digits hiding per-cell reversals)

Never delegate a launch, a `git push`, or anything that writes to `src/`,
`configs/` or `main.py` while a campaign is running.
