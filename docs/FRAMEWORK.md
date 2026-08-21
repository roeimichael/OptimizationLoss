# THE FRAMEWORK

**One file. Read it before proposing, running, or scoring anything.**
Everything else in `docs/` is history. If this file and any other document disagree, this file wins.

---

## 0. Where we actually are (2026-08-18)

We set out to build a **dual-loss** method that caps a class's prediction count while still
optimizing that class, beating both the dual baselines (Fioretto-LDF, Hounie-RCL) and the
post-hoc clippers.

**Current status: it does not beat a plain post-hoc clipper anywhere we have measured
-- but the SIZE of that statement is smaller than it used to read, and the p-values
behind it are dead.**

The DIRECTION stands and reproduces exactly. Re-scored on 2026-08-19 from
`evidence/predictions_mcbar_multiclass_2026-08-18.tar.gz` with the current scorer:
macro-F1 **+0.0022**, accuracy **+0.0015**, cc-F1 **-0.0035** -- all four decimals
identical to the archived record. A baseline with no constraint training at all still
beats `focal_clip` by more than our method does.

⚠️ **The SIGNIFICANCE does not stand, and cannot be recovered by any code in this
repository.** Three separate reasons, each sufficient on its own:

1. **"Pooled over 48 seed-pairs" violates this file's own rule** (section 3.5:
   *never pool across levels, backbones, or datasets; summaries count cells*), and it
   is precisely the pseudoreplication `full_panel.py` now rejects -- measured to
   inflate type-I error to 11-22% under the null. Under the current cell-level unit
   the same data gives p=0.844 / 0.688 / 0.438, not 0.478.
2. **Half the data does not exist.** 24 of those 48 pairs are `mcbar_regnet`, which
   has configs in the provenance archive but **no predictions in either tarball**.
3. **The "significant loss on eight other metrics" evaporates.** Re-scored on the
   half that survives, all eight -- cc-F1, AP, AUROC, ECE, Brier, NLL, macro-R,
   ConfGap -- return `tie` at BH q >= 0.573 (AUROC, the closest, is q=0.573 on a
   raw p=0.0625 and reads `lean loss`). That is structural, not luck: at 6 cells
   the exact Wilcoxon floor is 2^-5 = 0.031, and BH over the metric family pushes a
   lone 6/0 metric to q = 0.41.
4. 🚨 **29% of the `tralo_uniform` runs in that campaign were allocated by a
   DIFFERENT algorithm from `clip`.** Re-running the archive through the current
   scorer, the new allocator check reports *"tralo_uniform: 7 of 24 runs fell
   through to the LP fallback"* -- the Phase 3b defect fixed in `eb6b8897`, where
   the local fill spent the global budget, made the greedy allocation infeasible,
   and handed those runs to `_fallback_lp` while `clip` kept the greedy. So the
   comparison this section rests on is not purely arm-vs-arm: for 7 of 24 pairs it
   is also greedy-vs-LP. The flag was recorded on every run since the pipeline was
   written and read by nothing until now.

🛑 **So the honest headline is: the direction is a measured tie-or-loss, reproduced
exactly; no significance claim in this section survives, and nothing has been
re-measured end to end under the current protocol** (`results/` does not exist on
either machine). Section 4's "do not run" list inherits that caveat -- it is built on
this section.

There is a structural reason, and it is the single most important thing in this file:

> **Post-hoc allocation thresholds the ranking at the budget. The score IS the ranking.**
> cc-F1 is precision@K rescaled; AP is the integral of precision@k. A gradient that is a
> function of the *aggregate count* cannot change *which* items are on top, so it cannot
> beat post-hoc, structurally. A count says how many, never which.

Every arm that varied the count penalty tied for exactly this reason. **A winner must act
per-item at the operating point, or it must operate in a regime where post-hoc is not optimal.**

---

### The honest n is THREE, and it is the dataset count

Averaging seeds within a cell fixed the seed-level dependence. It did not fix the
one above it: **every cell inside one dataset shares that dataset's fixed test
set and the K derived from it.** Adding a backbone or a cap level buys resolution
on the cells we ran; only a new dataset buys an independent test set. The scorer
treated all cells as independent draws regardless, and `gen_campaign` told an
underpowered campaign to "add a backbone, a dataset or a cap level" as if the
three were interchangeable.

`full_panel.py` now prints a **dataset-clustered readout** beside the per-cell
table: the same deltas averaged to one value per dataset, tested by exact sign
flip over all `2^D` assignments. Re-scored on the archived `mcbar` campaign
(6 cells, 3 datasets, both clippers in-campaign):

| comparison | metric | cell p | clustered p | per-dataset |
|---|---|---|---|---|
| focal_clip vs clip | AUROC | 0.0312 | **0.2500** | derm +0.0016, oct +0.0071, tissue +0.0062 |
| focal_clip vs clip | NLL | 0.0312 | **0.2500** | derm -0.461, oct -0.833, tissue -2.067 |
| focal_clip vs clip | AP | 0.8750 | **1.0000** | derm +0.0107, oct +0.0075, **tissue -0.0184** |
| tralo_uniform vs clip | ccF1 | 0.4375 | **0.7500** | derm -0.0025, **oct +0.0042**, tissue -0.0122 |
| tralo_uniform vs clip | macroF1 | 0.8438 | **0.7500** | derm +0.0166, oct +0.0064, tissue -0.0164 |

**Every metric that read p=0.031 at the cell level reads p=0.250 clustered --
because 0.250 is the floor.** With `D` datasets the exact two-sided sign-flip
minimum is `2^(1-D)`, so at `D = 3` it is `0.25` and at `D = 2` it is `0.50`.

🛑 **No campaign this project can run will ever reach `p < 0.05` on the
generalization unit.** Three datasets is the entire universe, and three
same-signed clusters is the most extreme outcome available. Generality here is a
DIRECTION claim across datasets and a per-dataset consistency claim -- never a
significant one. `gen_campaign` prints this floor before a campaign launches and
`full_panel` prints it beside every comparison.

⚠️ It also shows how often the sign is NOT consistent. On `mcbar`, tissuemnist
carries the opposite sign to the other two on AP, ccF1, macroF1, macroP and acc
for both arms -- so the pooled cell-level mean is averaging a disagreement, which
is mistake pattern 15 operating one level up.

## 1-pre. THE HEADLINE BACKBONE IS **ViTB16**, decided 2026-08-20, A PRIORI

Recorded here BEFORE any hyperparameter sweep has been read, because when it was
decided is the only thing that makes it legitimate.

Roei's call: the headline backbone moves from MobileNetV3 to **ViTB16**. Any of
the four the paper claims is defensible, so choosing among them is a design
decision, not a result -- **provided it is made in advance.** Running all four
and headlining whichever one TraLO happens to win on is selection on the
outcome, and it is the mechanism behind more than one retraction already in
section 2d of this document.

So the rule that follows from it, and it binds:

🛑 **Every TraLO hyperparameter result from here is on ViTB16.** If a sweep is
later run on another backbone and TraLO does better there, that is a
GENERALIZATION check on a fixed headline -- it does not promote that backbone to
the headline. Changing the headline after seeing results re-opens exactly the
garden of forking paths this note closes.

⚠️ Practical consequence: ViTB16 is ~28 s/epoch, so a 30-epoch run is ~14
minutes and a full sweep is days on one GPU. Exploration therefore runs in two
stages -- **scout at 1-2 seeds to locate a region, then CONFIRM at 4 seeds
inside that region only.** Scouting output is not a result and must never be
quoted as one; only the confirmation stage produces cells.

## 1a. ⚠️ PROVENANCE OF THE PROTOCOL'S OWN NUMBERS (audited 2026-08-20)

**A frozen protocol is not a validated one.** Section 1 fixes these values so
that arms are comparable to each other -- that is what it is for, and it works.
It does NOT establish that any of them is a good value, and this document has
been read as if it did. Audited against every claim in this repository:

| value | set to | what actually backs it |
|---|---|---|
| `total_epochs` | 30 | ⛔ **nothing.** The stated reason is "equal compute: 30 optimizer epochs on both sides", but equal compute constrains the RATIO (`warmup_posthoc == warmup_trained + constraint_trained`), not the total. 30 could be 60 or 300 and still be equal. The paper's own protocol was 50 + 300. |
| `trained_warmup` | 1 | ⚠️ **partly.** warm-up 50 saturating CE IS measured (242/242 feasibility, section 3). "warm-up 5 is a dead zone" is **asserted twice in this file and measured nowhere** -- there is no campaign, no receipt, no corpus row. It is the stated reason for choosing 1 over anything between. |
| `constraint_epochs` | 29 | ⛔ **residue.** It is `total_epochs - trained_warmup`, never chosen. |
| `stable_count_threshold` | 31 | ⛔ **structurally unreachable.** It early-stops after N CONSECUTIVE satisfied epochs, and there are only 29 epochs. It can never fire. `fioretto_alm/train.py:48` admits this in a test comment ("5 against 31, low enough that the early stop would actually fire"). It is a disabled knob written as a live setting. |
| `constraint_grad_clip` | 1.0 | ⚠️ **the clip is load-bearing (measured); the VALUE is not.** protocol.yml itself says "Sweep 0.3 / 1.0 / 3.0 to get a dose-response curve" -- that sweep has never run. It was hardcoded in four trainers before it was a config key at all. |
| `lambda_step` / `lambda_global` / `lambda_local` | 0.05 / 0.01 / 0.01 | ⛔ one mention each, no sweep. Largely moot for MAGNITUDE (the clip cancels it) but NOT for the scope mix -- with a global scope plus 3 local groups the gradient is a weighted SUM, and these weights set the mix, which survives the clip. |
| `initial_rho` / `rho_target` | 0.5 / 100.0 | ⛔ no sweep. Same magnitude/direction split as above. |
| `lr` / `lr_constraint` | 1e-4 | ⚠️ the CONSTRAINT that they be equal is well-measured (the LR trap fabricated -16.7pp). The VALUE 1e-4 is inherited. |
| `dropout` | 0.3 | ⛔ no sweep. |
| `batch_size` | 64 | ⛔ **not mentioned once in this document.** |
| `hounie_alpha` | 10.0 | ⛔ **not mentioned once**, and it sets hounie's analytic lambda CEILING at `2*alpha*mean_l` -- the single number that decides whether that arm can reach a dose comparable to the others at all. |
| `fioretto_step_size` | 0.005 | ⛔ no sweep. |

🛑 **So: do not cite section 1 as evidence that a value is right.** Cite it as
the reason two arms are comparable. When a result depends on a value in this
table, the value is a free parameter that happened to be frozen, and saying so
is the honest report.

⚠️ **And nothing in this table can be swept until the noise floor is fixed.**
The same arm, same seed, same config scored macro-F1 0.6709 and 0.7015 on two
runs -- 0.0306 apart, ~18x the headline effect. An HP sweep read through that
much noise measures the kernels, not the hyperparameter.

## 1b-pre. THE INSTRUMENT WAS BROKEN UNTIL 2026-08-20. Four findings that gate everything.

### (1) The noise floor was 0.0358 macro-F1 -- 21x the effect being measured

Three runs of the SAME arm (`clip`), same seed, same config, same GPU, back to back
(`scripts/variance_probe.py`):

    F1 (Macro)  0.6524 .. 0.6882   SPREAD 0.0358   sd 0.0181
    warm-up     1178.5 / 1176.4 / 1176.4 s   (each really retrained)

Against a headline TraLO-vs-clip effect of 0.0017. Measured WITH
`cudnn.deterministic`, `benchmark=False` and `CUBLAS_WORKSPACE_CONFIG` already set,
so none of those was the answer. More seeds do not help: averaging shrinks the
standard ERROR, and the floor is what each draw is drawn FROM.

**Localised** by `scripts/bisect_determinism.py`, four processes per stage: model init
identical, batch order over a whole epoch identical (NOT the DataLoader), forward loss
at step 0 identical, **gradients at step 0 different in all four processes**. With the
fused SDPA backends disabled the same four agree bit for bit.

**The fix is one line, and the trap is that it reads backwards.**
`torch.use_deterministic_algorithms(True, warn_only=True)` is not a gentler setting --
PyTorch reads `deterministicAlgorithmsWarnOnly()` INSIDE the attention backward and
takes the NONdeterministic branch when it is true. `warn_only=False` gives one hash
across four processes with the fused kernel still on, at 5.5% (54.70s -> 57.72s per 126
steps; disabling the fused backends instead costs 62.97s).

**Verified**: three repeats now give one predictions md5 (`71aba83c`) and one weights
md5 (`df387dd2`), every metric spread 0.0000. **Floor 0.0358 -> 0.0000.**

⇒ Every arm-vs-arm number measured before this sits under a 0.0358 floor. It also means
**liveness is now a HASH COMPARISON, not a hypothesis test**: identical md5 is not a
small effect, it is no effect, at n=1.

### (2) The arms were not getting the same dose -- ~20x apart, invisible to every gate

`results/vit_diag` seed 1, same warm-up model, all three configs saying
`constraint_grad_clip: 1.0`:

    tralo     raw grad norm 0.638 .. 1826.5     clip binds  6 of 7
    fioretto  raw grad norm 17,667 .. 80,827    clip binds 18 of 18
    hounie    raw grad norm 0.005 .. 0.1105     clip binds  0 of 29

At the last epoch fioretto's constraint loss is 4390.838 and hounie's is 0.004204 --
1.04e6 apart. Faithful to each paper (hounie divides the violation by N, fioretto sums),
but the CONSEQUENCE is not a method difference: tralo and fioretto each deliver a
unit-norm step, hounie delivers its raw ~0.05-norm one. Fixed by
`constraint_grad_mode: normalize` in `src/training/constraint_step.py` -- ONE
implementation for all four arms, because four hand-rolled copies are how this drifted.

**And fioretto silently ran a 62%-length constraint phase**: 10 of 29 epochs lost to
non-finite gradients (6 NaN + 4 inf -- the RAW count; `dropna()` first hides the NaN and
reports 4), while writing `status: completed`. `constraint_fp32: true` decouples the
constraint pass from the CE loss scale. ⚠️ fp32 doubles the chunked-forward memory:
`constraint_chunk_size: 256` OOMs on a 22 GB card at ViTB16, 128 fits.

### (3) TraLO's constraint phase moves the count the WRONG way

`vit_diag` tralo seed 1, K=67: hard count **125 -> 121 -> 251 -> 250 -> 353 -> 205 -> 281**.
It starts the phase at 1.9x budget and ends at 4.2x, never satisfied on any of 29 epochs
(proved three ways, including that all 29 lambda ratchets fired). Final Precision@67 on
the capped class: **tralo 50/67 vs clip 57/67**. The constraint phase makes the
classifier WORSE, and lambda is a clock -- `0.01 + 0.05k` exactly, carrying no
information beyond "still violated".

✅ **ATTRIBUTED 2026-08-20 with a matched lambda=0 control** (`vit_diag .../null/seed_1`:
same cached warm-up `46e3754db799`, same allocator, only lambda differs). The count
gap splits, and so does the quality:

| arm | AUROC | raw macroF1 | **alloc macroF1** | count (K=67) |
|---|---|---|---|---|
| `clip` (warm-up 30) | 0.9601 | 0.6939 | **0.6709** | 130 |
| `null` (warm-up 1 + 29, lambda=0) | 0.9598 | 0.7010 | **0.6625** | 197 |
| `tralo` | 0.9610 | 0.7052 | **0.6720** | 267 |

⇒ of tralo's +137 count over clip, **+67 is the warm-up-1 training path and +70 is the
constraint**. ⇒ the constraint is worth **+0.0094** alloc macro-F1 over its own control,
but that control sits **-0.0084 below clip**, so TraLO spends its entire gain repaying a
deficit the regime created and lands at **+0.0011 = a tie**. 🚨 **Neither half is
attributable without the control** -- this section previously read the raw trajectory as
if the constraint owned all of it.

✅ The two-allocator confound is **bounded at ~0.0014**: `lp` and `clip` are the same
model (raw identical, 0.6939 / 261/130) and differ only in allocator, 0.6723 vs 0.6709.
Same order as tralo's margin over clip, which is why that margin is noise.

### (4) The penalty shape is nearly inert where the runs actually live

3,558 logged operating points from 428 archived dermmnist runs. The shape is EXACTLY
inert when every scope has the same relative excess (verified cosine 1.000000000000),
and the observed dispersion `max(u)/min(u)` has median **1.5x**. Rotation vs a linear
hinge: median cos **0.990 (8.2 degrees)**, q10 0.947. The 167:1 starvation is real
arithmetic but needs one scope 8x over while another sits near 58% -- which occurs in
**0 of 3,558** epochs. The dominant relative-weight skew is the shape-independent
**1/K** factor: group 2 (K=12) takes 72% of the squared gradient norm under ANY shape.

⚠️ Those 3,558 points are all SINGLE-capped-class. First multi-class measurement
(classes 2+4, L30_G20): u_2 = 4.50 and u_4 = 1.71, a **2.6x** dispersion -- materially
larger.

✅ **MEASURED 2026-08-20 for multi-class, against a lambda=0 control** (4 epochs, n=1,
`penalty_shape` now a knob). The shape is **live there, and what it controls is a
SEE-SAW**: deltas vs the control (233/201) --

| shape | d class 2 | d class 4 | AUROC | raw macroF1 | alloc macroF1 |
|---|---|---|---|---|---|
| `rational_bounded` | **+197** | -161 | -0.0103 | -0.0486 | **+0.0414** |
| `squared` | +112 | -77 | -0.0024 | -0.0295 | +0.0195 |
| `linear` | **+86** | -75 | -0.0021 | -0.0180 | +0.0255 |

**Every shape pushes one capped class down and the other up.** The penalty is a sum of
independent per-class terms, but the **softmax makes the capped classes compete**, so
mass pulled off one lands on the other -- and the class that should resist is the one
this shape starves. Shape sets the see-saw's SIZE in exactly the predicted order.
⛔ **It is not a fix**: total excess ROSE at all three shapes (+103 / +138 / +120). Keep
`penalty_shape` as the dial on the coupling, never as a remedy. And note the two views
disagree -- worse on every allocation-free measure, better after allocation.

### (8) What "CE saturates" MEANS: the boundary probability goes to 1 and the penalty loses its grip

A transductive count penalty differentiates `sum_i p_ic`, so an item's share of the
gradient scales with `dp/dlogit = p(1-p)`. The cut is decided by the **K-th ranked item**,
so the penalty can only move the cut when `p(1-p)` at that item is not vanishing.

Measured on dermmnist x ViTB16, 4 seeds per cell, penalty vs its own lambda=0 control:

| cell | class | p at the K-th item | `p(1-p)` | `d capF1` |
|---|---|---|---|---|
| `L30_G20` (4-epoch model) | 2 | 0.9730 | 0.0258 | **-0.012, 0/4 seeds** |
| `L50_G30` (4-epoch model) | 2 | 0.9389 | **0.0550** | **+0.008, 4/4 seeds** |
| `L50_G30` (30-epoch model) | 2 | **0.9990** | **0.0009** | -- |

⇒ a **2.1x** difference in the slope at the boundary is the whole distance between "no
signal at any shape or dose" and "signal at every seed". And converging the model drops
that slope by a factor of **60**.

🔑 **So "CE saturates and every method becomes identical" is this, stated mechanically.**
Saturation is the boundary probability approaching 1; at p = 0.999 the penalty's gradient
at the cut is 0.001. **Warm-up 1 is not an arbitrary protocol choice -- it is the setting
that leaves the largest reachable window**, and warm-up 50 is forbidden because by then
the window is shut. The archived "headroom grows with the cap" result is the same fact
from the other side: a looser budget puts the cut where the gradient can still act.

⇒ **`scripts/reachability.py` screens a cell before any GPU goes into it** -- but it must
be run on a model at the START of the constraint phase. Run on a converged model it says
OUT OF REACH everywhere, correctly and uselessly.

---

### (7) The two-allocator confound is EXACTLY ZERO -- except when an arm under-shoots

The trained arms post-process with `targeted_correction` (`src/utils/posthoc_adjustment.py`:
reduce over-limit, then fill under-limit, then local -- three sequential phases). `clip`
and `focal_clip` use `apply_allocation_heuristic` (`src/methodologies/heuristic/train.py`:
ONE joint pass over every (item, capped class) pair in descending probability, so the
capped classes compete for the same items). Different code, so every tralo-vs-clip number
in this project has carried a suspected confound.

**Measured 2026-08-21, item by item, same probabilities through both procedures:**

| run | raw counts vs budget | items differing | `d capF1` |
|---|---|---|---|
| mc29 `clip` / `null` / `tralo` (L50_G30) | both over | **0 of 2003** | 0.0000 |
| `seed2/3/4_bounded` (L30_G20) | both over | **0 of 2003** | 0.0000 |
| `seed1_bounded` (L30_G20) | **c4 = 40 UNDER K=45** | **12** | **+0.0149** |

⇒ **identical whenever every capped class is over budget, which is the normal case.**
They diverge only on the under-budget FILL path, and there the joint pass is better.
🎯 **So the confound is not general -- it is a penalty specifically on arms that
OVER-SUPPRESS a class below its own budget**, which is exactly what the shipped
`rational_bounded` shape does. Seed 1's -0.0149 for that shape is precisely the amount
the better allocator recovers.

⇒ **Comparisons where all arms stay over budget are allocator-clean and need no caveat.**
Where an arm under-shoots, re-score every arm through ONE allocator before reading it.
⚠️ Do NOT swap the trained arms' allocator mid-campaign -- it would make later seeds
incomparable to earlier ones.

---

### (6) At `L30_G20` a COIN does the same damage -- the direction carries no information

`constraint_random_direction` replaces the constraint gradient with a random vector of
the SAME norm: the dose is held exactly and only the information is removed. It is the
control that `separate_constraint_optimizer` never had -- that arm moved 8,900x further
and cost AP -0.0938, and nothing could say which half of the change did it.

dermmnist x ViTB16 x `L30_G20`, classes 2+4, 4 epochs, `d capF1` paired against the
same lambda=0 controls in every row:

| arm | mean `d capF1` | seeds positive |
|---|---|---|
| `rational_bounded` (the shipped penalty) | -0.0112 | 1/4 |
| `linear` | -0.0122 | 0/4 |
| **`randdir` -- a coin, same step size** | **-0.0093** | 0/4 |

**Indistinguishable.** At this cap the constraint does nothing a random step of the same
size would not do, and **no shape and no dose can fix that, because there is no signal to
tune.** This is the cleanest available statement of why ~13 penalty-shape arms and ~20
arms overall tied: they were all tuning the direction of a step whose direction does not
matter here.

✅ **AND IT IS CAP-SPECIFIC -- at `L50_G30` the coin FAILS and `linear` does not.**
Same control, same seeds, same cell:

| arm | mean `d capF1` | per seed | seeds positive |
|---|---|---|---|
| `rational_bounded` | -0.0185 | | 0/4 |
| **`linear`** | **+0.0078** (sd 0.0017) | +0.0104 +0.0069 +0.0070 +0.0069 | **4/4** |
| `randdir` -- a coin | -0.0130 (sd 0.0100) | -0.0243 -0.0139 +0.0001 -0.0137 | 1/4 |

**The distributions do not overlap**: `linear`'s worst seed is +0.0069, the coin's best
is +0.0001. ⇒ at this cap the penalty's DIRECTION carries real information -- a random
step of the same size actively hurts while the linear penalty helps, a separation of
0.021. It is corroborated by the mechanism: `linear` improves prec@K on class 2 at
**every** k from 20 to 150, which is a broad re-ranking and not an accident at the
budget. (Class 4 is noise in both directions, so the effect is one capped class, not
"the capped classes".)

🎯 **THE SHAPE AND THE CAP INTERACT, AND THE SHIPPED SHAPE IS NEGATIVE EVERYWHERE.**
A tight budget only reaches items the model already had right, so there is nothing for
a direction to earn; loosen it and a monotone penalty finds something a coin cannot.
⚠️ Still unestablished: this is ONE cell at 4 epochs against `null`. `clip` and
`focal_clip` at protocol length are the bar, and the gain does NOT show up in macro-F1
(+0.0015, 2/4 seeds) -- it is confined to the constrained classes.

---

### (5) No count trajectory is attributable without a lambda=0 control

At warm-up 1 the model is barely trained and every later epoch takes **126 CE steps
against the constraint's 1**. Measured with the penalty identically off (`L_Global=0`,
`Lambda_Global=0`, `Grad_Norm=0.0` on every row), the capped counts still swing
**242 -> 227 -> 324 -> 233** over four epochs, and **161 -> 278 -> 215 -> 177** at 29.
A count that moves by +-100 on its own cannot carry a claim about a treatment that
moves it by 70.

⇒ **Every campaign that will have its counts read carries the `_null` arm of its own
method** (`--arms all+null`, or `tralo_null`). It is the same warm-up, the same
allocator and the same seed with lambda set to 0, so it isolates the constraint and
nothing else -- and it doubles as a post-hoc clipper at equal compute with the
allocator held fixed, which is the one bar `clip` cannot provide without also
changing the allocator.

`scripts/dose_scan.py --with-null` runs it; `scripts/score_scan.py` prefers it as the
baseline row and says so loudly when it is absent.

---

## 1. THE PROTOCOL -- fixed, non-negotiable, applies to every run

Any campaign that violates a line here is invalid and gets deleted, not debugged.

### Regime

| knob | value | why |
|---|---|---|
| `warmup_epochs` | **1** for trained arms, **30** for post-hoc arms | warm-up 50 saturates CE before the constraint phase; warm-up 5 is a dead zone. Never interpolate. |
| `constraint_epochs` | **29** for trained arms, **0** for post-hoc arms | **30 total optimizer epochs on both sides -- equal compute.** |
| `lr_constraint` | **== `lr` == 1e-4** | unequal LR fabricated a -16.7 pp "finding" that was -1.7 pp when equalized |
| `stable_count_threshold` | 31 | |
| lambda toggle | **always on** | never `disable_lambda_toggle=True` |

`enable_ce_skip` and `alpha_kl` no longer appear here because the CE-skip and KL machinery
were **deleted from the pipeline** on 2026-08-18 (section 2f). They cannot be re-enabled by a config.

### Scope

- **Datasets: `dermmnist`, `octmnist`, `tissuemnist`.** Nothing else. No AIDER, no EuroSAT.
- **Backbones: `MobileNetV3` (headline), `MobileNetV2`, `RegNetY400MF`, `ViTB16`.** Nothing else.
  These are exactly the four the manuscript claims. `ShuffleNetV2`, `TinyCNN`, `SmallCNN` and
  `MediumCNN` were deleted on 2026-08-18 -- none appears in any `.tex` file, so no written
  result rests on them. `ViTB16` is the non-CNN check and is claimed in the paper, not optional.
- **Caps: at least two levels, always.** `L30_G30` and `L50_G50` minimum. A result from cells
  sharing one cap level has been retracted here **three times**. Headroom grows with the cap
  (0.024 at L20 vs 0.12 at L50), so never read a null at L20 as evidence against a method.
- **Seeds**: 1, 2, 3, 4.

#### The global cap does nothing at the tags we have always used

`L<local>_G<global>` sets the two scopes independently, but they are not independent in
effect. Local caps are per-group ceilings, so the most the model can ever predict for a
capped class is **the sum of the local caps**. A global cap at or above that sum can never
bind.

**Provenance of every number below**: measured on `dsisco01:~/OptimizationLoss/data/*/slice_1`,
re-derived independently on 2026-08-19 from `test_meta.csv` + `test_labels.npy` using the
pipeline's own `_round_to_K`. **The data is not on the Windows workstation** -- `data/`
there holds only `download_data.py`. `python -m scripts.verify_caps` run locally prints
`FAIL -- could not read the slice` for all three and exits 1, which is correct: a gate must
not pass on a dataset it never opened. Run it on the server.

| tag | dermmnist (MEL) | octmnist (drusen) | tissuemnist (GE) |
|---|---|---|---|
| `L30_G30` | global 67, local sum 67 -- **redundant** | global 75, local sum 76 -- binds | global 51, local sum 51 -- **redundant** |
| `L30_G50` | global 112 vs 67 -- **inert** | global 125 vs 76 -- **inert** | global 86 vs 51 -- **inert** |
| `L50_G50` | global 112 vs 111 -- **inert** | global 125, local sum 125 -- **redundant** | global 86, local sum 86 -- **redundant** |
| `L50_G30` | global 67 vs 111 -- **binds** | global 75 vs 125 -- **binds** | global 51 vs 86 -- **binds** |

So on the symmetric tags the global constraint is redundant almost everywhere, and on
`G > L` it is strictly inert. **Every result this project has produced was, in effect, a
local-cap-only result.** That is not wrong, but it is narrower than "global + local".

**Rule: to test the global scope, sweep `G < L`.** `L50_G30` is the cheapest tag that makes
the global cap the binding one on all three datasets while keeping a second cap level.
A campaign whose global caps are all redundant should say so rather than claim the
formulation was exercised.

#### The dermmnist test set shares lesions with its training set

**Receipt: `python -m scripts.check_lesion_leakage`** (server, or anywhere
`dermmnist_c_metadata.csv` is -- it needs the metadata CSV only, no images and
no GPU, because `StratifiedShuffleSplit` consumes `len(X)` and `y` so the exact
indices reproduce from the labels). It asserts its copy of `BASE_SEED` and
`TEST_FRACTION` against `create_slices.py` before reporting, so it cannot quote
a figure for a slice nobody trained on. These numbers were prose-only until
2026-08-19, and prose-only is how the ALM lambda stayed wrong.

**Measured 2026-08-19 on `dsisco01`, from `dermmnist_c_metadata.csv`, by replaying
`create_slices.py`'s own split with its own seeds.**

| slice | seed | test n | test images sharing a `lesion_id` with a TRAIN image | of the capped class (melanoma) |
|---|---|---|---|---|
| `slice_1` (the one every derm result uses) | 43 | 2003 | **776 = 38.7%** | **150 of 223 = 67.3%** |
| `slice_2` | 44 | 2003 | 787 = 39.3% | 150 of 223 = 67.3% |
| `slice_3` | 45 | 2003 | 790 = 39.4% | 161 of 223 = 72.2% |
| `slice_4` | 46 | 2003 | 743 = 37.1% | 147 of 223 = 65.9% |
| `slice_5` | 47 | 2003 | 801 = 40.0% | 161 of 223 = 72.2% |

HAM10000 photographs many lesions more than once: 10,015 images, **7,470 distinct
lesions**. `data/dermmnist/download_data.py` downloads *DermaMNIST-C* precisely
because it has corrected splits that keep a lesion whole -- its own docstring says
"fixed train/val/test splits that prevent same-lesion leakage across splits", and
measured, those splits have **exactly zero** lesion overlap across all three pairs.
`data/dermmnist/create_slices.py` then **pools train+val+test and re-splits 80/20
stratified on the label only**, which puts a different photograph of the same lesion
on both sides.

**Two consequences, and they pull in opposite directions.**

1. It is a shared confound **to first order, and only to first order**. Exposure is
   equal by construction: 30 optimizer epochs on both sides, and CE keeps running
   through every constraint epoch (`tralo/train.py:147`, inside the constraint loop),
   so no arm gets more opportunity to memorize than another. That is what makes the
   *paired* comparison survivable. It is **not** a proof: two arms with equal exposure
   can still convert memorization into test score differently, and the arm that departs
   less from the memorizing solution would keep more of it. Nothing here rules that out.

   What it invalidates outright is every **absolute** derm number -- cc-F1, AP, and the
   measured headroom of **0.0669** that the "there is room to win" argument rests on.

   It may also **compress** the gap: an item that is effectively memorized gets ranked
   correctly by both arms, so the arms can only differ on the ~61% that is not, and a
   shrunken denominator is one honest candidate explanation for why ~20 arms tied.

   **The test that settles both, and it needs no new training.** Replay the split to
   recover each test item's `lesion_id`, mark the 776 items that share one with a train
   image, and re-score any existing derm campaign twice -- once on the leaked subset,
   once on the clean 1,227. If the TraLO-vs-`clip` delta is the same on both, the
   confound is shared and the recorded nulls stand. If the arms separate on the clean
   subset, the null was a leakage artifact and the headline changes. Blocked only on
   having derm predictions to hand: `results/` is empty on both the workstation and
   `dsisco01`, and the only predictions that survive are inside
   `evidence/predictions_mcbar_multiclass_2026-08-18.tar.gz`.
2. It changes what the cap **means**. In the corrected split melanoma is
   **70/1227 = 5.7%** of test -- a genuine minority, matching the screening
   motivation the paper opens with. Pooling raises it to **223/2003 = 11.1%**, the
   figure the manuscript states. The leakage-free version is both more honest and a
   better fit to the paper's own story.

⚠️ **This is the headline dataset.** MobileNetV3 on dermmnist is the headline cell,
and dermmnist is the ONLY dataset with real groups, so it is the only place where a
local cap is a different constraint from the global one.

**DO NOT quietly re-slice.** Re-running derm on leakage-free splits invalidates every
derm result produced so far, which is most of the corpus. That is a call for Roei, not
a cleanup. Until it is made, the honest statement is: *the paired arm-vs-arm
comparisons stand; the absolute derm quality numbers are inflated by an unknown amount
and must not be quoted as achievable performance.*

⚠️ **tissuemnist uses the identical pooling pattern** (`data/tissuemnist/create_slices.py`,
same 80/20, same base seed) and **cannot be checked from what this repo stores** -- no
per-instance identifier survives into the slices. Whether TissueMNIST's source split
carries a donor/patient grouping needs checking at MedMNIST before its absolute numbers
are quoted either. octmnist is unaffected by this specific mechanism:
`scripts/prep_octmnist.py` keeps the official test split whole.

#### Only dermmnist has real groups

A local cap is a *different* constraint from the global one only if the groups
differ in class composition. `synth_group` is built by round-robin over array
order (`scripts/prep_octmnist.py:71`, `np.arange(len(y)) % 3`) or a random
permutation, so every group receives the same class mix. Measured as the
total-variation distance between each group's class distribution and the whole
test set's:

| dataset | group column | per-group TV distance |
|---|---|---|
| `dermmnist` | `loc_group` -- real HAM10000 anatomical sites | 0.091, 0.057, **0.507** |
| `octmnist` | `synth_group` | 0.045, 0.034, 0.029 |
| `tissuemnist` | `synth_group` | 0.016, 0.015 |

dermmnist's group 2 is a genuinely different population (37% class 2 against
17% class 5, where group 0 is 76% class 5). The two synthetic ones are uniform
to within a few percent, so each local budget is essentially `global / G`.

✅ **The paper's dataset description is exactly right, checked against the slices
on 2026-08-19.** Capped-class prevalence in the test split: tissuemnist GE
**171/2400 = 7.1%**, dermmnist melanoma **223/2003 = 11.1%**, octmnist drusen
**250/1000 = 25.0%** -- the three figures the manuscript states verbatim.

🔴🔴 **THE PAPER'S OCTMNIST MECHANISM DOES NOT HOLD FOR THE DATA WE TRAIN ON.**

`docs/paper/main_edited_by_roei.tex:2003` (and `main_clean.tex:2148`,
`main_rev.tex:2429`) says, as *"the property that makes OctMNIST the
hard-binding case"*:

> OctMNIST's training and test distributions disagree: drusen is roughly 8% of
> the training data but 25% of the balanced test split. A model warmed up on
> cross-entropy therefore *under*-predicts drusen on the test set before any
> constraint is applied, so a tight cap binds against a count the model was
> already reluctant to produce.

Measured on 2026-08-19 straight from the `medmnist` package:

| population | n | drusen (class 2) |
|---|---|---|
| official OCTMNIST train | 97,477 | 7,754 = **7.95%** |
| official train+val (what we pool) | 108,309 | 8,616 = **7.96%** |
| official test | 1,000 | 250 = **25.00%** |
| **our training slice** (`scripts/prep_octmnist.py`) | **12,000** | **3,000 = 25.00%** |

The 8% is exact -- **about MedMNIST**. It is not true of our data.
`prep_octmnist.py:16` takes `N_PER_CLASS_TRAIN = 3000` stratified, which
rebalances drusen from 7.95% to exactly 25%, the same as the test split. So the
prevalence disagreement the mechanism rests on **was removed by our own prep
script**, and a CE warm-up on our slice has no reason to under-predict drusen.

This matters beyond one sentence. The recorded insight is that *with one capped
class and identical train/test prevalence the cap carries no new information*.
OctMNIST was the dataset that was supposed to break that -- and in the data we
actually ran, its prevalences are identical too. All three datasets are in the
regime where the cap tells the model nothing it could not infer.

**Two ways out, and they are different experiments.** Either correct the paper
sentence to say the training subsample is rebalanced (and drop the mechanism it
supports), or rebuild the slice at the official prevalence and re-run octmnist.
Do not do the second quietly: it invalidates every octmnist result.

⛔⛔ **RETRACTED, same day it was written: my own "REFUTED: there is no 8% claim
about octmnist anywhere in the manuscript" (commit `871f9dfb`).** There is, in
three .tex files. I grepped for it, the shell ate the backslash in `8\\%`, the
grep returned nothing, and I published absence of evidence as evidence of
absence -- while the warning I was "refuting" sat correct two hundred lines
below in this same file. **A grep that finds nothing proves nothing until the
pattern is shown to match a case you know exists.** The prevalences I checked at
the same time (tissue 7.1%, derm 11.1%, oct test 25.0%) are correct and stand;
the refutation built on top of them does not.

**Put together with the cap finding above: on octmnist and tissuemnist the
"global + local" structure collapses to a single global budget** -- the global
cap cannot bind because the local caps sum to it, and the local caps are a
trivial equal partition of it. Only dermmnist exercises the local scope as a
distinct constraint.

This does not invalidate anything measured; the constraint was still enforced.
It bounds what those two datasets can *test*. If a claim rests on the local or
group-structured part of the formulation, dermmnist is the only dataset that
currently supports it -- and giving oct/tissue informative groups (stratify by
something real, or deliberately skew the synthetic split) is a cheap way to
widen that.

`scripts/verify_caps.py` prints the realized integer budgets and flags INERT / REDUNDANT
scopes. Run it whenever the cap tags, the constrained class, or the dataset slice changes --
it is the constraint-level version of the inert-flag check, and just as invisible without it.

### The arms -- every baseline the paper claims

`python -m configs.gen_campaign --arms all` emits the full panel. `clip` and `focal_clip` are
added to **every** campaign whether asked for or not: an arm-vs-arm delta is not a result until
the bar is in the same campaign.

| arm | methodology | training loss | allocator | epochs |
|---|---|---|---|---|
| `clip` | `heuristic` | CE | greedy threshold | 30 + 0 |
| `focal_clip` | `heuristic` | focal | greedy threshold | 30 + 0 |
| `lp` | `danits_lp` | CE | **LP-LG** (Shifman) | 30 + 0 |
| `focal_lp` | `focal` | focal | LP-LG | 30 + 0 |
| `cb_lp` | `class_balanced` | class-balanced | LP-LG | 30 + 0 |
| `la_lp` | `logit_adjust` | logit adjustment | LP-LG | 30 + 0 |
| `tralo` | `tralo` | CE + count penalty | greedy (post-hoc) | 1 + 29 |
| `fioretto` | `fioretto_ldf` | CE + dual ascent | greedy (post-hoc) | 1 + 29 |
| `hounie` | `hounie_rcl` | CE + resilient dual | greedy (post-hoc) | 1 + 29 |
| `alm` | `fioretto_alm` | CE + augmented Lagrangian | greedy (post-hoc) | 1 + 29 |

**Two allocators, and they are separate baselines.** `heuristic` is the greedy threshold;
`danits_lp` is the LP-LG allocator (local+global formulation of Shifman et al. 2025, which the
manuscript names LP-LG). `danits_lp` also ships that paper's Algorithm 1 greedy as a control.

**The LP is NOT the greedy clipper, and the gap grows with the number of capped
classes.** A reviewer claimed `danits_lp` is identical to `clip` by construction and
therefore unmeasurable forever. That is wrong, and the reason is worth stating because it
is the same reason multi-class is the one real opening.

Greedy ranks items by `p(i,c)` and thresholds at the budget. But assigning item `i` to the
capped class `c` **gains** `p(i,c)` and **loses** `p(i, best_uncapped(i))` -- greedy is blind
to that opportunity cost, the LP is not. A greedy that ranked by the MARGIN
`p(i,c) - p(i, best_uncapped(i))` would be the right greedy; the one we ship does not.
Measured, 25 synthetic instances of N=300 over 6 classes, LP objective set to the same
total-probability the greedy is thresholding (use a LINEAR cost `-p`; with `-log p` the LP
optimizes likelihood instead and the comparison is meaningless):

| capped classes | items assigned differently | LP - greedy, total probability |
|---|---|---|
| 1, global cap only | 136 | +0.154 |
| 1, global + local | 174 | +0.194 |
| 2 | 330 | **+0.495** |
| 3 | 524 | **+0.775** |

So the LP already beats greedy at ONE capped class, and its edge is **4x larger at three**.
The coupling is what grows it: with several capped classes an item is contested, and greedy
serves them in one joint descending pass with no ability to reconsider.

⚠️ **This refines "the score IS the ranking", it does not overturn it.** That result says a
gradient which is a function of the aggregate COUNT cannot reorder items and so cannot beat
post-hoc. Still true. What this adds is that *post-hoc greedy itself is not optimal*, so the
bar `clip` sets is beatable -- by a better allocator, which is what LP-LG is.

🛑 **But `full_panel.py` cannot currently see any of this.** The scorer rebuilds an
allocation from the probability matrix (`eq`) and computes all 13 metrics from it; the arm's
own shipped predictions are read only into the DIAGNOSTIC-ONLY family. `clip` and `lp` share
a warm-up by design, so they emit the same probabilities and therefore score IDENTICALLY on
every metric -- not because the allocators agree, but because the allocator is discarded
before scoring. Until the scorer grows a family computed on what the arm actually shipped,
an LP-vs-greedy campaign will report a tie it did not measure.

**Imbalanced-recipe hyperparameters are the PAPER's**: focal alpha=0.25 gamma=2,
class-balanced beta=0.9999, logit adjustment tau=1. (The `mcbar` campaigns ran focal at
alpha=1.0, which is **not** the paper's focal -- any focal number quoted from those runs is a
different arm from the one the manuscript describes.)

- **Historical dual runs are unusable** (300 epochs + the LR trap) -- re-run them in-campaign.

### The training recipe, stated plainly (swept and verified 2026-08-19)

A reviewer will ask, and none of this was written down. Every line was confirmed
by grep over `src/`, `scripts/`, `configs/` and `main.py`, not assumed.

| knob | value | where |
|---|---|---|
| optimizer | **Adam**, the only construction site in the repo | `src/pipeline/warmup.py:45-53` |
| learning rate | **1e-4, constant end to end** | `protocol.yml` `lr` and `lr_constraint`, forced equal |
| **LR schedule** | **NONE.** No scheduler of any kind exists in live code -- the only `lr_scheduler` hits in the repo are in gitignored `archive/` and are imported by nothing | -- |
| **weight decay** | **NEVER SET.** `weight_decay` appears **zero times** in the entire repository, and `make_optimizer` passes only `lr` and `fused`, so it sits at Adam's default of 0. No `AdamW`, no explicit L2 term | `src/pipeline/warmup.py:45-53` |
| **data augmentation** | **NONE.** `torchvision.transforms` is never imported. The only stochasticity in training is `shuffle=True` in the DataLoader | `src/pipeline/warmup.py:56-63` |
| regularization | **dropout 0.3 only**, plus the constraint-step gradient clip | `protocol.yml` |
| gradient clipping | **constraint step ONLY.** The CE step is unclipped in all four trained arms and in the warm-up loop | `constraint_grad_clip` |
| determinism | `cudnn.deterministic = True` and `cudnn.benchmark = False`, but **`torch.use_deterministic_algorithms` is never called**, so non-cuDNN nondeterministic kernels are not caught | `src/pipeline/setup.py:31,43` |

**Why this matters for the result rather than just for completeness.** The
comparison is a 30-epoch CE model (`clip`) against a 1+29 constrained model, on
train sets of ~8-12k images with ImageNet-pretrained backbones and **no
augmentation, no weight decay and no LR decay**. That recipe overfits, which is
the regime where the archive already records every large backbone saturating in
1-2 epochs. It is applied identically to both sides, so it does not bias the
paired delta -- but "the constraint phase has nothing left to redistribute" and
"the recipe has no regularization other than dropout" are the same observation,
and only the first is currently written down.

⚠️ Adding augmentation or weight decay would change the warm-up, so it
invalidates every cached model and every existing result. It is a new protocol,
not a tweak.

### Known asymmetries between arms -- decisions, not bugs

Three independent code audits (2026-08-19) found these. They are NOT fixed,
because fixing any of them changes what a baseline IS, and that is a call to
make deliberately rather than inside a bug-fix pass. Each is stated with its
measured magnitude so the decision can be made on numbers.

**1. The three duals do not share a normalization convention.**
`hounie_rcl` divides its primal constraint term by `n_test` / `N_g`;
`fioretto_ldf` and `fioretto_alm` do not. Each is internally consistent -- the
dual ascent matches its own primal -- but the effective weight on
`d(soft_count)/d(theta)` at epoch 29, simulated at `protocol.yml`'s step sizes
(N=2003, K=67, soft count 223):

| arm | lambda at ep29 | effective weight |
|---|---|---|
| `fioretto` | 22.62 | 22.6 |
| `alm` | 22.62 | **67.86** |
| `hounie` | 0.0225 | 1.12e-05 |

⚠️ **The ALM row used to read 701.2 / 701, which reproduces ONLY under
`lambda = max(0, lambda + eta*r) + mu_t*r+` stored back into lambda** -- the form at
`1d54de47:fioretto_alm/train.py:274`, which the current code documents as the bug it
fixed: *"the augmentation is added to the PRIMAL weight at use time, never stored back
into lam -- storing it compounds it every epoch."* Under the rule the code actually
runs, the effective weight is **67.86**, a 10.3x overstatement. The row's conclusion
changes from "ALM is 31x Fioretto" to **"ALM is 3x Fioretto"**.

✅ **MEASURED 2026-08-19, and it is stronger than the derivation above.** Running
`fioretto` and `alm` through the smoke harness at the same seed and hashing the
softmax output:

| capped classes | active constraints | `fioretto` | `alm` | |
|---|---|---|---|---|
| `[1]` | up to 4 | `4b71ff08bd` | `4b71ff08bd` | **BIT-IDENTICAL** |
| `[1, 2]` | up to 8 | `58ec9b1de1` | `58ec9b1de1` | **BIT-IDENTICAL** |

The paragraph above hedged this to "with a **single** active constraint". It
holds with eight. The unit-norm clip erases the 3x weight difference even when
several terms compete, so **`alm` is not a distinct baseline from `fioretto`** --
the paper claims nine methodologies and two of them emit the same predictions.

⚠️ Measured on the smoke harness (TinyNet, 120 test items, 2 epochs), not on a
real campaign. The MECHANISM is the derivation above and the measurement matches
it in the regime the derivation said might break it, but the scale is not the
paper's. `full_panel.py` md5s raw predictions across arms, so a real campaign
carrying both will say so on its own.

✅ **The four zero-dose siblings collapse to ONE model.** `tralo_null`,
`fioretto_null`, `hounie_null` and `alm_null` all hash to `c228d9b2b0` at the
same seed, and every treated arm differs from it. That is exactly what a correct
control does: with the treatment zeroed there is nothing left but warm-up 1 plus
29 CE epochs, which is the same object in all four arms. It also proves the
zeroing is COMPLETE -- `alm_null` in particular, where leaving `alm_mu0` at 0.01
would have left a live `mu0 * excess` augmentation on every epoch.

Fioretto is 2.0e6 times hounie. Both fioretto and ALM blow past the unit-norm
clip for any plausible `||dS/dtheta||`, so the clip renormalizes them to the
same norm-1 step -- with a single active constraint the two arms take a
**bit-identical update** and differ only in how they weight the local caps
against each other. Hounie never reaches norm 1, so its constraint phase is 29
epochs of CE plus a numerically negligible nudge.

Deciding this requires choosing a convention and re-deriving each paper's step
size in it. **Until then, do not claim these are three distinct dual
baselines.**

### ✅ ANSWERED 2026-08-20, on real data (`results/vit_diag`, ViTB16 x dermmnist x L30_G30, seed 1)

The paragraph above said the first campaign under this protocol would answer
empirically how often each arm's raw norm crosses 1.0. It did.

| arm | raw `grad_norm` over 29 epochs | clip=1.0 binds | non-finite epochs |
|---|---|---|---|
| `fioretto` | median **51,580**, max 80,827 | **18/19 finite (94.7%)** | **10 of 29 (34%)** |
| `hounie` | max **0.1105** | **0 of 29 (0%)** | 0 |

Same dataset, same backbone, same seed, same budget. The two arms are **five to
six orders of magnitude apart in the quantity the clip acts on**, which is the
derivation above confirmed at full scale rather than on the smoke harness.

Three consequences, all measured:

1. **`fioretto` loses a THIRD of its constraint phase to fp16 overflow.** Six
   NaN and four inf epochs; `GradScaler` records `found_inf` and `scaler.step()`
   skips. It ran a 19-epoch constraint phase where its config says 29 -- and how
   many it loses depends on the CARD, so this is a dose confound BETWEEN
   SERVERS, not just between arms.
2. **`hounie`'s constraint phase is 29 epochs of CE plus nothing.** `max_lam_g`
   reached 0.0168 against its own ceiling of `2*alpha*mean_l` = 0.829 -- 2% of
   the way -- and `constraint_loss` peaked at 0.0089 against fioretto's 7,601.
   `all_satisfied` was 0 on every epoch for both arms.
3. ⚠️ **Neither arm sits in the usable window.** For an arm to receive the
   treatment its config describes, its raw norm must be **above 1.0** (or the
   clip never binds and the arm is untreated) and **below ~1e4** (or fp16
   overflow eats steps). `hounie` is below the floor, `fioretto` is above the
   ceiling. **The window is roughly 1 to 1e3, and it is the target any
   calibration should aim at** -- not the largest norm that still runs.

### ✅ AND: `lp` IS a distinct arm -- the SCORER is what cannot see it

Same campaign, `clip` vs `lp`, which share a warm-up by design and differ only
in the allocator:

    final_predictions_raw.csv   clip dd71977d6177 == lp dd71977d6177   IDENTICAL
    final_predictions.csv       clip a24d4150d25e != lp cbd570b7fdfa   DIFFERENT

The raw probabilities are bit-identical, as designed. **The shipped predictions
are not.** So the LP-LG allocator is a real, separate treatment even at ONE
capped class -- which matches the synthetic measurement above (136 of 300 items
placed differently at a single capped class with only a global cap, because
greedy is blind to the opportunity cost the LP prices in).

🛑 A reviewer claimed `danits_lp` is identical to `clip` by construction and
therefore unmeasurable forever. **The arms differ; `full_panel.py` is what is
blind**, because it rebuilds an allocation from the probability matrix and
scores that, so two arms with the same probabilities score identically on all 13
metrics no matter what they actually shipped. Fixing the scorer is what makes
this arm measurable -- not changing the arm.

**2. The trained arms restart the optimizer at the warm-up boundary; the
post-hoc arms do not.** `run_warmup` builds an Adam and drops it; the
constraint phase constructs a fresh Adam at t=0. `clip` / `focal_clip` run one
Adam for all 30 epochs. Measured step-size kick at otherwise identical
parameters: 3.72x at the first step after the restart, 1.92x by step 3, 1.29x
by step 9. The trained arms get that burst at epoch 2 of 30, where the model is
most plastic, and the equal-compute baseline does not. Either carry the warm-up
optimizer state across the boundary, or restart the clipper's optimizer at
epoch 2 as well -- but say which in the paper.

**3. "Equal compute" is equal in optimizer epochs, not in FLOPs.** Each
constraint epoch adds a full FP32 forward over the test set plus a full AMP
forward+backward, on top of the CE epoch. The post-hoc arms pay neither. This
is a defensible definition -- optimizer epochs are what the regime finding is
about -- but a reviewer will ask, so state it rather than let it be found.

**Changed in the same pass, and it moves TraLO's numbers:** the constraint
gradient was being divided by `n_chunks = ceil(N_test / 256)`, which made
TraLO's effective constraint weight a function of the dataset (derm 8, oct 4,
tissue 10 -- 2.5x apart) and of a memory knob. The chunked-detach construction
already yields the exact full-N gradient, so the divisor was pure attenuation
and a cross-dataset confound. It is removed. **TraLO results from before
2026-08-19 are not comparable to results after it**, and the raw constraint
gradient is now 4-10x larger, so the unit-norm clip binds more often.

### Verify parity BEFORE launching

```bash
python -m configs.gen_campaign --root results/<name> --datasets ... --caps ... --arms all
python -m scripts.check_parity results/<name>      # exit 1 = do not launch
```

`check_parity.py` asserts the four things every retraction here traced to: equal total compute,
identical shared knobs (`lr`, `lr_constraint`, batch size, dropout, ...), identical cell and seed
coverage with at least two cap levels, and correct warm-up cache sharing. **Arms that share a
`base_model_id` share a trained model and must differ only in the allocator** -- an arm sharing a
warm-up with a different training loss is a dead flag, which has happened four times.

### Before reading any metric

1. **md5 the raw predictions across arms.** Inert flags are this project's most frequent
   failure mode -- it has now happened four times. Bit-identical output is a dead flag,
   not a null result.
2. **Check every arm completed.** A lagging arm used to silently delete pairs from every comparison.
3. **Drop non-finite runs loudly.** One run diverged to all-NaN with `status: completed`.

### Scoring

- **`scripts/full_panel.py` is the only scorer.** Never read `evaluation_metrics.csv`
  (two-allocator confound). ✅ A second, concrete reason found 2026-08-21: the RUNTIME
  allocator does not fill the budget exactly. On the stored evidence `targeted_correction`
  emits **K-1 on 22 of 88** (run, capped class) pairs -- never OVER, so no cap is ever
  violated, but not exactly K either, and one item is worth up to 0.006 capF1 here. The
  scorer re-equalizes from the probabilities and does hit exactly K every time (pinned by
  a test), so the shortfall never reaches a scored number -- but it does reach anything
  read off the stored predictions.
- **Atomic cell = (dataset, backbone, cap, method), averaged over the 4 seeds.**
  Never pool across levels, backbones, or datasets. Summaries **count cells**.
  A generator that sweeps a dimension must put that dimension **in the cell key**, not just
  the directory name.
- **Paired Wilcoxon on matched seeds.** Never unpaired pooled-std.
- **`flips`, `raw_count_over_K`, "proximity to feasibility", "less post-hoc surgery" are
  NOT METRICS.** Post-hoc filling to the boundary is free. They are one rejected metric under
  different names. When quality ties, `flips` is the one column with a small p-value -- the
  honest report is **"this arm produced nothing."**
- Quote multi-class accuracy against the **oracle-under-constraint ceiling**, not 1.0
  (oct L50 = 0.625, derm L30 = 0.809, tissue L30 = 0.795). ⚠️ PROSE-ONLY, no receipt.
- ccP / ccR / ccF1 are one metric in three costumes on single-class problems.
- **Read `d capF1` separately from `d macroF1` -- their precision differs by orders of
  magnitude.** Paired over 3 seeds on the multi-class cell, `d capF1` came out
  **-0.0149 / -0.0150 / -0.0149 (sd 0.0001)** while `d macroF1` had **sd 0.0371** and was
  positive in 1 of 3 seeds. macro-F1 there is dominated by the UNCAPPED classes, which
  swing with the seed; the capped classes are what the method is about and they are
  measurable. A mean whose sd exceeds it is not a result.
- **`d capF1` is QUANTISED, so check it against the integer.** When the allocator emits
  exactly K predictions for class c, `P = TP/K` and `R = TP/n`, so **F1 = 2TP/(K+n)** --
  linear in the number of correct items. Every capped-class F1 delta must therefore be an
  integer multiple of `1/(K+n)`; a value that is not is an arithmetic bug. The -0.0149
  above is exactly **4 correct predictions lost out of the 89 selected**, at every seed.
- **Check the cell's CEILING before spending GPU on it.** Recall on a capped class is hard
  limited to `K/n_true`, so `F1_cap <= 2K/(K+n)`. On dermmnist the entire headroom is
  **0.038 / 0.052 at L30_G20** and **0.052 / 0.072 at L50_G30**, against a paired seed sd
  of macro-F1 of **+-0.04**. ⇒ **at G20 the prize is smaller than the noise and the cell
  cannot resolve a win even in principle.** Prefer the looser cap; headroom grows with it.

### Infrastructure

- **Never run experiments locally.** SSH the server. `conda activate optloss` (base is CPU torch).
- **Max 2 GPUs.** `nvidia-smi` **with owner lookup** first. Never share a GPU with another user.
- Stop dispatchers with `kill -INT` (graceful; interrupted runs reset to `pending`). Never `pkill`.
- Delete stale model caches before new runs -- `base_model_id` omits normalization state.
- Any hyperparameter that changes what warm-up optimizes **must be in `compute_base_model_id`**,
  or the second arm silently loads the first one's cached model.

---

## 1b. THE PAPER'S PROTOCOL IS NOT THIS PROTOCOL

The manuscript's "frozen recipe" (Sec. setup) and this framework's protocol are **different
experiments**, and the generator deliberately cannot emit the paper's.

| | paper | this framework | why they differ |
|---|---|---|---|
| warm-up | **50 epochs** | 1 (trained) / 30 (post-hoc) | warm-up 50 saturates CE, so every method becomes identical -- section 3 |
| constraint phase | **300 epochs** | 29 | equal compute: 30 optimizer epochs on both sides |
| lr (warm-up) | 1e-4 | 1e-4 | same |
| lr (constraint) | **5e-6** | 1e-4 | **this is the LR trap.** Unequal lr fabricated a -16.7 pp finding that was -1.7 pp once equalized |
| TraLO `lambda_step` | 0.002 | 0.05 | |
| TraLO hinge `beta` | **0.5** | **deleted** | the undershoot hinge is rejected at every dose (section 2b) |
| Fioretto dual step | 0.005 | 0.005 | matches |
| Hounie eta_lambda / eta_u | 0.01 | 0.01 | matches |
| ALM eta / mu0 / mu_step | 0.005 / 0.01 / 0.01 | same | matches |

**So the paper's headline numbers were produced under warm-up 50 with an unequal
`lr_constraint` -- the two settings this framework forbids.** That is not a reason to re-run the
paper's config; it is the reason the framework exists. But any sentence comparing a new number to
a published one is comparing across protocols and must say so.

⚠️ **`fioretto_step_size` is REQUIRED** -- `fioretto_ldf/train.py:35` raises without it, because
the runner used to default to 0.01 while a generator defaulted to 0.005. **`hounie_rcl`'s inline
defaults are 0.1**, ten times the paper's 0.01, so an unset key silently runs a different method.
The generator now sets every per-method step explicitly; never rely on a default.

### Dataset scope check against the paper

| dataset | capped class | paper's share | on disk | status |
|---|---|---|---|---|
| DermMNIST | 4 (melanoma) | 11.1% of test | 223/2003 = 11.1% | matches exactly |
| TissueMNIST | 4 (GE) | 171/2400 = 7.1% | 171/2400 = 7.1% | matches exactly |
| OctMNIST | **2 (drusen)** | **~8% train / 25% test** | **25% train / 25% test** | ⚠️ **train split is balanced** |

⚠️ **The OctMNIST slice on disk does not have the property the paper attributes to it.** The paper
calls OctMNIST the hard-binding case *because* "drusen is roughly 8% of the training data but 25%
of the balanced test split" -- a train/test prevalence disagreement. The `slice_1` on disk is
balanced in **both** (25%/25%), so that disagreement is absent. Either the slice was rebuilt
differently from the runs behind the paper, or the paper describes the original MedMNIST split.
**Resolve this before quoting any OctMNIST claim.**

🔑 That same paragraph is the strongest lead in the project: a train/test prevalence gap is exactly
the **distribution shift** regime of section 4, the one setting where the cap carries information
the model does not already have. The paper already observed it; nothing has tested it.

---

## 2. WHAT FAILED -- the ideas, not the runs

Grouped by *why*, because the reasons repeat.

### (a) Anything that varied the count penalty -- ~13 arms, all ties

Penalty shape, rational vs quadratic, rho schedules, lambda schedules, granularity.
**Why:** none of them changed what the gradient is a *function of*. All are functions of the
aggregate count. See the structural claim in section 0.

- **Constraint granularity** (G group counts instead of 1) -- monotonically *worse* as it gets
  finer (cc-F1 -0.008 at G=1 to -0.018 at G=32).

### (a2) 🔴🔴 THE PENALTY'S GRADIENT VANISHES ON THE WORST VIOLATIONS

**Re-derived with autograd on the shipped `_penalty` (2026-08-19), K=67:**

| rho | at the boundary | at 57.7% over | at 8x over | peak / boundary | peak / deep | peak at |
|---|---|---|---|---|---|---|
| 0.5 (initial) | 0.014924 | 0.010849 | 0.000213 | 1.0x (monotone) | 70x | u -> 0 |
| **3.93** (after ONE constraint epoch) | 0.014934 | **0.044100** | 0.000406 | **3.0x** | **109x** | u = 0.532 |
| 100 (the target) | 0.015221 | **0.975433** | 0.005836 | **64.1x** | **167x** | u = 0.576 |

⚠️ **Corrected 2026-08-19** after a second reviewer re-derived it independently.
The first version of this table read `peak / boundary = 54.6x` at rho=100 and
gave boundary values from a slightly different evaluation point; and it claimed
the peak sits at `u = 1/sqrt(3)` **analytically**, full stop. That is the peak of
the QUADRATIC TERM ALONE. The rational term's own slope is decreasing, so it
pulls the combined peak left, and `1/sqrt(3)` is only the `rho -> infinity`
limit: at the operative rho of 3.93 the peak is at **u = 0.532**. The 167x
figure and the shape of the claim reproduce exactly.

So above `rho ~ 1` the gradient is **non-monotone in the violation** -- near-zero
at the boundary, peaking around 53-58% over, and decaying toward zero for
anything worse.

**A constraint violated by 8x its budget receives 167x LESS corrective pull than
one violated by 58%.** That is the opposite of what a penalty is for.

**And the terms compete for one clip.** Two capped scopes, one at its peak and
one 8x over, driven through `_sum` and the single unit-norm clip:

| scope A | scope B | A's share of the squared gradient | B's share |
|---|---|---|---|
| 58% over | **8x over** | **99.9964%** | **0.0036%** |
| 8x over | 58% over | 0.0036% | 99.9964% |
| 58% over | 58% over | 50% | 50% |

Symmetric under swapping the roles, so it is the violation DEPTH doing it, not
the scope. **The worst-violating group is starved by a milder one at 167:1**,
and dermmnist's 3 groups mean every single-capped-class run already carries 4
competing terms.

`rho_step` is derived as `(rho_target - initial_rho)/29 = 3.43`, so **rho is
3.93 after the first constraint epoch** -- the monotone regime exists for
exactly one epoch of twenty-nine.

**Does it bite? Yes, and here is the precise condition.** The unit-norm clip is
applied to the constraint gradient ALONE (`tralo/train.py:346-357`: CE and
constraint take separate `zero_grad`/`backward`/`step` sequences), so it
normalizes magnitude and preserves direction. With a **single** penalty term the
direction is therefore independent of the shape, and the shape is a no-op --
**which is the mechanical reason ~13 shape variants all tied (section 2a).**

But the loss sums a term per (capped class, scope): one global plus **one per
group**. dermmnist has 3 groups, so even a single-capped-class run carries **4
terms**, and their RELATIVE weights are exactly what the shape sets. A group
that is badly over budget is systematically down-weighted against one that is
mildly over. **This is live in every run we have made**, and it compounds with
multi-class caps, which is where section 4's open question lives.

⚠️ **It is the published formula, not a coding bug** -- it matches the
manuscript's Eq. 4. Changing it changes what the paper describes and invalidates
every existing result, so it is not a cleanup. Recorded, pinned by a test, and
left to Roei.

⚠️ **The stated justification for the shape is boundedness** -- but the
gradient clip already provides the only boundedness that reaches a parameter.
The shape's boundedness is redundant with the clip and buys nothing except this
defect. A plain or squared hinge has a constant or growing gradient with depth
and would not have it.

### (b) Anything that delivered MORE constraint gradient -- all significantly worse

- **More constraint steps per epoch** -- monotone: n=1 (incumbent) is the best setting in the
  sweep; n=4 costs AP -0.198; n=16 -0.182.
- **Dedicated constraint optimizer** (recovers ~10x more constraint gradient by measurement)
  -- AP -0.0938, p=0.0006, and it destroys the macro-F1 lean win.
- **`joint_objective`** -- holds the cap 98.8% of epochs vs 6%, but overfits (-0.067 AP);
  dead on multi-class at 13/13 metrics p=0.0000.
- **Undershoot hinge (`beta`)** -- worse at every dose, damage grows with beta, diverges at beta=100.

**This inverts the "the constraint phase is starved" narrative.** The constraint phase gets
~29 steps against CE's 3654, through a shared Adam that retains ~1% of its direction, with a
magnitude cancelled by the unit-norm clip. All three were read as defects to repair.
**They are not. The starvation is why the method is only mildly worse than a clipper.
Every attempt to deliver more constraint signal made it worse.**

### (c) Per-item losses -- all null so far

- **`rank`** (pairwise, transductive, top-K vs rest) -- null, 48/48. It is **self-referential**:
  no labels, so it can sharpen a cut but never reorder.
- **`rankpair`** (supervised pairwise hinge) -- null to negative.
- **`budget_margin`** (hinge at the cap's implied threshold) -- the knob is live but only AUROC
  moves, i.e. it improves the ordering in a region the cap never reads. Untested on multi-class.
  🔗 **2026-08-21: the SHIPPED penalty has the same failure mode**, which makes this a family
  rather than one arm's quirk. At protocol length the count penalty improves AUROC (+0.0049),
  ECE, Brier, NLL and ConfGap against `clip` while ccP falls **-0.0450**: better everywhere the
  cap does not read, worse in the only place it does. **⇒ before rebuilding `budget_margin`,
  state which metric it is supposed to move and check it is ccP.** An arm that moves AUROC has
  already been run, twice.
  ⚠️ But ccP is NOT immovable: in the `L50_G30` cell the same penalty with a `linear` shape
  gains **+0.030 ccP over its own lambda=0 control**. The pessimistic reading ("only AUROC can
  move") is a single-class result and does not survive the looser cap.

### (d) Retracted results -- claims that did not survive re-measurement

- **"TraLO beats the clipper"** -- beat `focal_clip` only; `clip` beats `focal_clip` by more.
- **"no-restore is a win"** -- a single-cap-level artifact; -0.0098, p=0.14 when swept.
- **"warm-up 1 gives +7 to +9 pp"** -- a compute artifact; -0.85 pp at equal compute.
- **"the constraint damages the representation"** -- the LR trap, invalid.
- **"the gain is on the uncapped classes"** -- vs `clip` that is +0.0010, p=0.90.

### (e) Dead code / inert flags found by audit

`rho_step` (log-only everywhere) - `base_loss`/`focal_alpha`/`focal_gamma` (dead in `arm_joint`,
so its `focal_clip` arms were a second `clip`) - `reset_optimizer_at_sat` (bit-identical no-op at
warm-up 1) - `tralo_uniform` class weights (documented no-op, it IS plain TraLO) -
`class_balanced`/`logit_adjust` (inert on oct) - `by_k` (inert on oct).

### (f) What was DELETED FROM THE CODE on 2026-08-18, and why

Every failed idea had left a flag, a branch and a default behind. The knobs are gone, not
merely defaulted off -- a config can no longer *imply* a knob that does not exist.

| removed | was | verdict that killed it |
|---|---|---|
| `hybrid_mode`, `fior_beta` | undershoot hinge | worse at every dose, diverged at beta=100 |
| `constraint_steps_per_epoch` | n steps per epoch | monotone: n=4 costs AP -0.198 |
| `separate_constraint_optimizer`, `post_sat_optimizer` | dedicated constraint Adam | AP -0.0938, p=0.0006 |
| `reset_optimizer_at_sat` | Adam reset at satisfaction | bit-identical no-op at warm-up 1 (16/16) |
| `constraint_class_weights` (`by_k`/`inv_k`) | per-class penalty weighting | the `uniform` branch was a documented no-op |
| `alpha_kl` + the whole KL anchor | KL to warm-up predictions | out of scope by decision |
| `enable_ce_skip` + CE-skip machinery | stop CE at saturation | reached only TraLO; fabricated a 0.22 cc-F1 artifact |
| `disable_freeze_on_satisfy` | ratchet/rho freeze ablation | never used; protocol freezes on satisfy |
| `cb_beta`, `logit_adjust_tau` | class-balanced / logit-adjusted losses | inert on oct; `focal` is the real baseline |
| 10 methodology packages | tralo_bounded, fioretto_rh/restart/alm, hounie_rh, alm_rh, danits_lp, focal, class_balanced, logit_adjust | dead arms |
| 55 config generators (5,481 lines) | one per campaign | replaced by `configs/gen_campaign.py`, which asserts the protocol |
| 47 analysis scripts (7,221 lines) | per-campaign figures/tables | replaced by `scripts/full_panel.py` |
| `src/evaluation/` (2,527 lines) | census, bootstrap, FDR, win-bar sensitivity | superseded by `full_panel.py` |
| 4 backbones | ShuffleNetV2, TinyCNN, SmallCNN, MediumCNN | absent from every `.tex` file; no written claim rests on them |
| 6 datasets from the loader | aider, retinamnist, bloodmnist, organamnist, octnative, tissuenative | out of scope; data deleted |

**Second pass (same day): 4,909 -> 4,680, and the remaining bloat was structural, not volume.**

| target | what was wrong | what it is now |
|---|---|---|
| `src/losses/transductive_loss.py` | the per-constraint penalty math was **duplicated verbatim** between the global and local paths; `penalty_mode` carried four shapes (`rational`/`quadratic`/`both`/`linear`) of which three are rejected arms | one `_penalty()` and one `_sum()` shared by both scopes; one shape. **Verified numerically identical** to the old code across 532 randomized comparisons (⚠️ PROSE-ONLY: the comparison script was not committed and no artifact survives) -- values, gradients, satisfaction flags, and the empty-constraint edge cases that must still return an autograd-connected zero |
| `main.py` (304 -> 164) | two dispatch paths; the threaded multi-GPU one ran several cards in **one process sharing one `model_cache`**, which races on the warm-up write | single GPU per process, matching how campaigns are actually launched (one process per card, own `EXPERIMENT_DIR`) |
| `configs/common.py` (94) | 94 lines for one live function | folded into `gen_campaign.py`; **`configs/` is now one file** |
| `src/training/__init__.py` (30) | re-exports nobody imported, including `ConstraintTrainer` which no longer exists | 4-line docstring |
| `ConstraintTrainer` re-exports | pointed at a class that no longer exists | gone |
| ~~`LogitAdjustedLoss`, `_class_counts`~~ | deleted here as orphans | **RESTORED** by the third pass below -- the paper claims `logit_adjust`. Live in `src/losses/imbalanced_losses.py` |

⚠️ **Third pass corrected an OVER-DELETION.** The first purge removed six methodologies that the
manuscript's Baselines paragraph actually claims: `danits_lp` (LP-LG), `fioretto_alm` (ALM),
`focal`, `class_balanced`, `logit_adjust`, and their shared `imbalanced_common` driver. They were
cut on the reasoning that `class_balanced`/`logit_adjust` are "inert on octmnist" -- but inert on
*one dataset* is not grounds for deletion, and `danits_lp`/ALM were never inert at all. All are
restored from `63c2b4cc` and wired into the generator. **Check the paper before deleting a
baseline**: the manuscript is the authority on what is in scope, exactly as it was for backbones.

⚠️ **`tralo_bounded` was NOT restored, deliberately.** The paper describes it as the ablation that
strips "the optimizer reset and the undershoot hinge" -- but both of those are now deleted from
`tralo` itself (section 2b), and the reset was already a bit-identical no-op at warm-up 1. Under
the current protocol `tralo_bounded` would be an exact duplicate of `tralo`. The ablation is only
meaningful at warm-up 50, which is a dead regime. **If the paper keeps that ablation, its text
needs to say it describes warm-up 50.**

🚨 **A latent collision was fixed while consolidating.** `compute_base_model_id` did not
include the warm-up objective, so **`clip` and `focal_clip` hashed identically** -- `focal_clip`
would load `clip`'s cached warm-up and silently become a second `clip`. That is the inert-flag
failure mode, occurrence five. Only `focal_clip`'s hash moves; the other 12 arm/dataset
combinations are bit-identical, so no other cached warm-up is invalidated.

✅ **`src/utils/posthoc_adjustment.py` (406 lines) was examined and left alone** -- every helper,
including the 139-line LP fallback, is reachable from `targeted_correction`. It is the algorithm
being compared against, not clutter.

⚠️ **"An AST reachability pass reports zero dead definitions" was true on 2026-08-15 and
false by 2026-08-18**, when 457 more dead lines came out (a whole unused `bounded_only` branch,
a `_infer_probs` duplicate, `score_arm.py`'s 176-line CLI). A one-time sweep is a snapshot, and
this one was quoted for three days after it stopped being true. **The durable version of this
claim is the gate, not the number**: `python -m scripts.audit_config` exits 1 on a hyperparameter
with no reader, and it runs before every launch.

**Result: 23,180 lines of Python -> 4,680 on 2026-08-15, and it has gone back UP since**, on purpose: the
six restored baselines, six new gate scripts, and 152 tests. **Do not quote a line count as a
quality measure** -- it has only gone UP since the purge while the repository got
strictly more correct, and every per-component figure written here has gone stale
within days. Measure it if you need it: `git ls-files '*.py' | xargs wc -l`.

What is actually load-bearing is that every one of those lines is reachable and every knob is
read: `audit_config` (no orphan hyperparameters), `smoke_arms` (every arm runs end to end; caps verified for the arms that emit predictions directly, and for the trained arms under `--matrix`),
`verify_caps` (the caps bind on the real slices), `check_parity` (equal compute, shared knobs,
no cross-objective warm-up sharing), and `pytest tests` (152 tests, ~35 s, no dataset needed).

**`rho_step` is still a DEAD KEY** and remains so by design: the ramp is derived from
`rho_target`. It is documented in `hp_defaults.py` rather than silently ignored.

---

## 3. WHAT WE KNOW WORKS -- regime beats method, every time

**The single most useful fact in this project: regime effects are ~8 pp. Method effects are ~0.1 pp.**
Every "win" that turned out to be a regime difference in disguise was bigger than every real
method effect.

1. **Warm-up 1 over warm-up 50.** Not because it wins, but because warm-up 50 makes every method
   identical -- CE is saturated, so the constraint phase is ~30 unit-norm steps on a frozen
   representation and all methods land within 0.1 pp.
2. **Equal compute is mandatory.** It is worth 7-9 pp on its own.
3. **`clip` is the strongest baseline on quality; `focal_clip` on calibration.** A base-loss swap
   on a clipper is free and buys a 16/16 calibration rout. Carry both.
4. **The unit-norm gradient clip is load-bearing.** Remove it and the count collapses to 0 --
   loses 4/4 cells. It binds 63-84% of the time.
5. **The lambda toggle is essential.**
6. **Post-hoc local adjustment never re-violates the global cap.** The
   original 199-run measurement was made by `paper/scripts/feasibility_check.py`,
   which was lost with that directory -- so this sat as a settled fact with no
   receipt, unlike `build_corpus.py` whose absence is flagged. Rebuilt at
   `scripts/feasibility_check.py` and re-run 2026-08-19: **128 archived runs,
   zero violations** (every run in the evidence tarballs that ships
   predictions). ⚠️ Those predictions carry no group column, so that pass
   verified the **GLOBAL** caps only; pass `--data-root` on the server to
   check the LOCAL caps, which is the direction the original warning was
   about. The check is mutation-tested: injecting 400 predictions of a
   capped class is caught as `GLOBAL c1 420>31`, exit 1.
   Retired as a concern.
7. **The scorer is validated.** `equalize_multi` is arm-independent, budget-constant, feasible
   144/144, order-independent, not calibration-convertible, and agrees with an exact LP.
8. **Multi-class caps are genuinely supported** by all three methods -- no `[0]` truncation anywhere.

### Measured and open, no clean answer yet

- **Warm-up 1 delays CE saturation, it does not prevent it**: dermmnist saturates at epoch 15
  (8/8 runs), octmnist 25, tissuemnist 30 (⚠️ PROSE-ONLY, no receipt). Across those three, TraLO does *best* on the *most*
  saturated dataset and *worst* on the least -- the opposite of the "room to move the boundary"
  prediction. **Confounded**: saturation timing tracks dataset difficulty. The clean test is
  within one dataset, using augmentation to hold CE unsaturated. Not yet run.

---

## 3b. THE MISTAKE PATTERNS -- how a wrong result got believed

Section 2 lists the ideas that failed. This section lists the *ways we fooled ourselves*, which
matter more, because the ideas are finished and the patterns repeat. Every retraction in this
project came from one of these. The right-hand column is the important one: a pattern guarded
only by discipline **will** recur.

| # | pattern | what it cost | guard today |
|---|---|---|---|
| 1 | **Inert flag** -- a config key emitted but never read, or read in one arm only | 5 occurrences. `focal_clip` was a second `clip`. 13 wave-1 arms tuned a quantity the code cancels. | **MECHANICAL** -- `scripts/audit_config.py`, AST and per-arm |
| 2 | **A claim from one cap level** | retracted 3x (no-restore, and two others) | **MECHANICAL** -- generator refuses <2 cap levels |
| 3 | **An arm-vs-arm delta with no baseline in the campaign** | both `rank` campaigns compared `rank_on` vs `rank_ctrl` and never contained a clipper | **MECHANICAL** -- `mandatory_arms: [clip, focal_clip]` |
| 4 | **Unequal compute** | worth 7-9 pp; fabricated "warm-up 1 gives +7-9 pp", which is -0.85 pp when equalized | **MECHANICAL** -- `check_parity` gate 1 |
| 5 | **A knob differing across arms** (the LR trap: `lr_constraint` 5e-6 vs 1e-4) | fabricated "the constraint damages the representation", -16.7 pp that was -1.7 pp | **MECHANICAL** -- `check_parity` gate 2 |
| 6 | **Pooling the axis being swept** | the granularity sweep's first read averaged over granularity itself | **DISCIPLINE** -- a swept dimension must be in the CELL KEY, not just the directory name |
| 7 | **Reaching for the one column with a small p-value** when quality ties | `flips` / raw count / "proximity to feasibility" are the same rejected metric renamed; relapsed ~10x | **MECHANICAL** -- `full_panel.py` refuses to headline them |
| 8 | **A scorer bug that reads as a tie** | `dropna` across ALL arms meant a lagging third arm deleted pairs from every comparison; at n=2 Wilcoxon floors at p=0.5, so in-flight campaigns ALWAYS read as ties. **Arms were abandoned on that.** | **FIXED**, but the class of bug is only guarded by testing the scorer against a known answer |
| 9 | **A diverged run recorded as `completed`** | `joint_b100` seed 4 went all-NaN, crashed the scorer, and hid 23 healthy runs | **MECHANICAL** -- `full_panel.py` drops non-finite runs loudly |
| 10 | **Deleting something the paper claims** | 6 baselines cut on "inert on octmnist"; inert on *one* dataset is not grounds for deletion | **DISCIPLINE** -- check the manuscript before deleting a baseline or a backbone |
| 11 | **Characterising the paper without reading it** | done twice; the paper already calls penalty shape "neutral" and lambda escalation "a symptom" | **DISCIPLINE** |
| 12 | **A cap that cannot bind** | the global cap has never bound at any tag we ran (section 1) | **MECHANICAL** -- `scripts/verify_caps.py` flags INERT / REDUNDANT |
| 13 | **Comparing arms across CAMPAIGNS** | measured cross-campaign drift is **0.027** (⚠️ PROSE-ONLY, no receipt), about **2x** the effects being argued over. The +0.0068 cc-F1 edge over focal+clip was cross-campaign; run in ONE campaign it vanished | **MECHANICAL** -- `mandatory_arms`, and `full_panel` pairs within a campaign |
| 14 | **Quoting a one-cell effect** | measured twice, and it shrinks by ~2x both times: restore -0.0477 (1 cell) -> **-0.0351** (4 cells); focal +0.0305 -> **+0.0156**. One cell always over-estimates | **DISCIPLINE** -- quote the multi-cell figure, always |
| 15 | **Averaging opposite signs across datasets** | the no-restore "AP win" of +0.0085 was derm **-0.027/-0.025** and oct **+0.027/+0.058** -- a dataset split averaged into a number that means nothing. p=0.56 | **MECHANICAL** -- `--percell` includes the cap, and the panel prints better/worse counts |
| 16 | **A treatment smaller than its own RNG noise** | `rank` at weight **1e-12** (gradient numerically nil) gave a parameter delta of 0.2238 -- *larger* than the 0.2222 at the real weight. The arm was pure nuisance | **DISCIPLINE** -- run every new arm at ~zero dose first and show the delta collapses to 0 |
| 17 | **A selector that compresses the effect being measured** | `ortho`'s checkpoint restore made 2 of 4 seeds **bit-identical** post-restore while pre-restore differed in 4 of 4 -- the restore compressed the measured effect **~13x** (AP +0.0003 vs +0.0041), and its criterion is total excess, exactly what the projection trades away | **DISCIPLINE** -- if a post-processing step selects on the quantity your treatment moves, report pre-selection metrics too |
| 18 | **Raw (uncentered) correlations** | `count_cv` reads rho = **-0.847** raw and **-0.165** within-cell. The raw number is dataset identity, not a relationship | **DISCIPLINE** -- centre within (dataset, model, cap) before correlating |
| 19 | **Mixing epoch conventions** | TraLO logs ABSOLUTE `Epoch`, the duals log RELATIVE from 0. Subtracting warm-up from a dual fabricated "Fioretto runs 250 epochs vs TraLO's 34" | **DISCIPLINE** -- per-method convention; and never use row count for epochs, TraLO logs sparsely |

**The meta-pattern behind 1, 8, 9 and 12: a thing that silently does nothing looks exactly like
a thing that does nothing useful.** An inert flag, a cancelled gradient, a non-binding cap and a
dropped pair all produce the same observable -- a tie -- as a real negative result. That is why
every guard above is a *pre-launch assertion* rather than a post-hoc analysis: after the fact the
two are indistinguishable.

### The practices that actually CAUGHT things

The archive documents failures exhaustively and successes only in passing. These
five are what found the defects, and they are cheap:

1. **md5 the raw predictions between arms before reading any metric.** On
   `steps5` it found two bit-identical pairs a plain delta would have shown as a
   null result. Now automatic in `full_panel.py`.
2. **Run a new arm at ~zero dose and show the delta collapses to 0.** This is
   what exposed `rank` as pure RNG nuisance, and it is the only check that
   distinguishes "small effect" from "no effect plus noise".
3. **Mutation-test the fix.** `rank`'s RNG repair was verified by REVERSING the
   two edits and confirming the pathology comes straight back -- so the check
   bites rather than passing vacuously.
4. **Predict the failure before the run, in writing.** `reweight`'s `[1/4, 4]`
   weight bound was called as a hard ceiling against a cell demanding 3.39x
   suppression, before it ran. It missed the cap exactly as predicted, which
   made the result a finding about the reweighting family rather than a bug
   hunt.
5. **Score the same run at two checkpoints rather than comparing two arms.** The
   restore probe measured -0.0351 AP with nothing confounded, because no arm
   comparison was involved at all -- and its three `restore_kind: none` runs read
   exactly 0.0000, an internal control that came free.

### The three things that actually made the pipeline work

1. **Regime beats method by ~80x.** Regime effects are ~8 pp, method effects ~0.1 pp. Every
   "win" that later evaporated was a regime difference in disguise. Fix the regime first, and
   never compare across regimes.
2. **The unit-norm gradient clip is load-bearing.** Remove it and the predicted count collapses
   to 0 and the arm loses 4/4 cells. It binds 63-84% of the time. This is also *why* rho and
   lambda are no-ops: the clip delivers exactly 1.000 against a raw norm of 2,560-12,400, so
   scaling the penalty changes nothing downstream.
3. **The constraint phase's starvation is PROTECTIVE, not a defect.** ~29 constraint steps
   against CE's 3,654, through a shared Adam that retains ~1% of the constraint direction. Every
   attempt to repair this -- more steps, a dedicated optimizer, a joint objective -- made results
   significantly worse, monotonically. **Read a weak constraint phase as the reason the method is
   only mildly worse than a clipper, not as the bug to fix.**

### What to avoid, stated as rules

- **Never vary the count penalty again.** Shape, schedule, granularity, magnitude: ~13 arms, all
  ties, because none changes what the gradient is a *function of*.
- **Never deliver more constraint gradient.** Monotone: more is worse, every time.
- **Never run warm-up 50**, and never interpolate to warm-up 5 (a dead zone).
- **Never let `lr_constraint` differ from `lr`.** The mechanism, read out of
  `src/methodologies/tralo/train.py`: the trainer builds its optimizer with
  `lr_constraint` and re-asserts `pg["lr"] = lr_constraint` on every param group at
  the top of each epoch's **CE** pass. So `lr_constraint` is not the constraint's
  learning rate -- it is the learning rate of the 126 CE steps too. Sweeping it does
  not isolate the constraint, it retrains the classifier, which is precisely how the
  LR trap fabricated -16.7 pp.
- **Never quote a metric that post-hoc filling can produce for free** (`flips`, raw count over K,
  proximity to feasibility). When quality ties, the honest report is "this arm produced nothing."
- **Never build a self-referential per-item term.** `rank` used top-K vs rest with no labels: it
  can sharpen a cut but cannot reorder, and ordering is the whole score.

---

## 4. THE ONE OPEN QUESTION

**STATE, 2026-08-21 -- read this before the 300 lines below.**

| | what | status |
|---|---|---|
| **built, not run** | `tralo_margin` + `tralo_st` (1b) -- the count's gradient on the decision boundary, decomposed from the count's value | 56-run campaign ready, all gates green, `docs/launch_margin1.sh` |
| **proposed, not built** | 1c -- optimise the metric at the budget **with LABELS**, via a jointly-trained SELECTION head | literature checked; **SelectiveNet (ICML 2019) beat exactly our `clip` baseline** in the analogous coverage setting |
| **needs Roei** | path 2 -- a test set whose prevalence DIFFERS from train | a load-time subsample, NOT re-slicing |
| **purged** | `budget_margin` (path 1) | must be rebuilt before it can run |

🛑🛑 **THE WHOLE EFFECT SPACE IS 2 TO 18 ITEMS. Everything else in this document is
downstream of that.** `F1 = 2TP/(K+n)` is linear in TP, so an F1 delta converts exactly
into items. Measured headroom from `clip` to the ANALYTIC CEILING on dermmnist:

    cell                headroom   items to close it   of K predicted
    class 1 @ 30%        0.0266           1.8                 31
    class 2 @ 30%        0.0315           4.5                 66
    class 4 @ 30%        0.0517           7.5                 67
    class 1 @ 50%        0.0452           3.5                 52
    class 2 @ 50%        0.0742          12.2                110
    class 4 @ 50%        0.1043          17.5                112

(`python -m scripts.headroom <root> --control clip`; corrected 2026-08-21 --
see the note at the cap-level table below for what the first version got wrong.
The correction WIDENS the range to 2-18 items; it does not change the argument,
because the seed noise widens with it and the ordering across cap levels holds.)

That is the gap to a PERFECT allocator, not to a better method -- no method can exceed
it. And `d capF1` is QUANTIZED at 0.67-1.68 items per 0.01, while the paired seed sd is
0.04 at L30_G20, worth ~2.7 items. ⇒ **seed noise is comparable to the entire headroom**,
which is why arms tie, why single-cell claims kept being retracted, and why the archived
result "losing EXACTLY 4 correct predictions of 89 in 3 of 4 seeds" looks so quantized:
it is counting items, because there is nothing else to count. **Convert every F1 delta to
items before believing it.**

🛑 **A tie is the pre-registered expectation for 1b, and three independent derivations say
so:** post-hoc is optimal GIVEN the probabilities; the cap adds no information the training
set lacks (proved from `create_slices.py`); and under `K << n_true` the Bayes rule is
top-K by probability, which IS the clipper. The only remaining lever is a better RANKING at
the operating point, and that needs LABEL information -- which no count penalty has. ⇒ if 1b
ties, go to 1c, **not** to a third count.

✅ **But the pessimism is about COUNT PENALTIES, not about the problem.** SelectiveNet
(ICML 2019) beats "a threshold over the prediction confidence of a pre-trained network"
-- our `clip`, exactly -- in the analogous coverage-constrained setting, by training a
selection head jointly so the network is "optimized over the covered domain". Post-hoc is
optimal GIVEN the probabilities; training can change which probabilities you get. **That
mechanism is available to 1c and to none of the arms run so far**, because every one of
them bolts a penalty onto a warm-started CE model instead of fitting the model to the
sub-population it will actually predict on.

Given section 0, these are the only things that can still win, and the only things worth
building. They are ordered by how much of the pessimism above they escape: **1c escapes
all of it, 1b escapes none of it** and is being run anyway because the analysis could be
wrong, and 2 changes the problem rather than the method.

1. **A per-item objective at the operating point** -- something whose gradient depends on an
   individual item's position relative to the budget, not on the aggregate count.
   Three attempts are null so far; `budget_margin` on multi-class is the one untried variant.
   ⚠️ **It has to be REBUILT first -- the code is purged** (`grep budget_margin src/` is empty).
   Section 2 records what the single-class version measured and what to change before rebuilding
   it; do not read this line as "it is sitting there ready to launch".

1b. **A count whose gradient lands ON the cut** -- derived 2026-08-21, not yet run.

   The soft count is `s_c = sum_i p_ic`, so `d s_c / d logit_ic = p_ic(1 - p_ic)`. The
   shape (`rational_bounded` / `linear` / `squared`) only scales `d penalty / d s_c` --
   for `linear` it is the constant `1/K` -- so **no choice of shape moves where the
   gradient lands**, which is why thirteen shape arms measured the same thing. Where it
   lands is fixed by `p(1-p)`.

   ⚠️ **Read (a) below before using that as a motivation.** The obvious next sentence --
   "so the gradient cannot reach the cut" -- is measured and it is WRONG as usually
   stated. `p(1-p)` at the K-th RANKED item is ~0 (0.94 early, 0.999 converged), but rank
   K is not where predictions change. At the decision boundary `p(1-p)` is near its
   maximum, and `sum` already puts 29.4% of its weight there. The real difference is
   narrower and is stated in (c).

   **The fix follows from the derivative, not from a new objective.** Keep the count, move
   its weight. An item is predicted `c` exactly when `p_ic > max_{c' != c} p_ic'`, so
   soften THAT instead of summing probabilities:

       s_c = sum_i sigmoid( m_ic / T ),    m_ic = p_ic - max_{c' != c} p_ic'

   The derivative peaks at `m = 0` -- at the decision boundary, on the items one step from
   flipping -- and vanishes for items buried inside a class. Summed, it tracks the HARD
   count rather than the probability mass, so it is also the tighter relaxation of the same
   constraint. `m` is a function of an item's own row, so it needs no order statistic, no
   label, and nothing from the other chunks: the chunked detach construction still gives the
   exact full-N gradient and `constraint_chunk_size` stays a free knob.

   Shipped as `soft_count_mode: margin` + `straight_through` + `cut_window_items`
   (tralo only -- the duals never form this count), as the arms `tralo_margin` and
   `tralo_st`. Default `sum` is bit-identical.

   ⛔ **A DEAD END, recorded because it is seductive.** The obvious version centres the
   window on the K-th largest probability, `sigmoid((p_ic - tau_c)/T)` with `tau_c` the
   K-th order statistic. That quantity counts how many items exceed the K-th largest,
   which is **K - 0.5 for any model whatsoever**. It is a constant, `relu(s - K)` is
   identically zero, and it produces **no gradient at all**. It was derived, wired into
   the trainer, and caught by the chunked-gradient test before it ever reached a GPU.
   A soft count must be able to EXCEED the budget or there is no violation to see.

   🛑 **THE MOTIVATING STORY IS PARTLY WRONG, AND THE PROXY FOR IT IS INCOMPLETE.**
   Both measured GPU-free on the stored evidence (`scripts/reachability.py` only
   READS predictions; 160 (class, cell) points, 3 datasets, 4 seeds, end-of-run).

   **(a) `sum` is NOT blind to the decision boundary.** It puts 29.4% of its total
   per-item gradient on the 20 items nearest `m = 0` -- 2% of the items, so 15x
   uniform. The older reasoning measured `p(1-p)` at the **K-th RANKED item** (0.94
   early, 0.999 converged, so ~0), but **rank K is not the decision boundary**: with
   a hard count of 300 against K = 44 the boundary is at item 300 and rank 44 is
   buried inside the class. Items flip at `m = 0`, where `p_ic` is near the
   runner-up and `p(1-p)` is near its MAXIMUM. Both numbers are right and they
   describe **different points**. ⇒ never repeat "the gradient cannot reach the cut"
   without saying which point it is about.

   **(b) On the target that actually matters, the window buys only 1.30x.** The set
   that must move is not "the 20 nearest zero" -- it is the `hard - K` items with
   the smallest POSITIVE margin, the ones that have to flip out. Median excess is 52
   of 157 predicted. Share of gradient landing on THAT set, 134 violating points:

       sum                    38.0%
       margin,   2 items      49.3%   1.30x   <- the best available
       margin,   5 items      41.9%   1.10x
       margin,  10 items      29.0%   0.76x
       margin,  40 items       8.5%   0.22x

   ⇒ **gradient placement is worth at most 1.30x**, and only at a 2-item window,
   which is a near-delta direction that `normalize` then scales to unit norm.

   **(c) The arm is DECOMPOSED, because it bundled two independent fixes.** The
   count's VALUE (`sum_i p_ic` tracks probability mass, not the hard count) and the
   gradient's PLACEMENT (`p(1-p)` vs the margin window) are separate defects, and a
   result from an arm that changes both cannot be attributed to either. Three arms:

       arm            count value      placement
       tralo          soft sum_i p     p(1-p)        the manuscript
       tralo_st       HARD             p(1-p)        value fix alone
       tralo_margin   HARD             margin        both

   `gen_campaign` warns if `tralo_margin` is selected without `tralo_st`.

   ⛔ **AND THE TOY DECIDES NOTHING -- I CHECKED AND IT DOES NOT REPLICATE.** The
   first run of `scripts/flag_live` gave 31 / 35 / 32 / 12 against K = 11, which
   reads as "placement does all the work". It is one seed. The same four arms:

       seed 1    null 31   sum 35   st 32   margin  12
       seed 2    null  0   sum  0   st  0   margin   0     cap never bound
       seed 3    null 120  sum 120  st 119  margin 119     everything saturated

   The harness has random labels and a 4-layer net that reaches chance accuracy, so
   ordering on it is noise. 🔑 **Only CONNECTEDNESS survives that harness** -- which
   is all `flag_live` claims, and the gate now runs three seeds, skips seeds where
   the cap never bound (there the penalty is identically zero and every arm is
   correctly bit-identical -- calling that "inert" was a false alarm it used to
   raise), and prints those three rows so nobody reads an ordering out of it.

   ⇒ **nothing is yet known about whether the window helps.** The pre-run evidence
   is: `sum` is already 15x uniform at the boundary, placement buys at most 1.30x on
   the items that must flip, and the concentration proxy is itself incomplete
   (`sum_i p_ic` can be reduced without moving one item across `m = 0`;
   `sum_i sigma(m_ic/T)` cannot -- and that difference is not a concentration
   statistic). The three arms exist so a real campaign settles it.

   🎯 **T IS DERIVED, NOT CONFIGURED, AND THAT WAS FORCED BY MEASUREMENT.** The
   first version shipped `cut_temp: 0.02`, a guess. On the same evidence **T = 0.02
   puts 1.4 to 1.9 items inside the window on every one of the three datasets**, so
   the arm would take a near-zero step, report a null, and write `completed`. The T
   holding 20 items spans **0.074 to 0.502 across cells** -- 6.8x -- and margins
   grow through the constraint phase as CE converges. One absolute number over a
   quantity whose scale differs per seed and per epoch is exactly the invisible dose
   gap `constraint_grad_mode: normalize` fixes on the other axis. So the knob is a
   width in ITEMS: dimensionless, and an empty window is impossible by construction.

   ✅ **A PREDICTED FAILURE MODE THAT DID NOT MATERIALIZE, recorded because it was
   predicted first.** `m_ic` contains the competitor explicitly, so reducing it can
   act by RAISING the runner-up -- and when the runner-up is the other capped class
   that is the see-saw (class 4 pulled to 57 while class 2 rose to 410). Measured on
   real stored predictions, penalty on capped class A only, one unit-norm step in
   logit space, 8 (cell, seed) points:

       count     A removed   B raised   collateral per A removed
       sum         -1.62       +0.25            0.15
       margin      -1.88       +0.25            0.13

   No amplification: `margin` removes 1.16x more per step at slightly LESS
   collateral. ⚠️ This is the ideal per-item direction in logit space, not what the
   network can deliver through shared parameters -- it rules the failure mode out
   as a property of the objective, not as a property of the training.

   ✅ **A side effect worth having: `straight_through` closes the K == 0 trap.** A
   group holding no true instances of the capped class gets `K == 0` legitimately,
   and on the soft value that constraint can NEVER be satisfied -- `sum_i p_ic` is
   strictly positive for any softmax even when the class is predicted for nobody in
   the group, so `relu(count - 0)` stays positive for the whole run, contributing
   nothing while holding the ratchet gate open for every other constraint. The hard
   count can be exactly zero. This is a standing warning in this project and
   `straight_through: true` removes it; a test pins the difference.

   🛑 **THE STRUCTURAL PRIOR, WRITTEN DOWN BEFORE THE RUN.** Post-hoc assignment is
   optimal GIVEN THE PROBABILITIES -- greedy over its candidate neighbourhood, the LP
   exactly. So a trained arm cannot win by extracting threshold information: post-hoc
   already extracts all of it. It can only win by producing DIFFERENT probabilities,
   i.e. by changing the representation, and the count penalty carries **no label
   information**. That makes every count-based arm a label-free REGULARIZER, and the
   closest prior for a label-free arm in this project is `rank` -- self-referential,
   top-K vs rest -- which was a **null**.

   And the margin count sharpens rather than softens that. `relu(count - K)` with
   `count = sum_i sigma(m_ic/T)` is minimized by pushing POSITIVE margins down and
   across zero; items already at `m < 0` contribute `sigma ~ 0` and are untouched. So
   the force is precisely "flip the smallest-positive-margin members of class c out"
   -- **which is what the greedy clipper does post-hoc, for free.** The arm is doing
   in training what the baseline does after it, plus a representation side effect.

   ⇒ **the pre-registered reading of a tie is: this is that trap, again**, and the
   next move is a genuinely per-item objective with LABEL information, not a third
   count. The pre-registered reading of a WIN is that the representation side effect
   is real, and it must then show up as **ccP at the operating point**, not as AUROC.

   ⚠️ One honest distinction from `rank`: `rank` had no notion of an operating point
   -- it separated top-K from rest globally. This acts only on items within `T` of
   the decision boundary, which is where a per-item objective would act. That is why
   it is worth one campaign and not more.

   ⚠️ **It is still an AGGREGATE count**, which is what path 1 says cannot win. The reason
   it is listed anyway is that path 1's own escape hatch is "per-item AT THE OPERATING
   POINT", and this is an aggregate whose WEIGHT is concentrated there. If it fails, it
   fails as evidence FOR path 1 stated strictly, and the next move is a genuinely per-item
   objective, not another count.

   ✅ **THE CAP LEVELS ARE CHOSEN BY MEASURED HEADROOM, not by habit.** Ceiling on
   capped-class F1 with exactly K predictions emitted is `2K/(K+n)` (recall <= K/n,
   precision <= 1). Measured against the stored dermmnist evidence, 4 seeds:

       cap        clip capF1   ceiling   headroom
       L30_G30       0.4255    0.4621     0.0366
       L50_G50       0.5942    0.6688     0.0746

   Reproduce with `python -m scripts.headroom <root> --control clip`.

   ⚠️ **CORRECTED 2026-08-21. The first version of this table read 0.0290 /
   0.0597** and came from a throwaway script that is now `scripts/headroom.py`.
   The script equalized the control against the GLOBAL budget only, dropping the
   local caps that `full_panel` passes -- a more permissive allocation, so the
   control scored too high and the headroom too low. Scoring the way the scorer
   scores, `achieved` reproduces the stored allocation's F1 to four decimals at
   both levels, which is the check that says the two agree. The CEILINGS were
   correct and are unchanged.

   ⚠️ **These are NOT the archived headroom numbers, and the difference is the metric,
   not a contradiction.** The archived table (0.048-0.059 at L30, 0.115-0.131 at L50,
   mean 0.0669) is **single-class**, class 4 alone, K = 67 and 112. The numbers above are
   **macro-averaged over two capped classes** on the multi-class campaign. The CEILINGS
   agree to four decimals -- 0.462 and 0.669 both ways -- which confirms the same formula
   is being applied; only the achieved value differs, because a macro average over two
   classes is a different quantity from one class's F1. ⇒ **never compare a headroom
   figure across campaigns without checking how many classes it averages.**

   🎯 **AND THE CAP LEVEL IS A TRADE-OFF, measured -- neither end is free.** How far over
   budget the model starts, same runs:

       cap        class   hard      K   excess   % over   headroom
       L30_G30      1    107.8     31     76.8     248%    0.0266
       L30_G30      2    233.5     66    167.5     254%    0.0315
       L30_G30      4    177.5     67    110.5     165%    0.0517
       L50_G50      1    107.8     52     55.8     107%    0.0452
       L50_G50      2    233.5    110    123.5     112%    0.0742
       L50_G50      4    177.5    112     65.5      58%    0.1043

   ⚠️ **ALSO CORRECTED 2026-08-21, and this one was internally impossible.** The
   first version gave class 1 a raw count of 79.8 at L30 and 68.0 at L50 -- but
   `clip` never sees the cap during training, so its model is **bit-identical
   across cap levels** (verified: md5 of `final_predictions_raw.csv` matches at
   all 4 seeds). Its argmax count therefore CANNOT depend on the cap, and the
   corrected column is the same 107.8 / 233.5 / 177.5 at both levels, exactly as
   that invariance requires. The old numbers came from averaging the raw count
   over every arm in the tree -- clipper and trained arm together -- which
   describes no model at all.

   A tight cap gives the constraint **2-3x more items to move and half the headroom to
   win**; a loose one the reverse. ✅ Two things follow for the campaign: at L40/L50 the
   excess is still 56-124 items, so the constraint is **not inert** there -- it is not a
   case of picking a cap so loose that nothing binds -- and every cell starts **58-254%
   over budget**, so the "seed already satisfied, takes no step" failure is unlikely at
   these levels.

   Headroom roughly DOUBLES from a 30% to a 50% cap, and the paired difference sd is
   0.0017 (L50_G30, linear) to 0.04 (L30_G20). ⇒ **L50 and L40 can resolve a win; L30
   is marginal and L20 cannot resolve one in principle.** That is why the campaign is
   `L50_G30` + `L40_G30` -- both above the resolvable line AND both with `G < L`, so the
   global scope actually binds. ⚠️ `tralo_uniform` shows the LARGEST headroom (0.048,
   0.071) purely because it scores the LOWEST -- headroom is not opportunity when the
   arm is simply worse.

   ⚠️ **Falsification conditions, fixed in advance:** it must (a) move **ccP**, not AUROC
   -- an arm that moves AUROC has been run twice, as `budget_margin` and as the shipped
   penalty; (b) beat `constraint_random_direction` at the same norm; (c) hold across
   4 seeds and >= 2 cap levels. `T` is a real dose knob and must be **dosed against the
   margin table**, not guessed: if `in win` is 0 the arm takes a zero step, and if it is
   the whole test set the sigmoid is flat and it has silently degraded back to `sum`.
   Either way the run measures nothing, and neither shows up as an error.

1c. **Optimise the metric at the budget, using LABELS** -- proposed 2026-08-21, not built.

   This is where the day's measurements point, and it is the only proposal here that
   escapes the trap 1b is stuck in. Three facts, each established above:

   - post-hoc assignment is optimal GIVEN the probabilities, so a trained arm can only
     win by changing the REPRESENTATION;
   - the count penalty carries **no label information**, so every count-based arm is a
     label-free regularizer, and the label-free prior in this project (`rank`) is a null;
   - the cap itself is recoverable from training prevalence, so it adds ~nothing.

   ⇒ **stop penalising the test count and start optimising the training objective AT
   THE OPERATING POINT, with labels.** The score is precision@K on the capped class
   (ccP; ccF1 is a rescaling of it). Nothing in the pipeline optimises that -- CE
   optimises average log-likelihood over all items, which weights the whole ranking
   equally and the cut not at all. A top-K surrogate on the TRAINING set does: for each
   capped class take the items near the rank the budget implies and push true positives
   above false positives across exactly that cut.

   Why it is not a repeat of section 2: it uses **labels** (every rejected arm was
   label-free), it acts **per item at the cut** (every rejected arm acted on an
   aggregate), and post-hoc **cannot replicate it** -- post-hoc is optimal given the
   ranking, and this changes the ranking using information post-hoc does not have.

   The transductive cap keeps exactly one job in this design, and it is a real one:
   **it says which K to optimise for.** That is the honest scope of the contribution
   the data can support, and it is smaller than the manuscript currently claims.

   📚 **A REVIEWER QUESTION TO PRE-EMPT, and the answer is in our favour.** The obvious
   objection to any "we know something about the test set" paper is: *if you know the
   test class proportions, the Bayes-optimal move is closed-form posterior reweighting,
   `p~(y|x) ~ p(y|x) * pi_test(y)/pi_train(y)`* -- the Saerens-Latinne-Decaestecker
   adjustment, and the whole prior-shift / quantification literature behind it (SLD 2002;
   "Dataset Shift and the Adjustment of Probabilistic Classifiers" 2018; "Adaptation of
   CNN Classifiers to Prior Shift" 2021; Sebastiani's quantification surveys). It needs
   no training at all, so it would dominate every arm here.

   **It does not apply, and the reason is worth stating in the paper.** Our cap is not the
   test prior. `K` is a FRACTION of the true test count -- measured on dermmnist, class 2
   has `n_true = 220` and `K` is 66 at a 30% tag, 110 at 50%. So the constraint is
   `K << n_true`: a BUDGET, not a prior match. Reweighting posteriors to a prior we do not
   have, and that is not what the cap states, answers a different question.

   ⚠️ **And that cuts against us too.** Under a budget `K < n_true` the optimal decision
   rule GIVEN the model is "take the K items most likely to be class c" -- which is
   exactly the clipper. That is the third independent derivation today of the same
   conclusion (the others: post-hoc is optimal given probabilities; the cap adds no
   information the training set lacks). ⇒ the setting is a **top-K selection problem**,
   which is why 1c is the proposal that fits it and why every count penalty does not.

   📚 **Literature checked so far (2026-08-21), and one lead still open.**

   - **Posterior regularization** (Ganchev, Graca, Gillenwater, Taskar) constrains the
     posterior's EXPECTATIONS during training, which is exactly `E[count_c] <= K`. It is
     the closest framework to TraLO itself. ✅ Already in `references.bib` as
     `ganchev2009posterior` and cited once in both live manuscripts.
   - **Prior shift / quantification** (SLD; Elkan 2001, in the bib): does not apply, see
     the budget-vs-prior argument below. ⚠️ Saerens/SLD, learning-from-label-proportions
     and BBSE-style label shift are **absent from the bib entirely** -- one pre-emptive
     sentence plus citations is cheap insurance against the obvious reviewer question.
   - **Top-K / precision@k surrogates**: well established (ICCV 2019 "Sampling Wisely";
     ICML 2023 "Weighted Sampling without Replacement for Deep Top-k Classification";
     NeurIPS 2024 "ST_k"), so 1c is not a new loss family.

   🔑🔑 **A PUBLISHED COUNTEREXAMPLE TO THE PESSIMISM ABOVE -- verified, and it changes
   the outlook.** "Predict at most K items as class c" is structurally a **COVERAGE
   CONSTRAINT**, and selective classification optimises coverage-constrained risk during
   TRAINING. **SelectiveNet** (Geifman & El-Yaniv, *ICML 2019*, "SelectiveNet: A Deep
   Neural Network with an Integrated Reject Option") states the setup we are in almost
   exactly:

   > "Existing rejection mechanisms are based mostly on a **threshold over the prediction
   > confidence of a pre-trained network**. In contrast, SelectiveNet is trained to
   > optimize both classification and rejection simultaneously, end-to-end. The result is
   > a deep neural network that is **optimized over the covered domain**."

   **Their baseline IS our `clip`** -- threshold a pretrained net's confidence -- and they
   report a consistently improved risk-coverage trade-off against it.

   ⇒ **the escape from "post-hoc is optimal given the probabilities" is in their own
   words: optimised over the COVERED DOMAIN.** Post-hoc is optimal given the probabilities,
   but training can change which probabilities you get, by fitting the model to the
   sub-population it will actually predict on rather than to all of it. That is a
   representation change thresholding cannot produce, and it is the one mechanism the
   three pessimistic derivations do not rule out.

   ⚠️ **The differences are real and must not be glossed.** Their coverage is one global
   rate chosen by the user; ours is per-class counts on a specific test set, coupled
   across classes and groups. Their metric is risk at coverage; ours is precision@K. And
   they train the selection head jointly from scratch, where every arm here bolts a
   penalty onto a warm-started CE model. ⇒ this makes 1c **plausible**, not proven, and
   the design to copy is the joint selection head, not the penalty.


   ⚠️ **Novelty is the pairing, not the loss.** Top-K / precision@k surrogates and
   learning-to-rank losses are well established; this is not a new loss family. The
   claim would be "a transductive budget tells you the operating point, and optimising
   there beats post-hoc thresholding of a CE model" -- which must be checked against
   label-shift, prior-correction and top-K optimisation literature BEFORE building it.

   🎯 **THE MINIMAL FORM, and it reuses machinery that already exists.** SelectiveNet
   trains a selection head jointly; that is a large change. The smallest thing with the
   same mechanism -- fit the model to the sub-population it will actually predict on --
   is to **re-weight the CE loss on the TRAINING set toward the operating point**:

       w_i  =  1 + eta * sum_{c capped} sigma'( m_ic / T )

   with `m_ic` the margin (`src/losses/transductive_loss.py: margins`) and `T` derived by
   `window_temp` from `cut_window_items`, exactly as `tralo_margin` does. Items sitting at
   a capped class's decision boundary get their CE up-weighted; items buried inside a
   class, which cannot flip, do not.

   Why this is not another count arm, point by point against path 1's requirements:
   it uses **labels** (it IS the CE loss), it is **per item**, it acts **at the operating
   point**, and it changes the RANKING -- which is the only thing post-hoc cannot do for
   itself. It is also ~20 lines, one hyperparameter (`eta`), and needs no new head.

   ⚠️ It differs from SelectiveNet in the way that matters: they *exclude* the uncovered
   domain, this only *down-weights* it. Whether that is enough is the experiment. And the
   operating point is defined on TRAIN margins while the budget lives on TEST -- the two
   coincide only because the split is stratified, which is the one place this design leans
   on the property proved in path 2.

   ⚠️ **And it may still lose to `clip`**, because CE + optimal post-hoc is a strong
   baseline and the training and test operating points differ by sampling noise. Build
   it only after 1b reports, and only with the same in-campaign bars.

2. **A regime where post-hoc is not optimal.** Post-hoc greedy is optimal only over its own
   candidate neighbourhood. It is weakest where the assignment is **coupled**: several capped
   classes plus local per-group caps. It is also uninformative where **train and test prevalence
   are identical**, which is true of every cap we have run -- the cap tells the model nothing it
   does not already know.

   🛑🛑 **AND THAT IS TRUE BY CONSTRUCTION, NOT BY ACCIDENT -- read `create_slices.py`.**
   Every slice is a `StratifiedShuffleSplit` on the label, and the script ASSERTS that each
   class's train and test percentages agree to within 1 point. So `n_test_c ~ 0.25 * n_train_c`
   to within a single item, and the cap `K = round(pct * n_test_c)` is recoverable from the
   TRAINING prevalence alone. ⇒ **the transductive cap carries essentially zero information the
   training set does not already contain** -- roughly the rounding, and nothing else. This is
   not an observation about our results; it is a property of how the data was made.

   ⚠️ **Be precise about what that does and does not say.** It does NOT say the cap is useless:
   `clip` binds on 63-84% of runs, so the trained model's argmax counts really do differ from
   the true test counts. But that is a MODEL error, not new information -- the model failed to
   absorb a prior that was in its own training data. So the cap is redundant with the TRAINING
   SET and not with the TRAINED MODEL, and correcting a trained model's count is exactly what
   post-hoc does, optimally, for free. ⇒ the honest form of the claim: **every bit the cap
   carries was available at training time, so any use of it that post-hoc cannot replicate has
   to come from the REPRESENTATION, not from the count.** That is the same conclusion path 1
   reaches from the other direction, and it is why they keep meeting.

   ⇒ **the only way to make the cap informative is a test set whose prevalence DIFFERS from
   train.** There are two routes and they are not equally costly.

   🛑 **Re-slicing is Roei's call and probably not needed.** New splits invalidate most of the
   stored corpus, and the same file already carries that warning for the leakage finding.

   ✅ **The safe route is a load-time SUBSAMPLE of the existing test set.** Drawing a subset of
   `slice_1`'s test items to hit a target prevalence touches **no file**, leaves every existing
   slice byte-identical, and invalidates nothing -- it is a selection in `src/utils/data_loader.py`
   (which today reads `test_labels.npy` whole and has no subsampling path), default off. Every arm
   in the campaign sees the identical subsample, so arm-vs-arm is valid; only comparison to the
   stored corpus is not, and that comparison is not what the regime is for. It also costs test-set
   SIZE, which is the real price: a shift big enough to matter on a rare class can leave few items,
   and the campaign must be powered for the subsample and not for the full test set.

   ⚠️ Not implemented, and not to be implemented unasked -- it changes the data path. **Distribution shift is the untested regime where the cap carries real
   information.** Novelty must be checked against label-shift / prior-correction work first.

**Anything that is not one of those is a repeat of section 2. Do not run it.**

### What 2026-08-21 changed about both of them

**Path 1 is weaker than stated, in the method's favour.** The claim was that only a
per-item objective can move the operating point, because a gradient that is a function of
the aggregate count carries no information about WHICH items should rank higher. Measured
against a matched lambda=0 control, the aggregate-count penalty **does** move it -- with
the right shape at the right cap. `linear` at `L50_G30`: prec@K +0.030 at protocol length
(n=1), `d capF1` +0.0078 over 4 seeds with sd 0.0017, and it beats a random step of the
SAME norm (-0.0130) with non-overlapping distributions. It lifts precision@k on class 2 at
every k from 20 to 150, which is a broad re-ranking.

⇒ **the count gradient is not information-free.** Pushing a class's mass down is not
uniform across items -- the items nearest the boundary move most -- so an aggregate
penalty does re-rank. It just has to be **monotone** (the shipped bounded shape is
negative at every cap) and it needs a cap **loose enough that the ranking has somewhere to
improve**. Path 1 should be widened from "per-item only" to "**anything that measurably
improves prec@K against its own lambda=0 control and against a same-norm coin**".

**Path 2 is narrower than stated, against the method.** "Post-hoc greedy is optimal only
over its own candidate neighbourhood, weakest on coupled multi-class + local caps" is now
quantified, and the neighbourhood is nearly the whole space. The greedy clipper does ONE
JOINT pass over every (item, capped class) pair, so it already solves the coupling; run
against the LP optimum on identical probabilities it gives up **~0.005-0.015 capF1**, and
**exactly 0** whenever every capped class sits above its budget.

⇒ **coupled multi-class + local caps does NOT make post-hoc meaningfully suboptimal.**
Do not build a method whose thesis is "post-hoc handles the coupling badly" -- there is at
most ~0.01 there, and the measurement says it is usually 0. What remains of path 2 is the
**distribution-shift** regime, where the cap carries information the training set does not.

🚨 **And the real obstacle is neither of these.** At protocol length the constraint gained
+0.030 prec@K over its own control while the warm-up-1 training path started it **-0.075
below `clip`** (n=1, seeds running). The method works; the regime it is required to run in
costs more than the method earns. That gap -- not the penalty -- is what a win has to
close.

---

## 5. Repository layout

```
main.py            dispatcher (kill -INT to stop; interrupted runs reset to pending)
configs/protocol.yml   EVERY experimental constant -- epochs, seeds, lr, caps, arms, backbones
configs/gen_campaign.py  THE generator: reads protocol.yml, holds no constant of its own
data/              dermmnist, octmnist, tissuemnist -- nothing else
docs/FRAMEWORK.md  this file
docs/archive/      history, not instructions
docs/paper/        the TMLR manuscript (main.tex is the professor's -- never edit)
results/           experiment outputs
scripts/full_panel.py  THE scorer (+ score_arm.py = the equalizer it calls)
scripts/audit_config.py  every hyperparameter has a reader; base_model_id is complete
scripts/smoke_arms.py    every arm runs end to end on synthetic tensors, ~40 s; --matrix adds the trained arms' caps
scripts/verify_caps.py   the caps bind, on the real dataset slices
scripts/check_parity.py  equal compute, shared knobs, warm-up cache sharing
scripts/prep_*.py        dataset preparation
src/               the pipeline: losses, methodologies, models, pipeline, training, utils
tests/             152 tests, ~35 s, no dataset required
evidence/          TWO tarballs that must be extracted into ONE tree to be scorable:
                   provenance_*.tar.gz  = config.json + evaluation_metrics.csv +
                     training_log.csv for 14,524 runs. NO predictions.
                   predictions_*.tar.gz = final_predictions{,_raw}.csv for 128 runs
                     (`mcbar` and `multiclass` only). NO configs.
                   full_panel globs **/config.json and needs BOTH prediction files,
                   so NEITHER tarball alone yields one scorable run, and only
                   128 / 14,524 = 0.9% can be re-scored at all. Every campaign
                   carrying a rejected-arm verdict -- nsteps, sepopt, granularity,
                   headroom, joint, beta, rank*, ortho, budgetprobe, mcbar_regnet,
                   mcbar_duals, mcjoint -- has configs and logs but NO predictions.
```

Nine methodologies: `tralo` - the duals `fioretto_ldf` / `hounie_rcl` / `fioretto_alm` -
the two allocators `heuristic` (greedy) / `danits_lp` (LP-LG) - and the imbalanced recipes
`focal` / `class_balanced` / `logit_adjust`, which are LP-clipped.

## 6. Evidence appendix

The full run-by-run record, with numbers, p-values and cell counts, is preserved at
`docs/archive/REJECTED_full_2026-08-18.md`. It is history, not instructions.
