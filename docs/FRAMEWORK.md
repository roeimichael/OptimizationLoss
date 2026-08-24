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

✅ **INDEPENDENTLY CONFIRMED 2026-08-23, and the ORDER is what makes this
quotable.** The choice was made a priori on 2026-08-20 for design reasons. Only
afterwards was the corpus asked which backbone best RESOLVES a method-vs-method
question, and ViTB16 wins on both terms of the ratio (macro-F1, warm-up 50,
`tralo - fioretto_ldf`):

| backbone | cells | method gap | wins | seed sd | gap / sd |
|---|---|---|---|---|---|
| **ViTB16** | 27 | **+0.27 pp** | 74% | **0.33 pp** | **0.82** |
| MobileNetV3 | 117 | +0.18 pp | 69% | 0.48 pp | 0.38 |
| RegNetY400MF | 42 | +0.13 pp | 58% | 0.57 pp | 0.23 |
| MobileNetV2 | 15 | +0.09 pp | 69% | 0.66 pp | 0.14 |

**ViTB16 has the largest method gap AND the smallest seed noise, so it resolves
the method question better than twice as well as MobileNetV3.**

🛑 **This is a confirmation, NOT the reason, and the distinction is the whole
point of the note above.** Had the corpus been read first and the backbone
chosen from it, this would be selection on the outcome. It was not; the decision
is dated 2026-08-20 and this measurement 2026-08-23. Quote it in that order or
not at all.

⚠️ **Note for the clipper contrast, which behaves differently.** Against
`heuristic` the best backbone is RegNetY400MF (+3.51 pp, 93% of cells), and
**ShuffleNetV2 REVERSES (-0.75 pp, 33%)** -- a second reversal beside aider's,
on a backbone since deleted from scope. The backbone that best answers "is TraLO
better than a clipper" is not the one that best answers "is TraLO better than
Fioretto", and neither is a reason to move the headline.

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
reports 4), while writing `status: completed`.

✅ **IT IS NOW COUNTED AND REPORTED. Fixed 2026-08-22.** `finish_constraint_step` has
always returned `applied` -- False when the norm is non-finite, because on the FP16 path a
NaN norm fails the `> 0` gate and an inf norm is skipped inside `scaler.step` -- and all
four trainers bound it to `_applied` and dropped it. Two arms in one campaign could take
29 and 19 steps with nothing able to say so, and **a dropped step leaves no trace in the
predictions except the effect it did not have**, so it is not recoverable from any metric.
Each run now writes `constraint_steps_applied` / `constraint_steps_attempted` into
`config['results']`, and `full_panel` prints a **CONSTRAINT DOSE** block that names any arm
which lost a step and refuses to let a >5pp gap in applied-fraction pass unremarked.
⚠️ The denominator is *epochs that formed a constraint gradient*, not
`constraint_epochs`: a satisfied cap yields a zero penalty and no step is attempted, which
is the zero-dose arms' normal state and must not read as a loss. `constraint_fp32: true` decouples the
constraint pass from the CE loss scale. ⚠️ fp32 doubles the chunked-forward memory:
`constraint_chunk_size: 256` OOMs on the 24 GB Quadro RTX 6000 (dsisco01) at
ViTB16, 128 fits. **The card is 24 GB, not 22** -- this file, `protocol.yml`
and `scripts/dose_scan.py` quoted three capacities for one OOM, and the
project owns exactly two GPUs (dsisco01's 24 GB Quadro RTX 6000 and dsisco02's
96 GB RTX PRO 6000 Blackwell), so a 22 GB card is not one of them and 96 GB
would not have OOM'd. A test now pins the three sites to one figure.

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

🚨🚨 **THE COIN WAS OVER-DOSED, SO THE SEPARATION ABOVE IS BIASED THE FLATTERING WAY.
Found and fixed 2026-08-21 -- treat the two coin rows as PROVISIONAL.**

`_randomize_direction` rescaled the random gradient to exactly `constraint_grad_clip`,
unconditionally. But under the protocol default `constraint_grad_mode: clip` the TREATMENT
delivers `min(raw, clip)`. So on every epoch where the clip did not bind, the control took
a **larger step than the thing it controls** -- measured by calling the real function:

| mode | raw norm | treatment delivers | coin delivered | over-dose |
|---|---|---|---|---|
| `clip` | 0.05 | 0.0500 | 1.0000 | **20x** |
| `clip` | 0.50 | 0.5000 | 1.0000 | 2x |
| `clip` | 5.00 | 1.0000 | 1.0000 | 1x |
| `normalize` | any | 1.0000 | 1.0000 | 1x -- clean |

⇒ the control varied **dose as well as information**, which is the one thing it exists to
hold fixed, and a bigger random step does more damage -- so the bias inflates the coin's
loss and flatters every shape it is compared against. Three documents asserted the
opposite ("same norm, no information"; "the dose is held exactly"). Fixed: the coin now
matches the DELIVERED norm in both modes, pinned by
`test_the_coin_control_matches_the_delivered_step_not_the_clip_bound`, which was verified
to FAIL on the old code.

⚠️ **The magnitude of the bias in these two tables cannot be recovered.** `randdir` appears
nowhere in the code (the arm is `tralo_coin`) and no `randdir` run survives on any disk, so
the mode those runs used is unrecoverable. It matters which: `scripts/dose_scan.py` forces
`normalize`, where the coin was always clean, while a `gen_campaign` campaign takes the
protocol default `clip`, where it was not. For `tralo` the recorded raw norms are
0.638-1826.5 with the clip binding 6 of 7 epochs, so at most ~1 epoch in 7 was over-dosed
there; for `hounie` the clip bound **0 of 29**, so every epoch was, by up to 20x.

✅ **WHAT SURVIVES.** `linear`'s **+0.0078 d capF1, 4/4 seeds** is measured against its own
`lambda=0` control, not against the coin, and does not depend on this at all. What is
provisional is the *second* claim -- "the direction carries information, a coin does not"
-- because that one rests entirely on the coin rows. ⇒ **re-run the coin arm before
repeating that sentence.**

🎯 **THE SHAPE AND THE CAP INTERACT, AND THE SHIPPED SHAPE IS NEGATIVE EVERYWHERE.**
A tight budget only reaches items the model already had right, so there is nothing for
a direction to earn; loosen it and a monotone penalty finds something a coin cannot.
⚠️ Still unestablished: this is ONE cell at 4 epochs against `null`. `clip` and
`focal_clip` at protocol length are the bar, and the gain does NOT show up in macro-F1
(+0.0015, 2/4 seeds) -- it is confined to the constrained classes.

---

### (13) ⛔⛔ THE CONSTRAINT'S EFFECT ON THE COUNT IS AT OR BELOW A DROPOUT RESEED

Measured on `results/dosefix`, 2026-08-22, and verified independently. RMS separation of
the capped-class hard count over epochs >= 4, per seed, then averaged:

| what changed | class 2 | class 4 |
|---|---|---|
| **the constraint** -- `tralo` vs its own lambda=0 twin @ L40_G30 | 75.6 | 85.3 |
| **the constraint** -- `tralo` vs its own lambda=0 twin @ L50_G30 | 74.6 | 95.2 |
| **nothing but the RNG stream** -- two pure-CE runs | **82.8** | **95.0** |

🔑 **THE NOISE FLOOR WAS ALREADY IN THE DATA.** `select_null` sets `select_eta: 0`, so its
loss is `ce + 0 * sel_loss` -- a PURE CE run on the same seed and the same warm-up cache as
`tralo_null`. It differs only in that its selection head draws from the global RNG, so the
dropout masks and batch order are a different stream. It is a free, in-campaign,
same-warm-up reseed control, and nobody had used it as one.

⇒ **Turning the constraint on perturbs the capped count by LESS than reseeding dropout
does** -- 0.90-1.00x, on both classes, at both cap tags. The constraint is not weak but
directional. Its whole measurable footprint on the count is a re-randomisation.

⚠️ **THIS IS THE CONTROL EVERY COUNT TRAJECTORY HAS BEEN MISSING.** "The constraint moved
the count 75 items" is not a result until it is stated as "the constraint moved it 75 and a
reseed moves it 83". Carry a reseed arm in every campaign that reads a count.

✅ **IT IS NOW AN ARM, AND THE GENERATOR REFUSES A CAMPAIGN WITHOUT IT (2026-08-22).**
The floor above came out of `select_null` by accident, and an accident is not a control:
that arm is `select`-methodology, is REJECTED (section 12), and nothing kept its
perturbation fixed. `tralo_reseed` makes it deliberate.

| | what it is |
|---|---|
| definition | `blocks: [constraint_phase, tralo_null, tralo_reseed]` -- it CARRIES `tralo_null`'s block and overrides ONE key, so the two cannot drift apart. Verified: the assembled hyperparameters differ in `rng_reseed` and nothing else. |
| dose | zero. `lambda_global = lambda_local = lambda_step = 0`, so `total_constraint == 0`, `has_constraint` is False and transductive pass 2 is skipped entirely. Pinned by spying on `constraint_backward` / `finish_constraint_step`. |
| perturbation | ONE draw from the global generator, `torch.rand(1)`, at the top of `tralo/train.py`. Everything downstream that consumes it -- the DataLoader's per-epoch shuffle seed, CPU dropout -- then runs on a different stream. |
| warm-up | SHARED with `tralo` and `tralo_null` (one `base_model_id`). The draw is inside the constraint phase precisely so it cannot reach the warm-up: `run_experiment` re-seeds after the warm-up, and a draw before it would change WHAT GETS CACHED depending on which of the three arms the dispatcher happened to run first. |
| scoring | `full_panel._zero_lambda_arms` reads it as UNTREATED from the config, not from the `_null` suffix -- which it does not have. |

🔑 **The gate is a REFUSAL, not an auto-add.** A trained arm is exactly what writes a
per-epoch capped-class count, so `gen_campaign` refuses any campaign holding a trained
arm without `tralo_reseed`, names the arm to add, and prints what the pair is for. It is
excluded from `--arms all` for the same reason the zero-dose siblings are -- adding a
trained arm is a compute decision, and silently growing what `all` costs is the scope
expansion this project has a rule against. `--arms all+null` carries it. Post-hoc-only
campaigns write no count trajectory and are not affected.

⚠️ **ONE reseed arm serves the whole campaign**, on the same argument that gives the coin
one arm: at lambda 0 the tralo path is 1 warm-up epoch + 29 CE epochs, which is the regime
every trained arm shares, and the floor being measured is a property of that regime rather
than of a dual rule. A `fioretto`-specific reseed would measure the same thing twice.

🔬 **MECHANISM, derived and verified by autograd.** With the shipped `sum` count the penalty
is `P = sum_c w_c * sum_i p_ic`, and one step on the logits changes the count by
`dS_j = -eta * sum_c w_c * sum_i p_ij p_ic (delta_jc - p_ij - p_ic + ||p_i||^2)`. The
cross-term (`j != c`, both capped) is NEGATIVE whenever `p_j + p_c > ||p_i||^2` -- i.e. on
exactly the items where the two capped classes hold most of the mass -- so pushing class 2
down pushes class 4 UP. Verified on a 2-vs-4 confusion item (p2 = p4 = 0.45): class 2 moves
-0.134 while class 4 moves **+0.040**, so 30% of the push is traded rather than evacuated.
**The mass is not removed, it is exchanged.** That is the see-saw, in closed form.

⚠️ **BUT THE PROPOSED FIX DOES NOT BUY WHAT IT LOOKS LIKE IT BUYS.** Renormalising each
capped class over `{c} + uncapped` zeroes the cross-term by construction, and on that item
class 4 flips from +0.040 to -0.043. Measured over a full 2014-item population under the
protocol's `constraint_grad_mode: normalize`, however, the aggregate movement is
**0.95x, not larger** -- the per-item see-saw correlation falls from **-0.319 to -0.166**,
but `d(S2+S4)` does not improve. A per-unit-eta probe on an UNNORMALISED gradient shows a
2-3x gain; `normalize` renormalises exactly that gain away, and `normalize` is what we run.
⇒ Worth testing for the see-saw, not for the dose. Predict a halved cross-class
correlation and NO increase in count movement.

---

### (12) ⛔⛔ `select` (1c, the jointly-trained SELECTION head) IS REJECTED

`results/selectrun`, 32 runs, dermmnist x ViTB16 x {L70_G30, L70_G50} x 4 seeds,
both clippers in-campaign, scored 2026-08-22. **It is the worst arm this project
has measured.**

| vs `clip` | AP | AUROC | ccF1 | macroF1 | acc | cells |
|---|---|---|---|---|---|---|
| `select` | **-0.1096** | -0.0326 | **-0.0804** = **-22 items** | -0.0873 | -0.0341 | **0 of 2 on every metric** |
| `select_null` | +0.0018 | +0.0020 | -0.0053 = -1.5 items | +0.0169 | +0.0025 | 1/1, a tie |

🔑 **THE NULL SEPARATES THE TWO EXPLANATIONS AND KILLS THE ARM, NOT THE SETUP.**
`select_null` is the same warm-up, the same 29 epochs, the same allocator, with
`select_eta: 0`. It TIES `clip`. So the loss is not the training setup and not
warm-up 1 -- it is **the selective term itself**, and it costs 22 items.

🚨 **It also destabilises training.** 2 of its 8 runs collapsed on the final
epoch -- 0.9835 -> **0.6968** and 0.9881 -> **0.8368** -- against zero collapses
in 8 `select_null` runs and one in 8 `clip`. The pipeline keeps the last epoch,
so those collapses are the scored models.

⇒ **Do not re-run it, at any `eta`, `tau` or `cov_weight`.** The direction is not
marginal and the mechanism is understood: the coverage term trains the network to
*abstain*, which is a different objective from ranking the capped class, and the
allocator then has a worse ranking to threshold. Gap A1 (the selector's ordering
is discarded at test time) was the reason to build it; the measurement says the
ordering it produces is worse than the one CE gives for free.

✅ **What survives is the null's reading**, which is worth more than the arm: at
warm-up 1 with 29 further epochs, a trained arm that does NOTHING lands on
`clip`. Together with section 9's corrected -0.06 items that is a SECOND,
independent measurement that the warm-up-1 setup carries no handicap.

---

### (11) ⛔⛔ THE CAPPED CLASSES DO NOT COMPETE -- "coupled multi-class" is CLOSED

Measured 2026-08-21 on the stored predictions, and independently re-verified.

**The top-K sets of the capped classes are PAIRWISE DISJOINT in every one of 16
dermmnist seed-cells**, at all three cap levels, computed from the RAW probabilities (not
from the allocation, which is disjoint by construction and would make the check vacuous):

| cap | K per class (1 / 2 / 4) | pairwise overlap |
|---|---|---|
| `L30_G30` | 31 / 66 / 67 | 0, 0, 0 -- all 4 seeds |
| `L50_G50` | 52 / 110 / 112 | 0, 0, 0 -- all 4 seeds |
| `L70_G70` | 72 / 154 / 156 | 0, 0, 0 -- all 4 seeds |

✅ **LIVENESS CONTROL** -- the probe can detect overlap, so the zero means something.
Sweeping the rank R on one cell: R=110 -> 0/0/0; R=150 -> 3/2/0; R=200 -> 23/19/6;
R=300 -> 98/83/79; R=900 -> 744/722/740 against a random expectation of 404.

🔑 **THIS IS THE MECHANISM BEHIND "greedy and LP give up EXACTLY 0".** The joint pass never
has a conflict to resolve, because no item is near the budget boundary of two capped
classes at once.

⛔ **THEREFORE the standing hypothesis that "with several capped classes the assignment is
COUPLED, so post-hoc greedy is only a HEURISTIC there" IS REFUTED ON OUR DATA.** It was
recorded as the one real opening. It is not one -- not weakly, but completely: there is no
coupling to exploit. ⚠️ Note this also strengthens the monotone-invariance closure: that
argument is strictly single-class, and one might have expected the coupled allocator to
break it. It does not, and now for a measured reason.

⚠️ Measured on the LEAKED slice (the evidence tarball predates the lesion fix). Disjointness
is structural and should survive, but re-check it once corrected-slice predictions exist.

---

### (9) 🛑🛑 THE BASELINE IS NOT NEUTRAL: `tralo_null` STARTS ~5 ITEMS BEHIND `clip`

Measured 2026-08-21 on `results/dosefix` (corrected lesion-disjoint derm x ViTB16 x
{L50_G30, L40_G30}, 3 seeds so far of 4, both clippers in-campaign).

`tralo_null` is lambda = 0: one warm-up epoch, then 29 constraint epochs whose constraint
term is identically zero. `clip` is 30 warm-up epochs and no constraint phase. **Both are
thirty epochs of cross-entropy on the same data, scored with the same allocator. They
should tie.**

⛔⛔ **RETRACTED AT 4 SEEDS -- AND THE REASON IS A COLLAPSED CONTROL.** The table below was
3 seeds. With all four in, `tralo_null` - `clip` is **ccF1 -0.0002 = -0.06 items**, and
macroF1 FLIPS to **+0.0221** in the null's favour.

| seeds | d items (null - clip) | reading |
|---|---|---|
| 1, 2, 3 | -3.9, -6.9, -4.2 | a consistent ~5-item handicap |
| 4 | **+14.8** | reverses it |
| **all 4** | **-0.06 items** | **~zero** |

🚨 **Seed 4's `clip` run COLLAPSED on its final epoch** -- train accuracy 0.9934 -> **0.9116**,
where every other control run ends 0.9935-1.0000 -- and the pipeline keeps the last epoch
UNCONDITIONALLY (no LR schedule, `enable_checkpoint_restore: false` by design). So the
corrupted run IS the baseline at that seed: it scores ~15 items below its siblings and
EVERY arm "beats" it there. **One collapsed control reversed the sign of a 4-seed headline.**
`clip`'s across-seed sd here is **8.2 items**, or 1.6 excluding seed 4, against the 2.7 this
document assumes elsewhere.

⇒ **The honest statement is neither "-5.2 items" nor "no effect":** the handicap appears on
3 of 4 seeds and the mean is zero because one control fell over. ✅ `full_panel` now detects
a terminal-epoch collapse and says so, naming the arm and warning explicitly when the
collapsed run is the CONTROL.

⚠️ **AND THE DETECTOR HAD TWO BLIND SPOTS, BOTH ON THE CONTROL'S SIDE. Fixed 2026-08-22.**

1. **A post-hoc arm that loads a cached warm-up writes NO `training_log.csv` at all.**
   `src/pipeline/warmup.py` returns early on a cache hit and the five post-hoc trainers
   log nothing, so the file never exists. `clip` + `lp` share one `base_model_id`
   (`check_parity` gate 4 prints exactly that), as do `focal_clip` + `focal_lp` -- so
   whichever of each pair the dispatcher runs SECOND was structurally invisible, and
   when that one is the `--control` the "ONE OF THESE IS THE CONTROL" warning could not
   fire even though its weights are byte-identical to an arm that did collapse.
   `_terminal_collapse` now returns THREE answers (`collapse` / `ok` / `nolog`) instead
   of a tuple-or-`None` that meant both "healthy" and "no trajectory", and
   `_collapse_report` resolves a log-less post-hoc run through its shared warm-up and
   prints anything still undetermined rather than skipping it.
2. **The 0.02 threshold was calibrated for a ONE-epoch interval and applied to a SIX-epoch
   one.** The warm-up logs `epoch < 3` and then every `max(1, warmup_epochs // 5)`-th
   epoch, so at `warmup_epochs: 30` a post-hoc arm's rows are 1,2,3,6,12,18,24,30 while a
   trained arm writes 29 adjacent ones -- the control and the treatment were held to
   different bars. Measured over the 4,862 `training_log.csv` files in this repository,
   converged tail only, the per-interval spread grows about as `sqrt(gap)`: **sd 0.00152
   at gap 1 (n=15,464) against 0.00300 at gap 5 (n=43,785)**, so the threshold is now
   `0.02 * sqrt(gap)` and the panel prints the span it judged.
   ⛔ **It cannot be fixed in the LOGGER.** `compute_train_accuracy` iterates the
   `shuffle=True` train loader, and a DataLoader iteration draws its permutation seed from
   the global RNG -- so logging a different SET of epochs changes every later epoch's
   batch order, the result, and every cached warm-up. **Logging density is part of the
   numerics here.**

*(superseded 3-seed table, kept so the retraction is legible)*

| | AP | AUROC | ccP | ccF1 | macroF1 | acc | cells won |
|---|---|---|---|---|---|---|---|
| `tralo_null` - `clip` | -0.0383 | -0.0087 | **-0.0404** | **-0.0188** | -0.0198 | -0.0078 | **0 of 2** on 12 of 13 metrics |

At this campaign's scale of **2.78 items per 0.01 ccF1**, that is **-5.2 items**. The whole
gap from `clip` to a PERFECT allocator is 1.9-9.9 items. ⇒ **the untreated arm gives away
more than half the available headroom before the constraint does anything**, and every
tralo-vs-clip number this project has produced sits on top of that handicap.

**THE ASYMMETRY IS STRUCTURAL, AND IT IS IN THE HARNESS, NOT THE METHOD.**
`src/pipeline/warmup.py` returns `(model, cached)` -- the Adam optimizer is a LOCAL and is
discarded -- and the trained arms then build a **fresh** Adam. So `clip` runs its 30 epochs
under ONE Adam, while `tralo_null` runs epoch 1 under Adam #1 and epochs 2-30 under Adam
#2, with the moments and the bias-correction step counter reset at epoch 2 of 30. Two
smaller asymmetries ride along: a second `GradScaler` restarts its back-off (fp16 only, so
exactly zero on dsisco02's bf16), and batch order diverges after the boundary.

🔑 **The phase boundary is an implementation detail of OUR method. It must not change what
the optimizer does.** Thirty epochs of CE should be thirty epochs of CE wherever the
boundary is drawn.

⚠️ **NOT YET ATTRIBUTED, AND DO NOT ASSUME THE OBVIOUS CAUSE.** An isolated measurement of
the optimizer restart alone (paired over 10 seeds, identical init, identical replayed batch
order) put it at **+0.0121 macroF1 / +0.0019 AP -- the WRONG SIGN**: on an under-fit model a
fresh Adam acts as a brief LR warm-up and HELPS. That was a small CPU model, and the
transient scales with parameter count, but the direction is a property of the mechanism.
⇒ the restart is **not** established as the cause. ⚠️ And the effect is close to this
project's own noise: -0.0198 macroF1 sits INSIDE the measured 0.0358 same-arm floor, and
5.2 items is ~2x the paired seed sd of 2.7 at n = 3 seeds.

✅ **MEASURED 2026-08-21. IT DOES NOT REPLICATE ACROSS SLICES -- and the variable that
separates the two campaigns is the DATA, not the optimizer.**

| campaign | slice | cap | d items (null - clip), per seed | mean | seeds + |
|---|---|---|---|---|---|
| `dosefix` | **lesion-corrected** | L40_G30 | -4, -7, -4 | **-5.0** | **0/3** |
| `dosefix` | lesion-corrected | L50_G30 | -4, -7, -4 | -5.0 | 0/3 |
| `mc29` | **leaked** | L50_G30 | **-10, +5, +6** | **+0.3** | 2/3 |

⚠️ `vit_diag` is excluded: its `clip` seed ran pre-determinism-fix (`warn_only=None`) and
its `null` seed post-fix, so the pair straddles the 0.0358 noise regime.

**What IS established:**
- **Epoch 1 is bit-identical** (`L_CE` 0.762392 on both arms), so divergence begins exactly
  at the boundary, as designed.
- **Epoch 2 costs the null arm CE progress in 7 of 7 runs** (dL_CE +0.0027 .. +0.0416,
  0.5-9.2% relative, sign test **p=0.0156**). ⚠️ Read `L_CE`, never `Train_Acc`: warm-up
  computes accuracy in `model.eval()` while the constraint phase accumulates train-mode
  logits with dropout on.
- **The transient is gone by epoch 3** and both arms end at train acc ~0.99.
- 🔑 **But the epoch-2 gap does NOT predict the scored delta**: Pearson r = **-0.10**
  (p=0.85). The largest CE gap has the smallest loss and vice versa. ⇒ **the restart
  transient is real and is NOT the explanation.**

⚠️ **The three candidate mechanisms cannot be separated, because all three are live in
100% of the runs that exist.** There is **no bf16 run** and **no warm-up-30 lambda=0 arm**
anywhere -- not in the five live campaigns, not in the 14,524-config archive (which
contains **zero** runs with `lambda_global = lambda_local = 0`). Both discriminating
experiments have to be RUN; they cannot be read off disk.

✅ **Repeat noise on this code is exactly ZERO** (`_variance_probe_strict`: three reps, one
md5, ccF1 0.3931 three times), against **4.0 items** pre-fix. So `dosefix`'s -5 items is
exactly reproducible and is not measurement noise. ⛔ **Seed noise is the binding
constraint**: sd 1.6 items on `dosefix` but **9.0 items on `mc29`** -- larger than the
entire effect space. At 3 seeds the exact sign-test floor is p=0.25. **This cannot be
resolved at n=3, and the answer is SEEDS, not cells.**

🎯 **THE LEAD.** The one controlled contrast -- `dosefix` vs `mc29` at L50_G30, same
backbone, cap, capped classes, seeds, GPU and AMP, **differing only in data slice** --
decomposes as: `clip`'s capped-class hit rate barely moves (84.0% corrected vs 82.7%
leaked) while **`null`'s falls 83.0% -> 80.1%**. That is this document's own leakage
prediction running forwards: leakage lets both arms rank the memorised items correctly and
**compresses** arm-vs-arm gaps. If so, the handicap was masked by leakage in every derm
result this project holds, and `dosefix` is the first place it is visible. **Not tested.**

🛑 **AND THE TWO CAP TAGS ARE ONE CAP LEVEL.** `L40_G30` and `L50_G30` both bind on the
GLOBAL scope (local sums 82 and 103 against a global 62), so class 2 gets **K=62** and
class 4 **K=67** in BOTH cells. Not just the six allocation-free metrics duplicate -- **the
budget-equalized family duplicates too**, and `dosefix`'s two cells return literally
identical ccP/ccF1/AP deltas. **"0 of 2 cells" is 0 of 1, counted twice; the effective n
across the whole campaign is 3 seeds.** `gen_campaign` refuses a single-cap campaign by
comparing tag STRINGS, which any two spellings satisfy; `verify_caps` now compares the
EFFECTIVE budgets and prints `*** SAME BUDGET, DIFFERENT TAGS`.

---

### (10) 🛑🛑 NO PROTOCOL-LEGAL COMPARISON OF THE DUALS AGAINST A CLIPPER HAS EVER RUN

Verified 2026-08-21 against the 14,524-config provenance archive, by classifying every
cell that holds both a dual arm (`fioretto_ldf` / `hounie_rcl` / `fioretto_alm`) and
`heuristic` at a matched (campaign root, backbone, dataset, cap, seed):

| | pairs |
|---|---|
| completed, equal compute, **`lr_constraint` != `lr` (THE LR TRAP)** | **2,972** |
| one side never completed | 104 |
| **completed + equal compute + `lr_constraint == lr`** | **0** |

The 48 pairs that ARE protocol-legal are the `mcbar_duals` campaign -- derm/oct/tissue x
MobileNetV3, dual at warm-up 1 + 29 against `clip` at 30 + 0, `lr_constraint == lr` -- and
**every one of them is `status: pending`. They were generated and never ran.**

⚠️ **Compute parity is NOT the disqualifier**, and an earlier reading that said 96.6% of
cells were at unequal compute was wrong: the clipper is given the full budget as warm-up,
exactly as the protocol prescribes, so 100% of completed pairs match on total optimizer
epochs. The disqualifier is the LR trap alone -- `lr_constraint` 5e-6 against `lr` 1e-4,
the same 20x gap that once fabricated a -16.7 pp "constraint damages the representation"
finding and is why the protocol now says `lr_constraint` **MUST** equal `lr`.

🎯 **CONSEQUENCE: "Fioretto and Hounie beat the clipper" is not a result this project
holds.** It is also not refuted -- there is simply no legal measurement either way. The
`mcbar_duals` campaign is the one that would settle it, and it is sitting `pending`.

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

⚠️ **And the `_null` arm alone is NOT enough for a count claim.** It says how much of the
count movement is the constraint rather than the regime; it says nothing about how much
movement is available for free. That is `tralo_reseed` (section 13), which is `tralo_null`
with the RNG stream perturbed and nothing else -- and it moves the capped count 0.90-1.00x
as far as the constraint does. `gen_campaign` REFUSES a campaign that holds a trained arm
without it.

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

- **Dataset: `iwildcam` (held-out camera traps).** Nothing else. 🛑 `dermmnist`,
  `octmnist` and `tissuemnist` were REMOVED 2026-08-22 -- section 2(n) has the measurement
  and the reasoning. They are not deprecated, they are UNRUNNABLE: the protocol declares
  only `iwildcam`, `data_loader.IMAGERY_DATASETS` holds only `iwildcam`, and the generator
  exits non-zero on a removed name, all three gated by
  `test_removed_datasets_cannot_be_selected_anywhere`. Restoring one requires a
  `scripts.dataset_screen` number first. No AIDER, no EuroSAT.
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

**2. `training_log.csv` IS TWO DIFFERENT FILES, AND ITS EPOCH COLUMN MEANS TWO
DIFFERENT THINGS.** Recorded 2026-08-22, pinned by a test. Same filename, same
directory layout, one axis with two definitions:

| | writer | epoch column | first constraint row at warm-up 1 | warm-up rows |
|---|---|---|---|---|
| `tralo`, `select` | `log_progress_to_csv` (`build_csv_header`) | `Epoch`, **absolute and 1-based** -- the loop runs `range(warmup_epochs, total_epochs)` and the writer adds 1 | `Epoch = 2` | **kept** -- `write_csv_header` rewrites the header and preserves existing rows |
| `fioretto`, `hounie`, `alm` | `open_epoch_log` | `epoch`, **relative to the constraint phase and 0-based** -- `range(constraint_epochs)`, logged raw | `epoch = 0` | **destroyed** -- the writer opens `"w"` and truncates the warm-up's rows |

⇒ **the same training step is row `Epoch = 2` in one arm and row `epoch = 0` in
the other**, a two-row offset, and the duals' log cannot answer a question about
warm-up at all. Both spellings differ in case as well, which is what stops a
naive `df["Epoch"]` from silently reading the wrong axis -- it raises instead.

⚠️ **What this does NOT break, checked:** `full_panel`'s terminal-collapse
detector reads the LAST row and accepts both spellings, so it is convention-free;
`df["Epoch"].max()` is only ever documented for TraLO. **What it does break:** any
cross-arm plot of a quantity against epoch, and any attempt to read epoch 1 out of
a dual's log -- section 9's "epoch 1 is bit-identical" check is a tralo-only
measurement for this reason, not by choice.

🛑 **Not unified, deliberately, and this is a decision not a deferral.** The two
schemas carry different columns (the duals log `total_excess` and `max_lambda_g`,
which the count schema has no place for), so merging them is a change to what
every stored log contains -- it would make the 14,524-run provenance archive
unreadable by whichever reader is kept. The asymmetry is documented and pinned
instead: a test asserts each arm keeps its own convention, so neither can drift
onto the other's meaning without saying so.

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
  ✅ **NOT an arm-vs-arm confound, checked 2026-08-21 across the whole stored evidence.**
  Two hypotheses were tested and BOTH REFUTED, which is what retires this as a worry:
  (1) *"the sequential fill favours the first capped class"* -- `targeted_correction`
  Phase 2 does loop `for c in constrained_classes` while `heuristic` runs one JOINT pass
  over all (item, class) pairs, so the asymmetry is real in the CODE. But the shortfall
  shows no ordering bias: `tralo_uniform` is short on 12 of 16 pairs at position 0 and
  0 of 16 at position 1. (2) *"only the trained arms under-fill"* -- they do not.
  **Post-hoc 72 of 272 pairs (26%), trained 47 of 160 (29%)** -- the same rate, and the
  cause in both is local caps exhausting the candidate pool, not the fill order.
  ⇒ the shortfall is a property of filling under local caps, it is symmetric across
  allocator families, and it therefore biases no comparison. Reproduce from any stored
  campaign; it needs no GPU.
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

### ⚖️ THE HEADLINE IS NOT SEED NOISE -- priced from the corpus itself, 2026-08-23

The noise-floor finding (three identical `clip` runs spreading 0.0358 macro-F1
against a 0.0017 headline effect) left an obvious objection open: **is the
paper's macro-F1 win just seed variance?** The corpus carries `seed` and
`f1_macro`, so the question is answerable directly rather than by analogy, and
the answer is **no.**

Paired `tralo` minus `heuristic` per seed, within cell, at the paper's own
protocol (`warmup_epochs = 50`), 236 cells with >= 2 seeds:

| | |
|---|---|
| within-cell paired seed sd | **1.47 pp** (median; IQR 1.11-2.29 over all 263 cells) |
| detectable at 4 seeds | **2.05 pp** |
| observed per-cell \|delta\| | 1.30 pp (median) |
| **cells clearing their OWN bound** | **32.2% -- 76 of 236** |
| aggregate direction | **tralo wins 184 of 236, sign p = 1.8e-18, mean +1.85 pp** |
| **restricted to the 76 resolvable cells** | **tralo wins 83%, mean +3.84 pp** |

🟢 **The robustness check that matters PASSES.** If the aggregate were carried by
noise, throwing away the cells too noisy to resolve would collapse it. It does
the opposite: the win rate holds and the effect roughly DOUBLES. A noise
explanation predicts the reverse, so it is refuted rather than merely
un-supported.

🛑 **What this does NOT do, and the distinction is the whole point.** Seed power
and comparison validity are different failures, and only the first is addressed
here:

* **Per cell, two thirds of the table is unresolvable.** The median cell reports
  1.30 pp against its own 2.05 pp bound, so **any individual per-cell macro-F1
  number below ~2 pp is inside its own seed noise** even though the aggregate is
  not. The paper reports per-cell numbers. Quote the aggregate, never a single
  sub-2 pp cell.
* **It says nothing about the confounds that actually retracted the win.**
  Unequal compute and `lr_constraint` 5e-6 vs 1e-4 (1b) are biases, not
  variance, and no amount of seed power touches a bias. The equal-compute bar
  still has `tralo` LOSING macro-F1 (0.6895 vs the clipper's 0.7069).

⚠️ **AND THE WARM-UP-1 ROW IS THE LR TRAP, NOT A RESULT.** The same computation
at `warmup_epochs = 1` returns **+15.20 pp, 10 of 10 cells** across 10 cells.
That is eight times the warm-up-50 effect and it is exactly the shape 1b
documents: unequal `lr_constraint` fabricated a 16.7 pp finding that became
1.7 pp once equalized. **Do not quote it.** It is recorded here so that nobody
rediscovers it and mistakes it for the regime effect of section 3.

**So the reviewer objection this answers is "your effect is seed noise" -- and
it is answered.** The objection it does not answer is "your baseline had less
compute and a 20x smaller learning rate", which remains the live one.

### 🔴 THE CORPUS NEVER MEASURED THE RANKING CHANNEL -- audited 2026-08-23

**Every outcome column in the paper's evidence base is either budget-equalized
or a banned non-metric. Not one row anywhere carries AUROC, AP, Brier, NLL or
ConfGap.**

`corpus_final.csv` -- 7,574 rows, the source of EIGHT of the eleven tables --
has exactly seven outcome columns: `acc`, `f1_macro`, `cc_f1`, `cc_rec`,
`cc_prec`, `flips`, `sat`. The first five are budget-equalized; the last two are
house-rule-5 non-metrics. Widen to all 17 files under
`docs/paper/data/corpus/` (~9,700 rows) and the ONLY allocation-free column that
appears anywhere is `ece`, in four auxiliary files. The paper of record mentions
`cc-F1` 45 times and `macro-F1` 41 times, and AUROC, AP, Brier and NLL zero
times each.

🔴 **AND IT CARRIES NO `_null` ARM EITHER -- verified against the file
2026-08-24.** The `method` column holds exactly six values: `danits_lp` (1247),
`fioretto_ldf` (1230), `heuristic` (1249), `hounie_rcl` (1238), `tralo` (1310)
and `tralo_bounded` (1300). **Not one lambda = 0 twin in 7,574 rows.** Every
trained method there gets 29 constraint epochs the post-hoc arms do not, so
each corpus number is `compute + constraint` as a single quantity and **no row
in the paper's evidence base can attribute its margin to the constraint**.
Section 2(s) separates the two on `results/xfam1`, which does carry the twins,
and finds the constraint half NEGATIVE for all three dual families on all eight
metrics. This is not repairable by re-analysis: the corpus is a frozen input
and the runs it would need were never made.

**Why this is structural and not an omission.** Section 2 established that a
post-hoc allocator thresholds the ranking at the budget, so the score IS the
ranking and training can only beat allocation by changing the ORDER. 2(p)
splits the allocation-free family accordingly: AP and AUROC read order and
nothing else, while ECE / Brier / NLL / ConfGap move under a rescale that
reorders nothing -- so **the one allocation-free metric the corpus does carry is
in the family that provably changes no allocation.** The corpus therefore cannot
exhibit the representation channel, and no re-analysis of it can recover one:
per `docs/paper/data/PROVENANCE.md` the corpus is a FROZEN INPUT, the runs are
gone from both machines and both building scripts with them.

⚠️ **"0.9% is re-scorable" IS ABOUT THE `evidence/` TARBALLS, NOT ABOUT THE
DISK -- corrected 2026-08-23.** Counted directly: `archive/raw_runs/` holds
**11,648** `final_predictions*.csv` and `archive/legacy/results/` a further
**442**, roughly 94x what the tarballs carry. The 0.9% figure is still exactly
right for what was PACKAGED as evidence, and every claim resting on the tarballs
stands. But do not read it as "the older work is unrecoverable" -- it is on
disk, and `archive/` is gitignored so it costs a clone nothing.
🛑 **It answers no current question, though**, and that is the part that
matters: every dataset under it is REMOVED -- aider, dermmnist, eurosat,
tissuemnist, plus ablation roots. Section 2(n) rules all of them structurally
incapable of testing a per-group count constraint, so more predictions on dead
datasets buy no measurement. Useful for auditing what the paper era DID, never
for a new result. ✅ Independently checked: the rejected-arm campaigns this
document names (nsteps, sepopt, granularity, headroom, joint, beta, rank, ortho,
budgetprobe, mcbar_regnet, mcbar_duals, mcjoint) have **0** prediction files
each, so the "cannot be re-scored" claim about THOSE is exact.
⚠️ **`ortho` is in that list by accident** -- this document names no verdict
for it anywhere, and its 8-run campaign violates three of the five rules.
Section 2(t) reopens it; do not read its presence here as a rejection.

🔎 **The one ranking measurement the paper era can still produce, and it is
negative.** `evidence/predictions_*.tar.gz` holds raw predictions for 128 runs
(0.9% of the corpus: `mcbar` 72 + `multiclass` 56), so AUROC and AP ARE
computable there. Scored 2026-08-23, five contrasts, 4 seeds: **10 POWERED
lines, 9 of them calibration. Exactly one POWERED RANKING line exists --
`tralo_byk` AUROC -0.0097, a LOSS.** ⚠️ It sits in a 2-cell contrast, where the
sign test cannot call anything (min attainable p = 0.5), so it is POWERED and
NOT CALLABLE at once: the seed noise is small enough to see the effect and there
are too few cells to attribute it. Quote it as "the only ranking effect the
paper's own evidence can resolve, and it points the wrong way", never as a
result.

**What follows for the manuscript.** Nothing in it is falsified by this -- it is
a gap in what was measured, not an error in what was reported. But a sentence of
the form "TraLO improves the model rather than the allocation" is not supported
by any number in the corpus, because the corpus contains no number of that kind.
`full_panel` records the six allocation-free metrics on every new run, so iwc1
is the first campaign that will carry them at all. See 2(p) for the seed cost.

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

### (b-post) READ-OUT: `results/dosefix` -- the first CLEAN TraLO measurement, and it is NEGATIVE

The first `tralo` runs ever to complete on the lesion-corrected slice. Read from
`training_log.csv` as the house rule requires -- counts against caps, satisfied-ever,
lambda, Grad_Norm, and the lambda=0 twin -- not from final metrics.

🔴 **THE CAP IS NEVER SATISFIED. 0 of 29 epochs, at BOTH cap tags.** The hard count sits at
**2.5-3.5x the budget for the entire run** and is still there at epoch 30 (183/177 against
K=62/67).

🔑 **THE TREATMENT IS LIVE BUT DOES NOT STEER.** `tralo` seed 4 against its own lambda=0
twin -- same warm-up, same seed, same allocator, differing only in the constraint:

| cap | class | K | d(count) mean | sd | epochs below twin | mean excess: tralo vs null |
|---|---|---|---|---|---|---|
| L40_G30 | 2 | 62 | **-0.34** | 65.5 | 14/29 | +141.3 vs +141.7 |
| L40_G30 | 4 | 67 | -36.9 | 97.5 | 17/29 | +104.1 vs +141.0 |
| L50_G30 | 2 | 62 | **+20.2** | 72.5 | 13/29 | +161.9 vs +141.7 |
| L50_G30 | 4 | 67 | -26.3 | 110.0 | 15/29 | +114.7 vs +141.0 |

The counts differ from the twin on **28 of 29 epochs**, so this is NOT an inert flag. But
**13-17 of 29 epochs below the twin is a coin flip**, the per-epoch sd is **2-3x the mean**,
and at L50 class 2 moves the count the WRONG WAY by +20. ⇒ the constraint injects variance
without direction.

🔑 **AND THE SEE-SAW IS BACK, ON CLEAN DATA.** At L50 class 2 goes **UP +20.2** while class 4
goes **DOWN -26.3**. That is section 2's recorded mechanism reproducing on the corrected
slice for the first time: the capped classes compete through the softmax, so pressure on one
is relief for the other, and the net movement is ~zero.

⚠️ **CE IS SATURATED BY EPOCH 10** -- train acc 0.99, L_CE 0.02-0.035 -- which is the regime
this document says makes every method identical. And the dose is the smallest the protocol
admits: under `normalize` + `sgd` the delivered step is exactly
`lr_constraint * constraint_grad_clip` = 1e-4 * 1.0 = **1e-4**, against Adam's ~0.93 x 126
CE steps per epoch. Grad_Norm reaches 150-215 before normalisation, so what the log shows
growing is the RAW penalty, not the delivered pressure -- **never read a rising L_Global or
Grad_Norm as pressure under `normalize`.** lambda ratchets 0.06 -> 1.435 with nothing to
show for it, which is the same no-op ratchet section 2 already records.

🎯 **THIS DOES NOT REOPEN THE DOSE AXIS.** That axis is CLOSED (2b): no dose both moves the
count and keeps the classifier. This run is the confirmation on clean data that the
mechanism was never dose-limited alone -- it is direction-limited, and the see-saw says why.

⚠️ **The campaign is ONE effective cap level** (see section 9), so this is 4 seeds at one
budget, not 8 cells. Report it as a single-budget negative.

---

### (b-pre) PRE-REGISTERED: `results/dosefix`, written 2026-08-21 BEFORE any run finished

Recorded here in advance because this document's own rule is *"state which metric it is
supposed to move, and check it is ccP"* -- and because section 2d holds five retractions
that all came from choosing the metric after seeing the numbers.

**Regime.** dermmnist `slice_1` **on the CORRECTED lesion-disjoint split** (0% leakage,
verified) x ViTB16 x {`L50_G30`, `L40_G30`} x {`clip`, `focal_clip`, `tralo`, `tralo_null`}
x 4 seeds = 32 runs. `penalty_shape: linear`, `constraint_grad_mode: normalize`,
`constraint_step_rule: sgd`, `constraint_fp32`, warm-up 1 / constraint 29. Gates green:
`verify_caps` (neither cap tag inert), `check_parity` PARITY OK, `audit_config` clean.

**Why this exact cell.** It is the ONE positive signal in the corpus: at `L50_G30` with a
`linear` shape the penalty gained **+0.030 ccP over its own lambda=0 control**. Everything
else about it is a replication attempt -- 4 seeds, two cap levels, both clippers
in-campaign, on data that is not leaking.

**PRIMARY ENDPOINT: `ccP` (equivalently `ccF1`) versus `clip`, paired by seed, converted
to ITEMS.** Secondary: the same against `tralo_null`, which isolates the constraint from
the warm-up.

🛑 **DECLARED IN ADVANCE, so it cannot be reinterpreted afterwards:**
- **AUROC / ECE / Brier / NLL / ConfGap moving is NOT a result here.** The shipped penalty
  already improves all five against `clip` while ccP falls -0.0450 -- better everywhere the
  cap does not read, worse in the only place it does. An arm that moves AUROC has been run
  twice. If ccP does not move, this arm produced nothing.
- **`flips`, raw count over K, and satisfaction rate are NOT results** and are not to be
  quoted (house rule 5).
- **A sub-item delta is a re-allocation, not a difference.** At this cell one item is worth
  ~0.007 ccF1, and the paired seed sd is worth ~2.7 items.
- **2 cells cannot reach significance** -- `gen_campaign` said so at generation time
  (exact Wilcoxon floor p=0.5). This campaign can report DIRECTION and per-seed
  consistency, never a starred verdict. If it is positive, the follow-up is **more SEEDS**,
  not more cells.

### (c) Per-item losses -- one null, one AUROC-only, one REJECTED, one REOPENED

- **`rank`** (pairwise, transductive, top-K vs rest) -- null, 48/48. It is **self-referential**:
  no labels, so it can sharpen a cut but never reorder.
- **`rankpair`** (supervised pairwise hinge) -- ⚠️ **THIS CLOSURE HAS NO RECEIPT AND IS
  HEREBY REOPENED.** Searched 2026-08-22: `results/rankpair` **does not exist**, on either
  machine or in any archive; there is no CSV in `docs/paper/data/corpus/`, and no table in
  `docs/archive/REJECTED_full_2026-08-18.md` -- which does carry a full receipts table for
  `rank`. The archive describes the campaign as PLANNED (line 261) and at line 889 still
  names it as the open question. "Null to negative" is a bare assertion that was then cited
  as settled, which is exactly how the ALM dual weight stayed wrong by 10.3x.
  ⇒ **The SUPERVISED per-item family is NOT closed.** `rank`'s null does not transfer to it:
  `rank` is self-referential (no labels), and the whole reason to try a supervised version is
  that labels are the one thing it lacks. Treat this line as an open question, not a wall.
- **`select`** (jointly-trained SELECTION head, path 1c) -- ⛔⛔ **REJECTED 2026-08-22, and
  it is the worst arm this project has measured**: -22 items vs `clip`, 0 of 2 cells on
  every metric, plus 2 of 8 runs collapsing on their final epoch. Its own `select_null`
  ties `clip`, so the selective TERM is what costs. **Do not re-run it at any `eta`, `tau`
  or `cov_weight`.** Full record and the measured tables: section (12) above.

🔑 **WHAT THIS LIST CLOSED, and how.** Every score-pushing arm here adds a term that moves
the SCORE ORDERING while leaving the classification loss untouched -- pairwise hinges and
threshold hinges are score arithmetic. **`rank` is null 48/48 and `budget_margin` moves
only AUROC** -- the region the cap never reads -- which is evidence that you cannot fix the
ranking by pushing on the scores. ⚠️ It used to read "three of them are null", counting
`rankpair`; that closure has no receipt and is reopened above, so the evidence here is TWO
arms, not three.

A **selective** loss was the one escape this list left open: it reweights the
CLASSIFICATION loss so the model is optimised to be accurate *on the items it selects*, so
the representation changes and not just the offsets -- the mechanism SelectiveNet reports
its gain from. ⛔ **That escape is now closed by measurement, not by argument.** `select`
was built for exactly this reading, inherited the same bar (**it must move ccP**), and
missed it on every metric including AP. ⇒ **"reweight the classification loss toward the
covered set" is a measured loss here, not an untested hope**, and the honest reading of
section (12) is that the coverage term optimises ABSTENTION, which is a different
objective from ranking the capped class.

⚠️ **The risk stated in advance is the one that materialised**, which is why it is kept:
the covered set is ~30-50% of ONE class, so the selective loss optimises accuracy on a
small, self-selected subset -- a recipe for overfitting it, the exact failure
`joint_objective` had (held the cap 98.8% of epochs, -0.067 AP). `select` lost 0.1096 AP.

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

### (e) Dead code / inert flags found by audit -- ALL FIXED, kept as a failure catalogue

⚠️ **Every entry here is CLOSED. This section is history, not a worklist** -- it was read
as an open list of known-bad things, which is exactly the failure mode of leaving a defect
documented instead of fixed. Verified against the tree 2026-08-21:

| flag | was | state now |
|---|---|---|
| `rho_step` | log-only, so the rho ramp was a no-op | ✅ **derived** in `tralo/train.py` from `rho_target`; no config key exists to get wrong |
| `base_loss`, `alpha_kl` | keys no reader ever read | ✅ **deleted**; `audit_config` (AST) now fails the build on any such key |
| `reset_optimizer_at_sat` | bit-identical no-op at warm-up 1 (16/16) | ✅ **deleted** |
| `constraint_class_weights` (`by_k`/`inv_k`/`uniform`) | `uniform` was a documented no-op | ✅ **deleted** |
| `enable_ce_skip` / `ce_skip_acc` | reached only TraLO; a 0.22 cc-F1 artifact | ✅ **deleted TWICE**. Re-added 2026-08-20 as a shared `CESaturationSkip` with `ce_skip_acc: 0.0`, which made the gate structurally unfireable: `self.skipping = True` is its only write and it sits under `if not self.enabled` where `enabled` is `threshold > 0.0`. Verified by md5 -- with the object stubbed out, 10 of 11 trained arms are bit-identical and the 11th (`tralo_coin`) is self-nondeterministic under its own negative control. **Deleted for good 2026-08-21**: an unfireable gate is a dormant re-add, not a knob. |
| `class_balanced`/`logit_adjust` inert on octmnist | a DATA fact, not dead code | ✅ explained: octmnist's groups are `index % 3` (see the dataset audit) |

🔑 **The standing rule this section exists to enforce: a defect is either FIXED or it is
not written down as an observation.** "This line is bad" in a document is a defect with a
comment attached, and it will be re-found, re-investigated and re-explained by whoever
reads it next. Fix it, then record the fix and the reason -- which is what the table above
does.

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
| `enable_ce_skip` + CE-skip machinery | stop CE at saturation | reached only TraLO; fabricated a 0.22 cc-F1 artifact. Re-added 2026-08-20 as a shared object defaulting to OFF, deleted again 2026-08-21 -- proved unfireable at the default, see section 2e |
| `disable_freeze_on_satisfy` | ratchet/rho freeze ablation | never used; protocol freezes on satisfy |

| 5 methodology packages | tralo_bounded, fioretto_rh, fioretto_restart, hounie_rh, alm_rh | dead arms |
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
six restored baselines, six new gate scripts, and 360 tests. **Do not quote a line count as a
quality measure** -- it has only gone UP since the purge while the repository got
strictly more correct, and every per-component figure written here has gone stale
within days. Measure it if you need it: `git ls-files '*.py' | xargs wc -l`.

What is actually load-bearing is that every one of those lines is reachable and every knob is
read: `audit_config` (no orphan hyperparameters), `smoke_arms` (every arm runs end to end; caps verified for the arms that emit predictions directly, and for the trained arms under `--matrix`),
`verify_caps` (the caps bind on the real slices), `check_parity` (equal compute, shared knobs,
no cross-objective warm-up sharing), and `pytest tests` (360 tests, ~105 s, no dataset needed).

**`rho_step` is still a DEAD KEY** and remains so by design: the ramp is derived from
`rho_target`. It is documented in `hp_defaults.py` rather than silently ignored.

### ✅ A run now records the commit that PRODUCED IT, not the one that wrote its config

Fixed 2026-08-22. `code_version` is stamped by `configs/gen_campaign` when a config is
CREATED, and `main()` never revisits a config it has already written (it explicitly skips
completed runs). Nothing in `src/` originated a version of its own -- verified by AST, not
grep. So the stamp describes the GENERATOR, and the failure it invites is routine:

> run half a campaign, land a change to a training file, resume the rest.

Every config still carries the original value. `full_panel`'s provenance gate then sees ONE
value across both halves and scores them as one comparison -- exactly what that gate exists
to refuse -- and `model_cache` hands the post-change runs the pre-change warm-up on the same
false agreement, because `base_model_id` hashes hyperparameters and not code.

`src/experiments/runner.py` now stamps **`run_code_version`** (git SHA + `-dirty`) at
EXECUTION time, written to disk BEFORE the status flips, since `update_experiment_status`
reloads `config.json` and would drop an in-memory key. `full_panel`, `check_parity` and
`model_cache` all prefer it. One implementation of the git call, in `src/utils/gitver.py`,
shared by the generator and the runner -- four hand-rolled copies of one step is how the
constraint dose came to differ 20x between arms.

⚠️ **The 14,524 archived runs carry no such field, and a missing value DEGRADES rather
than fails.** The gate falls back to the generator's stamp -- the old behaviour exactly --
and says loudly that for those runs it can separate two GENERATIONS but not a code change
landed mid-campaign. An archived warm-up cache is likewise never invalidated for lacking
the field; invalidating them would retrain every cached warm-up in the project.
⚠️ `-dirty` says the tree had uncommitted changes, not WHICH, so two dirty runs an edit
apart still carry the same string. That limit is unchanged, and it is why `check_parity`
warns on `-dirty` instead of passing it.

---

### (a3) 🔴🔴 AND THE MAGNITUDE CARRIES NO VIOLATION INFORMATION EITHER

Section (a2) shows the penalty's gradient DIRECTION is anti-correlated with the
violation depth. This is the other half, and together they close the argument.

**Verified from source 2026-08-22, `src/training/constraint_step.py`.** Under
`constraint_grad_mode: normalize` with `step_rule: sgd`:

1. `clip_grad_norm_(model.parameters(), max_norm=clip)` caps the norm at `clip`;
2. the branch below it scales anything with `raw_norm < clip` back **UP** by
   `clip / raw_norm`, so the delivered gradient norm is **exactly `clip`**;
3. the SGD step is `p.add_(p.grad, alpha=-lr)`.

So the parameter displacement per constraint step is **exactly `lr * clip`**, and
the total over the phase is **`lr * clip * n_steps`**. The violation magnitude
enters the update **only through the direction, never the size.**

🛑 **A model 200 items over budget takes the identical-size step to one 1 item
over.** That is not a feedback controller, it is a fixed bias. Consequences that
were each measured separately and are now one fact:

- **no dose-response to the cap level.** `alm`'s mean excess is 211 at `L50_G20`
  and 143 at `L50_G40` -- a 1.5x difference in violation -- yet its total bias
  push is -0.956 and -1.245 nats, flat and if anything inverted. Same for
  fioretto (-1.331 vs -1.295).
- **the two cap levels are EQUAL PRESSURE, UNEQUAL OPPORTUNITY.** They do not
  differ in how hard the constraint pushes, only in what the allocator is then
  asked to do (`L50_G40` has 33.0 items of headroom against `L50_G20`'s 11.0).
  A reading that treats the tighter cap as "more constrained training" is wrong.
- **the separation from the null is established by epoch ~6 and then flat**,
  because a fixed-magnitude step cannot settle at a boundary.

**MEASURED, not only derived.** Final soft count of the scored model, all eight
treated runs, `results/dualbar2` seed 1:

| cap | K (c2 / c4) | alm | fioretto | hounie | tralo | mean | the null |
|---|---|---|---|---|---|---|---|
| `L50_G20` | 41 / 44 | 97.5 / 154.2 | 139.2 / 90.6 | 90.1 / 104.1 | 102.8 / 79.1 | **107.4 / 107.0** | 165.3 / 223.5 |
| `L50_G40` | 82 / 89 | 153.3 / 82.6 | 105.2 / 118.6 | 99.8 / 91.9 | 114.5 / 118.3 | **118.2 / 102.9** | 165.3 / 223.5 |

**The budget DOUBLES and the arms land in the same ~100-120 band.** The
constraint is not steering toward K; it applies a fixed displacement and stops
wherever that lands. (Scatter is honest: within `L50_G40` class 2 spans
99.8-153.3, so between-arm variation is comparable to between-cap. The supported
claim is "the endpoint is set far more by the fixed step budget than by K", not
"the endpoint is identical".)

🛑 **A TRAP THIS SETS.** Because the landing zone is fixed and the target moves,
the same arms sit at ~2.5x budget at `L50_G20` and ~1.2x at `L50_G40`. Loosen the
cap enough and they would start reporting satisfaction; tighten it and they never
would -- **with no change to the method at all.** Any feasibility, excess or
satisfaction statistic compared ACROSS cap levels is reading the cap, not the
arm. That is `flips` in a new costume, and it is why section 5's metric rules
forbid the family.

⚠️ **And the constraint phase does not converge.** `tralo` at `L50_G40` takes
class 4's soft count down to 69.3 against K=89 -- feasible -- and back up to
141.8 by the final epoch, which is the model that gets scored. Feasibility is not
a metric and that is not a loss; it is evidence that the phase **oscillates with
an amplitude comparable to the entire effect being measured**, which is what a
fixed-magnitude step at a boundary must do.

⚠️ **This is not an argument against `normalize`.** Equalizing the delivered
norm is what makes the four arms comparable at all -- without it hounie delivers
a ~0.05-norm step against fioretto's clipped 1.0, a ~20x dose gap no config gate
can see. The point is that the equalization SPENDS the magnitude channel, and
nothing else was ever putting violation information into it, because the shape
(a2) had already inverted the direction channel.

### (g) 🔴🔴 TRANSDUCTIVE GEOMETRY IS A NULL -- and it was the last candidate

**The argument that produced this test.** Post-hoc top-K allocation is provably
optimal for expected TP *given the probabilities*, so training can only win by
improving the ORDERING; and the constraint's re-ranking is the same size as a
coin flip (below). An ordering gain therefore has to come from information the
allocator does not have. **The allocator consumes a 2014x7 SCORE matrix and has
never seen the 2014x960 GEOMETRY.** That was the last asymmetry left.

`scripts/graph_probe.py` tests it directly: a symmetric cosine kNN graph over
the stored penultimate features, Zhou-style diffusion of the model's own
probabilities (solved exactly, no iteration count to tune), then the run's OWN
endpoint. It reads **no labels** -- only test features and the model's scores --
so it is legitimate in a setting where the cap is already a transductive
statement about the test set. The classical method is the instrument, not a
claim.

**RESULT, 19 runs of `results/dualbar2`, paired, each against its own
undiffused scores, at the pre-registered default `k=10, alpha=0.5`:**

| variant | d items | sign | sign p |
|---|---|---|---|
| **diffused (the real geometry)** | **+0.50** | 10/19 | **1.0000** |
| C1 shuffled graph (same degree, random neighbours) | **-5.80** | 2/19 | 0.0007 |
| C2 shuffled features (real graph, permuted rows) | **-8.36** | 1/19 | 0.0001 |

🔑 **The controls make this a MEASUREMENT, not silence.** Destroying the
geometry costs 5.8-8.4 items decisively, so the instrument is live and the
geometry demonstrably carries ordering information. The real graph then buys
**nothing**: +0.50 items on a 10/19 coin flip. All of the information the
geometry holds is **already in the model's own scores** by the time the
allocator sees them.

**The whole grid is flat, so this is not a dose failure.** `alpha` swept
0.05-0.9 at k=10: +0.54, +0.31, +0.27, +0.34, +0.14, +0.60, +0.23, every sign
test between 0.45 and 1.00. `k` swept 3-1600 at alpha=0.05: -0.59, -0.46, +0.54,
+0.85, **+1.07**, +0.48, -0.35, -2.13, -2.86.

⚠️ **The `k=100` cell reads +1.07 items at p=0.049 and it is NOT a finding.** It
is the best of ~34 searched cells (uncorrected), the pre-registered point gave
+0.50, and the apparent monotone rise in `k` was tested and **turns over** --
which is what a searched maximum does. Recorded here so it is not rediscovered
and believed.

**DO NOT RUN** a GPU campaign for graph diffusion, nor a wider (k, alpha)
search. Cost of this closure: about twenty minutes of CPU on runs that already
existed.

### (h) THE CONSTRAINT RE-RANKS EXACTLY AS MUCH AS A COIN FLIP

Measured 2026-08-22 on `results/dualbar2` `L50_G20` seed 1, capped-class
probabilities against the shared lambda=0 twin:

| arm | dose | mean log-odds shift | rank rho | top-K set moved |
|---|---|---|---|---|
| tralo | full | -3.35 / -4.72 nats | 0.78 / 0.81 | 15/41, 19/44 |
| fioretto | full | -2.17 / -4.39 | 0.82 / 0.84 | 18/41, 16/44 |
| hounie | full | -1.68 / -3.04 | 0.79 / 0.83 | 12/41, 17/44 |
| alm | full | -3.16 / -3.05 | 0.80 / 0.83 | 14/41, 20/44 |
| **`tralo_reseed`** | **zero** | **+1.82 / +0.89** | 0.79 / 0.73 | **15/41, 20/44** |

⛔ **CLAIM 1 IS REFUTED. It read: "the SIGN of the mean shift separates cleanly
-- trained arms push the capped classes DOWN, the zero-dose reseed pushes them
UP." That rested on ONE zero-dose draw and the second draw killed it.** The
seed-2 numbers, same statistic, same cell:

| arm | dose | c2 | c4 | top-K set moved |
|---|---|---|---|---|
| tralo | full | -3.36 | -3.14 | 20/41, 13/44 |
| alm | full | -2.60 | -1.57 | 17/41, 15/44 |
| hounie | full | -2.15 | -1.66 | 15/41, 16/44 |
| fioretto | full | -1.96 | -2.54 | 15/41, 16/44 |
| **`tralo_reseed`** | **ZERO** | **-3.14** | **-3.40** | 16/41, 14/44 |

**At seed 2 the zero-dose control is MORE negative than three of the four
treated arms.** Its own shift swung +1.82 -> -3.14 nats on class 2 between
seeds -- about 5 nats -- so the control's variance is larger than the entire
treated range, and one draw per seed cannot estimate it. **The mean log-odds
shift is not shown to be a treatment effect.**

⚠️ Worth recording rather than hiding, because it is genuinely odd: the TREATED
shifts reproduce well across the two seeds (tralo c2 -3.35 -> -3.36, fioretto
-2.17 -> -1.96, hounie -1.68 -> -2.15, alm -3.16 -> -2.60) while the zero-dose
control does not. That is a reason to keep measuring it, not a reason to claim
it -- a stable treatment against an unstable control at n=2 per arm is exactly
the shape a lucky control draw produces.

✅ **CLAIM 2 STANDS, and it is the one that mattered.** Every treated arm moves
29-45% of the selected set; the pure RNG reseed at zero dose moves the same --
15/41 and 20/44 at seed 1, 16/41 and 14/44 at seed 2. **The constraint re-ranks
no more than re-seeding does, at both seeds.**

⚠️ **Correcting a plausible mis-reading of this table.** The shift is *not* a
near-uniform translation, and it must not be described as one: the per-item sd
is **5.1-5.9 nats against a mean of 1.7-4.7**, so the spread is larger than the
shift. An allocator IS invariant to a uniform per-class shift, but that is not
what this is -- the re-ranking above is real, it is simply not aimed.

### (i) ALLOCATION noise is maximal where the budget cuts the CONTESTED middle (NOT a cap recommendation -- see the correction)

**How to choose a cap level, measured 2026-08-22.** `tralo_null` and
`tralo_reseed` differ by exactly one `torch.rand(1)` call -- same warm-up, same
data, same lambda 0 -- and both are cap-blind, so ONE pair of models can be
re-allocated at any budget. Their ccF1 gap, in items, seed 1 of
`results/dualbar2`:

| K (class 2) | 10 | 20 | **41** | 62 | **82** | 123 | 164 | 205 |
|---|---|---|---|---|---|---|---|---|
| gap, items | -1.04 | -0.96 | **-0.16** | -6.09 | **-12.02** | -11.06 | -1.20 | -1.13 |

(205 is the class's true count, so the last column is a budget that takes
essentially everything.)

🔑 **Pure RNG noise is NOT monotone in K. It peaks in the middle and collapses at
both ends.** The mechanism is direct: at a tight budget the selected set is the
high-confidence head that every model ranks the same way, so two models cannot
disagree much; at a budget approaching the true count both are forced to take
nearly everything and the sets re-converge; in between, the budget cuts through
the ranks where they actually disagree.

**Consequence for campaign design, and it inverts the obvious reading.**
`L50_G40` has 3x the headroom of `L50_G20` (33.0 items against 11.0), which
makes it look like the better place to find an effect. It also has **75x the
noise** on this draw:

| cap | headroom | RNG noise | noise as a fraction of headroom |
|---|---|---|---|
| `L50_G20` | 11.0 items | **0.16** | **1.4%** |
| `L50_G40` | 33.0 items | **12.02** | **36%** |

⚠️⚠️ **CORRECTED THE SAME DAY, BEFORE IT WAS ACTED ON. The first version of this
section concluded "the tight cap is the better-powered cell". That over-reached
and the conclusion is WITHDRAWN.** What the curve above measures is a pair of
models differing ONLY in the RNG STREAM, with the warm-up SHARED -- so it is the
noise the ALLOCATOR contributes, not the noise a seed contributes. A seed change
also redraws the warm-up, and section (j-pre)'s two-seed data shows that
component is far larger and does NOT follow this curve: at `L50_G20`, where the
stream-only gap is 0.16-0.8 items, `fioretto`'s isolation swung **8.8 items**
across seeds, against that cell's 11 items of total headroom.

**What the measurement DOES license**, and it is still worth having:

- the ALLOCATION-induced component of noise is non-monotone in K, peaking where
  the budget cuts the contested middle. That is mechanism and will hold.
- therefore `L50_G40` carries a large allocation-noise term that `L50_G20` does
  not, **on top of** whatever seed noise both share.

**What it does NOT license:** any ranking of the two cells by total power. That
needs the seed component at both caps, which arrives with the four-seed cells.

🛑 **This section is currently IN TENSION with section 1's ceiling rule**
("at G20 the prize is smaller than the noise ... prefer the looser cap; headroom
grows with it"). Both are unresolved and neither should be acted on yet:

- section 1's rule compares the CAPPED-CLASS headroom against the **macro-F1**
  seed sd (+-0.04), and section 1 itself says macro-F1 is carried by the UNCAPPED
  classes -- so it is the wrong noise for that prize;
- this section's curve uses the right metric but the wrong noise SOURCE
  (stream, not seed).

⇒ **Neither cell is currently known to be better powered.** The four-seed cells
settle it by giving the seed sd of the ccF1 isolation at both caps, which is
exactly what `full_panel`'s RESOLUTION block now prints. Until then, sweep both
and read both, which the protocol requires anyway.

### (j) 🔴🔴🔴 THE COMPLETE MECHANISM: the constraint is a PRIOR SHIFT, and top-K is invariant to prior shifts

**This is the capstone of sections (a2), (a3), (g) and (h), measured 2026-08-22, and
it also CLOSES PATH 2 without spending a campaign.**

**Step 1. Post-hoc top-K is nearly invariant to a per-class prior correction.** Multiply
the capped classes' odds by `w` on the shipped `clip` model and re-allocate:

| w (odds multiplier) | 0.1 | 0.25 | 0.5 | 2 | 4 | 10 | 100 |
|---|---|---|---|---|---|---|---|
| top-K items moved, c2 | 2/41 | 1/41 | 1/41 | 0/41 | 1/41 | 2/41 | 4/41 |
| top-K items moved, c4 | 3/44 | 2/44 | 1/44 | 1/44 | 2/44 | 3/44 | 7/44 |

**A 1000x prior correction moves at most 4 and 7 items -- fewer than a single RNG
reseed does.** The within-class ranking is what the allocator consumes, and a per-class
reweighting barely disturbs it.

**Step 2. The per-class shift, whatever causes it, explains almost none of the
re-allocation.** Section (h) measures the shift at -1.7 to -4.7 nats, i.e.
odds x 0.009 to 0.18 -- squarely inside the range above. ⚠️ Section (h) now shows
that shift is **not established as a treatment effect** (the zero-dose control
produced one just as large at seed 2), which does not weaken this step; it
strengthens it. The comparison below is per-arm and asks only how much of THAT
ARM's re-allocation its OWN mean shift accounts for. Build a synthetic twin of the null whose ONLY difference is that
same mean shift, and compare it to what the arm actually did:

| arm | measured shift (c2, c4) | items moved, ACTUAL | items moved, PURE SHIFT |
|---|---|---|---|
| tralo | -3.35, -4.72 | 15/41, 19/44 | **4/41, 3/44** |
| fioretto | -2.17, -4.39 | 18/41, 16/44 | 3/41, 2/44 |
| hounie | -1.68, -3.04 | 12/41, 17/44 | 2/41, 1/44 |
| alm | -3.16, -3.05 | 14/41, 20/44 | 2/41, 2/44 |
| **`tralo_reseed`** (zero dose) | +1.82, +0.89 | **15/41, 20/44** | 1/41, 2/44 |

🔑 **THE DECOMPOSITION. Each arm's own mean shift accounts for 1-4 of the 12-20 items
it re-allocates -- and that shift is the ONE transform top-K is built to ignore. The
other 11-18 items are UNAIMED, and a zero-dose RNG reseed reproduces them exactly, at
both seeds.** Roughly 80% of what any arm does to the allocation is unexplained by its
own systematic component, and the remaining 20% is the part the allocator discards.

⚠️ **Stated WITHOUT attributing the shift to the constraint**, because section (h)'s
second zero-dose draw showed it is not established as a treatment effect. That makes the
conclusion stronger, not weaker: the constraint's attributable footprint is SMALLER than
the decomposition's already-small systematic term.

That is why every arm ties, stated once and completely. It is not a dose problem, a
shape problem, an optimizer problem or an estimator problem -- all four were swept and
all four are downstream of this.

### 🛑 THEREFORE PATH 2 (a test set whose prevalence differs from train) IS CLOSED

Section 4 lists the prior-shift regime as the one setting where the cap carries real
information, marked "needs Roei" because it touches the data path. It does not need to
be run, and here is why in two parts:

- **the CHEAP version cannot create the regime.** A load-time subsample changes the
  prior and nothing else: every item's score is unchanged, so the within-class ranking
  is unchanged, and step 1 above shows the allocator is invariant to exactly that.
- **the EXPENSIVE version does not create a GAP.** Training on a shifted prior would
  genuinely degrade the model's ranking and so raise the headroom -- but post-hoc
  allocation is optimal for expected TP **given the probabilities**, and that optimality
  is distribution-free. A bigger headroom is a bigger prize for the CLIPPER too. Shift
  changes how much is on the table; it does not change who can reach it.

⇒ **Do not ask for new slices, and do not implement the load-time subsample.** The
question it was going to answer is answered.

### (j-pre) PRE-REGISTERED: what section (j) predicts for `results/dualbar2`

**Written 2026-08-22 with 28 of 88 runs complete and NO cell at four seeds.**
Committed before the data exists so the mechanism can be wrong rather than
merely fitted. Section (j) says the constraint is a per-class prior shift the
allocator is nearly invariant to, plus unaimed noise. If that is right, the
COUNT is where the reproducible signal lives and QUALITY is where it does not.

Already visible at two seeds, and it is what prompted this:

| arm | excess separation vs own null, s1 -> s2 | ccF1 isolation, items, s1 -> s2 |
|---|---|---|
| `alm` | -237.9 -> -257.1 (reproduces to 8%) | -4.0 -> -3.8 |
| `fioretto` | -209.4 -> -244.8 (reproduces to 17%) | +0.8 -> **-7.9** |

**THE PREDICTIONS**, to be read at four seeds in `L50_G20`:

1. **The count separation reproduces.** Per arm, the coefficient of variation of
   the mean excess separation from its own null, across the four seeds, is
   **below 25%**.
2. **The quality isolation does not.** For at least three of the four trained
   arms, the seed sd of the ccF1 isolation **exceeds its own |mean|**.
3. 🔑 **The count does not predict the quality -- this is the sharp one.** Across
   the sixteen (arm, seed) pairs, |Spearman rho| between the excess separation
   and the ccF1 isolation is **below 0.5**. A monotone per-class shift moves the
   count and is exactly what a threshold-at-budget allocator ignores, so a
   strong correlation here would REFUTE section (j).

**What would falsify (j).** Any of: a count separation that varies more than the
quality isolation; an arm whose ccF1 isolation is stable across four seeds at a
magnitude above 4 items; or |rho| >= 0.5 in prediction 3. Prediction 3 is the one
to weight -- 1 and 2 are close to restatements of what two seeds already show,
while 3 is a relationship not yet looked at.

⚠️ **This pre-registers a MECHANISM, not a win.** None of the three predictions
becoming true makes any arm beat `clip`; they would only establish why it does
not. The primary endpoint remains ccF1 vs `clip`, seed-paired, per cell.

### (k) INTERIM, n=2: no arm beats a ONE-`torch.rand(1)` reseed, and in the loose cap none comes close

**Recorded 2026-08-22 at 46 of 88 runs, verified independently of the analysis
agent, and NOT callable -- two seeds per cell.** It is written down now so it
cannot be retrofitted once the four-seed cells land.

Isolation = arm minus its OWN zero-dose null at the SAME seed, in items:

| arm | G20 s1 | G20 s2 | G20 mean | G40 s1 | G40 s2 | **G40 mean** |
|---|---|---|---|---|---|---|
| **`tralo_reseed`** (ZERO dose) | +0.2 | -2.0 | -0.9 | +12.0 | +4.0 | **+8.0** |
| hounie | +4.0 | -1.8 | +1.1 | +11.9 | -1.8 | +5.0 |
| alm | -3.0 | -3.8 | -3.4 | +7.9 | -0.7 | +3.6 |
| fioretto | +1.8 | -7.9 | -3.1 | +8.0 | -3.1 | +2.4 |
| tralo | +2.8 | -4.0 | -0.6 | -0.2 | -7.2 | **-3.7** |

**In `L50_G40` the zero-dose arm has the best mean isolation of any arm, is the
only arm positive at both seeds, and is the only arm with positive AP isolation
at both seeds** (+0.0181 / +0.0043; every treated arm is negative at seed 2 in
both cells). `tralo` is the worst arm overall despite looking mildly positive at
seed 1.

⚠️ **DO NOT read this as "RNG is better."** In `L50_G20` the reseed does NOT win
(-0.9, third of five). The correct reading is that **every treatment effect sits
inside the RNG band**, and in one of two cells the RNG draw happened to land
above all of them. That is a statement about RESOLUTION, not about the reseed.

🔑 **Two independent findings agree here, which is why it is worth recording at
n=2.** The reseed's OWN seed-to-seed swing is **8.0 items at K=82 and 2.2 at
K=41** -- the same shape as section (i)'s allocation-noise curve, which peaks
where the budget cuts the contested middle and collapses at both ends. The cell
with 3x the headroom is the cell where a coin flip moves furthest, and that is
why the loose cap flatters everything including the control.

### (l) 🛑🛑 THE LOCAL CAP HAS NEVER BOUND THE OUTPUT -- the mirror of the 2026-08-18 bug

**Found 2026-08-22, Roei's question: "are we monitoring multiclass constraints,
on both global and local?"** The answer is that we monitor multiclass correctly
and have **only ever tested the GLOBAL scope.**

⚠️ **STATE THIS PRECISELY -- there are two senses of "bind" and they disagree
here.** The local term is LIVE IN THE LOSS: `Lambda_Local` ratchets 0.06 -> 1.02
and the groups sit far over their ceilings all through training (`dualbar2`
`L50_G40` seed 1 ends group 2 class 2 at **63 against a limit of 38**). What has
never happened is the local cap binding **at allocation**, which is the sense
that reaches every metric -- the allocator imposes the tighter GLOBAL total
first, and the resulting per-group split has landed inside every local ceiling
every time. So training-time local pressure exists and is then **overwritten**:
whatever it did to the distribution, the allocator re-derives that distribution
under a budget the local caps do not constrain.

**THE ARITHMETIC.** Local caps are per-GROUP ceilings, so their sum is
`L * total_true` while the global is `G * total_true`. Therefore:

| regime | binding scope |
|---|---|
| `G > L` | LOCAL binds, global inert |
| `G = L` | identical, global redundant |
| **`G < L`** | **GLOBAL binds, local slack** |

🛑 **This is the exact mirror of the 2026-08-18 finding.** That one said the
global cap had never bound, and the fix was "sweep `G < L`". The fix worked --
and silently made the LOCAL scope inert. **Nobody checked the other side.**
`results/dualbar2` runs `L50_G20` and `L50_G40`, both `G < L`:

| cap | class | global K | local K per group | local sum | slack |
|---|---|---|---|---|---|
| L50_G20 | 2 | **41** | 35 / 30 / 38 | 103 | 2.5x |
| L50_G20 | 4 | **44** | 48 / 48 / 15 | 111 | 2.5x |
| L50_G40 | 2 | **82** | 35 / 30 / 38 | 103 | 1.3x |
| L50_G40 | 4 | **89** | 48 / 48 / 15 | 111 | 1.2x |

⚠️ Note both `dualbar2` caps are `L50`, so their **local ceilings are
IDENTICAL** (35/30/38 and 48/48/15 in both rows above). The campaign varies only
`G`: it is a pure GLOBAL sweep, and no campaign to date has varied the local
budget at all.

✅ **CONFIRMED EMPIRICALLY, not only by arithmetic:** `lp_fallback_used` is
**False in all 52 completed runs, with 0 candidates**. The LP fallback fires only
when the greedy allocation violates a local group ceiling, so a campaign-wide
zero means the local ceilings were never once the binding constraint at
allocation time.

🔑 **WHY THIS IS THE MOST PROMISING UNTESTED REGIME.** Section 3's standing
objection is *"a count says how MANY, never WHICH"*, which is why post-hoc top-K
is optimal and every arm ties. **That objection is about the GLOBAL cap.** A
binding LOCAL cap says "at most 35 in g0, 30 in g1, 38 in g2" -- it constrains
the DISTRIBUTION across groups, not just the total. That is strictly more than a
scalar, and on this data the groups are genuinely different populations:

| group | size | class 2 | prevalence | class 4 |
|---|---|---|---|---|
| g0 | 1101 | 70 | **6.4%** | 97 |
| g1 | 695 | 60 | 8.6% | 95 |
| g2 | 218 | 75 | **34.4%** | 30 |

A **5.4x** class-2 prevalence spread across groups, and a 5x size spread. ⚠️ It
does NOT follow that a binding local cap beats a clipper -- the allocator can use
local caps too, which is what `lp_fallback` exists for. But it is the only part
of the formulation that has never been the binding constraint in any campaign.

**TO TEST IT: sweep `L < G`** (e.g. `L20_G50`, where the local sum is 41 against
a global 102). That is a cap-tag change, not new code.

🔑 **AND THE CONTRAST IS EXACTLY CONTROLLED, which was not obvious.** Per-group
`L%` sums to `L%` of the total, so `L20_G50` imposes the SAME TOTAL BUDGET as
`L50_G20` -- 41 for class 2 and 44 for class 4 in both -- and differs only in
whether the split across groups is also pinned:

| cap | class 2 budget | class 4 budget | what is constrained |
|---|---|---|---|
| `L50_G20` | 41 total, split free | 44 total, split free | the TOTAL only |
| `L20_G50` | 41 as **14 / 12 / 15** | 44 as **19 / 19 / 6** | total AND distribution |

Holding the total fixed and moving only the scope is the clean form of the
experiment, so **the two caps belong in ONE campaign** -- comparing against
`dualbar2` instead would cross a `code_version` boundary. Verified that no group
budget rounds to 0 at either level (a `K = 0` constraint is silently SKIPPED in
the loss).

⚠️ **And the capped classes are mid-frequency, not rare** -- class 2 is 10.2% and
class 4 is 11.0% of a set whose largest class is **67.7%** and whose imbalance
ratio is **62:1**. The genuinely hard classes (3 at 1.1%, 6 at 1.7%, 0 at 3.4%)
are not capped, and cannot easily be: `K = 0.2 * 22 = 4` for class 3, and a cap
rounding toward 0 is SKIPPED in the loss.

### (m) 🔴🔴 THE SPLIT CARRIES NOTHING -- the local-cap direction is CLOSED, for 0 GPU-hours

Section 2(l) proposed a 120-run campaign at `L20_G50` to make the local scope
bind. **`scripts/scope_probe` priced it first and it is dead.** Not the campaign
-- the direction.

The contrast is exactly controlled (2(l)): `L20_G50` and `L50_G20` impose the
SAME TOTAL, so the scope question is answerable from stored probabilities with
the MODEL HELD FIXED. Three measurements on `results/dualbar2`, every one with a
live control:

| measurement | d items | sign | what it was |
|---|---|---|---|
| **pin the split at true proportions** | **-0.86** | 15/56 | what a binding local cap does |
| C1 rotated ceilings | -5.46 | 0/56 | same total, wrong shape |
| C2 reversed ceilings | -5.30 | 0/56 | same total, worst shape |
| **best split found WITH labels** | **+4.18** | -- | an oracle, over 14 runs |
| **the same oracle, TRANSFERRED** | **-0.89** | 12/30 | fit on run A, scored on run B |
| **per-group prior calibration** | **-0.20** | 0/24 | label-free method candidate |

**1. THE INSTRUMENT IS LIVE.** A wrong-shape split of the right size costs
5.3-5.5 items at 0/56. That is more than half the entire `clip`-to-perfect
headroom, so an effect of that size would have been seen. The null is a
measurement.

**2. THE ORACLE GAIN IS SELECTION NOISE, AND THIS IS THE TRAP.** Choosing the
best of ~900 splits on the same 2014 labels that then score it gains +4.18
items, which looks like the largest lever ever measured here. It transfers at
**-0.89**, 12/30. The splits it picks are not even stable -- group 2's class-2
budget ranges 14 to 33 across runs. 🛑 **`scope_probe` therefore prints the
transfer UNCONDITIONALLY beside the oracle**; an own-oracle number alone is a
headroom claim built out of noise, and with a seed sd near 2.7 items the
selection alone buys several.

**3. THE FREE SPLIT IS ALREADY THE BEST LABEL-FREE CHOICE.** Pinning at the true
proportions costs 0.86 items; another run's oracle costs 0.89. Every alternative
split costs about the same ~0.9, because the greedy allocator maximises expected
TP given the probabilities and its own split is that optimum. A local cap does
not add information -- it OVERRIDES the allocator's optimum with the proportional
split, and the proportional split is not the good one.

**4. AND THE PER-GROUP PRIOR IS NOT A LOOPHOLE IN 2(j).** 2(j) closed the prior
shift because ONE multiplier per class is monotone and cannot reorder anything.
A PER-GROUP multiplier is not monotone over the full set and CAN reorder across
groups, which is exactly what sets the split -- so it needed its own measurement
rather than an appeal to 2(j). It got one: corrections of 1.7x to 3.1x were
applied, and **6 items moved.** The residual miscalibration is only 1.68x
between groups while the top-K items are further apart than that. The model's
per-group priors are already nearly right.

🔑 **WHAT THIS COSTS AND SAVES.** Zero GPU-hours, against a 120-run campaign
of roughly ten. This is the third direction closed by a CPU probe against runs
that already existed (`frozen_head_probe`, `graph_probe`, now `scope_probe`), and
the pattern is worth stating as a rule: **when a contrast can be made exactly
controlled, price it offline before generating a campaign.**

⚠️ **WHAT IT DOES NOT CLOSE.** All of this holds the MODEL fixed, so it prices
the allocator half. Training under a binding local cap could in principle yield
different probabilities. But the standing finding is that the constraint prunes
and never re-ranks, and 2(l) measured the training-time local term acting purely
on the TOTAL with the per-group SHARES unchanged (class 4 group 2, whose ceiling
is punitively tight at 15 against 48/48, held share 0.146 treated vs 0.137
zero-dose, p=0.296). Both halves say the same thing.

### (n) 🟢 THE DATASET CRITERION -- what a dataset must have for ANY of this to work

**Roei, 2026-08-22: "the whole issue that stands between us and making the dual
method work is finding a correct dataset."** Correct, and the criterion is now
measurable. `scripts/dataset_screen` computes it from labels and metadata alone
-- no images, no model, no GPU.

**THE ARGUMENT, from the nulls.** Post-hoc top-K is provably optimal for
expected TP *given the probabilities*, so training can only win by producing
better probabilities. The constraint's sole contribution is information: it
states counts. That can only help if it is information the TRAINING SET DOES NOT
ALREADY CARRY. Two measurements pin which kind:

* **2(j): a GLOBAL count cap cannot help on ANY dataset.** One multiplier per
  class is monotone, so it cannot reorder two items; a 1000x correction moved
  fewer than an RNG reseed. **That route is shut permanently, not per-dataset.**
* **2(m): a PER-GROUP multiplier is not monotone and CAN reorder across
  groups.** That is the only live route, and it is what a local cap is.

🛑 **THEREFORE THE METRIC IS THE *DIFFERENTIAL* PER-GROUP SHIFT, AND NOTHING
ELSE.** If every group shifts by the same factor, that is one global multiplier
wearing N hats -- dead by 2(j). `dataset_screen` reports `NET`: the per-group
deviation after rescaling each group by the observed GLOBAL shift, minus a
simulated sampling-noise null.

**⚠️ BOTH SUBTRACTIONS ARE LOAD-BEARING, and the first version had neither.**
Without the noise null the screen scored dermmnist at "62x the seed noise" --
a group of 218 manufactures tens of "novel" items by binomial noise alone.
Without the global rescale it scored `shift_1` at 160.

| slice | NET | LOCAL | GLOBAL | verdict |
|---|---|---|---|---|
| `dermmnist/slice_1` | **+65** (z=2.9) | +70 | -10 | stage 1 pass |
| `dermmnist/shift_1` | **+50** (z=2.5) | +160 (z=7.1) | +154 (z=7.4) | **110 of its 160 is global shift in disguise** |
| `octmnist/slice_1` | **-7** (z=-0.4) | -7 | -46 | **DEAD** |
| `tissuemnist` | **-55** (z=-1.9) | -55 | -90 | **DEAD** |

✅ **The screen passes its negative control:** oct and tissue come back dead,
which they must -- their `synth_group` is literally `np.arange(len(y)) % 3`
(`scripts/prep_octmnist.py`), so the groups are i.i.d. draws from one
distribution and the local scope is empty **by construction**. Any screen that
called them live would be broken. `shift_1` has **never been used in any
campaign**, and now should not be: its differential content is no better than
the slice we already run.

**🔑 STAGE 1 IS NECESSARY, NOT SUFFICIENT -- and dermmnist is the proof.** It
carries **65 items** of genuine differential per-group information, and 2(m) fed
a model the TRUE per-group counts and moved **6 items**, for -0.20. Information
existing is not the same as being convertible into ORDERING. A per-group
multiplier only flips items whose scores sit within its ratio of the cut; on
dermmnist the residual factor is **1.68x** and the top-K items are further apart
than that. **Stage 2 is `scripts.scope_probe --calibrate`** and it needs one
trained model.

**WHAT TO LOOK FOR, stated so it can be checked before downloading anything:**

1. **Test groups ABSENT from training.** The maximal case: the model holds no
   prior for them, so the cap is the ONLY source and the correction is
   unbounded rather than 1.68x. **Every dataset we own has `unseen = 0`.**
2. **Per-group class distributions differing by orders of magnitude**, not the
   5.4x on dermmnist -- ideally a class common in one group and absent in
   another.
3. **A real, semantic group variable.** `index % 3` is not one. dermmnist's
   `loc_group` (body site) genuinely is, which is why it is the only one of the
   three to pass stage 1.
4. **Meaningful multi-class imbalance with rare classes**, and every per-group
   budget must round above 0 -- **a `K = 0` constraint is silently skipped in
   the loss**, so a cap that rounds a rare class to zero disables itself.

**SHORTLIST, ranked by criterion 1**, which is the one no dataset we own
satisfies:

| candidate | classes | group | test groups unseen? | note |
|---|---|---|---|---|
| **iWildCam (WILDS)** | 182 species | camera trap | **YES, explicitly disjoint** | long tail; per-camera species sets barely overlap. Best structural fit. Non-medical. |
| **FMoW (WILDS)** | 62 land-use | region x year | partly -- OOD split is by YEAR, regions recur | regions are natural, well-sized groups |
| **RxRx1 (WILDS)** | 1139 | experimental batch | yes | too many classes; batch is not a meaningful "local" constraint |
| **ISIC 2019** | 8 | acquisition source / body site | no, but sources differ sharply | keeps the medical framing and the derm continuity |
| **Terra Incognita / CCT** | ~10-16 species | camera trap location | **YES, by design -- test locations have NO training data** | 20 traps, Beery et al. ECCV 2018. Same structure and same authors as iWildCam, which is the problem as well as the appeal -- see below |
| Camelyon17 | 2 | hospital | yes | **binary -- fails multi-class** |

**✅ MEASURED 2026-08-22 -- iWildCam CLEARS STAGE 1 BY 48x.** Built a held-out
camera slice from the iWildCam 2020 COCO annotations (`data/iwildcam/oodslice`):
8 species, 150 training cameras, **7 test cameras with ZERO overlap**, 2,943 test
images.

| dataset | NET | z | LOCAL | GLOBAL | unseen |
|---|---|---|---|---|---|
| `dermmnist/slice_1` | +65 | 2.9 | +70 | -10 | 0 |
| `octmnist/slice_1` | -7 | -0.4 | -10 | -46 | 0 |
| `tissuemnist` | -56 | -1.9 | -52 | -90 | 0 |
| **`iwildcam/oodslice`** | **+3131** | **97.4** | +3169 | +638 | **7** |

And the novelty is genuinely DIFFERENTIAL, which is the part that matters: NET
(3131) is nearly all of LOCAL (3169), with only 638 attributable to the global
shift. Contrast `shift_1`, where 110 of 160 was global in disguise.

#### 🟡 CANDIDATE #2: Caltech Camera Traps / Terra Incognita -- SCREENED 2026-08-23, images NOT acquired

The only other dataset to clear stage 1. Screened metadata-only, exactly as
this section prescribes, before a single image was fetched:

| dataset | NET | z | LOCAL | GLOBAL | unseen |
|---|---|---|---|---|---|
| **`cct/oodslice`** | **+2546** | **75.8** | +2810 | +540 | **5** |
| `iwildcam/oodslice` (same run) | +3133 | 96.3 | +3531 | +994 | 7 |

NET (2546) is most of LOCAL (2810), so the novelty is differential rather than
a global shift replicated across cameras -- the same test `shift_1` failed.

**The structure is verified from the metadata, not asserted.** 128 train
cameras against 5 test cameras with **overlap 0**, and the per-camera class
counts give:

| capped class | test n | cameras present | ZERO ceilings | support |
|---|---|---|---|---|
| 0 | 612 | 2 of 5 | **3** | {51, 78} |
| **2 (rabbit)** | 426 | 3 of 5 | **2** | {51, 78, 135} |
| **6 (bobcat)** | 382 | 3 of 5 | **2** | {37, 51, 78} |
| 5 (bird) | 249 | 5 of 5 | 0 | ⛔ no zero ceiling, local scope adds nothing |
| 7 (cat) | 401 | 1 of 5 | 4 | ⛔ local collapses onto the global scope |

Classes **2 and 6** satisfy the criterion this section sets: a K=0 ceiling
binds regardless of sum slack, so a class ABSENT from some test cameras and
present in others makes the LOCAL scope bind. Their supports also DIFFER
({51,78,135} vs {37,51,78}), so the two local budget vectors are not one number
divided up -- the failure mode that would make two capped classes a single
constraint wearing two hats. 4 of 10 per-group ceilings are zero, against
iwildcam's 7 of 14.

📌 **Class 0 has strictly MORE binding structure than either pick** -- 3 zero
ceilings and the largest test count (612) -- and is the first thing to
reconsider if a CCT campaign is ever designed. It was not chosen; note the
choice rather than assume it was optimised.

#### 📡 THE FULL CAMERA-TRAP SWEEP -- six datasets screened in ONE run, 2026-08-23

LILA hosts ~30 COCO-CameraTraps sets and `prep_iwildcam`'s reader parses them
unmodified, so screening is cheap and was done exhaustively rather than one at
a time. All metadata-only, no images, no GPU. **Every row below was re-run
locally, not taken from a report:**

| dataset | NET | z | LOCAL | GLOBAL | **NET/LOCAL** | unseen |
|---|---|---|---|---|---|---|
| `cct/oodslice` | +2546 | 75.8 | 2810 | 540 | **90.6%** | 5 |
| `iwildcam/oodslice` | +3133 | 96.3 | 3531 | 994 | **88.7%** | 7 |
| `idaho/oodslice` | +2291 | 77.0 | 2946 | 1262 | 77.8% | 3 |
| `wcs/oodslice` | +3440 | 103.7 | 4640 | 2611 | 74.1% | 5 |
| `wellington/oodslice` | +1331 | 52.3 | 2027 | 1093 | 65.7% | 2 |
| `serengeti/oodslice` | +1646 | 55.9 | 2914 | 2136 | 56.5% | 5 |
| `ena24` | -- | -- | -- | -- | -- | ⛔ REPORTED to have no group variable (`KeyError: 'location'`). **NOT reproduced -- its slice is not on disk**, so re-screen before relying on the rejection |

🔑 **RANK ON NET/LOCAL, NOT ON NET, and a large GLOBAL column is a NEGATIVE
signal.** NET/LOCAL is the fraction of the novelty that is genuinely
DIFFERENTIAL across groups; the remainder is the global shift replicated
across them, which 2(j) proved is one multiplier per class and therefore
cannot reorder ANY top-K set. `wcs` has the largest headline NET of all
(+3440) and is fourth on the measure that matters, because 26% of its local
signal is global. This is the `shift_1` trap (110 of 160 global in disguise)
in a new costume.

⚠️ **AND THE RATIO MUST COME FROM ONE RUN.** Dividing this run's LOCAL into
the stored NET in the table above gives iwildcam 99% and puts it first;
measured consistently in a single invocation it is 88.7% and **`cct` edges it
at 90.6%**. The two are close and both are far clear of the rest, so nothing
downstream changes -- but the ordering of the top two comes from mixing a
stored number with fresh ones, which is the same apples-to-oranges this
document has been burned by before. Compute the ratio inside one screen.

✅ **NO LEAKAGE, checked rather than assumed.** iWildCam draws partly from
WCS-contributed data, so a shared image between candidates was a real risk.
Filename overlap between `iwildcam` and `cct` is **0** of 22,943 vs 22,985,
and each dataset's own train/test overlap is **0**. Re-check this for any
further candidate before it is admitted; dermmnist shipped with a 38.7% leak.

Two candidates carry a caveat that shows up only on inspection:

* **`idaho`** -- 4 of its 8 classes are camera artefacts (`snow on lens`,
  `foggy lens`, `vegetation obstruction`, `malfunction`). That IS real
  per-group label shift, since a broken camera produces many broken frames,
  but it is degenerate rather than ecological and a reviewer may say so. Only
  3 test cameras.
* **`wellington`** -- only 2 test groups, so the local scope has almost
  nothing to differentiate between.

⛔ **NONE OF THEM BUYS INDEPENDENCE.** Every survivor is a camera-trap set
read through the same COCO-CameraTraps schema, and several share contributors.
They raise the dataset count, so section 0's exact sign-flip floor improves
(1 dataset p=1.000, 2 datasets p=0.500), and a reviewer may still fairly call
the whole family one generalization unit. **A genuinely independent third has
to leave camera traps entirely** -- ISIC 2019 is the standing candidate and is
NOT yet screened.

🛑 **AND NOTE WHAT THIS SWEEP CANNOT FIX.** There is no dataset on which the
GLOBAL scope carries information: 2(j) is structural, not empirical. Whether
the global cap BINDS is a cap-tag choice (`G < L`, e.g. `L50_G30`), never a
dataset property. So the sweep can only ever improve the LOCAL differential,
which is the one live route -- and (p-post) has just measured that route
negative on iwildcam.

🛑 **CCT IS NOT AN INDEPENDENT GENERALIZATION UNIT.** Same authors, same
modality, same schema, same COCO-CameraTraps reader as iwildcam. It moves the
dataset count from 1 to 2, so section 0's exact sign-flip floor improves from
p=1.000 to p=0.500, and a reviewer may still fairly call the two one unit. It
buys CONSISTENCY, not independence. A genuinely independent third would have to
leave camera traps entirely.

⛔ **The images are NOT on disk and no run is possible.** Only
`train_meta.csv` / `test_meta.csv` exist (20,000 / 2,985 rows). Acquiring the
images is a ~8 GB fetch from LILA and needs Roei's approval.

⚠️ **The WIRING is already committed (`3ea7d35f`) but MUST NOT BE DEPLOYED
while `results/iwc2` runs.** It is purely additive -- `protocol.yml` gains a
`cct` block (38 insertions, 0 deletions) and `IMAGERY_DATASETS` goes from
`{'iwildcam'}` to `{'iwildcam', 'cct'}` -- so it cannot change what an
iwildcam run does. That is why it is safe to have committed, and it is still
not safe to deploy: `code_version` is a git hash, so moving the campaign
checkout off `3bb7e8b4` mid-flight splits iwc2 into two non-comparable halves
whatever the diff contains. Deploy after the last run, never during.

**WHY THE STRUCTURE IS RIGHT, in one line:** the median camera sees **10 of 216
species**, and camera 501 is impala/elephant/cattle while camera 408 is ocellated
turkey/tapir/puma -- different continents. A species that dominates one camera is
**absent** from another, so per-group ratios are effectively infinite against
dermmnist's 5.4x, and the residual correction factor cannot be the 1.68x that
made 2(m) a null.

🛑 **ONE FIX WAS REQUIRED TO MEASURE THIS AT ALL, and it is the kind of bug
that silently confirms a prior.** The screen originally SKIPPED groups absent
from training. On a fully held-out-domain split -- the exact design this section
recommends -- no unit survives the sum, so it returned `net_items 0.0,
net_z nan`: the criterion would have rejected its own best case. A model that has
never seen a group holds no group-specific prior and must fall back to the
GLOBAL one, so that is the baseline the deviation is now measured against. Gated,
with the skip re-injected as the negative control.

⚠️ **STAGE 2 IS STILL OUTSTANDING.** dermmnist cleared stage 1 at +65 and still
nulls. What makes iWildCam different in kind rather than degree is that the
correction factor is unbounded here, where dermmnist's was 1.68x -- but that is
an argument, not a measurement, until `scope_probe --calibrate` is run on a
trained model.

**THE SECOND-DATASET SEARCH -- where FMoW stands, 2026-08-22.** FMoW is the
right shape on paper: 62 classes, heavily imbalanced, and a real domain axis
(geography from the `polygon` centroid, time from `timestamp`). Two obstacles
were measured rather than guessed, and both are recorded so the next attempt
does not rediscover them.

1. 🛑 **The official WILDS archive is GONE, not down.** The CodaLab bundle
   answers **HTTP 500** while the CodaLab root answers 200 -- re-confirmed here
   after the same result during the iWildCam work. Do not plan around it.
2. 🚨 **The popular HuggingFace mirror `EVER-Z/fMoW_rgb` has a CORRUPT LABEL
   COLUMN.** Its `category` field is truncated at the first underscore:
   `place_of_worship` -> `place`, `oil_or_gas_facility` -> `oil`,
   `recreational_facility` -> `recreational`. **This MERGES CLASSES** --
   `airport`, `airport_hangar` and `airport_terminal` all become `airport`, so
   three of the 62 collapse into one. A count-constrained experiment run on that
   column would be capping a class that does not exist, and nothing downstream
   would look wrong. ✅ **The full label survives in `image_name`**
   (`place_of_worship_3475_4_rgb.jpg`), so derive the label by stripping the
   trailing `_<id>_<seq>_rgb.jpg` and **never read `category`.**

⚠️ **Screening it is not free the way iWildCam's was.** `dataset_screen` needs
labels and groups only, but this mirror stores them inside 296 image-bearing
parquet shards, and a column-selective read over HTTP did not finish in two
minutes for ONE shard. So FMoW cannot be screened before downloading, which
inverts the 2(n) discipline. **Do the download on the server, derive labels from
`image_name`, and screen before training** -- the screen is still the gate, it
just cannot come first here.

**TRIAGE A CANDIDATE BEFORE DOWNLOADING IT -- the group's DEFINITION decides.**
`dataset_screen` is cheap but still needs the metadata in hand, and FMoW showed
that getting metadata can cost more than the screen. Most candidates can be
settled earlier than that, because the thing that killed octmnist and
tissuemnist is visible in one sentence of the dataset's own documentation:
`synth_group` is `np.arange(len(y)) % 3`, so groups are i.i.d. draws from one
distribution and the local scope is empty **by construction**. Three questions,
answerable from a README:

1. **Is the label space multi-class and imbalanced?** Binary fails outright --
   there is no allocation problem across classes to constrain.
2. **Are the groups defined by something the LABEL DEPENDS ON** -- geography,
   time, institution, device -- **or by an index, a randomisation, or a
   balanced assay design?** The second kind is dead however large the dataset
   is, and no amount of domain shift in the FEATURES repairs it: we need the
   per-group CLASS distribution to differ.
3. **Can whole groups be held out?** If the standard split is label-stratified,
   the splitter has to be replaced (see the warning below).

Applying it, ⚠️ **REASONED from the dataset designs, not measured** except where
a NET figure is quoted above:

| candidate | verdict | why |
|---|---|---|
| **iwildcam** | ✅ **LIVE, measured** | NET +3131, z=97.4, 7 unseen cameras. In use |
| fmow | 🟡 live in principle | region/year; class mix genuinely varies by geography. Blocked on acquisition, above |
| **rxrx1** | ⛔ **DEAD BY CONSTRUCTION** | 1,139 classes and real batch effects make it look ideal, but it is a **plate-based screen: every siRNA appears in every experiment by design**, so the per-group class distribution is uniform across groups. This is octmnist's failure in a prestigious costume -- the shift is in the FEATURES, and we need it in the per-group LABEL counts |
| **terra_incognita** | 🟡 **live in principle, and the only candidate SCREENABLE BEFORE DOWNLOAD** | see below |
| camelyon17 | ⛔ dead | 2 classes |
| povertymap / globalwheat | ⛔ dead | regression / detection, no class budget |
| geolifeclef, iNaturalist | 🟡 live in principle | geographic groups, species mix differs sharply by region; long tail is a feature here |
| civilcomments | 🟡 live but out of scope | toxicity rate does differ by identity group, but the pipeline is imagery-only (`IMAGERY_DATASETS`) |

**TERRA INCOGNITA -- the candidate that can be screened for the cost of a JSON,
2026-08-23.** Beery, Van Horn and Perona, ECCV 2018 (Caltech Camera Traps):
twenty camera traps, and the benchmark exists specifically to measure
"generalization to new locations where no training data is available". Criterion
1 is satisfied by the dataset's own design rather than by a slice we build.

⚡ **Why it is worth naming despite the shortlist already being long: it is
screenable BEFORE acquisition, and as of 2026-08-23 that is a real command
rather than an aspiration.** FMoW is blocked because its metadata only comes
with the shards -- 2(n) records that reading one shard costs minutes. Terra
Incognita ships COCO-CameraTraps annotations separately from the images, and
`prep_iwildcam.build_split` reads exactly that schema
(`categories[].id/name`, `annotations[].image_id/category_id`,
`images[].id/file_name/location`), which is the format iWildCam inherited.

⚠️ **The chain did not actually work end to end until it was fixed.** 2(n)
presents stage 1 as the pre-GPU, pre-image screen, but `prep_iwildcam` wrote the
two CSVs `dataset_screen` reads from INSIDE the shard-download loop -- so
pricing a candidate cost the full acquisition the screen exists to avoid, on
every dataset not already on disk. `--meta-only` stops after the split:

```bash
python -m scripts.prep_iwildcam --annotations <cct.json>     --out data/<name>/oodslice --meta-only     # no images, no GPU
python -m scripts.dataset_screen data/<name>/oodslice
```

Gated end to end on a synthetic COCO-CameraTraps file, with the octmnist failure
as its negative control: give every camera the SAME class mix and the screen
returns **DEAD, NET -10 items, z=-0.8** -- while STILL reporting the held-out
cameras. Criterion 1 without criterion 2 looks exactly like a live dataset until
the NET column is read, which is how two of the original three were run for
months against a question they could not test.

⚠️ A meta-only NET is the INTENDED slice, not the delivered one: if shards fail
during a later real acquisition the slice shrinks. So it is an upper bound --
good enough to REJECT a candidate, never to accept a borderline one.

🛑 **BUT IT BUYS LESS INDEPENDENCE THAN IT LOOKS.** Section 0's clustered floor
is about DATASETS being independent draws, and two camera-trap corpora by the
same authors, in the same modality, with the same failure mode are not two draws
-- a reviewer will read them as one. So:

* For **resolution** (more cells, tighter CIs) Terra Incognita is cheap and fine.
* For **generality** it is weak, and FMoW's satellite imagery is worth its
  acquisition cost precisely because it is not another camera trap.

And criterion 2 still has to be MEASURED, not assumed. rxrx1 is the standing
warning: designed-in domain shift with a per-group label distribution that is
uniform by construction. iWildCam scored NET +3131 because its species mix
genuinely differs per camera; Terra Incognita plausibly does the same, and
"plausibly" is what the screen exists to replace.

⏭️ **NEXT ACTION, and it needs a human:** fetching the CCT annotation JSON is a
download, so it is not something to do unasked. Once it is in hand the screen is
one command and needs no GPU.

🧮 **AND THIS IS NOW THE CRITICAL PATH, not a nice-to-have.** 2(p) records the
two power floors: at one dataset the clustered sign-flip floor is p=1.000, and
**no number of seeds, cells or backbones moves it.** Generality is blocked on a
second dataset and on nothing else. That reorders the queue -- a second backbone
buys resolution the campaign can already almost see, a second DATASET buys the
only thing it structurally cannot.

🔑 **The rule worth keeping: a dataset famous for DOMAIN SHIFT is not
automatically a dataset with PER-GROUP LABEL SHIFT, and only the second is what
a per-group count cap can use.** rxrx1 is the clearest example -- it would have
survived every generic "is this a real shift benchmark" check and still tested
nothing.

⚠️ **A NEW DATASET ALONE WILL NOT FIX THIS.** The splitter is half the problem:
`create_slices.py` stratifies on the LABEL, which forces test prevalence to
match train prevalence and is exactly why the global cap carries nothing. A
WILDS-style dataset run through a label-stratified splitter would reproduce
every null in this document. **Split BY GROUP, holding groups out.**

### (o) 🔧 THE REACHABILITY CEILING -- `straddle_probe`, an INSTRUMENT not yet a result

**The gap this closes in our own accounting.** `scripts/headroom.py` reports the
distance from `clip` to a PERFECT allocator, 1.9-9.9 items, and that number has
been quoted throughout this document as "the prize". It is an ORACLE quantity:
it assumes the ranking can be rewritten arbitrarily. **Ours cannot.** 2(a3)
measured that under `constraint_grad_mode: normalize` the delivered displacement
is exactly `lr * clip` per step, so the constraint moves scores by a BOUNDED
amount, and an item misranked by a wide margin is unreachable at any dose. So
part of the headroom we have been chasing was never available to any arm, and
nothing in the repo said how much.

**The quantity.** With exactly K predictions emitted for a capped class,
improving the endpoint means SWAPPING -- a false positive above the cut leaves,
a true positive below it enters -- so within a displacement budget `delta`,

    reachable(delta) = min( #FP in [t, t+delta], #TP in [t-delta, t) )

with `t` the K-th largest score. In ITEMS, comparable to everything else here.
`contested(delta)`, how many items lie within `delta` of the cut at all, is the
LABEL-free version, readable on a test set whose labels are not to be touched or
on a fresh unlabelled set under an existing model. ⚠️ It is **not model-free** --
without a model there is no ranking and therefore no cut -- so it does not screen
a candidate dataset before training. `dataset_screen` (2(n)) is the pre-GPU one. ⚠️ It is an UPPER bound twice over: it assumes every near-cut
item moves the RIGHT way, and it ignores the per-group ceilings, which can forbid
a swap the global count allows.

**`delta` is MEASURED, not assumed.** Given a treated run and its `_null` twin
at the same seed -- same warm-up, same allocator, same RNG, lambda=0 -- the
per-item difference in the capped-class score IS the displacement the constraint
delivered. That makes `reachable` at the measured delta the ceiling for the
constraint AS CONFIGURED. Assuming a delta instead would have made the whole
statistic unfalsifiable, which is how `rho_step` and the lambda ratchet were
tuned against a quantity that turned out to be cancelled.

⚠️ **THE SHUFFLED CONTROL POINTS THE OTHER WAY -- do not read it as
must-collapse.** Permuting the scores keeps their DISTRIBUTION and destroys the
ORDERING, and `reachable` then RISES, because a random top-K scatters positives
on both sides of the cut. It came out at 10.8 vs 11.6 items across two regimes
whose true error structures differ 5x, so it depends on n, K and prevalence only
-- a reference, not a second measurement. The SIGN of the deviation is the
reading:

| observed | means |
|---|---|
| `reachable << ctrl` | the ranking already took the easy swaps; what remains at the cut is genuinely hard |
| `reachable ~= ctrl` | no ranking information at the cut -- the statistic is reading the score distribution and reports NOTHING |
| `reachable >> ctrl` | positives are parked BELOW the cut beyond chance -- **the one configuration in which a cut-local method has something real to win** |

**THE GATE, and its negative control.** `--self-test` runs two synthetic regimes
whose error GEOMETRY is known: `matched` (clean labels, residual errors sit AT
the cut) must show a HIGH reachable share, `tailnoise` (positives planted among
the lowest-scoring items) a LOW one. Measured: oracle gap 8.60 vs 44.40 items,
and the share separates on **5 of 5 resolved deltas** (0.05/0.00, 0.14/0.05,
0.26/0.14, 0.72/0.42). The gate deliberately reads the whole delta ladder rather
than the widest band, where almost anything is reachable in either regime. Both
directions are pinned in `tests/`, including the negative control that matters:
replacing the band with a position-BLIND one makes the share equal in both
regimes and the gate FAIL, so it has been shown capable of failing.

⚠️ **THE MEASURED DELTA IS A NET DISPLACEMENT, NOT A PATH LENGTH.** It is the
treated-minus-null difference after all 29 constraint steps. For "what did this
arm achieve" that is exactly right -- only the final ranking reaches the
allocator, so an item that moved out and back contributes nothing. For "what
COULD a bounded-step method achieve" it is a LOWER bound on the budget, since a
non-monotone path covers more ground than its endpoints show. A small
`reachable` therefore says THIS arm did not have the reach; it does not by
itself prove no schedule could. State which question is being answered.

**FIRST REAL-DATA EXERCISE, 2026-08-22 -- and it produced a METHOD warning, not
a result.** Run over the 128 stored-evidence runs (`evidence/`, extracted into
one tree: dermmnist 56, octmnist 48, tissuemnist 24, MobileNetV3, caps
L30_G30 / L50_G50 / L70_G70, arms clip / focal_clip / tralo_uniform / tralo_byk).
Nothing here is a claim about the METHOD -- all three datasets are removed, derm
is leaked, oct and tissue have `index % 3` groups, and every cap has L = G so
the global scope is redundant throughout. What it bought is the instrument.

🚨 **THE POOLING TRAP, WALKED INTO TWICE.** The first run pooled all 128 into
one table -- three datasets and three cap levels, where "class 1" names a
different class in each. The second pooled the four ARMS inside each cell, and
the arm IS the ranking, the whole object being measured. Both are rule 4, and
neither was caught by care: they were caught by the numbers looking too clean.
**The fix is in the tool** -- `straddle_probe` now keys every aggregate on
(dataset, backbone, cap, ARM, class), and a test builds two datasets with
deliberately different geometry and fails if they land in one row.

🔴 **THE FINDING THAT DIED IN ITS OWN CONTROL -- keep this one.** With delta
swept as a FRACTION OF THE SCORE RANGE, the reachable share of the oracle gap
FELL as the cap loosened, in **24 of 33** (arm, dataset, class) series. That
reads as a geometry result: loose caps have more headroom but it sits farther
from the cut. Then the control: `contested` -- how many items lie in the band at
all -- falls too, in **22 of 33**. Thinning density explains the same numbers.

So the delta was re-parameterised to hold the CONTESTED MASS fixed at 50 items
(`--match-contested`), and **the direction REVERSED**: reachable items now RISE
with the cap in **23 of 33** series. The original trend was the confound, end to
end. ⚠️ **Never read a cross-cap comparison off the fraction-of-range ladder** --
it is fine within one cell and meaningless across cap levels, because the cut
moves into a differently-dense part of the score distribution.

🛑 **AND THE REVERSED VERSION IS NOT A FINDING EITHER.** dermmnist and octmnist
rise; tissuemnist falls. The honest n is **3 datasets**, not 33 series -- series
inside a dataset share its test set, and arms share a warm-up. Two datasets
agreeing is not a result, it is section 0's exact sign-flip floor. **No claim is
made here about whether tight or loose caps are more reachable.** The usable
output is the method warning above and a ladder that can answer the question
properly when a campaign with `_null` twins exists.

🛑 **STATUS: no real-data number is in hand.** The instrument is built and gated;
it has been run on synthetic regimes only. The first real reading is due on
`results/iwc1`, which is the first campaign carrying `_null` twins on a dataset
that clears 2(n). **Do not quote a reachability figure until that lands.**

### (p) 📌 PRE-REGISTRATION for `results/iwc1` -- written 2026-08-22, BEFORE the data

Recorded before the campaign lands so the read cannot be rationalised
afterwards. This document has twice had a result explained after the fact and
retracted later; a prediction on the record is the cheapest guard against a
third.

**PREDICTION 1 -- the ALLOCATION channel will null, and the binding cap will not
save it.** 2(n) chose iwildcam because 7 of its 14 per-group ceilings are K=0,
which makes the LOCAL scope bind at every cap level for the first time. ⚠️ **A
binding cap is not an informative one.** A K=0 ceiling is a per-group multiplier
of ZERO -- the strongest reordering 2(m) allows -- and the post-hoc allocator
applies it exactly and for free, by zeroing that cell and reassigning each item
to its best ALLOWED class. Given the probabilities that reassignment is optimal,
so the clipper collects the entire benefit of the structure that was the reason
for picking this dataset. Expect `tralo` vs `clip` to tie on every
budget-equalized metric.

**PREDICTION 2 -- the REPRESENTATION channel is the live one, and iwc1 is the
first dataset on which it can be tested at all.** The caps are transductive:
they are computed from the TEST set and applied during training, so the
constraint is a weak label on the target domain and training under it is
transductive adaptation, not allocation. On dermmnist that channel was empty by
construction -- train and test are the SAME domain (2(k): the cap is recoverable
from training prevalence to within one item), so there was nothing to adapt to.
On iwildcam the test cameras are DISJOINT and unseen, which is a real domain
shift. **This is the one mechanism in the project that the "top-K is optimal
given the probabilities" argument does not cover**, because it changes the
probabilities rather than the allocation.

**THE DISCRIMINATING MEASUREMENT, and it already exists.** The two channels
separate cleanly on the metric families `full_panel` already prints:

| channel | signature | does it change an allocation? |
|---|---|---|
| allocation only | budget-equalized metrics move, allocation-free metrics flat | yes, and post-hoc gets it for free |
| **calibration only** | **ECE / Brier / NLL / ConfGap move, AP / AUROC flat** | **NO -- provably none** |
| representation | **AP / AUROC move**, and no post-hoc step can touch them | yes, and only training can produce it |

⚠️ **THE MIDDLE ROW IS NEW, 2026-08-23, AND IT IS THE ONE THAT WOULD HAVE BEEN
MISREAD.** "Allocation-free" was treated as one family, and it is two. AP and
AUROC read the ORDER of the score column and nothing else, so a strictly
monotone rescale leaves them BIT-identical; ECE, Brier, NLL and ConfGap move
under a rescale that reorders nothing. A top-K allocator reads order alone, so a
calibration-only move -- a temperature or prior shift -- **provably changes no
allocation**, which is 2(j) restated in the metric panel. Gated by
`test_a_temperature_rescale_moves_calibration_and_NOT_the_ranking`, whose
negative control checks that a genuine reordering DOES reach AUROC.

So the verdict on iwc1 is read from **`d AP` and `d AUROC` of `tralo` against
`tralo_null`** -- same warm-up, same allocator, same seed, lambda=0 -- and NOT
from ccF1 against `clip`. Allocation-free metrics are computed from
probabilities alone, so no amount of post-hoc filling can manufacture them; this
is the family that survived the quota-fill audit. A ccF1 win with AP flat is the
allocation channel and means nothing new.

⚠️ **Pre-register the null too.** `tralo_reseed` moves the capped count RMS
83-95 items on its own, so a count change is not evidence of anything. And per
2(o), `straddle_probe` must be run on the same campaign to say how much of the
oracle gap a step our size could even reach -- a tie is uninterpretable without
it, because "no effect" and "no reachable effect" are different conclusions.

🛑 **If AP and AUROC are both flat against `tralo_null` at 4 seeds AND the
allocation-free RESOLUTION block says POWERED, the representation channel is
measured and closed** -- and with it the last mechanism the structural argument
leaves open. Say so plainly rather than looking for a fifth slice.

⚠️ **The POWERED clause is not a formality, and it did not exist when this
section was first written.** The RESOLUTION block converts to items via
`items_per_001`, an F1 identity that does not apply to AP or AUROC, so the
scorer printed a seed cost for `d ccF1` alone -- the one family post-hoc filling
CAN reach -- and none for the family this verdict rests on. `full_panel` now
prints a second block in native units for AP and AUROC. **A flat AP that comes
back UNDERPOWERED closes nothing**; it says the campaign needs more seeds, and
the block prints how many.

📏 **WHAT "POWERED" WILL COST -- measured 2026-08-23 on the stored evidence, so
the clause above is a NUMBER before the data lands, not a hope.** The
allocation-free block had never run on real data when it was written. Run it on
the 128 prediction-bearing runs (`mcbar` 72 + `multiclass` 56, MobileNetV3,
4 seeds) and the within-cell seed sd is stable across all five contrasts:

| family | metric | seed sd (5 contrasts) | median | **MDE at the protocol's 4 seeds** |
|---|---|---|---|---|
| RANKING | AP | 0.0202 - 0.0274 | 0.0252 | **~0.035** |
| RANKING | AUROC | 0.0058 - 0.0108 | 0.0094 | **~0.013** |
| CALIBRATION | ECE | 0.0125 - 0.0303 | 0.0276 | ~0.039 |
| CALIBRATION | Brier | 0.0242 - 0.0583 | 0.0512 | ~0.072 |
| CALIBRATION | NLL | 0.2544 - 0.5198 | 0.3660 | ~0.51 |
| CALIBRATION | ConfGap | 0.0085 - 0.0212 | 0.0199 | ~0.028 |

`seeds_needed = ceil(z^2 sd^2 / d^2)` with `z = 1.960 + 0.842`, so `n <= 4`
exactly when `d >= sd * sqrt(z^2/4) = 1.401 sd`. **AUROC is resolved ~2.7x better than AP on identical runs**, which
is a fact about the estimators and not about any method: AP integrates over the
whole precision-recall curve and inherits the seed-to-seed churn of the tail,
AUROC does not.

🔴 **THE CENSUS IS THE RESULT: 10 POWERED lines across the 5 contrasts, and 9
of them are CALIBRATION.** Exactly one RANKING line clears 4 seeds in the entire
stored evidence -- `tralo_byk` AUROC **-0.0097**, a LOSS -- while `focal_clip`
clears ECE, NLL and ConfGap on BOTH campaigns, unanimously and in the improving
direction, with AP and AUROC never once resolved. So this instrument, at this
seed count, **routinely resolves recalibration and almost never resolves
reordering**. That is not a property of the methods; it is the sd column above,
and it is the single most important thing to know before reading iwc1.

Two consequences for the read below, both of them binding:

1. **The verdict leads with AUROC, and a calibration move is NOT a substitute
   for it.** AUROC is the best-resolved RANKING metric by a factor of 2.7 over
   AP, and ranking is the only channel a top-K allocator can see. The temptation
   the census above sets up is precise: iwc1 will very likely come back with
   POWERED calibration lines and flat ranking lines, because that is what this
   instrument does. **That pattern is a recalibration, and 2(j) says it moves no
   top-K set.** Reporting a flat AP as "closed" while its own block says
   ~1.4 million seeds is the other half of the same failure.
2. **A representation effect smaller than ~0.013 AUROC is invisible at 4 seeds,
   and the honest report is "not measured", not "no effect".** If iwc1 comes
   back flat and UNDERPOWERED, the next move is seeds on the SAME cells, not a
   fifth slice.

⚠️ These sd's come from dermmnist / octmnist / tissuemnist -- **removed
datasets** (2(n)), so they are a PRIOR on the instrument, not a prediction about
iwildcam. iwildcam's test cameras are held out entire, which can widen the seed
spread rather than narrow it. Read the block iwc1 prints; do not substitute this
table for it.

⚠️ **`--campaign` takes ONE campaign root.** The evidence tarball holds `mcbar`
and `multiclass` side by side, and pointing the scorer at the tree above them
lands both campaigns' `clip/seed_1` on one (cell, seed, arm) key. `_one` refuses
-- correctly -- and now names which of the two causes fired and prints the
colliding paths, because the old message blamed the pairing key and sent the
reader into the scorer instead of into the path they passed.

🧱 **TWO POWER FLOORS BIND, NOT ONE, AND THEY ARE INDEPENDENT.** The table above
is the SEED floor -- can a cell see an effect of this size. `gen_campaign`
prints the other one, the CELL floor, and it is harsher. Regenerated
2026-08-23 for `--datasets iwildcam --models MobileNetV3 --caps L20_G50 L30_G50`:

    2 cells -> exact Wilcoxon floor p=0.50000; a lone mover needs p < 0.00455
    *** UNDERPOWERED: with 2 cells NO single metric can reach a *** verdict,
        whatever the effect size. 9 cells is the minimum for one.
    GENERALIZATION: 1 dataset(s) -> exact sign-flip floor p=1.000

**So no iwc1 contrast can ever be significant, at any effect size.** That is
arithmetic, not pessimism, and it was true before the campaign launched.

⚖️ **This does NOT sink the pre-registered verdict, because the verdict is a
NULL.** The two floors answer different questions and only one of them applies
here:

* The Wilcoxon floor governs REJECTING the null -- claiming an effect. iwc1
  cannot do that, so a positive iwc1 headline is unavailable whatever lands.
* The seed floor governs BOUNDING the null -- saying how big an effect would
  have had to be to show up. That is what "closed" means, and it is exactly the
  MDE table above.

**So state the null as an equivalence, with the number in it.** Not "AUROC was
flat" but **"any AUROC effect larger than ~0.013 would have been seen, and none
was"**, and the same for AP at ~0.035. A flat result with no bound attached is
the "tie means no effect" conflation that the RESOLUTION block exists to stop.

✅ **The number is now printed, so it does not have to be derived by hand.**
`full_panel`'s allocation-free block carries a `detectable` column --
`z*sd/sqrt(n)`, the exact inverse of `seeds_needed` -- which is what the seeds
actually present WOULD have caught. Read it off the campaign, not off the table
above: the table is the stored-evidence prior and iwc1 prints its own. Note the
sqrt: quadrupling the seeds only halves the bound, which is why "add seeds" is
expensive advice and why a backbone (a new cell) usually beats it.

🛑 **AND SCOPE IT TO WHAT WAS RUN.** Two cells from ONE dataset and ONE backbone
support "closed for MobileNetV3 on iwildcam at these two caps". They do not
support "the representation channel is empty" -- cells inside one dataset share
its test set and its warm-up, so two of them agreeing is section 0's sign-flip
floor and not independent replication. Widening needs a BACKBONE (resolution) or
a DATASET (independence); more seeds buy neither. If the verdict comes back flat
and POWERED, the honest next step is a second backbone on the same slice, which
is the cheapest cell the protocol allows -- and **that backbone is ViTB16**,
which is the a-priori headline (1-pre) and, independently, the best-resolved
backbone for a method contrast in the corpus (gap/sd 0.82 against MobileNetV3's
0.38). ⚠️ If iwc1 itself is running on MobileNetV3, then by 1-pre it is a
GENERALIZATION check and not the headline cell, and the verdict must be worded
that way.

**THE EXACT READ, in order.** Verified 2026-08-22 that `full_panel` accepts an
arbitrary control and validates it against the arms present, so the twin
contrast needs no new scorer:

```bash
python -m scripts.log_health results/iwc1                          # 0. did it RUN
python -m scripts.reachability <one-completed-run>                 # 0b. IS CE SATURATED HERE
python -m scripts.full_panel --campaign results/iwc1 --control tralo_null  # 1. THE VERDICT
python -m scripts.full_panel --campaign results/iwc1 --control clip        # 2. the bar
python -m scripts.straddle_probe --campaign results/iwc1           # 3. was it REACHABLE
```

🛑 **STEP 0b IS NOT OPTIONAL ON THIS DATASET, AND IT IS NEW.** Rule 1 fixes
warm-up at 1 because at warm-up 50 CE saturates and every method becomes
identical -- but that regime boundary was calibrated on **dermmnist**, a hard
7-class problem. **iWildCam's warm-up reaches 95.6% accuracy in ONE epoch**
(observed 2026-08-22 in the iwc1 log): eight camera-trap species are far more
separable than skin lesions. If CE is already saturated at warm-up 1 here, then
warm-up 1 on iwildcam is warm-up 50 on dermmnist, the regime protection is void,
and a tie across all arms would be the SATURATION and not the methods -- the
single most-repeated failure in this document, arriving through a door rule 1
does not cover. `reachability` measures `p(1-p)` at the cut; 2(a4) records that
converging the model drops it 60x. **Read it before reading any contrast.**

⚖️ **AND IT MAY CUT THE OTHER WAY -- do not pre-judge it.** The 95.6% is
`Train_Acc`, and iwildcam's test cameras are DISJOINT from training. A converged
TRAIN accuracy on a held-out-domain split is perfectly compatible with an
uncertain model on the TEST set, which is where the cut lives and where the
constraint acts -- and that is the very property 2(n) selected this dataset for.
So the saturation question is genuinely OPEN here in a way it was not on
dermmnist, where train and test are the same domain. `reachability`, measured on
the TEST predictions, is what answers it. Do not read `log_health`'s convergence
flag as a verdict in either direction.

✅ **THE ORDER IS GATED, 2026-08-23.** Three of the four commands had never run
against a campaign carrying `_null` twins, because no such campaign exists yet
-- so the first execution of this order would have been the night the data
landed. `test_the_preregistered_iwc1_read_runs_end_to_end` now runs all four
against an iwc1-shaped fixture (iwildcam x MobileNetV3 x 2 caps x 4 seeds, the
capped class ABSENT from 4 of 7 cameras) and asserts each prints the block read
out of it. Its negative control removes `tralo_null` and requires step 1 to
REFUSE, because a verdict against some other control answers a different
question.

⚠️ Building it surfaced a real near-miss worth keeping. The first fixture
crashed `full_panel` inside `_round_to_K`: a per-camera budget of 0.20 x 2 items
rounds to K=0 and the trainer refuses that by design. It is NOT a live risk --
`gen_campaign` calls the SAME `compute_local_constraints`, so a campaign whose
caps would raise at scoring time cannot be generated at all -- but the one
function shared by generator, trainer and scorer is what makes that true, and it
is the reason not to reimplement the rounding anywhere.

Step 0 first, every time: a dead arm reads as `pending`, and all eight tralo
runs of one campaign once OOM'd while the campaign merely looked unfinished.
Step 1 is the pre-registered verdict and it is read from the **ALLOCATION-FREE**
block (AP, AUROC) -- not from ccF1, which is the allocation channel. Step 2 is
the quality bar and answers a different question; a step-1 win with a step-2
loss is still a loss. Step 3 is what makes a TIE interpretable, per 2(o).

⚠️ `paired_seeds` reports `d AUROC` but **no AP**, and reads the ALLOCATED
predictions file. That is sound -- post-hoc rewrites `Predicted_Label` and never
the `Prob_Class_*` columns, so its AUROC is genuinely allocation-free -- but it
does not carry the second half of the pre-registered verdict. Use `full_panel`
for the verdict and `paired_seeds` only for the per-seed spread.

### (q) ⛔ THE FROZEN-HEAD PROBE DOES NOT TRANSFER TO iwildcam -- measured 2026-08-23

`frozen_head_probe` is one of the five pre-GPU pricing tools, and every number
ever quoted from it (`topk`/`ptopk` +1.2-1.3 items at ~24-36 seeds/cell) came
from **dermmnist**, a REMOVED dataset. Run on iwildcam embeddings it cannot be
read at all, and the tool says so itself.

Run on `results/iwc1` (MobileNetV3, L30_G50, `tralo_null`, the CE-only
representation), 8 seeds, corruption ladder 0.1 / 0.5 / 1 / 2:

    alpha 0.1      +0.00 items   0/8 negative   -
    alpha 0.5      +0.00 items   0/8 negative   -
    alpha 1       -35.09 items   8/8 negative   resolved
    alpha 2       -72.04 items   8/8 negative   resolved

🛑 **The probe RESOLVES 35.09 items on this feature space, and the entire
question is 1.9-9.9 items.** So every `NO DIFFERENCE` it prints here --
`topk` -0.28, `pauc` +0.00, `ptopk` -0.70 -- is **not a null, it is an absence
of measurement**, and none of them may be quoted. The resolution is a property
of the FEATURE SPACE, not of the harness, so it must be re-read on every
embedding file rather than inherited from the dermmnist runs.

**WHY iwildcam is the hard case for this instrument, and it is the same fact as
everywhere else in (p-post):** the top-K set here is unambiguous (`ccP` 0.999,
oracle gap 0.00-1.50 items). Small perturbations move nothing at all, and then
the head collapses between alpha 0.5 and 1. There is no graded middle for the
probe to sit in. A dataset on which the cheap probe works is one where the cut
is contested -- which is exactly the dataset property the whole project has
been unable to find.

⇒ **The loss-family direction (top-K surrogates, pAUC, perturbed top-K) cannot
be priced on iwildcam by this route.** That is a statement about the
instrument, not a verdict on the losses. Do not report it as one, and do not
spend a GPU campaign on the strength of a row this probe printed on iwildcam
features.

⚠️ **AND THE ADVERTISED INVOCATION COULD NEVER HAVE PASSED THE GATE.** The
liveness test is a two-sided sign test, so n seeds agreeing give
`p = 2^(1-n)`; at the default `--max-sign-p 0.01` it needs **8** non-zero
seeds. `CLAUDE.md` shows `--seeds 1 2 ...` and the natural reading is the
protocol's 4. At 4 seeds the probe measured a **72-item** corruption -- ten
times the whole headroom -- and still printed `NOTHING DETECTED AT ANY ALPHA`,
which reads as "the probe saw nothing" when what it could not do was clear its
own floor. It now prints the required number, the number actually run, and the
largest damage that failed to clear it. Gated by
`test_the_probe_says_how_many_seeds_its_own_liveness_gate_needs`, whose
negative control moves `--max-sign-p` to 0.2 and requires the printed
requirement to fall to 4 -- so the number is derived, not hardcoded.

### (p-post) 🔴🔴 THE READ, EXECUTED 2026-08-23 -- the representation channel is MEASURED, and it moved the WRONG WAY

`results/iwc1`, MobileNetV3 x iwildcam x {L20_G50, L30_G50} x 4 seeds, 9 arms,
all 72 runs completed. Every number below was re-derived from the campaign on
the day it was written, not carried over from a note.

**The pre-registration expected a TIE and said a powered tie would close the
channel. It is not a tie.** Both RANKING metrics moved, both are POWERED against
their own detectable threshold, and both moved in the LOSING direction on 2 of 2
cells.

**STEP 1 -- THE VERDICT. `tralo` against `tralo_null` (same warm-up, same
allocator, same seed, lambda=0):**

    family        metric   control    tralo     delta   cells   seed sd  detectable  verdict
    RANKING       AP        0.9585   0.9279   -0.0306    0/2     0.0155     0.0217   POWERED
    RANKING       AUROC     0.9903   0.9793   -0.0110    0/2     0.0064     0.0089   POWERED
    CALIBRATION   ECE       0.1549   0.1805   +0.0256    0/2     0.0184     0.0258   underpowered
    CALIBRATION   Brier     0.3453   0.3961   +0.0508    0/2     0.0365     0.0511   underpowered
    CALIBRATION   NLL       1.2319   1.4400   +0.2082    0/2     0.2141     0.2998   underpowered
    CALIBRATION   ConfGap   0.1053   0.0961   -0.0093    0/2     0.0201     0.0282   underpowered
    EQUALIZED     ccF1      -- d = -0.0018 = -0.95 items, ~17 seeds needed: underpowered

=> **State it as the equivalence the section demanded: any AUROC effect larger
than 0.0089 would have been seen, and what was seen was -0.0110 -- a LOSS. Any
AP effect larger than 0.0217 would have been seen, and what was seen was
-0.0306 -- a LOSS.** The channel is not empty. It is negative.

⚠️ **THE PRE-REGISTERED CONTROL IS WHAT MADE IT READABLE, and that is the
transferable lesson.** The same `tralo` effect measured against `clip` instead
of against its own twin is **AUROC -0.0108 with sd 0.0123, detectable 0.0172:
UNDERPOWERED**. Identical effect, opposite power verdict. The twin shares the
warm-up, so pairing against it removes the warm-up seed variance and halves the
sd (0.0064 vs 0.0123); pairing against `clip` leaves that variance in the
residual. **A contrast is not powered or unpowered on its own -- it is powered
against a particular control**, and choosing the control BEFORE the data is what
bought the resolution here. 2(p) picked the twin for attribution reasons and got
the power for free.

**STEP 2 -- the bar. Every arm against `clip`, allocation-free only:**

    arm                d AP   verdict         d AUROC   verdict
    lp              +0.0000   DEAD FLAG       +0.0000   DEAD FLAG (bit-identical)
    tralo_null      -0.0030   underpowered    +0.0002   underpowered
    tralo_reseed    -0.0086   underpowered    +0.0008   underpowered
    focal_clip      -0.0130   underpowered    +0.0012   underpowered
    tralo           -0.0336   POWERED         -0.0108   underpowered
    alm             -0.0639   underpowered    -0.0133   underpowered
    fioretto        -0.1035   underpowered    -0.0291   underpowered
    hounie          -0.2156   POWERED         -0.0739   POWERED

Three things fall out of that column, and none of them is the headline anyone
wanted:

1. 🟢 **`lp` is BIT-IDENTICAL to `clip` on all six allocation-free metrics, and
   that is the instrument liveness control passing on real data.** A post-hoc
   allocator rewrites `Predicted_Label` and never the probability columns, so it
   provably cannot move this family -- and now the panel has DEMONSTRATED it on
   the very campaign the verdict is read from, rather than the reader taking the
   argument on faith. If a future allocator ever shows a non-zero AP here, the
   scorer is reading an allocated column and the verdict is void.
2. **Both lambda=0 twins sit at zero** (`tralo_null` -0.0030/+0.0002,
   `tralo_reseed` -0.0086/+0.0008, all four underpowered). So the -0.0336 is
   attributable to the CONSTRAINT and not to the warm-up, the allocator or the
   RNG stream. Rule 1 of the three-rules block, doing exactly its job.
3. 🔴 **The losses are ORDERED BY CONSTRAINT PRESSURE.** null -0.0030, reseed
   -0.0086, `tralo` -0.0336, `alm` -0.0639, `fioretto` -0.1035, `hounie`
   -0.2156. More dual machinery, monotonically worse ranking, across four
   independently-implemented families. That is a dose-response curve pointing
   the wrong way, and it is the cleanest statement this project has of what the
   constraint phase actually does to the model: it does not re-rank toward the
   budget, it degrades the ranking it was given.

**STEP 3 -- was there anything to win. `straddle_probe`, re-run 2026-08-23:**

    cell / class                          emits K   true   ORACLE gap   reachable   shuffled ctrl
    L20_G50 / class 2                          74    370      1.00         1.00        25.00
    L20_G50 / class 7                          92    456      0.00         0.00        68.75
    L30_G50 / class 2                         111    370      1.00         1.00        18.50
    L30_G50 / class 7                         137    456      1.00         1.00        55.25

**The oracle gap is 0.00 to 1.00 items**, and `ccP` is 0.999. At these caps the
allocator is already perfect to within one item, so the 2(o) question answers
itself: there was nothing to win, and the measured effect is a loss thirty times
that size.

⛔ **AND THE CONTROL SAYS THE CUT IS NOT WHERE THE WIN IS.** `reachable` is far
BELOW its shuffled control in every cell (1.00 vs 18.50-25.00, 0.00 vs 68.75),
and 2(o) is explicit about the sign: `reachable << ctrl` means **the ranking
already took the easy swaps**. The one case worth training for is
`reachable >> ctrl`, positives parked below the cut, and iwc1 is its opposite in
all four (cell, class) combinations. A cut-local method has nothing to collect
here however it is tuned -- a measurement, and cheaper than the campaign it
saves.

🛑 **WHAT MAY AND MAY NOT BE CLAIMED FROM THIS.** The CELL floor binds exactly
as 2(p) predicted: 2 cells gives a minimum attainable Wilcoxon p of 0.500, so
every line in both tables prints `NOT CALLABLE` and **no significance claim is
available in either direction** -- including the negative one. What IS available
is the within-cell power statement, and it is unanimous: 0 of 2 cells, both
POWERED, same sign. Say "on both cells measured, the constraint cost the ranking
more than this campaign could have missed"; never "the constraint significantly
degrades the ranking", which 2 cells cannot support at any effect size.

📌 **SCOPE, per 1-pre.** iwc1 is MobileNetV3, so it is a GENERALIZATION check
and NOT the headline cell. The a-priori headline backbone is ViTB16, which is
also the best-resolved backbone for a method contrast in the corpus (gap/sd 0.82
against MobileNetV3 0.38). `results/iwc2` is that cell -- same slice, same two
caps, ViTB16, `clip` / `focal_clip` / `tralo` / `tralo_null` / `tralo_reseed` --
launched 2026-08-23 08:50 on dsisco01 GPU 3 at `code_version 3bb7e8b4`. **It is
the first genuinely pre-registered test of this verdict**, because the
prediction now exists in writing before its data: if the mechanism above is
real, iwc2 must show `tralo` LOSING AP and AUROC to `tralo_null`, and the
lambda=0 twins sitting at zero against `clip`.

⚠️ **STEP 0 FOUND A DEFECT IN THE READOUT ITSELF, now fixed.** The `log_health`
starvation warning fired on `tralo_reseed` -- a lambda=0 twin with
`constraint_steps_applied: 0` on all 8 runs -- attributing the 2(a2) penalty
mechanism to an arm that has no penalty, on the one arm this project uses as its
NOISE FLOOR. The warning is now gated on positive evidence that a constraint
step was applied, which also covers a live arm whose cap never bound in a seed.
Gated by `test_the_starvation_warning_is_never_made_about_an_arm_with_no_penalty`
(both arms carry byte-identical trajectories, so the steps key is the only
possible cause of a difference) and by a second control asserting that runs
predating the key keep the diagnostic rather than losing it silently.

📎 **STEP 0b, for the record.** `reachability` reports 2 of 2 (class, cell) cuts
sitting where `p(1-p)` has gone flat. Read it with the caveat the script itself
prints: the cut is the K-th RANKED item and predictions change at the DECISION
BOUNDARY, which is a different item whenever the hard count exceeds K. So this
bears on how much HEADROOM the cap leaves -- and step 3 independently measured
that headroom at under two items -- not on whether the penalty had anywhere to
push. The saturation question 2(p) raised is therefore answered in the direction
that matters: there was little to win here, and the arm lost anyway.

### (r) 🔴🔴🔴 THE CONSTRAINT EVICTS THE CORRECT ITEMS -- measured 2026-08-24, and it is the mechanism

`scripts/order_probe.py --evictions`, `results/iwc2`, `tralo` against its OWN
lambda=0 twin, 16 (cell, class, seed) points. The allocator is a top-K, so it
reads the ORDER of the capped class and nothing else. At the arm's own budget:

| | `tralo` | `tralo_reseed` (control) |
|---|---|---|
| items moved per cell | 73.1 of K=399 | 62.8 |
| precision of what it **EVICTED** | **0.6880** | 0.4860 |
| precision of what it **ADMITTED** | **0.3007** | 0.4844 |
| **NET items per cell** | **-30.44**, 16/16 negative | **+0.38** |

⇒ **attributable to the constraint on iwc2: -30.81 items per cell, a 38.6 pp
precision gap.**

🛑 **AND THE PRIZE ON iwildcam IS 0.2 TO 2.0 ITEMS -- SMALLER THAN dermmnist,
not larger. A TOOL BUG SAID OTHERWISE FOR ONE HOUR ON 2026-08-24.**
`scripts/headroom.py` set `K = int(G[c])`, the GLOBAL cap alone. Local caps are
per-group ceilings so their SUM already bounds the count, and on iwildcam the
global is INERT above it -- `gen_campaign` prints exactly that for every cap it
emits ("INERT GLOBAL: K=185 is above the local sum 111, so it can never bind").
The ceiling `2K/(K+n)` then read 0.667 where the reachable one is 0.462, and the
tool printed **59-114 items of headroom where the real gap is 0.2-2.0**.

Fixed (`effective_budget`, gated both directions). Corrected, `results/iwc2`:

| cap | class | n | K (binding) | ceiling | achieved | headroom |
|---|---|---|---|---|---|---|
| L20_G50 | 2 | 370 | 74 | 0.3333 | 0.3266 | **1.5 items** |
| L20_G50 | 7 | 456 | 92 | 0.3358 | 0.3349 | **0.2 items** |
| L30_G50 | 2 | 370 | 111 | 0.4615 | 0.4532 | **2.0 items** |
| L30_G50 | 7 | 456 | 137 | 0.4621 | 0.4604 | **0.5 items** |

Independently confirmed by counting the equalized top-K straight off the stored
predictions: 0.0-4.0 items. And it agrees with (q)'s "oracle gap 0.00-1.50
items", which was right all along and which the buggy number contradicted.

⇒ **`clip` is already within TWO ITEMS of a perfect allocator on iwildcam.**
Section 4's pessimism is not merely intact, it is stronger here than on
dermmnist. Nothing in this document is reopened by headroom, and the
"re-price every closed direction" note that briefly stood here is WITHDRAWN.

⇒ **It also restores the scale of (r) itself.** The constraint costs 3.4 items
(MobileNetV3) to 30.8 (ViTB16) against a prize of 0.2-2.0. That is not a
fraction of the headroom -- it is **2 to 150 times it, spent backwards**, which
is the largest measured effect in this project and the one thing worth removing
before any win is attempted.

🔑 **THE CONTROL IS THE WHOLE MEASUREMENT.** EVICTED items sat in the twin's
top-K *by construction*, so they outrank ADMITTED ones and **any** perturbation
nets negative on this statistic. `tralo_reseed` moves a COMPARABLE number of
items and nets +0.38, with evicted and admitted precision equal to three
decimals -- a perturbation of no consequence swaps items of equal quality.
Never quote the arm's number without it.

⛔ **IT IS NOT A BOUNDARY OR PLACEMENT EFFECT.** The cut sits at p=0.5359 while
evicted items average **p=0.7884** and admitted ones **p=0.2510**. The damage
spans the whole range, not the margin. ⇒ **`tralo_margin` is not predicted to
fix this**: 4(b) prices placement at <=1.30x on the items that must flip, and
placement is not what is wrong. `docs/launch_margin1.sh` still answers its own
question (value vs placement) but it does not answer this one.

⚠️ **READ THE BAND, NOT THE GLOBAL RHO.** Pooled Spearman against the twin is
0.6953 for `tralo` and 0.6694 for the reseed -- globally the constraint
preserves order slightly BETTER than noise. Restricted to the contested band
(ranks K/2..2K) it is 0.6134 against 0.7011, **worse than noise, 16/16**. A
probe reporting only the global number calls this a null.

🟢 **THE FIX, BUILT AND PRE-REGISTERED: `soft_count_mode: uniform`
(`tralo_uniform`).** The cap is satisfiable with ZERO reordering -- drop the
class logit by a constant and every `p_ic` falls monotonically while the order
is exactly preserved -- so a harmless path always exists and the shipped loss
simply does not take it. `d(sum_i p_ic)/dz_ic = p(1-p)` differs per item, which
is what singles items out. `uniform` keeps the count's VALUE exact and makes the
per-item gradient CONSTANT in log-odds, where `du_c/dz_c = 1` exactly, so a
uniform step is a bias shift and cannot reorder. Same dose (`w` is the mean of
`p(1-p)`), different distribution. Live by md5 on every binding seed
(`flag_live`), gated on the gradient itself with autograd rather than on the
smoke harness, which cannot see it (rho=0.999965 for both modes).

**PREDICTED**: recovers the ~30 items and lands `tralo` ON its own null.
**NOT predicted**: that it BEATS the null -- a uniform shift is a prior shift and
2(j) says top-K is invariant to those. The claim is "the constraint becomes
free". **FALSIFIED IF** net items vs the twin stays materially negative, which
would put the damage in the SHARED BACKBONE rather than the per-item output
term, and move the next lever to which parameters the constraint may touch.

Launch: `docs/launch_uniform.sh` (9 cells, 6 arms, 4 seeds = 216 runs; 9 cells
is deliberate -- sign-test floor 0.00391 against BH 0.00455, so unlike iwc1/iwc2
it can return a CALLABLE verdict). Read with `order_probe --evictions` FIRST.

### (s) 🔴🔴🔴 EVERY DUAL'S MARGIN IS THE 29 EPOCHS, AND THE FAMILY ORDERING IS COLLATERAL DAMAGE

`results/xfam1`, dsisco02, scored 2026-08-24 at 141 of 324 runs. iwildcam x
{MobileNetV2, MobileNetV3, RegNetY400MF} x {L20_G50, L30_G50, L50_G30} = 9
cells, 16 matched cell-seeds, nine arms including **a lambda=0 twin for every
dual family** -- the thing the 7,574-row corpus does not have in a single row
(see the corpus audit in section 1), and without which no published number in
this project could be attributed to a constraint rather than to compute.

Read with `python -m scripts.family_split --campaign results/xfam1`.

🔑 **THE POSITIVE CONTROL IS EXACT, AND IT COMES FIRST.** At lambda = 0 the
dual family is irrelevant: same cached warm-up, same allocator, same seed, no
constraint gradient. So `tralo_null`, `fioretto_null` and `hounie_null` must be
the SAME RUN. They are -- **byte-identical raw predictions in 12 of 12
cell-seeds** (md5, CLAUDE.md rule 3), while `tralo_reseed` and `clip` differ
from them everywhere. Three consequences, all load-bearing:

* the compute term is **one number**, not one per family, by construction;
* the per-family constraint term is therefore a clean difference;
* two thirds of the null runs in a cross-family campaign are **redundant
  compute** -- the next one wants ONE full null plus a one-cell identity
  check across families, not one null per family, which frees 64 of 324 runs.

`family_split` checks the digests and **refuses to print a table** if they ever
diverge, because then something other than lambda differs between the families
and every constraint term below would be contaminated.

#### The decomposition

`total = compute + constraint`, identically, per seed:

    total      = arm      - clip        what the manuscript reports
    compute    = arm_null - clip        29 epochs at lambda = 0
    constraint = arm      - arm_null    the method's OWN contribution

| metric | compute (all three) | constraint `tralo` | constraint `fioretto` | constraint `hounie` | reseed floor |
|---|---|---|---|---|---|
| macroF1 | **+0.0145** | -0.0113 | -0.0026 | -0.0092 | -0.0044 |
| uncF1 | +0.0194 | **-0.0144** | **-0.0027** | **-0.0114** | -0.0058 |
| ccF1 | -0.0003 | -0.0020 | -0.0023 | -0.0028 | -0.0001 |
| AP | -0.0060 | **-0.0609** | **-0.1218** | **-0.1362** | +0.0241 |
| AUROC | -0.0047 | -0.0102 | -0.0454 | -0.0474 | +0.0089 |
| ECE | -0.0006 | -0.0296 | -0.0497 | -0.0504 | -0.0036 |
| Brier | -0.0020 | -0.0566 | -0.1003 | -0.0973 | -0.0032 |
| NLL | -0.0477 | -0.3692 | -0.6093 | -0.5467 | -0.0969 |

Signs are oriented so **+ is better** on every row, including the three where
lower is better natively. `ccF1` converts at **5.2 items per 0.01** here.

🛑 **NOT ONE OF THE 24 CONSTRAINT TERMS IS POSITIVE.** Eight metrics, three
families, and every method's entire margin over `clip` is the 29 epochs that
every trained arm gets and the post-hoc clipper does not. This is section 3's
"regime beats method" measured directly instead of inferred -- and it is the
first time the inference has had the twin it needs.

🔑 **THE PUBLISHED ORDERING IS A DAMAGE RANKING WEARING A BENEFIT RANKING'S
CLOTHES.** On macro-F1 the totals are `fioretto` +0.0118 > `hounie` +0.0052 >
`tralo` +0.0031, which is exactly the manuscript's ordering. The compute term
is identical across the three, so that ordering is `0.0145` minus the damage --
i.e. it ranks the families by how little each one spoils a gain none of them
produced. **TraLO is not "improving less". It is subtracting more.**

⚠️ **AND `fioretto`'s ADVANTAGE SITS BELOW THE NOISE FLOOR.** `tralo_reseed`
-- the same null with one RNG draw perturbed and nothing else -- moves macroF1
**-0.0044**, and `fioretto`'s whole constraint term is **-0.0026**. So
"fioretto's constraint is gentle on macro-F1" is not distinguishable from
"fioretto's constraint does nothing to macro-F1", and the metric the paper
headlines cannot separate the two.

#### The ordering REVERSES on the only channel an allocator can see

A top-K allocator reads the ranking and nothing else; a calibration move
provably leaves every top-K set untouched (section 2(j)). On AP the constraint
terms are `tralo` **-0.0609** against `fioretto` -0.1218 and `hounie` -0.1362
-- TraLO damages the ranking **2.0x and 2.2x less**. On AUROC it is -0.0102
against -0.0454 and -0.0474: **4.5x and 4.6x less**.

**TraLO is the gentlest of the three on the representation and the harshest on
the composite, and nobody was scoring the channel where it wins.** That is not
a rescue -- all three are still negative, and negative is the finding -- but it
says the shortfall is not in the constraint machinery TraLO was designed
around.

#### Where the difference actually lives

`ccF1` and `uncF1` split macro-F1 into the classes the constraint names and the
six it does not. The capped-class terms are **-0.0020 / -0.0023 / -0.0028** --
about one item, near-identical, and all three inside each other's noise. The
uncapped terms are **-0.0144 / -0.0027 / -0.0114**, a **5.3x spread**.

🔑 **All three dual families do the same negligible thing to the classes the
constraint is about, and differ five-fold in what they do to the classes it
never mentions.** The entire cross-family story is collateral damage.

#### Mechanism: the leak is real, the obvious fix is a NULL, and that is the result

The shipped count is `S_c = sum_i softmax(z)_ic`, so
`dS_c/dz_k = -sum_i p_ic p_ik` is **nonzero for every uncapped k**: one push on
a capped class moves all eight logits. That looked like the whole story, and a
one-vs-rest count `S_c = sum_i sigmoid(z_ic)`, whose gradient is **exactly
zero** outside the capped columns at any dose, was staged as the fix.

⛔ **IT IS A NULL, measured for 0 GPU-hours, and the algebra says why.** The
update adds `+eta * p_ic * p_ik` to `z_k`, which is **monotone increasing in
`p_ik`** -- it widens the gaps in the uncapped block in the direction they
already point. It SHARPENS the existing order and cannot invert it.

`scripts/collateral_probe.py`, 16 stored runs, effect matched at 20 / 50 / 100
/ 200 capped predictions removed (dose is matched on EFFECT, not step size --
at one unit-norm step no mode flips anything and an equal-dose read returns
0 vs 0 and says nothing):

| target | `sum` eta | `sum` uncapped logit moved | `sum` unc->unc flips | `ovr` flips |
|---|---|---|---|---|
| 20 | 7.5 | 0.77 | **0.00** | 0.00 |
| 50 | 87.2 | 9.76 | **0.00** | 0.00 |
| 100 | 149.7 | 12.03 | **0.00** | 0.00 |
| 200 | 1091.7 | **79.12** | **0.00** | 0.00 |

Zero flips across a 50x dose range, at the end of which the uncapped logits
have moved **79 units**. The cross-term perturbs the uncapped block and
provably cannot reorder it, so `ovr` removes a leak that costs nothing.

🔑 **THEREFORE THE uncF1 DAMAGE DOES NOT COME THROUGH THE OUTPUT LAYER.** It
comes through the **shared backbone** -- 29 epochs of constraint gradient
flowing into the features, which moves every class because every class reads
the same representation. That is the lever `docs/launch_uniform.sh` named in
advance as the fallback if the output-space fix failed: **the parameter set the
constraint is allowed to touch, not the count it is computed from.**

⚠️ **This is NOT section 2(a)'s renormalisation.** 2(a) zeroes the
**capped-vs-capped** cross-term and was scored on *count movement*, not a
metric (rule 5); it came out 0.95x and was shelved. This is the
**capped-vs-uncapped** term scored on *uncapped F1*. Different quantity,
different objective, same verdict.

⚠️ **A SECOND FINDING FELL OUT: the shipped count SATURATES.** `sum` and
`uniform` could not remove 100 capped predictions in 10 of 16 cells, or 200 in
20 of 32, at any `eta <= 4096` -- by which point the logits have moved tens of
units. `ovr` reached every target up to 200 and 8 of 16 at 400. So the softmax
count has a hard ceiling on how much cap it can enforce through the output
layer, which is 2(a2)'s vanishing gradient measured from the other side. It is
not a reason to run `ovr`: enforcement is not the binding problem here, the
allocator already fills to exactly K.

Gated by `test_the_softmax_cross_term_CANNOT_reorder_the_uncapped_classes`
(negative control: random noise on the same block MUST reorder it, or the
assertion is passing for a trivial reason) and by
`test_ovr_count_has_ZERO_gradient_outside_the_capped_columns`.

#### What this does NOT support

* **Nothing outside iwildcam.** One dataset, so the clustered-by-dataset unit
  is p = 1.000 by construction. Every p below is a CELL-level sign test.
* **The callable subset is smaller than the table.** Surviving BH at 0/9 cells,
  p = 0.0039: ECE, Brier and NLL for all three families, and AP and AUROC for
  `fioretto` and `hounie`. `tralo`'s AP loss is 1/8 cells, p = 0.0273, BH
  q = 0.060 -- **directional, not called**. Every macroF1 and ccF1 contrast is
  UNDERPOWERED at the seed level (53, 9 and 19 seeds per cell needed).
* **16 matched cell-seeds is 2 seeds in 6 cells and 1 in 3.** The campaign is
  at 141 of 324 and the table must be re-read at 4 seeds before any of the
  underpowered lines is quoted.

### (t) 🟡 `ortho` IS NOT REFUTED, IT IS UNTESTED -- and 2(s) points straight at it

⚠️ **CORRECTION TO THIS DOCUMENT'S OWN RECORD, made 2026-08-24.** Section 1
names `ortho` in the list of "rejected-arm campaigns this document names".
**This document never named a verdict for it.** Grep the file: `ortho` appears
three times, once in that list, once in a mistake-pattern row about a
checkpoint selector, once in the repository layout. There is no result section,
no table and no reason. A direction cannot be closed by appearing in a list of
closed directions.

**What `ortho` is.** The constraint gradient projected onto the orthogonal
complement of the CE gradient, so enforcing the cap cannot undo CE progress.
`ortho_project: true`.

**What was actually run**, recovered from `evidence/provenance_2026-08-18.tar.gz`
(all 14,524 configs): `ortho_project` appears in **24 configs, true in 4**. The
campaign is `newdirections/arm_ortho/results/ortho`:

| | |
|---|---|
| runs | **8** -- `ortho_on` x 4 seeds, `ortho_off` x 4 seeds |
| cell | **one**: dermmnist x MobileNetV3 x **L30_G30** |
| regime | warm-up 1 / constraint 29, `lr_constraint` 1e-4 -- the LIVE regime |
| clipper in campaign | **none**. Neither `clip` nor `focal_clip` |
| prediction files | **0** -- it cannot be re-scored, ever |

That campaign violates three of the five rules at the top of `CLAUDE.md`:
**rule 2** (both clippers in the campaign -- it has neither), **rule 4** (at
least two cap levels -- it has one, and `L30_G30` is the level where the global
cap is REDUNDANT by section 1's arithmetic), and the dataset is **dermmnist**,
which section 2(n) rules structurally incapable of carrying a per-group count
constraint and which is 38.7% test-set leaked.

🔑 **AND ITS ONE NUMBER IS POSITIVE.** The archive records
**AP +0.0041 pre-restore** for the projection, against +0.0003 after the
end-of-run checkpoint restore -- and states the reason in its own words: the
restore's criterion is *total excess*, **exactly the quantity the projection
deliberately trades away**, so the treatment arm was systematically handed worse
candidates. The selector compressed the effect ~13x and it was biased against
the treatment.

⇒ **The only measurement of the orthogonal projection is in its favour, made
under a selector designed to disfavour it, on 8 runs in one cell of a dead
dataset at the one cap level where the global scope does nothing.** That is not
a refutation. It is an absence of measurement, and it has been filed as a
refutation since 2026-08-18.

#### Why it matters NOW

2(s) measured that the constraint's damage to the six uncapped classes does
**not** arrive through the output layer -- the softmax cross-term perturbs
those logits but provably cannot reorder them, zero flips across a 50x dose
range. It arrives through the **shared backbone**, which is precisely what a
projection onto the complement of the CE gradient acts on. Of every
intervention this project has tried, it is the only one aimed at the mechanism
2(s) actually found, and the only one whose recorded sign is positive.

🛑 **SIZE IT HONESTLY BEFORE SPENDING A GPU.** 2(s) puts TraLO's AP constraint
term at **-0.0609**. A projection worth +0.0041 recovers **~7% of that**. On
its own that is a nibble, not a fix -- and the +0.0041 was measured where the
cap barely binds (`L30_G30` on dermmnist, `lp_fallback_used` False with 0
candidates on all 52 derm runs), whereas on iwildcam 7 of 14 per-group ceilings
are K = 0 and the local scope binds at every cap level. The effect there could
be larger or smaller; **nothing in hand predicts which**, and quoting +0.0041
as the expected iwildcam effect would repeat 2(m)'s error.

🔧 **THE FLAG HAD BEEN PURGED, AND IS NOW BACK.** `grep -rn ortho src/ configs/ main.py`
`ortho_project` was purged in the 2026-08-18 code deletion **without an entry
in 2(f)'s table of what was deleted and why**, which is the record a deletion
is supposed to leave -- so the flag vanished and the direction was filed as
closed with neither a verdict nor a deletion reason anywhere.

It is re-implemented as of 2026-08-24: `snapshot_grads` takes the CE gradient
off the parameters after the CE loop (free -- it is cleared on the next line
anyway), and `project_out` removes the constraint step's component along it
**before** the norm bound, so under `constraint_grad_mode: normalize` the
projected and unprojected arms deliver exactly the same step size and differ in
direction alone. Projecting after the bound would shorten the treatment's step
and confound direction with dose; that is gated, with the post-bound version as
the negative control.

⚠️ The reference is **one CE minibatch**, not the epoch's CE direction -- the
cheap estimate, and the honest description of what is removed is "the component
along the last CE step actually taken". A full-epoch reference costs a second
pass over the training set every epoch.

⚠️ Do not confuse this with `separate_constraint_optimizer`, which IS rejected
with a verdict (AP -0.0938, p = 0.0006, 2(f)) and which `CLAUDE.md` names in
its do-not-run list. That one gives the constraint its own Adam state. This one
changes the DIRECTION of the shared step and leaves the optimizer alone. They
are different interventions and only one of them was measured.

⛔ **THERE IS NO OFFLINE PRICE FOR THIS ONE.** Every probe in section 2 that
closed a direction for 0 GPU-hours read stored *outputs*. A projection acts on
*parameter gradients during training*, which no stored artefact records. So
unlike the one-vs-rest count in 2(s), this cannot be killed from `evidence/`
and a campaign is the only instrument. That raises the bar on the campaign
rather than lowering it:

* **both clippers in-campaign** (rule 2), **at least two cap levels with
  `G < L`** so the global scope binds (rule 4 + section 1);
* **`tralo_null` and `tralo_reseed`**, so the projection is read against
  lambda = 0 and against the RNG floor, not against `ortho_off` alone;
* **liveness first**: `scripts/flag_live tralo tralo_ortho` must show the raw
  predictions DIFFER, and `Ortho Fired Frac` must be non-zero. `ortho_project`
  would be the fifth inert flag in this project's history if it were not
  checked, and a campaign whose 4-run predecessor left no predictions cannot be
  audited after the fact;
* **pre-restore metrics**, or the restore re-runs the same compression.

Until that campaign exists, the honest entry for `ortho` is **OPEN**, and this
section replaces its listing among the rejected.

---

## 3. WHAT WE KNOW WORKS -- regime beats method, every time

**The single most useful fact in this project: regime effects are ~8 pp. Method effects are ~0.1 pp.**
Every "win" that turned out to be a regime difference in disguise was bigger than every real
method effect.

📐 **MEASURED ON THE PAPER'S OWN CORPUS, 2026-08-23 -- the ~0.1 pp is now a
number with 260 cells behind it, and the win DECOMPOSES.** Paired per seed
within cell, `warmup_epochs = 50`, macro-F1:

| contrast | trained? | mean | wins | sign p | per-cell detectable at 4 seeds |
|---|---|---|---|---|---|
| tralo - **heuristic** | vs POST-HOC | **+1.85 pp** | 184/236 | 1.8e-18 | 2.05 pp |
| fioretto_ldf - heuristic | vs POST-HOC | +1.64 pp | 177/234 | 1.7e-15 | |
| tralo_bounded - heuristic | vs POST-HOC | +1.54 pp | 187/252 | 7.3e-15 | |
| hounie_rcl - heuristic | vs POST-HOC | +1.13 pp | 159/234 | 4.2e-08 | |
| **danits_lp - heuristic** | **POST-HOC vs POST-HOC** | **-0.06 pp** | **84/257** | 3e-08 | |
| tralo - fioretto_ldf | same compute | **+0.15 pp** | 170/262 | 1.7e-06 | 0.71 pp |
| tralo - tralo_bounded | same compute | **+0.13 pp** | 164/260 | 2.9e-05 | 0.68 pp |
| tralo - hounie_rcl | same compute | +0.73 pp | 185/264 | 5.7e-11 | 0.88 pp |

**Read the danits_lp row first: it is the control.** Every TRAINED method beats
the clipper by 1.1-1.9 pp, and the one method that is also POST-HOC does not --
it loses. So the effect tracks *having a constraint phase*, not *which* one.

🛑 **AND THE +1.85 pp IS NOT UNIFORM -- it REVERSES on one of the four
datasets.** The pooled figure is rule 4's hazard in its purest form, so break it
out (macro-F1, warm-up 50):

| dataset | tralo - heuristic | cells positive | danits_lp - heuristic | tralo - fioretto_ldf |
|---|---|---|---|---|
| tissuemnist | +3.24 pp | 91% | -0.04 pp | +0.09 pp |
| octmnist | +2.28 pp | 97% | -0.35 pp | +0.43 pp |
| dermmnist | +1.02 pp | 78% | -0.03 pp | +0.02 pp |
| **aider** | **-0.53 pp** | **14%** | +0.02 pp | +0.45 pp |

**TraLO LOSES on aider, on 86% of its cells.** The clustered unit is the
DATASET, so the honest generalization statement is **3 of 4**, which is section
0's exact sign-flip floor -- p=0.25, not significant at any cell count. The
"184 of 236 cells, p=1.8e-18" above is a statement about CELLS and must never be
quoted as if it were about datasets.

📄 **THE PAPER DOES NOT REPORT aider AT ALL** -- `main_edited_by_roei.tex`
names OctMNIST 58 times, DermMNIST 31 and TissueMNIST 18, and aider zero. So
nothing in the manuscript is contradicted by the reversal. But the shape is
worth stating plainly, because a reviewer will: **the three datasets reported
are the three where TraLO wins, and the one excluded is the one where it loses.**

⚖️ **This was deliberated, not hidden -- and the corpus now settles it.**
`docs/archive/PROFESSOR_REVIEW.md` item 2 asks in as many words whether to "keep
it as the saturated-regime ablation ... or drop it from the headline and move to
an appendix", and item 3 records a known "AIDER F1 gap" for Hounie. So the
question was open and documented; what was missing was the number. It is
**-0.53 pp on 86% of cells**, and that turns an open decision into a disclosure
obligation: a paper reporting 3 of 4 datasets should say which the fourth was
and why it was dropped, in its own words, rather than leave the ratio to be
discovered.

⚠️ **And do not read `docs/archive/REJECTED_full_2026-08-18.md`'s "aider already
a winner" as a result.** In context it is a 2026-05 note about which datasets
are still worth SEARCHING for -- "already in the set", not "TraLO wins there".
The corpus says the opposite of the second reading. The archive is history and
is not being edited; the correction lives here, where it is live.

✅ **What DOES hold in every dataset separately:**
* `danits_lp` -- the post-hoc control -- wins **46 / 33 / 28 / 29%** of cells.
  Below half everywhere. The control is not carried by one dataset.
* `tralo - fioretto_ldf` stays under **+0.45 pp** everywhere. The
  method-specific part is small in all four, including the one where TraLO beats
  the clipper by 3.24 pp and the one where it loses.
* Dropping dermmnist entirely -- its test set is LEAKED (2, 38.7%/67.3% MEL) --
  changes nothing: `tralo - heuristic` **+2.33 pp** (115/148),
  `danits_lp - heuristic` **-0.09 pp** (51/157), `tralo - fioretto_ldf`
  **+0.24 pp** (121/163). The decomposition does not depend on the contaminated
  dataset.


🔢 **The decomposition: of TraLO's +1.85 pp over the clipper, about +1.7 pp is
"being a trained method at all" and about +0.15 pp is TraLO.** That is roughly
92% / 8%, and the 8% is BELOW its own per-cell bound of 0.71 pp -- it reaches
significance only by being consistent across 262 cells (65% of them), never
because any single cell can see it. Section 3's "~0.1 pp" was right, and this is
what it looks like with 260 cells and a p-value.

🔴🔴 **THE THIRD CONTROL, AND IT IS THE ONE THAT SETTLES IT: 63% OF THE
HEADLINE'S CELLS HAVE A LOCAL SCOPE THAT IS EMPTY BY CONSTRUCTION -- AND THE WIN
IS BIGGER THERE.** The corpus records which group variable each cell used, and
only ONE dataset used a real one:

| dataset | group_column | cells | tralo - heuristic | cells positive |
|---|---|---|---|---|
| tissuemnist | **synth_group** | 91 | **+3.24 pp** | 91% |
| octmnist | **synth_group** | 29 | **+2.28 pp** | 97% |
| dermmnist | loc_group | 83 | +1.00 pp | 77% |
| dermmnist | sex | 5 | +1.37 pp | 100% |
| aider | **synth_group** | 28 | -0.53 pp | 14% |

| | cells | mean | wins |
|---|---|---|---|
| **`synth_group` -- local scope empty by construction** | **148 (63%)** | **+2.33 pp** | 115/148 (**78%**) |
| real groups (`loc_group` / `sex`) | 88 | +1.02 pp | 69/88 (**78%**) |

`synth_group` is `np.arange(len(y)) % 3` (2(n)). The groups are therefore i.i.d.
draws from ONE distribution, so every group's class distribution is identical by
construction and a per-group cap is exactly the global cap divided by three --
which 2(j) proved cannot reorder a top-K set at any size. **On 63% of the
headline's cells the local constraint provably carries zero information, and
those cells show a LARGER win at an IDENTICAL win rate.**

🛑 **The point is that the group definition makes NO difference.** aider is also
`synth_group` and loses, so "synthetic groups cause wins" is false and is not
the claim. The claim is the reverse and it is stronger: the one variable that
decides whether a local cap can carry information at all -- how the group is
defined -- **does not move the result**, 78% against 78%. A mechanism that
depends on per-group information cannot produce that.

🔎 **AND THE SAME HOLDS ON THE CAPPED-CLASS METRICS, where the constraint's
signature should be STRONGEST -- this was looked for, not assumed.** Repeating
all three controls on `cc_f1`, `cc_prec` and `cc_rec` (tralo vs heuristic,
warm-up 50, synth-group cells against real-group cells):

| metric | synth (scope empty) | real groups |
|---|---|---|
| cc_f1 | +1.13 pp, 62% | +0.59 pp, 67% |
| cc_prec | +2.25 pp, 67% | +1.27 pp, 70% |
| cc_rec | +0.83 pp, 62% | +0.41 pp, 64% |

Larger on the cells where the local scope is empty, in all three; win rates 2-5
points apart, which is inside the standard error of a proportion at n=88. Cap
slopes within dataset are near zero and MIXED in sign (-0.006 to +0.019). Every
one of the three tracks the DATASET instead (tissuemnist > octmnist ~ dermmnist
> aider, negative). **There is no metric in the corpus on which the count
constraint leaves a fingerprint.**

⚠️ And the "real" arm is not much of a control either: 2(n) measured dermmnist's
`loc_group` at NET +65 items, z=2.9 -- it clears stage 1 and still nulls,
because its test groups ARE its training groups. **No dataset in the corpus has
a local scope carrying real information. iwildcam is the first**, at NET +3131,
z=97.4, with 7 test cameras absent from training entirely.

**So the corpus supports "training beats post-hoc allocation" and supports
nothing about the count constraint.** That is not a retraction of the paper's
numbers -- they are what they are -- it is a statement about what they can be
attributed to, and the answer is: not the mechanism the paper is named for.

✅ **AND THE PAPER'S OWN CLAIMS SURVIVE ALL OF IT -- checked 2026-08-23, because
an audit that only looks for failures is not an audit.** Recomputed
independently from the corpus, the abstract's two specific claims hold:

* **"constraint-time training ... improving overall classification quality by
  1.6 to 5.3 pp over satisfying the cap after training".** Note the subject:
  **constraint-time TRAINING**, not TraLO. That is exactly the attribution the
  92/8 decomposition above supports. The abstract does not claim the margin for
  TraLO's mechanism, and the loose reading that it does is the reader's error,
  not the paper's.
* **"one regime shows a consistent advantage across all backbones tested:
  OctMNIST at tight caps, with the largest gains on a ViT-B/16".** In its own
  regime (octmnist, `cc_f1`, warm-up 50):

  | contrast | cells | mean | wins | sign p |
  |---|---|---|---|---|
  | tralo - hounie_rcl | 29 | +2.25 pp | 27/29 | 1.6e-06 |
  | tralo - fioretto_ldf | 29 | +1.18 pp | 23/29 | 0.0023 |
  | tralo - tralo_bounded | 29 | +1.09 pp | 24/29 | 0.00055 |
  | tralo - heuristic | 29 | +0.56 pp | 21/29 | 0.024 |

  and by backbone: **ViTB16 +1.90 pp (8/9)**, RegNetY400MF +1.15, MobileNetV3
  +0.62 -- ViT largest, as written. By cap level the effect peaks at L30 (+4.60)
  and L40 (+2.88) and is ~0 at L10 and L20; the paper's "tight cap" is L40
  (Sec. Results), so its definition and its data agree.

🔑 **The two findings are consistent, and it is worth being precise about why.**
octmnist is a `synth_group` dataset, so its local scope is empty by
construction -- and the paper never attributes this advantage to per-group
information. It attributes it to **the optimizer reset and the undershoot
hinge**, two portable components, and explicitly reports the saturating penalty
shape as NEUTRAL. Today's audit independently supports that attribution: the
effect does not depend on the group definition, does not scale with cap
tightness, and disappears against a post-hoc arm. **What the corpus refutes is a
claim the paper does not make.**

### 🔬 THE HINGE ABLATION IS NOT BUDGET-EQUALIZED -- audited 2026-08-23

The paper's central mechanistic finding names the undershoot hinge as
load-bearing. `ablation_no_hinge.csv` (24 runs, octmnist, 3 backbones x
{L30_G30, L40_G40} x 4 seeds) pairs exactly against `paper_final`'s `tralo` on
the same cells, so the ablation recomputes directly:

| metric | hinge - no_hinge | cells | sign p | detectable at 4 seeds |
|---|---|---|---|---|
| **cc_f1** | **+3.23 pp** | **6/6** | 0.031 | 3.05 pp -- **clears it** |
| cc_rec | +3.10 pp | 6/6 | 0.031 | 2.52 pp -- clears it |
| **cc_prec** | **-3.31 pp** | **0/6** | 0.031 | 3.50 pp |
| f1_macro | +1.09 pp | 5/6 | 0.219 | 1.70 pp -- does not clear |

**So the paper's claim reproduces.** But precision and recall move in OPPOSITE
directions, and with exactly K predictions emitted that is arithmetically
impossible -- `prec = TP/K` and `rec = TP/n_pos` are both monotone in TP, so
they must move together. The arms are not at the same budget.

🛑 **They are not: `rec/prec = K/n_pos`, and the hinge arm emits 16.3% MORE
predictions, on 24 of 24 pairs.** House rule 5 says filling to the boundary is
FREE, so part of that +3.23 pp is the fill and not the hinge. This is the
flips-is-not-a-metric failure appearing inside the paper's own central ablation.

📐 **How much of it? There is an exact threshold.** At the SAME emitted count
both arms share the F1 denominator, so the hinge wins iff
`rec_h > rec_n + (K_h - K_n) * q`, where `q` is the precision of the items the
no-hinge arm did not emit. Solving for the break-even `q*`:

| backbone | cap | **q\*** | that arm's own precision |
|---|---|---|---|
| MobileNetV3 | L40_G40 | **0.099** | 0.938 |
| MobileNetV3 | L30_G30 | 0.457 | 0.961 |
| RegNetY400MF | L40_G40 | 0.441 | 0.885 |
| RegNetY400MF | L30_G30 | 0.733 | 0.934 |
| ViTB16 | L40_G40 | 0.762 | 0.930 |
| ViTB16 | L30_G30 | **0.986** | 0.929 |

**The verdict is CELL-DEPENDENT and the corpus cannot decide it.** On
MobileNetV3/L40_G40 the hinge survives equalization only if its next-ranked
items would be correct under 10% of the time, against an arm averaging 94% --
implausible, so that cell is very likely fill. On ViTB16/L30_G30 the marginal
items would have to exceed 98.6%, so that cell very likely is not. Pooled
`q* = 0.580` against an average precision of 0.929.

**And nothing in the corpus can measure `q`.** `cc_prec`, `cc_rec` and `cc_f1`
are the only constrained-class columns and none of them sees an item that was
not emitted. Settling it needs the RAW predictions, and the evidence tarball
holds `mcbar` + `multiclass` only -- not `paper_final`, not `g5_hinge_oct`.

🔴 **AND THE SAME TEST ON THE FULL LEAVE-ONE-OUT ABLATION IS THE REAL RESULT:
THE TWO "LOAD-BEARING" COMPONENTS ARE EXACTLY THE TWO THAT CHANGE THE EMITTED
COUNT.** `extra_robustness_corpus.csv`'s `B_loo_ablation` block carries
`cc_support`, so the emitted count `K = rec*support/prec` is EXACT rather than
proportional. Full `tralo` against each leave-one-out arm, 32 pairs each,
dermmnist + octmnist x 3 backbones x 2 caps x 4 seeds:

| ablation | d cc_f1 | **K_full / K_ablated** | d prec | d rec |
|---|---|---|---|---|
| **no_reset** | **+8.80 pp** | **1.440** | -0.018 | +0.071 |
| no_freeze | +0.13 pp | **1.002** | +0.001 | +0.001 |
| no_rho | +0.10 pp | **1.000** | +0.002 | +0.001 |
| plus_kl | -0.48 pp | **1.011** | -0.016 | -0.003 |

**The three components the paper reports as NEUTRAL leave the emitted count
untouched (1.000-1.011). The one it reports as load-bearing raises it 44%.**
Together with the hinge's 16%, both of the paper's two carried components move
the budget and none of the neutral ones does. That is exactly the pattern
"cc-F1 is measuring native satisfaction" predicts -- and the paper's own
appendix already says native satisfaction is a deployment property and **not a
headline**.

⚖️ **The reset comes off far better than the hinge, and the numbers say why.**
Its break-even `q*` is 0.742-0.942 against ablated-arm precisions of
0.870-0.943 -- so `q*` sits JUST BELOW the arm's average in 7 of 8 cells. Items
ranked below the cut are normally less precise than the average, so the reset's
advantage plausibly survives equalization. The hinge's `q*` ran 0.099-0.986 with
several cells needing implausibly low marginal precision. **Reset: probably
real. Hinge: cell-dependent, often not.** Neither is decidable from the corpus.

⏭️ **So there are exactly two honest resolutions**, and the hinge being deleted
from the codebase (2b) does not remove the need for one:
1. Restore the hinge, re-run the ablation under the current protocol, and score
   it with `full_panel`, whose `equalize` emits exactly K by construction so the
   fill cannot contribute. That is what the budget-equalized family is FOR.
2. Or state in the paper that the hinge ablation compares arms at different
   emitted counts, and that the constrained-class gain is therefore an upper
   bound.

Doing neither leaves the paper's central mechanistic claim resting on a
comparison its own house rules forbid.

⚠️ The live divergence is elsewhere and is already recorded in 1b: the paper's
mechanism includes the **undershoot hinge**, which this framework REJECTED and
DELETED. A paper whose central mechanistic finding names a component the
codebase no longer has is the thing to reconcile -- not the headline.

🟢 **AND THE CONSTRUCTIVE READING, WHICH IS THE IMPORTANT ONE: the mechanism has
never been tested, so it has not been refuted either.** Line up what the four
corpus datasets could have shown:

| dataset | local scope | could a per-group cap carry information? |
|---|---|---|
| tissuemnist | `synth_group` = `index % 3` | **no, by construction** |
| octmnist | `synth_group` = `index % 3` | **no, by construction** |
| aider | `synth_group` = `index % 3` | **no, by construction** |
| dermmnist | `loc_group` | NET +65, z=2.9 -- clears stage 1 and still nulls; test groups ARE training groups |

**Not one of them.** So three years of campaigns measured a constraint that had
nothing to constrain, and the honest summary of the corpus is not "the mechanism
failed" but "the mechanism was never given a chance to act". Those are different
conclusions and only the second is recoverable.

🎯 **That is exactly what makes `results/iwc1` the first real experiment rather
than the next one.** iwildcam is the first dataset that clears 2(n): NET +3131
at z=97.4, seven test cameras absent from training ENTIRELY, and 7 of 14
per-group ceilings at K=0 -- a zero ceiling binds regardless of sum slack, so
the LOCAL scope constrains the output at every cap level for the first time.
2(p) already frames iwc1 as the first test of the REPRESENTATION channel; this
section adds that it is also the first test of the LOCAL CONSTRAINT channel, for
the same reason.

⚠️ Which sharpens the stakes rather than softening them. iwc1 cannot produce a
significant positive at 2 cells (2(p)), so the best available outcome is a
bounded null on a dataset where the mechanism CAN act -- and that would be worth
more than every positive cell in the corpus, because it would be the first one
measured where the question is answerable.

🧭 **AND IT DOES NOT SCALE WITH CONSTRAINT PRESSURE, which is the second
control.** If the +1.7 pp came from the count constraint doing work, a TIGHTER
cap -- more binding, more pressure -- should buy more of it. Regressing the
per-cell `tralo - heuristic` delta on the local cap percentage, warm-up 50:

| dataset | cells | cap levels | slope (pp per cap-point) | p |
|---|---|---|---|---|
| dermmnist | 88 | 9 | **+0.023** | 0.010 |
| tissuemnist | 91 | 9 | +0.018 | 0.198 |
| octmnist | 29 | 9 | +0.014 | 0.328 |
| aider | 28 | 5 | +0.001 | 0.875 |

**All four slopes are >= 0 -- the effect is flat or grows as the cap LOOSENS**,
which is backwards for a constraint mechanism and doubly so because the clipper
should be hurt MORE at tight caps (it has to delete more predictions), which
would push the delta the other way.

⚠️ Read it as a null with a consistent sign, not as a positive result: r is
0.03-0.27 and only dermmnist is individually significant. The claim is "cap
tightness does not drive the effect", 4 of 4 datasets agreeing, not "loose caps
are better". Per rule 4 the slope is computed WITHIN each dataset; the pooled
version (+0.017, p=0.021) is reported only to show it is not hiding a
cancellation.

**So two independent controls point the same way.** `danits_lp` says the effect
needs a training phase; the cap sweep says it does not need the cap to bind.
What the corpus CANNOT do is separate "extra epochs" from "extra epochs with
some auxiliary objective", because it holds no trained arm at lambda=0. That is
exactly the gap `tralo_null` exists to fill, and why `gen_campaign` refuses a
campaign that schedules a trained arm without it.

🔧 **An instrument fact worth keeping.** The seed sd of a dual-vs-dual contrast
is **0.48-0.63 pp**, a THIRD of dual-vs-clipper's 1.47 pp: two trained methods
share most of their variance and the paired difference cancels it. So a
method-vs-method question is roughly 3x cheaper in seeds than the same question
asked against the clipper -- but it answers a different one, and the
clipper is the bar that has to be cleared.

⚠️ **On cc-F1 the method-specific part vanishes entirely.** `tralo - fioretto_ldf`
is +0.54 pp at **p=0.073** and `tralo - tralo_bounded` +0.51 pp at **p=0.072** --
neither callable. Only `tralo - hounie_rcl` (+2.06 pp) survives. TraLO is
indistinguishable from Fioretto and from its own bounded variant on the
constrained-class metric.

1. **Warm-up 1 over warm-up 50.** Not because it wins, but because warm-up 50 makes every method
   identical -- CE is saturated, so the constraint phase is ~30 unit-norm steps on a frozen
   representation and all methods land within 0.1 pp.

   ⚠️ **QUALIFIED 2026-08-23 -- warm-up 1 buys HEADROOM, not method
   discrimination, and the stated rationale conflates them.** The corpus carries
   the whole warm-up axis, and the two contrasts behave completely differently:

   | warm-up | tralo - heuristic (compute + method) | tralo - fioretto_ldf (method only) |
   |---|---|---|
   | 1 | **+15.20 pp** (10 cells) | **+0.21 pp** (10 cells, 70%) |
   | 2 / 3 / 4 / 5 | +4.59 / +4.74 / +2.19 / +0.59 | +0.27 pp at 3 |
   | 50 | +1.85 pp (236 cells) | **+0.15 pp** (262 cells, 65%) |

   **The vs-clipper gap moves 8x across the regime. The method-vs-method gap
   does not move at all** -- 0.15 to 0.27 pp everywhere, and at warm-up 1 it is
   +0.21 +- 0.15 pp (sd 0.47 over 10 cells), statistically indistinguishable
   from the warm-up-50 value and from zero.

   So "warm-up 50 makes every method identical" is true, and **so is warm-up 1**.
   What warm-up 1 actually provides is a large, readable gap against the
   POST-HOC baseline -- headroom for the constraint phase to act in -- and Rule 1
   stands on that. It does NOT make TraLO easier to tell from Fioretto, and a
   campaign run at warm-up 1 to separate two duals is spending the regime on the
   wrong question.

   🛑 Two caveats, both real: **10 cells against 262**, and the warm-up-1 rows
   carry the LR trap of 1b. Treat the direction as established and the magnitude
   as provisional. It is recorded because the natural misreading -- "methods
   separate at warm-up 1" -- is the one that would size a campaign wrongly.
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

**STATE, 2026-08-22 -- read this before the 300 lines below.**

| | what | status |
|---|---|---|
| **built, not run** | `tralo_margin` + `tralo_st` (1b) -- the count's gradient on the decision boundary, decomposed from the count's value | `docs/launch_margin1.sh`. ⚠️ **"all gates green" was stale**: the script targeted `dermmnist`, a REMOVED dataset that `gen_campaign` now refuses outright, so it could not have run. Retargeted 2026-08-23 to iwildcam with capped classes 2/7, pointed at the real campaign checkout, and given a refuse-if-a-dispatcher-is-running guard. Re-run its gates before trusting it again |
| ⛔ **RUN AND REJECTED 2026-08-22** | 1c -- optimise the metric at the budget **with LABELS**, via a jointly-trained SELECTION head | `results/selectrun`, 32 runs: **-22 items vs `clip`, 0 of 2 cells on every metric**, 2 of 8 runs collapsed on the final epoch, and `select_null` TIES `clip` so the selective term owns the loss. **Do not re-run at any `eta`, `tau` or `cov_weight`.** Section (12); code kept at `src/methodologies/select/train.py` so the campaign stays readable |
| 🔴 **RUN 2026-08-23, NEGATIVE and POWERED** | the REPRESENTATION channel (2(p)) -- the last mechanism the structural argument left open | `results/iwc1`, 72 runs: `tralo` vs its own lambda=0 twin is **AP -0.0306 (detectable 0.0217) and AUROC -0.0110 (detectable 0.0089), both POWERED, 0 of 2 cells**. Losses ordered by constraint pressure across four dual families. Section (p-post). ⚠️ 2 cells => NOT significance-testable in either direction |
| **running** | the same test on the a-priori headline backbone | `results/iwc2`, ViTB16, launched 2026-08-23 on dsisco01 GPU 3 |
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
the operating point, and that needs LABEL information -- which no count penalty has.
⛔ **The old sentence here read "if 1b ties, go to 1c". 1c HAS BEEN RUN and it is
rejected** (section 12), so a tie on 1b no longer has a labelled successor waiting: the
remaining move is path 2, the regime change.

⛔ **THE COUNTEREXAMPLE DID NOT REPRODUCE HERE, AND THAT IS A MEASUREMENT.** SelectiveNet
(ICML 2019) beats "a threshold over the prediction confidence of a pre-trained network"
-- our `clip`, exactly -- in the analogous coverage-constrained setting, by training a
selection head jointly so the network is "optimized over the covered domain". Post-hoc is
optimal GIVEN the probabilities; training can change which probabilities you get, and that
mechanism was available to 1c and to no count-based arm. **We built it and it lost 22
items.** The two standing caveats on that non-reproduction are stated where they belong,
in 1c below: the dose is 23x outside the published regime (tau ~ 0.03 against 0.70-1.00),
and SelectiveNet's own advantage is contested -- by Feng et al. (arXiv:2206.09034) on the
mechanism, and by Jaeger et al. (ICLR 2023) on how strong a plain softmax baseline is;
neither is a refutation of SelectiveNet and 1c states exactly what each one does say.
⇒ **cite this as a measured negative on OUR setting, never as a refutation of
SelectiveNet.**

Given section 0, these were the only things that could still win. **Path 1c is now closed
by measurement**, so what remains is 1b (built, not run, and pre-registered to tie) and
path 2, which changes the problem rather than the method. Ordered by how much of the
pessimism above they escape: 1c escaped all of it on paper and none of it in practice;
**1b escapes none of it** and is being run anyway because the analysis could be wrong.

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

   ⚠️ **THE K == 0 TRAP, CORRECTED 2026-08-22 -- it is HALF of what this section
   used to claim.** A group holding no true instance of the capped class gets
   `K == 0` legitimately, and on the soft value that constraint can never be
   satisfied: `sum_i p_ic` is strictly positive for any softmax even when the class
   is predicted for nobody in the group, so `relu(count - 0)` stays positive for the
   whole run. That much is true and `straight_through: true` removes it.

   🛑 **What is NOT true is the consequence this file and the loss docstring both
   asserted -- that it "holds the ratchet gate open for every other constraint".**
   Read against `src/methodologies/tralo/train.py`: `snapshot_local_satisfied`,
   `snapshot_global_satisfied` and `ratchet_gate` are all computed from
   `total_local_hard` / `total_global_hard`, which CAN be exactly zero. A K == 0
   group therefore neither blocks the freeze nor holds the ratchet open. The two
   soft-count flags that did encode the wrong notion (`local_constraints_satisfied`,
   `global_constraints_satisfied`) were assigned on every forward and **read nowhere
   in the repo** -- inert flags five and six -- and are now deleted, with an AST test
   preventing their return. **This matters because the wrong version nearly got
   `results/iwc1` condemned as corrupt while it was healthy.**

   ⚖️ **So `straight_through` is a KNOB here, not a fix.** What K == 0 really does is
   contribute a permanent, non-vanishing gradient pushing `p_ic` down in that group
   -- and for a group with genuinely no instances of the class **that direction is
   correct**. `straight_through: true` makes the term satisfiable on the hard count
   and thereby switches that pressure OFF. On iwildcam, where **seven of the fourteen
   per-group ceilings are K == 0**, switching it off discards pressure that is
   pointing the right way, so the protocol default (`false`, which is what every
   `iwc1` arm runs) is the defensible setting rather than an oversight. Decide it on
   the measurement, not on the word "trap".

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
   these levels. ⛔ **TRUE ON dermmnist ONLY -- see the `binds` table just below, where it
   is FALSE on both other datasets.**

   🛑🛑 **THE CAP DOES NOT BIND IN EVERY SEED, AND dermmnist IS THE ONLY DATASET WHERE IT
   DOES.** Measured 2026-08-21 across all five stored MobileNetV3 campaigns
   (`python -m scripts.headroom <root> --control clip`, `binds` column):

       dataset        cell                  binds
       dermmnist      every cell             4/4     <- clean
       octmnist       L50_G50 class 2        3/4
       octmnist       L70_G70 class 2        2/4
       tissuemnist    L30_G30 class 1        2/4
       tissuemnist    L50_G50 class 1        1/4     <- 76 / 51 / 34 / 18 vs K=56

   The penalty is `relu(soft - K)`, so **a seed already under budget gets an identically
   zero constraint gradient** -- the treated arm IS its own null in that seed, and
   averaging over it dilutes a real effect toward zero and reports a tie. ⇒ **check
   `binds` before scoring a cell, and never average over a cell that does not bind 4/4.**

   ⚠️ **The mean excess HIDES this and I nearly filed it as fine.** tissuemnist L30 class 1
   averages a healthy-looking **+10.8 items over budget while binding in only half its
   seeds**; the L50 sibling averages **−11.2**, which reads as "never binds" when it in
   fact binds in one seed. Only the per-seed count shows either. ⚠️ This also **corrects
   the claim two paragraphs up** that the "seed already satisfied, takes no step" failure
   is unlikely at these levels -- that was derived from dermmnist alone, and it is FALSE
   on the other two datasets.

   🔑 **AND HEADROOM IS RANKING QUALITY, NOT ALLOCATOR SLACK.** `equalize` already takes
   the top K by probability, which is optimal for expected TP *given those probabilities*,
   so the allocator gives up ~nothing and essentially the whole gap is the RANKING. That
   is what makes headroom the right target for a method that changes TRAINING -- and it
   means **a large headroom can just mean a weak classifier**: tissuemnist shows 0.3626 at
   L50 against dermmnist's 0.0746, and that is the model being bad at tissuemnist, not
   opportunity. ⇒ **never pick a dataset because its headroom is large.**

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

1c. **Optimise the metric at the budget, using LABELS** -- built 2026-08-21,
   ⛔⛔ **RUN AND REJECTED 2026-08-22. THIS ENTRY IS A CLOSED PATH, NOT A PROPOSAL.**

   🛑 **Read section (12) first.** `results/selectrun`, 32 runs: **AP -0.1096, ccF1
   -0.0804 = -22 items, macroF1 -0.0873, 0 of 2 cells on every metric**, and 2 of 8 runs
   collapsed on their final epoch (the pipeline keeps the last epoch, so those collapses
   are the scored models). `select_null` -- same warm-up, same 29 epochs, same allocator,
   `select_eta: 0` -- **TIES `clip`**, which is what makes the loss attributable to the
   selective term and not to the regime. **Do not re-run it at any `eta`, `tau` or
   `cov_weight`.**

   Everything below this point is the DESIGN RATIONALE, kept because it states what was
   pre-registered and what the two standing caveats on the result are. **None of it is an
   argument to resume the path.** The mechanism is understood: the coverage term trains
   the network to *abstain*, which is a different objective from ranking the capped class,
   and the allocator is then handed a worse ranking to threshold.

   `src/methodologies/select/train.py`; arms `select` / `select_null`; `blocks: [chunked,
   select]` with `constraint_step: false`, since this arm takes no constraint step at all
   and inheriting those ten keys would emit eight nothing reads.

   ⚠️ **THE DOSE IS 23x OUTSIDE THE REGIME THE METHOD WAS PUBLISHED IN, AND THAT IS NOT
   FIXABLE BY TUNING.** SelectiveNet's coverage targets are 0.70-1.00. Ours is
   `tau_c = K_c / n_test` ~ 0.03, because a BUDGET is not a coverage rate -- so a batch of
   64 carries ~2 covered items. Both of the published estimators degrade there: dividing
   the selective risk by `g.sum()` is a ratio estimator whose denominator is a small
   random variable, and `(g.mean() - tau)^2` has expectation `Var(cov) + bias^2` with the
   variance dominating. Both are now stabilised (risk normalised by the EXPECTED covered
   mass; the coverage term takes its value from a running estimate and its gradient from
   the batch), and the arm PRINTS its covered-items-per-batch before the first step and
   warns when it is small. ⇒ **a null from this arm must be read as "underpowered" before
   it is read as "the method does not work"** -- `cut_temp: 0.02` already produced exactly
   that silent null with 1.4-1.9 items inside its window.

   ⚠️ **AND SELECTIVENET'S OWN ADVANTAGE IS CONTESTED -- but check WHAT each paper
   actually says, because an earlier draft of this passage overstated both.** Verified
   against the abstracts 2026-08-22:

   * **Jaeger, Lueth, Klein & Bungert, "A Call to Reflect on Evaluation Practices for
     Failure Detection in Image Classification" (ICLR 2023, oral; arXiv:2211.15259).**
     Their benchmark spans confidence-scoring methods across research silos, and the
     load-bearing sentence is *"The revelation of a simple softmax response baseline as
     the overall best performing method underlines the drastic shortcomings of current
     evaluation."* 🛑 That is a statement about CONFIDENCE SCORING broadly. **The abstract
     does not establish that SelectiveNet is in the benchmark at all**, and it states no
     coverage range -- so do NOT cite this paper for "beats learned selection heads at
     matched coverage". It supports the weaker and still useful claim that a plain softmax
     threshold is a stronger baseline than the literature treats it as.
   * **Feng, Ahmed, Hajimirsadeghi & Abdi, "Towards Better Selective Classification"
     (arXiv:2206.09034).** This is the one that speaks to the mechanism, and its claim is
     STRONGER for us than what was written here: *"the superior performance of
     state-of-the-art methods is owed to training a more generalizable classifier rather
     than their proposed selection mechanisms."* ⚠️ The abstract gives no accuracy figures,
     so quote the mechanism claim and never a percentage-point number from it.

   Between them that is the same suspicion as our `clip` baseline, and it is the reason the
   falsification was pre-registered rather than optional: **if `select` beats `clip`,
   apply the identical head to `clip` and re-run.** If the head helps both equally, the
   result is the head, not the joint training. ✅ **That branch never had to be taken:
   `select` did not beat `clip`, it lost 22 items to it.** The pre-registration is kept
   because it is what made the negative readable -- and Feng et al.'s reading, that the
   gain is the classifier and not the selector, is the one our own data now supports.

   **This was where 2026-08-21's measurements pointed** -- the only proposal that
   escaped the trap 1b is stuck in, built on three facts each established above. The
   facts still hold; the arm built on them does not. Kept so the next proposal is
   checked against the same three rather than re-deriving them:

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
   which is why 1c was the proposal that fit it and why every count penalty does not.
   ⚠️ Being the right SHAPE of proposal did not make it work: 1c was measured and lost.

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
   penalty onto a warm-started CE model. ⇒ this made 1c **plausible**, not proven -- and
   the measurement came back negative. ⚠️ **The last difference is the one that was never
   removed**: SelectiveNet trains the head jointly FROM SCRATCH, `select` bolts it onto a
   1-epoch warm-up like every other arm here, so the non-reproduction is a result about
   OUR setting and is not a refutation of theirs.


   ⚠️ **Novelty is the pairing, not the loss.** Top-K / precision@k surrogates and
   learning-to-rank losses are well established; this is not a new loss family. The
   claim would be "a transductive budget tells you the operating point, and optimising
   there beats post-hoc thresholding of a CE model" -- which must be checked against
   label-shift, prior-correction and top-K optimisation literature BEFORE building it.

   🎯 **THE MINIMAL FORM -- NOT WHAT WAS REJECTED, AND NOT THEREBY OPEN.** Be exact
   about the scope of section (12): what lost 22 items is the SELECTION HEAD, whose
   coverage term trains the network to abstain. The re-weighting below has no coverage
   term and no abstention, so section (12)'s mechanism does not transfer to it -- but it
   was never built, it shares 1c's core premise (fit the model to the sub-population it
   will predict on), and that premise now has one measured negative against it.
   ⛔ **So it is not a next step and must not be read as one.** It is recorded so that a
   later proposal is not re-derived from scratch, and it would need its own case, its own
   pre-registered bar and Roei's call before any GPU goes into it.

   SelectiveNet trains a selection head jointly; that is a large change. The smallest
   thing with the same mechanism -- fit the model to the sub-population it will actually
   predict on -- is to **re-weight the CE loss on the TRAINING set toward the operating
   point**:

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
   baseline and the training and test operating points differ by sampling noise. ⛔ The
   line here used to read "build it only after 1b reports"; with 1c measured and rejected,
   building it is not scheduled at all.

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
data/              iwildcam -- nothing else (the other three removed 2026-08-22, section 2(n))
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
scripts/dataset_screen.py     CAN a count cap carry information here? labels + metadata only
scripts/frozen_head_probe.py  refit only a linear head on frozen features, verdicts in items
scripts/graph_probe.py        diffuse scores over a kNN graph of the stored embeddings
scripts/scope_probe.py        local-vs-global SCOPE at a fixed total budget
scripts/straddle_probe.py     how much oracle headroom a step OUR size can reach; --self-test
src/               the pipeline: losses, methodologies, models, pipeline, training, utils
tests/             360 tests, ~105 s, no dataset required
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

⚠️ **`src/methodologies/` holds TEN packages, and the tenth is not a tenth
baseline.** `select` is path 1c, it is **rejected** (section 12), it appears in no
`.tex` file, and `gen_campaign` subtracts it from `--arms all`. So "nine
methodologies, all claimed in the paper" stays exactly true -- but anyone counting
directories finds ten, and this is the line that says why.

## 6. Evidence appendix

The full run-by-run record, with numbers, p-values and cell counts, is preserved at
`docs/archive/REJECTED_full_2026-08-18.md`. It is history, not instructions.
