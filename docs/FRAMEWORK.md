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

🛑🛑 **READ THIS BEFORE ANY DUAL-vs-DUAL SENTENCE IN THIS FILE (2026-09-04).**
`fioretto` and `hounie` are **DEAD ARMS** in `dom1`, `dom1b` and `equaldose1` --
28.00 attempted constraint steps per run against 29.00 -- and `tralo_lam0` is
one in `equaldose1` too (2(z40)). They are the only campaigns on the recipe that
carry rival duals, so **every "TraLO beats the duals" claim in this repository
rests on at least one arm that may not be contrasted with anything.** Once they
are dropped the surviving field is `tralo` vs `alm`, and as deployed **TraLO is
#1 in 0 of 15 cells** (2(z43)). The dose objection that 2(z19) recorded as
CLOSED is **REOPENED**: the control built to close it, `tralo_lam0`, is itself
a dead arm.

🟢🟢 **AND THE HEADLINE LEDGER IS UNTOUCHED -- SAY THIS AS LOUDLY AS THE
LOSSES.** `scripts/paper_rows.CONTRASTS` is exactly `vs_clip`, `vs_null`
(family-resolved) and `vs_reseed`, and **none of them touches a dead arm**. So
`tralo` vs `clip` **4/4 units p=0.0625**, vs its own `_null` **4/4**, vs
`tralo_reseed` **3/4**, and the task-restricted **3/3 p=0.125** all stand
exactly as written. **0 of 15 cells are lost**; 144 of 792 runs (18.2%) are
touched. **And the paper of record is entirely unaffected** -- it is built on a
disjoint MedMNIST corpus with zero iwildcam rows and no quarantined campaign
(verified 2026-09-04, grep count 0). See 2(z43).


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

⛔ **RETRACTION, 2026-08-25, of an edit made to this section EARLIER THE SAME
DAY.** That edit claimed the coin null has a mechanism -- that under a shared
Adam the treatment and its control deliver the same vector
(`cos = 0.9937`), so the campaign "could not have resolved the constraint
direction whatever it contained". **That inference is withdrawn.** It was wrong
in two independent ways, and the disproof was already sitting four paragraphs
below it in this same section:

1. **The data refute it.** At `L50_G30`, in the same cell with the same seeds
   and the same control, `linear` scores **+0.0078 (4/4 seeds, sd 0.0017)** and
   the coin **-0.0130 (1/4)** -- *"the distributions do not overlap"*. If the
   two arms delivered indistinguishable steps that could not happen.
2. **The inference does not follow even from its own number.** A per-step cosine
   of 0.994 is not outcome-indistinguishability: a 0.6% consistent directional
   difference **compounds over 29 steps**, and the effects at issue here are
   0.008-0.013 `capF1`. I converted a per-step geometry into a claim about a
   29-step trajectory without doing the compounding.

**And the premise was never checked.** The measurement assumes
`constraint_step_rule: shared`. That is the protocol default
(`configs/protocol.yml:82`, no block overrides it), but **I did not verify what
these diagnostic campaigns ran**, and the contemporaneous `results/dosefix`
(section 2(b-pre)) used `constraint_step_rule: sgd` -- under which the step is
`p.add_(p.grad, alpha=-lr)`, the direction is delivered at `cos = 1.0`, and none
of the momentum argument applies. `evidence/provenance_2026-08-18.tar.gz` holds
**zero** `constraint_random_direction` configs, so the campaign postdates the
archive and its step rule cannot be recovered from it.

✅ **WHAT SURVIVES, because it was measured rather than inferred:** under
`constraint_step_rule: shared`, the per-step update is `~92.6%` stale CE
momentum and `cos(real step, coin step) = 0.9937`
(`python -m scripts.ortho_survival`). That is a true statement about the
default step rule and it is why `tralo_ortho`'s guarantee dies (2(t)) --
`tralo_ortho` provably runs under `shared`. It is **not** a re-reading of the
campaign below, and the conclusion that section reaches from its own data is
unaffected.

⛔⛔ **A SECOND RETRACTION IN THIS SECTION, 2026-08-25, OF AN EDIT MADE TO IT
EARLIER THE SAME DAY -- AND IT IS THE SAME MISTAKE AS THE FIRST.**

That edit said the compounding had finally been done, and claimed: Adam
accumulates a consistent count-gradient difference as `(1 - b1^k)`, so the
recorded "~7.4% channel" is a STEP-1 number that "decays away", reaching 0.953
by k=29; the two arms' updates therefore open to 4.7-16.5 degrees with 8-31% of
the trajectory separating them. **All of those numbers are withdrawn.**

**The premise was never checked, again.** `(1 - b1^k)` is the accumulation for
**consecutive** steps. The constraint steps are not consecutive:
`src/methodologies/tralo/train.py:192-212` runs the entire CE batch loop with one
`optimizer.step()` per batch and calls `finish_constraint_step` **once per
epoch** at line 404. So about **126 CE steps sit between two constraint steps**,
and `b1^126 = 1.7e-6`. With `c` CE steps between, the difference present at a
constraint step is

    (1 - b1) / (1 - b1^(c+1))     = 1.000 at c=0,  0.1457 at c=10,  **0.1000 at c=126**

i.e. at the pipeline's real spacing it is **exactly the single-step value,
forever**. The momentum does not compound it at all. *The file was right before
I corrected it.*

🔑 **WHAT DOES COMPOUND IS THE WEIGHT TRAJECTORY, and it is small.** Each
constraint step displaces `w` a little differently and those displacements add
even though the momentum resets between them -- which is what this section meant
by "compounds over 29 steps" in the first place. Measured with the interleaving
modelled (`ortho_survival --compounding`), cumulative trajectories at a
realistic input angle:

| CE-direction model | after 1 step | after 29 | separation / length |
|---|---|---|---|
| fresh each step | 0.44 deg | **2.31 deg** | 0.040 |
| half-correlated | 0.08 deg | **0.08 deg** | 0.0013 |
| highly correlated | 0.05 deg | 0.04 deg | 0.0007 |
| *(consecutive -- the retracted model)* | *2.28 deg* | *25.39 deg* | *0.441* |

So compounding is real at ~5x under an uncorrelated-CE assumption and **absent
under a correlated one**, and consecutive-vs-interleaved is a ~10x error in the
end separation. The spread across the CE-correlation assumption is **31x**, not
the 3.8x the retracted edit reported.

✅ **AND THAT ASSUMPTION IS NOW A MEASUREMENT** (`ortho_survival --compounding`,
which runs it). A real net under real `torch.optim.Adam` at the trainer's own
spacing -- `batch_size: 64` from `configs/protocol.yml`, and 8064/64 = **126
steps per epoch**, exactly what runs between two constraint steps -- gives a
lag-1 cosine between consecutive CE minibatch gradients of

| epoch | lag-1 cos |
|---|---|
| **1 (the warm-up-1 regime)** | **0.128** |
| 2 | 0.056 |
| 3 | 0.025 |

It **falls as the model fits**, so warm-up 1 is its high point, and the batch
size drives it (0.057 at 32, 0.128 at 64, 0.395 at 256, 0.580 at 512) -- which
is the tell that it is minibatch noise rather than curvature, and which doubles
as the probe's liveness control.

🛑 **AT THE MEASURED VALUE THERE IS ESSENTIALLY NO COMPOUNDING: 0.26 -> 0.30
degrees over 29 steps, 0.5% of the distance travelled.** The ~5x growth belongs
entirely to the `ce_rho = 0` assumption. So the per-step compression IS the
story -- which is what this project recorded before I "corrected" it, and the
second independent reason the retraction above was necessary.

⚠️ A synthetic MLP is not MobileNetV3 on iwildcam: take ~0.1 as an order of
magnitude with a mechanism attached, not as the campaign's number. And this is
still a POWER consideration and **not a predicted null** -- parameter separation
is not items, and 1b-pre(6) is the standing warning against exactly that leap.
Gated by `test_the_CE_autocorrelation_is_MEASURED_and_the_probe_responds_to_batch_size`.

✅ **ONE PART SURVIVES, because it is geometry and not dynamics: the input
angle is never 180 degrees.** `sum`'s per-item gradient is `p(1-p)` and
`uniform`'s is their mean; **both are elementwise non-negative**, so the angle
between them is bounded below 90 by construction -- **18.7 to 49.6 degrees**
measured over plausible `p` distributions, ~28 for a trained-like split. 180 was
the abstract extreme the probe is swept over and it was being quoted as if it
described `tralo` vs `tralo_uniform`. Both launch scripts keep that correction
and lose the rest.

🛑 **THE PREMISE IS NOW A GATE, NOT A SENTENCE.**
`test_the_constraint_step_is_NOT_inside_the_CE_batch_loop` reads
`tralo/train.py` by AST and fails if `finish_constraint_step` ever moves inside
the batch loop -- because that single edit would make the geometric accumulation
correct and reverse everything above. Shown to FAIL by making exactly that move.

🔑 **THE LESSON, which is the reason this retraction is kept in full.** A
mechanism that explains a result is not evidence for itself. This one was
derived, written into the operational document, and committed **before reading
the rest of the section it was reframing** -- which contained the disproof.
Section 3b already lists "chose the metric after seeing the numbers" as a
mistake pattern; this is its sibling: *chose the mechanism after seeing the
null, then stopped reading.*

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

**Receipt: `git show 61e34c0a^:scripts/check_lesion_leakage.py`** -- the script was DELETED by that commit ("purge: remove dermmnist, octmnist and tissuemnist from the runnable path") along with the dataset it checks, so it is no longer a runnable command. The receipt is still retrievable and still reproduces; it needs (server, or anywhere
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
`prep_octmnist.py` (deleted by `61e34c0a`) keeps the official test split whole.

#### Only dermmnist has real groups

A local cap is a *different* constraint from the global one only if the groups
differ in class composition. `synth_group` is built by round-robin over array
order (`prep_octmnist.py:72`, `np.arange(len(y)) % 3` -- retrieve with `git show 61e34c0a^:scripts/prep_octmnist.py`, the commit that purged it) or a random
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
| **our training slice** (`prep_octmnist.py`, deleted by `61e34c0a`) | **12,000** | **3,000 = 25.00%** |

The 8% is exact -- **about MedMNIST**. It is not true of our data.
`prep_octmnist.py:15` takes `N_PER_CLASS_TRAIN = 3000` stratified, which
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
  integer multiple of `2/(K+n)` -- **not** `1/(K+n)`, because `TP` is an integer so
  a delta of half an item cannot occur; a value that is not is an arithmetic bug. The -0.0149
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
  🛑 **PRICED 2026-08-25, and the price closes it ON iwildcam.** The receipt it was
  missing is not a null result -- it is the size of the prize. `headroom` on `results/iwc3`
  reports the gap from `clip` to a **PERFECT RANKING** as **0.0 to 1.0 items**, and
  **exactly 0.0 in four of the six (cap, class) combinations**: see 2(v). A supervised
  pairwise hinge is a ranking arm, so that is its ceiling, and iwc3's own paired seed sd
  is 2.11 items. It cannot be measured here at any affordable seed count. The family stays
  open as an IDEA and is unaffordable on the only dataset this project has.
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

### (y) THE REGIME STEP IS REAL AND ATTRIBUTABLE. THE GEOMETRIC EXPLANATION FOR IT IS NOT.

⚠️ **This section was written on 2026-08-30 claiming a MECHANISM and is
corrected the same day.** The claim was tested and mostly failed. What survives
is stated first; what does not follows it. Tool: `scripts/cut_gap.py`.

#### ✅ What is solid: the regime step, paired within the warm-up

The CNN warm-ups are **shared across campaigns** -- one warm-up per
(model, seed) spans `uniform1` (TIGHT), `loose1` and `dom1` (LOOSE). That is
usually a nuisance; here it is a gift, because it gives a **within-model**
tight-vs-loose contrast with no backbone, host or amp confound. Paired on the
**12 CNN warm-up models present in both regimes**:

| contrast | result |
|---|---|
| `tralo` delta LOOSE > TIGHT | **12/12, +6.24 items, sign p = 0.00049** |
| floor-corrected (`delta - floor`) | **12/12, +5.30 items, p = 0.00049** |
| **the FLOOR itself**, LOOSE > TIGHT | **5/12, +0.94 items, p = 0.774** |
| `tralo - tralo_uniform`, LOOSE > TIGHT | 11/12, +3.67 items, p = 0.0063 |

**The floor does not move with regime, so the step belongs to the constraint.**
p = 0.00049 is the exact sign floor at n = 12, i.e. the best this design can
report. This is the cleanest attributable result the project has.

⛔ But the step is a STEP, not a gradient. Within-warm-up dose-response over K
is +0.614 (22/24, p = 3.6e-5) across the full K = 74..411 range, and **+0.250
(10/16, p = 0.45) -- null -- for warm-ups living in only one regime**, i.e.
over a narrow K range. There is no dose axis here, only tight versus loose.

#### ⛔ What failed: the geometric explanation

The proposed account was that the penalty aims at the DECISION BOUNDARY while
the metric reads the CUT, and that `gap = hard_count - K` is therefore the
cause. The geometry itself is not in dispute -- at K/n = 0.20 the cut sits at
p = 0.9999 where `p(1-p)` = 0.0001, against 0.59-0.99 at K/n = 0.90. **The
causal reading is.**

1. **`gap`, `slope_K` and `K/n` are one variable in three costumes.** Within a
   warm-up model the hard count is constant across every cap tag (verified,
   40/40 groups), so `gap = hard - K` is an exact decreasing affine function of
   K: measured `rho(gap, K) = -1.0000`. `slope_K` follows K at mean rho +0.937.
   **They cannot be separated even in principle on this design.**
2. **The only prying-apart variation is between backbones at a fixed cap** --
   10 strata of n = 4, where a PERFECT ordering reaches at best p = 0.083.
   Stratified permutation over 20k draws: `gap~delta` p = 0.30,
   `slope_K~delta` p = 0.55. **Null.**
3. **`gap` and `slope_K` REVERSE SIGN when the cap is held fixed.**
   `rho(gap, delta)` is -0.291 pooled and **+0.590 within TIGHT (p = 0.002)**.
   Their pooled correlations are between-cap artefacts. Only `K/n` orders the
   effect robustly, and `K/n` is a restatement of the cap, not a mechanism.
4. **⛔ THE SHARP PREDICTION FAILED.** The account predicted `tralo_uniform`
   would order OPPOSITELY in `gap`. It does not: both arms' slopes have the
   SAME sign at every level, and uniform's one nominally significant result
   dies against its own floor (+0.451 p = 0.012 raw -> +0.126 p = 0.492
   floor-corrected).

⇒ **Cite the geometry as an unrefuted account, never as a measured cause.**
On these five campaigns it is observationally identical to "loose caps help,
tight caps hurt". Testing it requires varying `gap` at FIXED `K/n`, which needs
a dataset or backbone whose calibration differs far more than the 3-126 item
spread available here, or a deliberate miscalibration arm.

#### 🛑 And the absolute loose-cap claim does NOT survive honest units

At the cell level `tralo - tralo_null` at loose caps reads 15/20 (p = 0.041)
and beats its floor 14/20. **At the 16 distinct warm-up models it is 11/16
(p = 0.21), and beats the floor 11/16 (p = 0.21).** The cell-level significance
is manufactured by counting cap levels as independent replicates of one model.

| loose caps, 16 warm-up units | result |
|---|---|
| `tralo - null` > 0 | 11/16, p = 0.2101 |
| `tralo` beats the reseed floor | 11/16, p = 0.2101 |
| `tralo` > `tralo_uniform` | 12/16, p = 0.0768 |
| **d macroF1 > 0** | **5/16, p = 0.2101, mean -0.0009** |
| **d uncapped F1 > 0** | **5/16, p = 0.2101, mean -0.0030** |

⇒ **`tralo` is not established as beating its own RNG floor at loose caps**,
and macroF1 and uncapped F1 are NEGATIVE in 11 of 16 cells. The relative
statement (loose minus tight, paired on the warm-up) survives; the absolute one
does not.

#### ⚠️ TWO STRUCTURAL FACTS THAT CHANGE EVERY n

* **`dom1`'s L80_G95 and L90_G95 cells are BYTE-IDENTICAL to `loose1`'s** --
  80/80 `final_predictions.csv` across 5 arms x 2 models x 2 caps x 4 seeds.
  `dom1` contributes only **L95_G80**. Any count that treats `dom1` and
  `loose1` as separate evidence double-counts 8 cells.
* **Only 20 distinct warm-up models exist** across `uniform1`, `vitu1`,
  `loose1`, `loosevit1` and `dom1` -- 16 in TIGHT, 16 in LOOSE, **12 in both**.
  Only ViTB16 has separate warm-ups per regime.
* 🐛 `cut_gap.summarise` originally grouped by (campaign, cap, class),
  **pooling across backbones** in violation of the seed-only averaging rule.
  Fixed 2026-08-30 to key on the model; the geometry reproduces disaggregated,
  but the first table printed backbone-averages under per-cap labels.

### (z12) 🔴🔴🔴 THE SHIPPED COUNT FUNCTION PUTS **0.00%** OF ITS
GRADIENT AT THE CUT -- the mechanism behind z11, and the fix it forces

Measured 2026-08-31 on REAL stored `test_embeddings.npz`, `iwc1`, 24 (run,
class) pairs. `scripts/step_direction_probe.py`, `--self-test` gated in BOTH
directions (it reports 0.701 on a steerable geometry, so a collinearity here is
a measurement).

**(a) THE COUNT-FUNCTION FAMILY IS THREE DIRECTIONS, NOT ONE.** The gradient of
any `S_c = sum_i phi(p_ic)` w.r.t. the head weights is `sum_i g_i f_i`, a
g-weighted mean of the test features. Cosines between those directions, on real
features:

| cluster | members | mutual cosine |
|---|---|---|
| A flat / decreasing | `uniform`, `1-p` | **0.986** |
| B peaked at p=0.5 | **`sum` p(1-p)**, `margin` | **0.989** |
| C increasing in p | `p`, `linear` z | **0.989** |

Between clusters 0.58-0.87. ⚠️ **A GAUSSIAN TOY SAYS 1.0000 FOR ALL OF THEM
AND IS WRONG** -- real post-ReLU features are non-negative and anisotropic and
give `uniform` vs `sum` = **0.7479**, not 1.0. Never price this on synthetic
features.

✅ **CONSEQUENCE, PRE-REGISTERED**: `tralo_margin` sits at cosine **0.989**
from `tralo`, so running it will mostly REPRODUCE `tralo`. It is the arm held
in reserve and it is now predicted to be a near-duplicate. Do not spend a
campaign on it before the cut window below.

**(b) AND NONE OF THE THREE AIMS AT THE CUT.** Fraction of total gradient mass
on the 40 items straddling rank K -- the only items whose movement can change
the emitted top-K set:

| cap | cls | K | p at the cut | `uniform` | **`sum` (SHIPPED)** | `p` | cut-window |
|---|---|---|---|---|---|---|---|
| L20_G50 | 2 | 74 | 1.00000 | 0.0136 | **0.0000** | 0.148 | 0.830 |
| L20_G50 | 7 | 92 | 1.00000 | 0.0136 | **0.0000** | 0.072 | 0.111 |
| L30_G50 | 2 | 111 | 0.99984 | 0.0136 | **0.0005** | 0.168 | 0.700 |
| L30_G50 | 7 | 137 | 1.00000 | 0.0136 | **0.0000** | 0.068 | 0.158 |

⚠️ **REPRODUCIBILITY.** The `cut-window` column was first produced by an
ad-hoc inline script, i.e. by tooling that was not committed -- the exact defect
this project's own naming gate exists to catch. `scripts/step_direction_probe.py`
now carries the cut-centred weighting (`--n-items`), so every column above
regenerates from committed code. Fixed 2026-09-01. The per-cell table is a
deduped L20/L30 subset; the committed probe's POOLED figure over all 24 pairs
is the reproducible receipt, and it is sharper still:

| weighting | mass at the cut | min | max |
|---|---|---|---|
| **`cut_window`** | **0.3486** | 0.0950 | 0.7786 |
| `p` | 0.1039 | 0.0637 | 0.1976 |
| `linear_z` | 0.0816 | 0.0560 | 0.1284 |
| `uniform` | 0.0136 | 0.0136 | 0.0136 |
| **`sum` (SHIPPED)** | **0.0001** | 0.0000 | 0.0034 |
| `margin_sech2` (the BOUNDARY window) | **0.0000** | 0.0000 | 0.0000 |
| `one_minus_p` | 0.0000 | 0.0000 | 0.0000 |

🛑 **`margin_sech2` IS EXACTLY 0.0000.** The boundary window -- the count
`tralo_margin` would run -- puts LITERALLY NONE of its gradient at the cut, and
`margin2` is 432 runs staged against it. That is the conflation in CLAUDE.md
rule 3 costing a campaign, and it is the strongest single argument for running
the cut window first.

🔴 **`p(1-p)` IS MAXIMAL AT p=0.5 AND VANISHING AT p=1, AND THE TIGHT-CAP
CUT SITS AT p=0.99984-1.00000.** So the shipped penalty spends its entire
budget deep inside the class where movement cannot change the emitted set, and
**nothing** where the metric reads. Every item it moves is collateral. That is
z11 explained: the reordering measures at the RNG floor because it is, with
respect to the metric, arbitrary.

✅ **AND IT DERIVES THE REGIME REVERSAL WITH NO NEW ASSUMPTION.** At loose caps
the cut falls to p=0.59-0.99, where `p(1-p)` finally has mass -- which is
exactly where `sum` wins (+0.0253 AP at L80/L90) and `uniform` is the reverse.
The reversal was an observation; it is now a consequence.

⚠️ **THIS IS RULE 3 OF `CLAUDE.md` BITING FOR REAL.** `margin_window` targets
the DECISION BOUNDARY (p=0.5). The metric reads at RANK K. At tight caps those
are the same conflation the rule warns about, and here they are ~300 items and
~10 logits apart. A window at the boundary is not a window at the cut.

⇒ **THE FIX: WINDOW THE CUT, NOT THE BOUNDARY.** Weight
`sech^2((z_i - z_(K)) / T)` centred on the K-th RANKED logit. 🛑 **AND SET T
IN ITEMS, NOT IN LOGITS** -- at a fixed T=1.0 the cut window already collapses
from 0.830 to 0.111 between class 2 and class 7 of the SAME run, purely because
the logit spread near the cut differs. ⚠️ **An earlier version of this
line named `_window_from_items` as the helper to reuse. THAT FUNCTION DOES NOT
EXIST** -- the real one is `window_temp`, and it is NOT directly reusable: it
measures distance from ZERO (the boundary), not from `tau` (the cut).
`cut_window_count` therefore inlines its own sort. Corrected 2026-09-01.

⚠️ **WHAT IS NOT YET SHOWN.** That aiming at the cut HELPS. It is necessary,
not sufficient: `ceiling_screen` bounds the whole prize at 1.9-9.9 items and
`headroom` reads 0.0-1.0 on iwildcam's tight cells, so a correctly-aimed
gradient can still find nothing to win there. This closes the question "why has
nothing worked", and opens "does aiming fix it" -- do not report the second as
answered by the first.

### (z13) 🔴🔴 THE CONSTRAINT IS NOT WEAK AT TIGHT CAPS. IT IS STRONG
AND AIMED AT THE WRONG POINT -- and my own prediction had the sign backwards

Measured 2026-09-01 from `final_predictions_raw.csv` (the mandated source --
the logged `Hard_Class*` disagrees with the model's predictions on every
trained arm, 3(0c)). Each arm differenced against its OWN `tralo_null` twin,
then stated as a RATIO to the RNG floor, which normalises WITHIN campaign.

| campaign | cls | pairs | \|tralo-null\| | \|reseed-null\| | ratio |
|---|---|---|---|---|---|
| **TIGHT** `iwc1` K/n=20-30% | 2 | 8 | 76.4 | 19.0 | **4.02x** |
| **TIGHT** `iwc1` K/n=20-30% | 7 | 8 | 67.6 | 23.0 | **2.94x** |
| **LOOSE** `loose1` K/n=80-90% | 2 | 24 | 44.9 | 47.5 | 0.94x |
| **LOOSE** `loose1` K/n=80-90% | 7 | 24 | 57.5 | 50.2 | 1.14x |

⛔ **I PRE-REGISTERED THE OPPOSITE AND WAS WRONG.** MISSION queue item 1
predicted `tralo_cut` would move the capped COUNT more at tight caps because
`sum` supposedly could not reach there. `sum` reaches tight caps fine -- 3-4x
the floor.

✅ **AND IT IS NOT A CONTRADICTION OF 2(z12). IT IS 2(z12) SHARPENED, AND THE
ERROR WAS CLAUDE.md RULE 3.** The HARD COUNT counts items whose argmax is the
capped class, i.e. items at the DECISION BOUNDARY -- exactly where `p(1-p)` is
MAXIMAL. So `sum` is very good at moving the hard count and simultaneously puts
**0.0001** of its gradient at rank K, ~300 ranks away. Both are true; they
describe different points. The hard count is a BOUNDARY readout and cannot
test a claim about the CUT.

⇒ **THE DEFECT RESTATED, and this is the version to quote**: the constraint
at tight caps is not weak. It is **strong, and pointed 300 ranks from where the
metric reads**. That is why 2(z11) finds its item-level effect at the RNG floor
while the count moves 4x the floor: it is moving the wrong items, hard.

✅ **IT ALSO EXPLAINS THE LOOSE COLUMN WITHOUT A NEW ASSUMPTION.** At K/n=0.9
the cap barely binds, so the violation and hence the gradient are small and the
count sits at the floor -- the "where the constraint BINDS nothing is
measurable, and where something is measurable the constraint hardly
constrains" trade-off already recorded in `paired_noise`.

🛑 **CONSEQUENCE FOR `cutwin1`: THE READOUT CHANGES.** The pre-registered
prediction must NOT be about the hard count. It is:

> `tralo_cut` changes the **EMITTED top-K set** at TIGHT caps by more, relative
> to the `tralo_reseed` floor, than `tralo` does -- measured by
> `boundary_probe --control tralo_null` on `final_predictions.csv`, in net
> items. The hard count is expected to move LESS than `tralo`'s, not more,
> because the cut window deliberately removes gradient from the boundary.

⚠️ **CONFOUND, stated because the ratio does not remove it**: tight and
loose come from DIFFERENT campaigns and different backbones, so this is a
cross-campaign DIRECTION, not a within-cell contrast. `cutwin1` carries both
cap levels in ONE campaign and is the clean test.

### (z14) ✅ THE CUT GEOMETRY HOLDS ON ALL FOUR BACKBONES -- but its SIZE is
regime-dependent, which repurposes half of `cutwin1` as a negative control

Measured 2026-09-01 on LIVE campaigns only (`uniform1` + `vitu1` tight,
`loose1` + `loosevit1` loose), balanced: the 6 arms present on all four
backbones x 4 seeds = 48 (run, class) pairs per cell, 20 cells. K is
model-independent and identical across backbones at each cap, so the backbone
contrast is exactly controlled.

| backbone | regime | `cut_window` mass | **`sum` mass** | ratio | cos(p, cutw) |
|---|---|---|---|---|---|
| MobileNetV3 | TIGHT | 0.3795 | 0.0010 | **361x** | 0.975 |
| MobileNetV3 | LOOSE | 0.7319 | **0.1527** | 4.8x | 0.902 |
| MobileNetV2 | TIGHT | 0.5518 | 0.0019 | 283x | 0.983 |
| MobileNetV2 | LOOSE | 0.7489 | **0.1537** | 4.9x | 0.905 |
| RegNetY400MF | TIGHT | 0.7038 | 0.0019 | 361x | 0.976 |
| RegNetY400MF | LOOSE | 0.7487 | **0.1371** | 5.5x | 0.926 |
| **ViTB16** | **TIGHT** | **0.7652** | **0.0066** | **116x** | 0.978 |
| **ViTB16** | **LOOSE** | 0.7440 | **0.1386** | 5.4x | 0.949 |

✅ **THE THREE CLUSTERS HOLD ON EVERY BACKBONE, and are SHARPEST ON ViTB16**
(within-minus-between cosine 0.363-0.399 vs 0.174-0.253 on the CNNs). Clusters
A `{uniform, 1-p}` and B `{sum, margin}` never lose a member anywhere.

🛑 **BUT THE ADVANTAGE IS 116-361x AT TIGHT AND ONLY 4.8-5.5x AT LOOSE**,
uniformly. At loose caps `sum` ALREADY carries 0.07-0.21 of its gradient at the
cut, and `cut_window` MIGRATES from cluster C to cluster B on all four
backbones (cos to B 0.950-0.985 vs to C 0.885-0.918). That is the decision
boundary collapsing onto the cut as K/n rises -- the same fact that makes `sum`
win loose -- and it means **at loose caps the cut window IS approximately
`sum`.**

⇒ **THIS REPURPOSES `cutwin1`'s L90_G95 HALF AS A BUILT-IN NEGATIVE CONTROL**,
and makes the pre-registration two-sided and much harder to satisfy by chance:

> at **L30_G50** `tralo_cut` moves the emitted top-K set more than `tralo`
> does relative to the `tralo_reseed` floor; at **L90_G95** the two are
> approximately the SAME arm (cos 0.90-0.95) and must behave alike. A
> `tralo_cut` that "wins" in BOTH regimes is reading noise, not the mechanism.

⚠️ **AND MobileNetV3 UNDERSTATES THE EFFECT 2x ON THE HEADLINE BACKBONE**
(tight `cut_window` mass 0.380 vs ViTB16's 0.765). `cutwin1` is deliberately
run on the WEAKER backbone: if the mechanism shows there it should show harder
on ViTB16, and a null there is the conservative reading. Do not promote a
MobileNetV3 result to ViTB16 without running it.

⚠️ `linear_z` moves C->B on **ViTB16 at tight caps only** -- the one genuinely
backbone-specific reassignment (to-B 0.883 vs to-C 0.855, replicated on the
independent `iwc2` ViT campaign, and never on any CNN). Thin margin; noted, not
leaned on.

**CORRECTION TO 2(z12).** Its table came from `scripts/step_direction_probe`
when that tool still had a **silent `[:12]` cap** on `--glob`. `sorted()` is
alphabetical, so it kept the first two or three ARMS of one cap tag and called
it 24 pairs. The cap is removed (`--limit`, loud when used). The geometry
SURVIVES the correction -- the full `iwc1` MobileNetV3 set reads `cut_window`
0.3841 / `sum` 0.0004 against the quoted 0.3486 / 0.0001 -- and 2(z12) also
drew on **`iwc1`, which is QUARANTINED**; the live replacements agree (0.3655
`uniform1`, 0.4313 `iwc4`), so the quarantine did not distort it. The table
above is the live-campaign basis and is the one to quote.

⚠️ **SIDE FINDING, and it affects a past claim.** `dom1`'s CNN LOOSE runs are
**byte-identical to `loose1`'s** -- md5 of `final_predictions.csv` matches
**96/96** on MobileNetV3 + MobileNetV2 at L80_G95 and L90_G95, embeddings too.
`dom1b` RegNet vs `loose1` is 0/12, i.e. genuinely independent. Determinism
would explain identical output from identical configs, so this is not yet
established as a defect -- but either way **`dom1` is NOT an independent
replication of `loose1` on those cells and the two must never be counted as
separate evidence.**

⛔ **THE LAST SENTENCE HERE READ "The duals (`alm`, `fioretto`, `hounie`) are
unique to `dom1`, so the dominance contrast itself is untouched" UNTIL
2026-09-04. IT IS NOW EXACTLY INVERTED.** Being unique to `dom1` is precisely
what makes the dominance contrast fragile: `fioretto` and `hounie` are `dom1`
`dead_arms` at 28.00 steps (2(z40)) and exist nowhere else at valid dose, so
there is no second campaign to fall back on. The sentence argued from
uniqueness to safety; uniqueness is the exposure. **`alm` alone survives, and
it is the whole of what "the dominance contrast" now means.**

### (z15) ⛔⛔⛔ AT TIGHT CAPS ON iwildcam THE PRIZE IS **EXACTLY ZERO**, AND
CE SATURATION IS A SEPARATE FACT THAT MERELY COINCIDES WITH IT

Measured 2026-09-01 on `uniform1` + `loose1`, MobileNetV3, `clip` arm, all
seeds. This is the question 2(z12)-2(z14) never asked, and it governs them.

| cell | cls | K | p@K | prec@K | errors <= K | errors in +-20 band |
|---|---|---|---|---|---|---|
| TIGHT L20_G50 | 2 | 74 | 1.00000 | **1.0000** | **0.0** | 0.0 |
| TIGHT L20_G50 | 7 | 92 | 1.00000 | **1.0000** | **0.0** | 0.0 |
| TIGHT L30_G50 | 2 | 111 | 0.99999 | **1.0000** | **0.0** | 0.8 |
| TIGHT L30_G50 | 7 | 137 | 1.00000 | **1.0000** | **0.0** | 0.0 |
| TIGHT L50_G30 | 2 | 111 | 0.99999 | **1.0000** | **0.0** | 0.8 |
| LOOSE L80_G95 | 2 | 296 | 0.79122 | 0.9493 | **15.0** | 10.2 |
| LOOSE L80_G95 | 7 | 364 | 0.99929 | 0.9533 | **17.0** | 13.5 |
| LOOSE L90_G95 | 2 | 333 | **0.38433** | 0.9144 | **28.5** | 17.5 |
| LOOSE L90_G95 | 7 | 411 | 0.99253 | 0.9264 | **30.2** | 11.0 |

⛔ **AT EVERY TIGHT CELL THE CLIPPER'S SELECTION IS 100% CORRECT.** Zero
errors inside K. There is no swap that improves it, only swaps that damage it.
No loss, count function, dual, allocator or optimizer can beat a perfect
selection. This is `headroom`'s "0.0 items in 4 of 6 cells" seen directly.

🛑 **SO IT DOES NOT MATTER THAT `sum` CANNOT REACH THE CUT AT TIGHT CAPS**
(2(z12)) -- **there is nothing at the cut to reach.** 2(z12) diagnosed a real
mechanism in a regime that has no prize. Both statements stand; the second one
governs.

⚠️ **AND IT DOWNGRADES THE CUT WINDOW.** `cut_window` beats `sum` on aim by
**361x at tight** and **4.8x at loose** (2(z14)). Tight is where it wins the
comparison and where the prize is 0.0. Loose is where the prize is 15-30 items
and where it is only 0.90-0.95 from `sum`, i.e. barely a different arm.
⇒ **The cut window is unlikely to help anywhere on iwildcam**, and
`cutwin1`'s L30_G50 half is dead by construction. Predicted BEFORE running it.

✅ **CE SATURATION IS REAL BUT IS NOT THE BINDING CONSTRAINT.** Measured on
the same logs: `L_CE` falls **0.4603 -> 0.0044 (epoch 16) -> 0.0010 (epoch 30)**
and `Soft/Hard` reaches **1.000 by epoch 16**. Warm-up 1 buys only ~2 epochs,
because ~126 CE steps run between consecutive constraint steps, so the model is
collapsed for ~25 of the 29 constraint epochs.
⚠️ **BUT precision@K is a RANKING property, not a confidence property.** A
perfectly calibrated, unsaturated model that ranked the same 111 items would
still have prec@K = 1.0000 and nothing to fix. **Un-saturating CE cannot create
errors to win back, so it does not open the tight regime.** Saturation and the
zero prize coincide here; they are not cause and effect. Do not spend a
campaign on a CE schedule expecting it to unlock tight caps.

⇒ **WHERE THE WORK IS.** The only iwildcam cell with BOTH slack and errors is
**LOOSE, class 2**: at L90 `p@K = 0.384` (not saturated at all), prec@K 0.9144,
**28.5 errors inside K and 17.5 in the reachable band**. Class 7 stays saturated
(p@K 0.992-0.999) even at loose. So the live target is narrow and specific, and
it is the same place `tralo` already measured +0.0253 AP.

⇒ **AND WHAT IT MEANS FOR DATASET #2.** The reason to want `fmow` is now
quantified from the other side: iwildcam gives a model that is PERFECT at the
tight cut. The screen number to go and get is fmow's `prec@K`, and anything at
or above 1.0 there means the same dead end. 2(w2).

### (z16) 🛑🛑🛑 THE CAP IS ONLY A TASK IN A NARROW WINDOW, AND IT IS A
DIFFERENT WINDOW PER CLASS. **Most of this project's campaigns tested a
non-task.**

Roei's question, 2026-09-01: put the cap somewhere it actually forces a
choice, before running any more grids. Measured on `loose1`, MobileNetV3,
`tralo_null` (the unconstrained twin), all seeds.

Three conditions, and **all three must hold** or the cell measures nothing:

| | condition | fails when |
|---|---|---|
| **BINDS** | `hard_count > K` | K above the model's own count -- the cap is free |
| **PRIZE** | `errors@K > 0` | the top-K is already perfect, nothing to swap |
| **WIGGLE** | `p@K < 0.99` | the cut sits in saturated territory |

**class 2** -- n_true 370, model predicts **336** unconstrained:

| K/n | K | errors | p@K | binds | verdict |
|---|---|---|---|---|---|
| 0.30 | 111 | 0.0 | 0.99996 | yes | no prize |
| 0.50 | 185 | 1.0 | 0.99859 | yes | saturated |
| 0.60 | 222 | 2.5 | 0.99395 | yes | saturated |
| **0.70** | 259 | 8.0 | 0.95510 | yes | **TASK** |
| **0.80** | 296 | 15.0 | 0.71779 | yes | **TASK** |
| **0.90** | 333 | 27.8 | 0.58649 | yes | **TASK** |
| 1.00 | 370 | 46.0 | 0.28713 | **NO** | cap slack |

**class 7** -- n_true 456, model predicts **490** unconstrained:

| K/n | K | errors | p@K | binds | verdict |
|---|---|---|---|---|---|
| 0.30-0.80 | 137-365 | 3.2-16.5 | >= 0.99968 | yes | saturated |
| **0.90** | 410 | 27.5 | 0.98881 | yes | **TASK** |
| **1.00** | 456 | 43.5 | 0.71844 | yes | **TASK** |
| 1.10 | 502 | 70.8 | 0.37616 | **NO** | cap slack |

⛔⛔ **CORRECTION, SAME DAY: "BINDS" MUST BE A COUNT, NOT A BOOLEAN, AND THAT
REMOVES THE OVERLAP ENTIRELY.** `hard_count > K` is passed by a cap that evicts
ONE item. At K/n=0.90 on MobileNetV3 class 2 the model predicts **336** against
K=**333**, so the cap forces out **three predictions** and constrains
essentially nothing while looking binding. Requiring >= 10 forced evictions:

| backbone | class 2 window | class 7 window | overlap |
|---|---|---|---|
| **MobileNetV3** | **K/n 0.70-0.80** | K/n 0.90-1.00 | **NONE** |
| ViTB16 | K/n 0.60-0.90 | K/n 0.90-1.00 | only 0.90 |

⛔ **ON MobileNetV3 NO SINGLE CAP FRACTION MAKES BOTH CAPPED CLASSES A TASK.**
The protocol applies one fraction to every capped class, so **the correct
experiment is currently INEXPRESSIBLE** there, and on ViTB16 it is expressible
only at one marginal point. Per-class caps are not a refinement, they are
REQUIRED for the two-class setting to pose a question at all.

✅ **THE CAPS TO USE**, both mid-window on both backbones:

| class | K/n | K | forced out | errors@K | p@K | verdict |
|---|---|---|---|---|---|---|
| 2 | **0.80** | 296 | 40 | 15.0 | 0.718 | TASK |
| 7 | **1.00** | 456 | 34 | 43.5 | 0.718 | TASK |

Note class 7 needs **K/n = 1.00** -- a budget EQUAL to the true count. That is
not a degenerate cap: the model predicts 490 against 456 true, so it still
evicts 34, and its cut sits at p=0.718. A cap above `n_true` is legitimate
whenever the model over-predicts, and the protocol has never used one.

🛑 `scripts/task_window.py` is the gate, `--self-test` covers all five
verdicts including a negative control that a genuine task IS reported as one.

🛑 **THE WINDOW IS SQUEEZED FROM BOTH SIDES.** From below by saturation and
by the top-K already being perfect; from above by the cap simply not binding.
Class 2: **K/n 0.70-0.90**. Class 7: **K/n 0.90-1.00**. The ONLY overlap is
**exactly K/n = 0.90**, i.e. `L90_G95` -- one of the TWO cap tags TraLO's
+0.0253 AP came from (`loose1`, 5/1 over **6** cells, 2(w3); it is not a
single cell).

⛔ **CONSEQUENCE: EVERY L20 / L30 / L50 CAMPAIGN TESTED A NON-TASK.** At those
tags class 2 has 0.0-1.0 errors inside K and class 7's cut sits at p >= 0.9999.
A null there is not evidence about the method; it is the absence of a question.
This is the single best explanation on record for why so many arms tied.

⚠️ **AND THE PROTOCOL CANNOT EXPRESS THE RIGHT CAP.** `L<local>_G<global>`
applies ONE fraction to EVERY capped class, so at any single tag at least one
class is in the wrong regime -- at 0.90 class 2 sits at the top of its window
and class 7 at the very bottom of its. **A per-class cap fraction (class 2 at
0.80, class 7 at 0.95) is the fix**, and it is a `src/training/constraints.py`
change, not a knob.

⇒ **BEFORE ANY FURTHER GRID**: verify the three conditions on the intended
cells. A cap that does not bind, has no errors, or sits at p@K ~ 1 cannot
distinguish any two methods, however well aimed either of them is.

### (z17) 🛑🛑🛑 THE NON-TASK IS UNIVERSAL: **24 OF 24** (backbone x class x cap)
CELLS AT L20 / L30 / L50 POSE NO QUESTION, ON ALL FOUR BACKBONES

2(z16) measured the task window on MobileNetV3 and inferred the consequence for
the rest. Measured 2026-09-01 on all four backbones, `tralo_null` (the
unconstrained twin), iwildcam, 8-12 reference runs each. ViTB16 comes from
`iwc2`, which is quarantined for constraint DOSE -- irrelevant to a lambda=0
arm, whose ranking is a valid reading of what an unconstrained ViTB16 does.

**Every backbone HAS a task window. None of them is anywhere near the caps this
project ran.** The lowest window edge across all four is **K/n = 0.60**.

| backbone | unconstrained count, class 2 | window, class 2 | unconstrained count, class 7 | window, class 7 | overlap |
|---|---|---|---|---|---|
| **ViTB16** | 347 / 370 | 0.60-0.90 | 487 / 456 | 0.90-1.00 | only **0.90** |
| MobileNetV3 | 355 / 370 | 0.70-0.90 | 478 / 456 | 0.90-1.00 | only **0.90** |
| MobileNetV2 | 405 / 370 | 0.80-1.00 | 421 / 456 | 0.80-0.90 | 0.80, 0.90 |
| RegNetY400MF | 345 / 370 | 0.60-0.90 | 498 / 456 | 0.80-1.00 | 0.80, 0.90 |

⛔ **THE VERDICT AT THE CAPS THE CORPUS ACTUALLY RAN.** 4 backbones x 2 classes
x {L20, L30, L50} = **24 cells, and not one is a task**: 12 read `no prize`
(zero errors inside K) and 12 read `saturated` (p@K >= 0.9938). The same 4 x 2
grid at K/n = 0.90 is **8 of 8 a TASK**.

| K/n | errors@K, range over the 8 (backbone, class) | p@K, range | tasks |
|---|---|---|---|
| 0.20 | **0.0 - 2.5** | 0.99978 - 1.00000 | **0 / 8** |
| 0.30 | **0.0 - 3.0** | 0.99945 - 1.00000 | **0 / 8** |
| 0.50 | **0.0 - 7.8** | 0.99381 - 1.00000 | **0 / 8** |
| **0.90** | **14.5 - 43.8** | **0.48820 - 0.96096** | **8 / 8** |

🔑 **THIS IS THE QUANTIFIED FORM OF THE CE-SATURATION WORRY, AND IT LOCATES THE
FAULT IN THE CAP, NOT IN THE MODEL.** The model is not uniformly saturated. The
cap was placed deep inside the region where it is. Move the cut from K/n=0.20 to
0.90 and p@K falls from ~1.0 to 0.49-0.96 while the errors available to fix rise
from 0.0-2.5 to 14.5-43.8. The wiggle room was always there; every campaign cut
above it.

⛔ **SO THE HEADLINE BACKBONE IS COVERED TOO.** 2(z16) left open whether ViTB16
escaped. It does not: at L20 and L30 **both** its capped classes read `no prize`
outright -- zero errors inside K, so there exists no swap that improves the
selection and only swaps that damage it. A null in those cells is the absence of
a question, and the corpus is full of them.

✅ **AND THE TWO INDEPENDENT LINES AGREE ON THE SAME CAP.** `paired_noise`
priced K/n=0.90 at **~7 seeds per cell** against 546-2607 at L20/L30/L50
(2(v)); the task window independently says 0.90 is the only single fraction
that is a task for both classes on all four backbones. The cheap regime and the
answerable regime are the same regime. The standing caveat in 2(v) -- "where the
constraint BINDS nothing is measurable" -- is now **too pessimistic as stated**:
at 0.90 the cap still forces out 11-88 predictions on every backbone, which is
binding by any count-based reading.

⚠️ **The single-fraction protocol has exactly ONE legal cap on iwildcam, and it
is a corner.** At 0.90 class 2 sits at the TOP of its window on three backbones
and class 7 at the BOTTOM of its on two. Per-class cap fractions
(`src/training/constraints.py:cap_fraction_for`, 2(z16)) are what let the
campaign sit mid-window on both classes at once; `taskwin1` is the first
campaign to use them.

⇒ **NO GRID AT L20 / L30 / L50 IS TO BE RUN AGAIN ON iwildcam.** It is not a
weak regime, it is a regime with nothing to measure, and that is now established
on every backbone the paper claims.

### (z18) 🔑🔑 THE AIM FIX AND THE TASK WINDOW ARE ANTI-CORRELATED. Where
`tralo_cut` has the most room, the cap poses no question; where the cap poses a
question, `sum` already aims reasonably well.

Priced BEFORE the GPU, 2026-09-01, `step_direction_probe` on real stored
features, MobileNetV3, `tralo_null`, unlimited (6-8 run-class pairs per row).
Tight rows from `iwc3`, loose rows from `equaldose1`.

| cap | K/n | cos(`cut_window`, `sum`) | cut mass, `sum` | cut mass, `cut_window` | ratio | a TASK? |
|---|---|---|---|---|---|---|
| `L20_G50` | 0.20 | **0.716** | 0.0000 | 0.3034 | unbounded | ⛔ no |
| `L30_G50` | 0.30 | **0.728** | 0.0000 | 0.3222 | unbounded | ⛔ no |
| `L50_G30` | 0.30 | **0.728** | 0.0001 | 0.3222 | ~3000x | ⛔ no |
| `L80_G95` | 0.80 | **0.926** | 0.0972 | 0.6733 | **6.9x** | ✅ yes |
| `L90_G95` | 0.90 | **0.951** | 0.1589 | 0.7086 | **4.5x** | ✅ yes |

✅ **`L30_G50` and `L50_G30` give byte-identical probe rows**, which is the
expected consequence of `K_eff = min(global, sum of local)`: both land on
K/n=0.30, and `tralo_null` shares one warm-up across cap tags, so identical p
and identical K give identical everything. A useful confirmation that the
effective-budget arithmetic in `gen_campaign` matches what the runs deploy.

🛑 **THE CONSEQUENCE FOR `tralo_cut`, STATED BEFORE THE CAMPAIGN RUNS.** In the
task window it is **not** a distinct method from `tralo` in any strong sense:
0.926-0.951 cosine is 18-22 degrees of separation, against 0.716-0.728 (44
degrees) at the tight caps where it was designed to matter. It is not INERT --
the cut mass still differs 4.5-6.9x and `flag_live` reports md5-distinct raw
predictions -- but the expectation is a small difference, not a new arm.

⇒ **`taskwin1` IS NOT A `tralo_cut` VS `tralo` EXPERIMENT, AND MUST NOT BE
REPORTED AS ONE.** Its question is the one 2(z17) made askable for the first
time: **with the cap finally inside the measured window, does ANY trained arm
clear the `tralo_reseed` floor?** Every previous null carried the escape hatch
"the cap was in the wrong place". These two cells remove it.

⚠️ **AND IT PRICES THE WHOLE CLUSTER-C DIRECTION DOWNWARD.** Aiming at the cut
was derived (2(z12)) from the tight-cap geometry, where `sum` puts 0.0001 of
its gradient at the cut. That geometry is real and it is measured -- but it
lives entirely in cells that pose no question. In the cells that do, the aim was
never far wrong. Cluster C is therefore a fix to a defect that only exists where
nothing can be won, unless the task window itself moves (a different dataset,
or a backbone whose unconstrained count sits much further from K).

🔑 The honest summary of 0-NOW: **defect (3) is real but is largely confined to
the non-tasks that defect (4) identifies.** Neither was visible without the
other, and the pair together explains the tie history better than either alone.

### (z19) 🟢🔴⛔ THE TASK-WINDOW QUESTION IS ANSWERABLE ON DATA ALREADY IN HAND.
⛔ **THE "`tralo` LEADS EVERY RIVAL IN THE TASK CELLS" HALF OF THIS HEADER IS
RETRACTED 2026-09-04** -- two of the three rivals it counted are dead arms.
✅ THE "STILL DOES NOT CLEAR THE RNG FLOOR" HALF IS UNTOUCHED AND STANDS.

`equaldose1` was built for a different question (is TraLO's lead a 3.4% step
head start?), but its caps are LOOSE, and 2(z17)'s measured windows put **4 of
its 6 cells inside the window on both capped classes**: all three MobileNetV2
caps, plus `MobileNetV3/L90_G95`. So the question `taskwin1` was staged to ask
is answerable now, at 4 cells with 9 arms and no GPU.

Integrity first: `check_parity` OK, one `code_version` `10d375183f8c`,
`n_md5 == n_seeds` in every cell (no inert flag). All four task cells hold 4
seeds; the 12 unfinished runs sit in `MobileNetV3/L95_G80`, a NON-task cell.

⛔ **THIS LINE SAID "dose 100% on every arm, ... not quarantined" UNTIL
2026-09-04. BOTH CLAUSES ARE FALSE.** `equaldose1` is PARTIALLY QUARANTINED
(`scorable=True` with `dead_arms=['fioretto', 'hounie', 'tralo_lam0']`), and
its dose is 29.00 steps/run for `tralo` and `alm` against **28.00** for those
three (2(z40)). The "100%" was applied/attempted computed WITHIN each arm,
which is exactly the statistic that cannot see a between-arm gap -- the same
error as 2(x). **Three of this section's nine arms may not be contrasted with
anything.**

**Each arm MINUS its own `tralo_null` twin, mean over cells, ccF1 in items:**

| arm | TASK cells (4) dAP | dccF1 items | dmacroF1 | NON-task (2) dccF1 items |
|---|---|---|---|---|
| **`tralo`** | **+0.0275** | **+2.32** | **-0.0011** | +3.96 |
| ~~`tralo_lam0`~~ ⛔ **DEAD ARM, 28.00** | ~~+0.0287~~ | ~~+2.67~~ | ~~-0.0110~~ | ~~+2.37~~ |
| ~~`fioretto`~~ ⛔ **DEAD ARM, 28.00** | ~~+0.0218~~ | ~~+1.62~~ | ~~-0.0015~~ | ~~+0.58~~ |
| `alm` | +0.0219 | **-0.73** | -0.0036 | **+5.85** |
| ~~`hounie`~~ ⛔ **DEAD ARM, 28.00** | ~~+0.0243~~ | ~~-2.30~~ | ~~-0.0080~~ | ~~+1.63~~ |
| `clip` | +0.0044 | -2.78 | -0.0080 | -2.91 |
| `focal_clip` | -0.0069 | -2.80 | -0.0141 | -2.59 |
| **`tralo_reseed` (RNG FLOOR)** | **-0.0157** | **-3.39** | -0.0063 | -0.45 |

⛔⛔⛔ **THE DOSE QUESTION IS REOPENED, 2026-09-04. UNTIL THEN THIS READ:**

> ✅ **THE DOSE QUESTION IS ANSWERED, IN TraLO'S FAVOUR.** `tralo` (+0.0275 AP)
> and `tralo_lam0` (+0.0287) are indistinguishable, and `tralo_lam0` is the
> dose-matched arm whose first constraint step carries a zero gradient exactly
> as the duals' does. **The 3.4% step head start is NOT the source of TraLO's
> lead.** That was `equaldose1`'s designed question and it is now closed.

🛑 **THE CONTROL BUILT TO CLOSE THE OBJECTION IS THE ARM THE DEFECT LANDED ON.**
`tralo_lam0` attempts **28.00** constraint steps per run and is a `dead_arm` in
`equaldose1` (2(z40)). So the single contrast that closed the dose objection is
itself a contrast against a dead arm, and under the registry as written
`scripts.quarantine.drop_dead_runs` removes those runs before any scorer sees
them. **The claim cannot be recomputed at all, not merely recomputed smaller.**

🛑 **AND NO REPLACEMENT EXISTS.** Every arm in `equaldose1` at 28.00 steps --
`fioretto`, `hounie`, `tralo_lam0` -- is a dead arm. There is no surviving
dose-matched control for `tralo` anywhere in the corpus. **`vitdual2` is the
only campaign that can close this objection** (all four duals at 29.00,
verified), and on 2026-09-04 it is **32 of 88 runs complete**. Until it lands,
the honest statement is: *the dose objection is OPEN, and `dom1`'s ordering has
never been shown to survive equalisation.*

⚠️⚠️ **AND THERE IS A REAL TENSION HERE THAT IS NOT MINE TO RESOLVE -- IT IS
RECORDED, NOT DECIDED.** `scripts/quarantine.py` calls `tralo_lam0`'s 28.00 a
DEFECT ("a lambda=0 arm that still gates its backward on a multiplier, so it
loses epoch 0 exactly as the duals do"). But `docs/MISSION.md`'s launch note
for `equaldose1` says the 28 was **DELIBERATE**: the arm exists precisely to
match the duals' 28 so that TraLO's extra step can be priced, and its void
check ("epoch-1 `Grad_Norm` 0.0 for `tralo_lam0`, ~3.09 for `tralo`") was
recorded as PASSING on exactly that basis. Both readings are on file and they
disagree about whether a *deliberately* dose-matched control counts as a dead
arm.
**UNVERIFIED. This needs a human decision, not a recount.** What would settle
it: decide whether an arm mismatched BY DESIGN is a dead arm, and write the
decision into the `equaldose1` registry entry. Until then the operative fact is
the one above -- the registry as written drops the arm, so the number is not
recomputable.

⛔ **THIS READ "AND IN THE TASK CELLS `tralo` LEADS EVERY RIVAL ON ccF1:
+2.32 items against `fioretto` +1.62, `alm` -0.73, `hounie` -2.30" UNTIL
2026-09-04.** Two of those three rivals are dead arms. **Against the one
surviving rival the statement holds and is worth stating exactly:** in the 4
task cells `tralo` is **+2.32 items against `alm`'s -0.73**, with both clippers
below -2.7. That is a lead over ONE rival dual, not over a field.

✅ **UNAFFECTED:** `tralo` is still **the only arm whose macroF1 damage is near
zero** (-0.0011 against -0.0036 to -0.0141), and the clipper comparisons are at
equal dose. Neither depends on a dead arm.

🔴 **BUT IT DOES NOT CLEAR THE RNG FLOOR, AND RESTRICTING TO TASK CELLS MAKES
THAT WORSE.** A pure reseed moves ccF1 by **3.39 items** in the same cells,
against `tralo`'s **2.32** -- a ratio of **0.68x**. Over all 6 cells the same
comparison reads 2.63 vs 2.39 (1.10x), so **the task cells are where the
constraint looks WEAKEST relative to noise**, not strongest. The prize being
real does not make it reachable.

🔑🔑 **THE ORDERING CHANGES WITH THE CELL SELECTION, WHICH IS THE POINT.**
`alm` is the best arm on ccF1 in the 2 NON-task cells (+5.85) and the second
WORST in the 4 task cells (-0.73); `tralo` is the reverse. **Which method looks
best depends on whether the cell poses a question at all.** That is direct
evidence the task window is not a bookkeeping refinement: it selects the
answer. Every historical ranking in this project was computed over cells that
2(z17) shows were mostly non-tasks.

⚠️ **WHAT THIS IS NOT.** These are means over cells, not paired tests --
`full_panel`'s paired version over all 6 cells reads `tralo` +0.0256 AP 6/0,
+0.0035 ccF1 5/1, and calls every line UNDERPOWERED (9-17 seeds needed). And
the two nulls are ONE run per (dataset, backbone, seed) replicated across cap
levels, so their effective n is cells / n_cap_levels: the reseed floor here
rests on 2 independent units, not 6. Quote the direction and the ordering;
do not quote the ratio as settled. FRAMEWORK 2(v), and the standing rule that
signs hold early while ratios do not.

⇒ **`taskwin1` KEEPS ITS VALUE BUT LOSES ITS URGENCY.** It adds per-class caps
mid-window on BOTH classes (which no cell here achieves -- every cell above sits
at a window EDGE for at least one class), and it adds `tralo_cut`. It is no
longer the only way to ask the question.

### (z20) 🟢🟢🟢 ON THE HEADLINE BACKBONE, IN THE ONE CELL THAT IS A MEASURED
TASK, `tralo` IS THE ONLY ARM POSITIVE ON EVERY METRIC AND IT CLEARS THE RNG
FLOOR BY 2.8x.

🛑 **CORRECTED BY 2(z24) 2026-09-01, AND THIS ONE IS THE SHARPEST CASE.**
The "one cell that is a measured task" was classified from a window that is a
MEAN over seeds. Re-measured on `loosevit1`'s OWN 8 `tralo_null` runs, PER SEED:
ViTB16 class 2 is a strict task at K/n **0.60 and 0.70 only**, and at 0.90 the
cap binds in **6 of 8** seeds (`** PARTIAL 6/8 **`). Class 7 is a task at 0.90
only. So `L90_G95` is a PARTIAL cell, not a clean one, and **ViTB16 has ZERO
strict task cells at any cap ever run**.

⚠️ Read the result below as a DIRECTION measured on a cell whose cap poses its
question to three quarters of its seeds. The dilution biases +1.41 items TOWARD
zero, so the sign is if anything conservative, but "the one cell that is a
measured task" is no longer an accurate description of it, and n=1 cell was
never a significance claim.

🔑 **THIS IS WHY `vittask1` EXISTS AND WHY IT IS NOT A REPEAT OF `loosevit1`.**
The two ViTB16 classes' per-seed windows are 0.60-0.70 and 0.90 (the yml is
the only place either is a number) and do NOT overlap,
so no single-fraction tag can express a strict cell. `vittask1` runs the
per-class tags `L60-90_G95` and `L70-90_G95`, which are the only two that sit
inside both. `loosevit1` could not have tested them: the per-class cap form did
not exist when it was generated.

`loosevit1` (ViTB16, iwildcam, 48 runs, all completed) lives in
`~/optloss-loosevit`, one of the **fourteen** worktrees sharing the object
store (2(z22)) and one that no doc listed until 2026-09-01. Its
two cap tags are `L80_G95` and `L90_G95`, and 2(z17)'s ViTB16 windows (class 2
**0.60-0.90**, class 7 **0.90-1.00**) split them:

| cell | class 2 K/n | class 7 K/n | verdict |
|---|---|---|---|
| **`L90_G95`** | 0.900 | 0.901 | ✅ **TASK on both classes** |
| `L80_G95` | 0.800 | 0.798 | ⛔ class 7 below its window |

Gates first: dose **116/116 on both trained arms in both cells**, `check_parity`
OK, one `code_version` `74f858657154`, not quarantined, `n_md5 = 4` for every
arm in every cell (no inert flag). 4 seeds everywhere.

**Each arm MINUS its own `tralo_null` twin. ✅ TASK CELL `L90_G95`:**

| arm | dAP | dAUROC | dccF1 | dmacroF1 | ccF1 items |
|---|---|---|---|---|---|
| **`tralo`** | **+0.0083** | **+0.0023** | **+0.0018** | **-0.0008** | **+1.41** |
| `tralo_uniform` | -0.0162 | -0.0019 | -0.0045 | -0.0129 | -3.50 |
| `clip` | -0.0230 | -0.0037 | -0.0077 | -0.0137 | -6.05 |
| `focal_clip` | -0.0070 | -0.0001 | -0.0039 | -0.0169 | -3.07 |
| **`tralo_reseed` (RNG FLOOR)** | -0.0113 | -0.0010 | **-0.0006** | -0.0384 | **-0.51** |

🟢 **`tralo` is the ONLY arm positive on all four**, and its ccF1 movement is
**1.41 items against an RNG floor of 0.51 -- 2.8x.** This is the first time in
this project that a trained arm's ccF1 movement clears its own reseed floor in
a cell whose cap is a MEASURED task. Its macroF1 damage is **-0.0008**, i.e.
none, while the floor itself sits at -0.0384 and every other arm at -0.0129 to
-0.0169.

⛔ **AND `tralo_uniform` IS BELOW THE FLOOR HERE**, -3.50 items against -0.51,
which is 2(z11)'s tight-cap verdict reproduced in a task cell. The uniform count
is not the fix.

**The NON-task cell `L80_G95`, same campaign, same arms:**

| arm | dAP | dccF1 | ccF1 items |
|---|---|---|---|
| `tralo` | +0.0045 | +0.0015 | +1.13 |
| **`tralo_reseed`** | -0.0113 | +0.0012 | **+0.91** |

⇒ `tralo` is +1.13 items against a floor of +0.91 -- only **1.24x**. **The task
cell separates the constraint from the noise more than twice as well as the
non-task cell does** (2.8x vs 1.24x), on the same campaign, the same backbone
and the same arms, with only the cap moved from K/n 0.80 to 0.90.

⚠️ **AND IT DOES NOT REPRODUCE THE SIGN OF 2(z19).** On MobileNetV2/V3
(`equaldose1`) restricting to task cells made the ratio WORSE (1.10x -> 0.68x);
on ViTB16 it makes it BETTER (1.24x -> 2.8x). So "task cells help" is NOT
established as a general rule -- what IS established is that the cell selection
changes the answer, in both directions, which is why it must be measured rather
than assumed.

🛑 **ONE CELL. NOT CALLABLE.** A single cell has a minimum attainable sign-test
p of 0.500, and 4 seeds give 0.125 at best. This is a DIRECTION on the
pre-registered headline backbone, not a result, and it must never be quoted as
significant. The floor it clears is itself 0.51 items, which is small enough
that a second cell could move the ratio a long way.

⇒ **THE NEXT CAMPAIGN IS ViTB16 AT SEVERAL TASK CELLS.** ViTB16's two windows
overlap only at K/n=0.90, so more cells require the per-class cap form
(2(z16)): e.g. `L70-90_G95`, `L80-95_G95`, `L85-100_G95` all sit mid-window on
both classes. This supersedes MISSION queue item 2's "more loose cap tags",
which would have added cells that are not tasks.

### (z21) 🟢🟢 `dom1` READ ON ITS TASK CELLS: `tralo` CLEARS ITS OWN RNG FLOOR
IN **4 OF 4**, LEADS EVERY RIVAL DUAL ON ccF1, AND `tralo_uniform` IS BELOW THE
FLOOR IN 4 OF 4.

🛑 **CORRECTED BY 2(z24) 2026-09-01, READ IT FIRST.** The task cells below
were classified from a window that is a MEAN over seeds spreading 105 items,
and imported from another campaign's model. On the strict per-seed rule `dom1` has 3 task cells, not 4, all MobileNetV2, so
its 2 independent units are **1**. The SIGNS below are unchanged;
the unit count and the p-value are not.

`dom1` (384 runs, **16 arms**, MobileNetV2 + MobileNetV3, `L80_G95` `L90_G95`
`L95_G80`) is complete, 4 seeds in every cell, `n_md5 == n_seeds` in all 96
cells (no inert flag). Its value here is that **each dual carries its OWN
lambda=0 null** (`alm_null`, `fioretto_null`, `hounie_null`, `tralo_null`), so
every number below is that method's constraint term, not its compute.

2(z17)'s windows make 4 of its 6 cells tasks: all three MobileNetV2 caps plus
`MobileNetV3/L90_G95`.

**ccF1 in ITEMS, each arm minus ITS OWN null, per cell. T = task cell:**

| cell | | `tralo` | `alm` | `fioretto` | `hounie` | `uniform` | **FLOOR** |
|---|---|---|---|---|---|---|---|
| MobileNetV2/L80_G95 | **T** | **13.23** | 12.95 | 6.61 | 2.98 | 7.53 | 9.34 |
| MobileNetV2/L90_G95 | **T** | **11.26** | 7.86 | 9.14 | 8.47 | 4.78 | 6.84 |
| MobileNetV2/L95_G80 | **T** | 11.61 | **12.64** | 6.38 | 10.23 | 2.11 | 5.63 |
| MobileNetV3/L90_G95 | **T** | 13.72 | 10.46 | **15.17** | 11.82 | 10.62 | 10.68 |
| MobileNetV3/L80_G95 | | 7.21 | **11.35** | 2.09 | 6.10 | 8.49 | 8.89 |
| MobileNetV3/L95_G80 | | 7.30 | 6.41 | 7.02 | 7.37 | **9.51** | 5.86 |

**Above its own floor, in the 4 TASK cells:**

🛑 **AND THE 4 CELLS ARE NOT 4 INDEPENDENT UNITS.** Cells at different cap
levels on the SAME backbone share one lambda=0 warm-up model -- `tralo_reseed`'s
dAP is literally constant across caps within a backbone -- so `dom1`'s four task
cells (MobileNetV2 x 3 caps + MobileNetV3 x 1) are **2 independent units**, not
4. The 4/4 below is a per-cell count and must never be turned into a sign-test
p. The independent statistic is in 2(z23).

| arm | cells above floor | mean items | ratio to floor |
|---|---|---|---|
| **`tralo`** | **4 / 4** | **12.46** | **1.53x** |
| `alm` | 3 / 4 | 10.98 | 1.35x |
| `fioretto` | 3 / 4 | 9.32 | 1.15x |
| `hounie` | 3 / 4 | 8.37 | 1.03x |
| **`tralo_uniform`** | **0 / 4** | 6.26 | **0.77x** |
| `tralo_reseed` (the floor) | -- | 8.12 | 1.00x |

⚠️ **TWO OF THESE FOUR TASK CELLS ARE GRID-SNAPPED** (2(z22)):
MobileNetV2 `L80_G95` (class 7 K/n 0.798) and MobileNetV2 `L90_G95`
(class 7 0.901) sit 0.002 and 0.001 outside a window edge and are tasks
through the tolerance. Both edges are measured grid points, so this is
rounding rather than extrapolation -- but the 4/4 below is 2 strictly
inside plus 2 snapped, and should be quoted that way.

🟢 **`tralo` is the only arm above its floor in every task cell**, and it leads
every rival dual on mean ccF1 items. `alm` leads on AP (+0.0426 vs +0.0403), so
the ordering is metric-dependent and must be quoted as such. 4/4 is a sign-test
p of 0.0625 -- a DIRECTION, not significance.

⛔ **`tralo_uniform` IS BELOW THE FLOOR IN 4 OF 4 OF *THESE* TASK CELLS**, and
2(z11) found it below the floor at tight caps too. ⚠️ **But it is NOT below the
floor everywhere**: `dom1b` puts it 2/3 ABOVE (1.61x) and `loose1` 1/5, so
across all 8 measured task cells it clears its floor in **3 of 8** (2(z23)).
The verdict that survives is the weaker one -- it never LEADS, and it is the
worst arm in the campaign with the most cells -- not "refuted everywhere".

🔴 **AND THE ORDERING SCRAMBLES IN THE 2 NON-TASK CELLS**, exactly as 2(z19)
found on `equaldose1`: `tralo` clears its floor in only 1 of 2 there, and
`tralo_uniform` -- bottom of the table in the task cells -- is the best arm in
`MobileNetV3/L95_G80`. Cell selection changes the winner. Every historical
ranking in this project pooled cells without asking which posed a question.

⚠️ **THE macroF1 SIGN FLIPS WITH THE CELL TYPE, AND THE FLOOR FLIPS WITH IT.**
In the task cells every method's macroF1 vs its own null is neutral-to-positive
(`tralo` +0.0008, `alm` +0.0044, floor **+0.0047**); in the non-task cells every
one is negative (-0.0064 to -0.0170, floor **-0.0124**). So the universal
macroF1 damage of 2(xfam1) is confined to the non-task cells in ABSOLUTE terms
-- but the RNG floor moves the same way and by the same amount, so **relative to
the floor the damage does not disappear** (`tralo` is -0.0039 vs floor in task
cells and +0.0041 in non-task). Quote the ratio, never the raw delta.

🛑 **THE FLOOR IS NOT PORTABLE BETWEEN CAMPAIGNS.** `dom1`'s reseed floor is
**5.63-10.68 items** per task cell; `equaldose1`'s, on the SAME two backbones at
the SAME three cap tags, is **3.39** and of the opposite sign. A floor measured
in one campaign says nothing about another -- it must come from the reseed arm
inside the campaign being scored. This is 2(v)'s "say which of the four noise
numbers you mean", now with a fifth axis: WHICH CAMPAIGN.

### (z22) 📊 THE CENSUS: **1376 OF 2112** COMPLETED iwildcam RUNS SIT AT A CAP
THAT POSES NO QUESTION, AND 46% OF THE ONES THAT DO NOT ARE WITHIN 0.002 OF A
WINDOW EDGE.

Counted 2026-09-01 across all FOURTEEN worktrees, every `results/` tree, from
`config.json` status only. Buckets are disjoint; precedence is
quarantined > removed dataset > unmapped cap > task/non-task.

| bucket | completed runs | cells |
|---|---|---|
| QUARANTINED | 817 | 36 |
| iwildcam **NON-TASK** | 750 | 27 |
| iwildcam **TASK** | 736 | 17 |
| **total completed, whole project** | **2303** | **80** |

Setting quarantine aside (a quarantined run still sits in a cell): of **2112**
completed iwildcam runs, **1376 are NON-TASK and 736 are TASK**. ⛔ **All 626
quarantined iwildcam runs are NON-TASK -- not one quarantined run sits in a
task cell.** Every dermmnist run (191) is inside a quarantined campaign, so the
removed-dataset bucket reads zero only because quarantine absorbs it; flipping
precedence to dataset-first gives no-window 191, quarantined 626, and the same
totals.

⚠️ **THE 0.005 TOLERANCE IS LOAD-BEARING, AND HERE IS WHY IT IS STILL HONEST.**
336 of the 736 task-cell runs (**46%**) qualify only through it. They are
`L80_G95` (class 7 K/n = **0.798** against a window edge of 0.80) and
MobileNetV2 `L90_G95` (class 7 **0.901** against an edge of 0.90). Those are
0.002 and 0.001 from grid points that were **themselves measured and found to
be tasks** -- `fraction_grid` has a step of 0.1, and a cap TAG cannot produce a
round K/n because K is an integer budget over an integer class count. The
tolerance SNAPS to an already-measured point; it does not extrapolate past one.
`configs/task_cells.py` now asserts `tolerance <= grid_step / 10` so raising it
without re-measuring on a finer grid FAILS, and `classify` reports a per-class
`margin` and `snapped` flag so an edge case is visible rather than folded into
a boolean.

🛑 **AND THE TWO CAMPAIGNS EVER GENERATED WITH TASK-WINDOW CAPS HAVE ZERO
COMPLETED RUNS.** `taskwin1`'s `L70-90_G95` / `L80-100_G95` are the only
per-class cap tags in the project's history and it is 48/48 pending, blocked on
a GPU.

**Three record corrections the census forced:**

1. **FOURTEEN campaigns carry `QUARANTINE.json`, not ten.** CLAUDE.md said ten;
   the extra three were `dosefix`, `vit_ceskip`, `vit_diag`. `taskwin1` was
   added 2026-09-01 (no `--constraint-fp32`, 20/29 = 69.0%, superseded by
   `taskwin2` at 100% on the same host). Corrected.
2. **11 `config.json` sit at depth 6 rather than 5**, all under
   `optloss-audit/vit_diag` in `_hp_liveness/` and `_variance_probe*/` instead
   of `seed_N/`. They are diagnostic sub-runs, not protocol runs, and any tool
   that globs at a fixed depth silently omits them. Excluded above; including
   them gives 2313 completed.
3. **420 `seed_*` directories hold no `config.json` at all** (`uniform1_VOID`
   240, `mnv3bar` 62, `vit_ceskip` 46, `vit_diag` 40, `mc_sgd` 32). All are in
   quarantined campaigns and hold only `error_log*.json` or `training_log.csv`,
   so they conceal no completed runs -- but a tool counting DIRECTORIES rather
   than configs would over-report this project's output by 18%.

⇒ **The tie history now has a denominator.** Roughly two thirds of every
iwildcam run ever completed was spent in cells where no two methods can be
distinguished. That is not a re-interpretation of the results; it is a
measurement of what was asked.

### (z23) 🟢🟢 THE TASK-CELL RESULT REPLICATES ON A FOURTH BACKBONE AND IN A
SECOND CAMPAIGN, AND THE HONEST SIGNIFICANCE IS **p = 0.0625 ON 4 INDEPENDENT
UNITS**, NOT 7/8 ON CELLS.

🛑 **CORRECTED BY 2(z24) 2026-09-01, READ IT FIRST.** The task cells below
were classified from a window that is a MEAN over seeds spreading 105 items,
and imported from another campaign's model. `loose1` and `dom1` are BYTE-IDENTICAL on MobileNetV2's lambda=0 arm (8/8), so
they are one model, not two campaigns; `loose1`/MobileNetV3 and
`loose1`/RegNetY400MF have ZERO strict task cells. **4 of 4, p=0.0625 becomes
3 of 3, p=0.125.** The SIGNS below are unchanged;
the unit count and the p-value are not.

`dom1b` (192 runs, RegNetY400MF, 16 arms) and `loose1` (144 runs, MobileNetV2 +
MobileNetV3 + RegNetY400MF, tralo family only) were both complete and both
unread. Gates first: `check_parity` OK, one `code_version` each
(`1d92117363d2`, `74f858657154`), `n_md5 == n_seeds` in every cell.

⛔ **THIS SAID "dose 100% on every trained arm in both ... neither quarantined"
UNTIL 2026-09-04, AND BOTH CLAUSES ARE FALSE FOR `dom1b`.** It is PARTIALLY
QUARANTINED (`dead_arms=['fioretto', 'hounie']`), and its dose is 29.00
steps/run for `tralo`/`alm`/`tralo_uniform` against **28.00** for those two
(2(z40)). As in 2(x) and 2(z19), the "100%" is applied/attempted computed
WITHIN an arm and is structurally blind to a between-arm gap. `loose1` is
separately off-recipe (`constraint_grad_mode: clip`).

`dom1b` is **3 of 3 task cells**; `loose1` is **5 task, 1 non-task**
(MobileNetV3 `L80_G95`, class 7 at K/n 0.798 against a 0.90-1.00 window).

**ccF1 in ITEMS, arm minus ITS OWN null, task cells:**

| campaign | arm | mean items | above own floor | ratio |
|---|---|---|---|---|
| `dom1b` (3 cells) | **`tralo`** | **4.38** | **3/3** | **2.49x** |
| | `alm` | 2.87 | 2/3 | 1.63x |
| | ~~`fioretto`~~ ⛔ **DEAD ARM** | ~~2.82~~ | ~~2/3~~ | ~~1.60x~~ |
| | `tralo_uniform` | 2.84 | 2/3 | 1.61x |
| | ~~`hounie`~~ ⛔ **DEAD ARM** | ~~1.33~~ | ~~2/3~~ | ~~0.76x~~ |
| | `tralo_reseed` | 1.76 | THE FLOOR | 1.00x |
| `loose1` (5 cells) | **`tralo`** | **9.65** | **4/5** | **1.54x** |
| | `tralo_uniform` | 5.39 | 1/5 | 0.86x |
| | `tralo_reseed` | 6.27 | THE FLOOR | 1.00x |

⛔ **THIS READ "THE ARM ORDERING REPLICATES EXACTLY. `dom1b`: tralo 4.38 >
alm 2.87 ~ fioretto 2.82 > hounie 1.33. `dom1`: tralo 12.46 > alm 10.98 >
fioretto 9.32 > hounie 8.37" UNTIL 2026-09-04.** Two of the four rank positions
in each list are dead arms, so a four-place ordering cannot be read here at
all. **What survives is a two-place ordering, and it does replicate:** `dom1b`
tralo 4.38 > alm 2.87; `dom1` tralo 12.46 > alm 10.98. Same direction on a
different backbone.

✅ **UNAFFECTED:** `tralo` is still the only arm above its OWN floor in every
`dom1b` cell (3/3, 2.49x) -- that is a `vs_null` / `vs_reseed` contrast and
touches no dead arm.

✅ **AND `loose1`'s MobileNet SUBSET IS A NEAR-EXACT REPLICATION OF `dom1`**:
`tralo` 12.74 items, 3/3 above floor, 1.42x, floor 8.95 -- against `dom1`'s
12.46, 4/4, 1.53x, floor 8.12. **Different campaign, different code version,
same two backbones, same answer.**

✅ **THE FOURTH BACKBONE AGREES WITH `dom1`, NOT WITH `equaldose1`.** RegNet
reads 2.49x in `dom1b` (fp16, `1d921173`) and 2.24x in `loose1` (bf16,
`74f85865`) -- two different GPUs, two different AMP regimes, two different
commits, reproducing independently. That is the `loosevit1`/ViTB16 regime
(2.8x), not `equaldose1`'s 0.68x.

🛑🛑 **THE INDEPENDENCE CORRECTION, AND IT REVISES 2(z21).** Cells at different
cap levels on the SAME backbone within a campaign **share one lambda=0 warm-up
model** -- confirmed directly: `tralo_reseed`'s dAP is literally constant across
caps within a backbone. So the 8 task cells are **not 8 independent units**.
The independent units are the 4 **(campaign, backbone)** pairs, and
`tralo - tralo_reseed` is positive in all four:

| unit | items |
|---|---|
| `dom1b` RegNetY400MF | +2.62 |
| `loose1` MobileNetV2 | +4.16 |
| `loose1` MobileNetV3 | +3.03 |
| `loose1` RegNetY400MF | +2.78 |

**4 of 4, sign p = 0.0625.** Counting the 8 cells instead gives 7/8 and
p = 0.035, and **that number is anticonservative and must not be quoted.**
This is the same clustering error as 2(z19)'s "the nulls' effective n is
cells / n_cap_levels", now shown to bite the TREATED arms too.

⛔ **NO SINGLE CELL IS SIGNIFICANT ON ITS OWN.** Per-seed paired
`tralo - floor` at 4 seeds gives t from -0.79 to +3.34; the per-cell paired sd
is **1.4 to 21.4 items**, so the SEM at 4 seeds is 0.7-10.7 against effects of
1.6-13.7.

⚠️ **AP DOES NOT CORROBORATE ccF1 ON RegNet.** In `dom1b` `tralo` is **0.97x
its floor on AP** (1/3) while `alm` leads at 1.42x (3/3). So the RegNet result
is an ALLOCATION-quality result at these caps, not a ranking result -- the two
channels disagree, and which one is quoted changes the winner.

⚠️ **TWO RATIO COLUMNS ARE UNREADABLE AND MUST NOT BE PRINTED AS RATIOS.**
Where the floor mean is near zero or negative (`loose1` dAP floor **-0.0060**,
`loose1` dmacroF1 floor -0.0005) the ratio is arithmetic garbage (-4.32x) and
only the sign count carries information. On that sign count `tralo` is above the floor on dAP in **5 of 5** `loose1`
cells -- but those five collapse to **3 independent (campaign, backbone)
units**, where the attainable floor is p = 0.25. 🛑 **Quote it as 3/3,
direction only.** Printing p = 0.031 off those five cells is the same
anticonservative count this section rules out 17 lines above.

✅ **A POSITIVE CHECK ON THE LAMBDA TOGGLE.** In `dom1b`, `alm_null`,
`fioretto_null`, `hounie_null` and `tralo_null` are **byte-identical in 12 of
12 groups**. At lambda=0 the family term vanishes and all four share one
warm-up and one allocator, so this is exactly right -- and it means `dom1b`'s
per-family nulls are nominally present but informationally degenerate
(`alm - alm_null` IS `alm - tralo_null`), and that `tralo_reseed` is a
legitimate shared floor for all four families there.

⚠️ Also byte-identical in `dom1b`: `cb_lp == clip == lp` and
`focal_clip == focal_lp`. The first is FRAMEWORK 2(x1)'s documented `cb_lp`
inertness; the rest are allocator-only siblings sharing a pre-allocator raw
file, which is expected and is why allocators must be compared on
`final_predictions.csv`. **All five trained arms and `tralo_reseed` are
pairwise distinct in every group.**

⛔⛔ **THIS PARAGRAPH IS WHERE THE DEFECT WAS SEEN AND WAVED THROUGH. IT READ,
UNTIL 2026-09-04:**

> ⚠️ `fioretto` and `hounie` attempt **28** constraint steps per run against
> `tralo`/`alm`/`tralo_uniform`'s **29** in both campaigns -- the familiar 3.4%
> gap. It is under `full_panel`'s 5-point refusal threshold so the comparison
> stands, but the duals are not at literally identical dose. 2(z19) closed this
> objection directly with `tralo_lam0`.

**Both conclusions are false, and the reasoning behind each is the lesson.**
(a) *"Under the refusal threshold, so the comparison stands"* -- a threshold
that does not fire is not a finding of equality. It is the absence of a
detector, and 2(z40) records that this exact argument let four campaigns run.
(b) *"2(z19) closed this objection with `tralo_lam0`"* -- `tralo_lam0` is
itself at 28.00 and is a dead arm, so it closed nothing. **The gap was written
down, correctly, in this very paragraph, and was then argued away twice.** The
arms are `dead_arms` in `dom1` and `dom1b`; the objection is OPEN; see 2(z19)
and 2(z43).

### (z24) 🛑🛑🛑 THE TASK WINDOW WAS A **MEAN OVER SEEDS THAT SPREAD 105 ITEMS**,
AND TWO OF THE CAMPAIGNS BEHIND 2(z23) SHARE **ONE** lambda=0 MODEL,
BYTE-IDENTICAL. THE INDEPENDENT-UNIT COUNT FALLS FROM **4 TO 3**, AND ViTB16
HAS **ZERO** STRICT TASK CELLS AT ANY CAP EVER RUN.

Found 2026-09-01 while re-deriving 2(z16)'s window on `loose1`'s own reference
arm instead of importing `iwc3`'s row. Three separate defects, each of which
alone would have been enough.

**1. THE STATISTIC WAS WRONG.** `scripts/task_window.sweep` averaged the
unconstrained hard count across runs and then applied `MIN_FORCED = 10` to
that mean. On iwildcam/MobileNetV3 the four lambda=0 seeds predict

    class 2:  278   329   354   383      mean 336, SPREAD 105 items

so at `L90_G95` (K=333) the mean says `forced = 3` and the cap reads "barely
binds", while the cap actually evicts **50 items in one seed and is entirely
slack in two others**. No seed resembles the mean. The same is true of every
backbone: the per-seed range is 60 to 106 items wide everywhere measured.

✅ **FIXED.** `task_window` now computes `forced` PER SEED, prints the range
and a `binds n/N` column, and emits a new verdict `** PARTIAL n/N **` where
the cap poses its question to some seeds only. `recommend()` and the
`-> TASK WINDOW` line accept `** TASK **` only. Gated in both directions:
per-seed `[5, 60, 60, 60]` must read PARTIAL 3/4 **and** the same numbers
passed as their own mean must read `** TASK **`, so the check cannot pass by
refusing everything.

⛔ **2. AND HERE IS THE INFERENCE THAT DOES NOT FOLLOW, TESTED AND REFUTED.**
The natural reading of a slack seed is "the penalty is `relu(hard - K) = 0`,
so the treated arm IS its own null there, so that seed dilutes the contrast to
zero." That is what this file believed on 2026-08-21. **It is false on
iwildcam.** md5 of `final_predictions_raw.csv`, dom1/MobileNetV3, 12
(cap, seed) pairs: `tralo` and `tralo_null` are **DISTINCT in 4 of 4 slack
seeds**, identical in none. Two structural reasons, both specific to this
dataset: the binding scope is the LOCAL per-group ceiling and **7 of its 14
are K = 0**, so a camera carrying any prediction of that class violates its
ceiling however slack the class TOTAL is; and the penalty reads SOFT counts,
which exceed K while the hard count does not. So `forced` is a statement about
the class total ONLY. Read PARTIAL as "the cap does not pose the same question
to every seed", never as "those seeds are free nulls".

🛑 **3. THE WINDOW WAS IMPORTED FROM ANOTHER CAMPAIGN'S MODEL.**
`configs/task_windows.yml` is keyed by (dataset, backbone), but a row is
measured from ONE campaign's unconstrained arm. Measured per campaign on
MobileNetV3 class 2, all on **the same four cached warm-up checkpoints**:

| campaign | lambda=0 count | window (per-seed rule) |
|---|---|---|
| `dom1`, `loose1` | **336** | 0.70 only |
| `equaldose1`, `iwc3` | **355** | 0.70 only |

The yml carries `iwc3`'s 355. At K=333 that is 22 evicted items against
`dom1`'s 3, i.e. the difference between a task and a non-task, and the warm-up
cache does NOT explain it: `base_model_id` is shared across all four
campaigns, so the divergence is in the 29 CE-only epochs after warm-up.
✅ `configs.task_cells.classify` now returns the row's `provenance` so a
caller can print whose model the verdict came from.

🛑🛑 **4. `dom1` AND `loose1` ARE NOT TWO CAMPAIGNS ON MobileNetV2.** Their
`tralo_null` raw predictions are **byte-identical in 8 of 8 (cap, seed)
pairs** (`7f1ff13ebc`, `1df6ab42f8`, `b51c30725d`, `7ab05f80c4`). 2(z23)'s
"`loose1`'s MobileNet subset is a near-exact replication of `dom1`" is not a
replication: it is the same model read twice.

#### The recount, on the strict per-seed rule

A cell is a TASK only where BOTH capped classes bind in EVERY seed.

| campaign / backbone | strict task cells | of |
|---|---|---|
| `dom1` / MobileNetV2 | **3** (L80, L90, L95_G80) | 3 |
| `dom1b` / RegNetY400MF | **2** (L80, L95_G80) | 3 |
| `loose1` / MobileNetV2 | 2 (L80, L90) | 2 |
| `equaldose1` / MobileNetV2 | **2** (L80, L95_G80) | 3 |
| `dom1` / MobileNetV3 | 0 | 3 |
| `loose1` / MobileNetV3 | 0 | 2 |
| `loose1` / RegNetY400MF | 0 | 2 |
| `loosevit1` / **ViTB16** | **0** | 2 |
| `equaldose1` / MobileNetV3 | 0 | 3 |

⇒ **3 distinct lambda=0 models carry every strict task cell in the project**:
`dom1`+`loose1`/MobileNetV2 (one model), `dom1b`/RegNetY400MF, and
`equaldose1`/MobileNetV2. 2(z23)'s "4 of 4 independent units, sign
p = 0.0625" is **3 of 3, p = 0.125**, and two of the three are the same
backbone. 2(z21)'s 4 task cells are 3, all MobileNetV2, so its 2 units are 1.

✅ **WHAT SURVIVES, AND IT IS NOT NOTHING.** Every sign is unchanged: `tralo`
is above its own reseed floor in all three surviving units, and the arm
ordering `tralo > alm ~ fioretto > hounie` is untouched because it was never
computed from the window. The dilution in a PARTIAL cell biases a measured
effect TOWARD zero, so the positive readings are conservative rather than
inflated. What falls is the COUNT of independent units and therefore the
p-value, which was already only 0.0625.

🛑 **AND THE HEADLINE BACKBONE HAS NEVER RUN A TASK CELL.** ViTB16's per-seed
windows are class 2 `0.70` and class 7 `0.90`, which **do not overlap**, so no
single-fraction tag can express one. Both caps ever run there (`L80_G95`,
`L90_G95`) are PARTIAL 6/8 on class 2. The tag that works is the per-class
form **`L70-90_G95`**, which is exactly what `taskwin2` is running on
MobileNetV3 and what `vittask1` must run on ViTB16.


### (z11) 🔴🔴🔴 AT THE ITEM LEVEL THE CONSTRAINT IS AT THE RNG FLOOR, AND
`tralo_uniform` IS BELOW IT IN BOTH REGIMES

Measured 2026-08-31, `boundary_probe --control tralo_null`, so every arm is
against its OWN lambda=0 twin with the warm-up, seed and allocator held fixed.
**This is the control that decides 2(z10), and it was run second.**

| regime | arm | swapped | net items |
|---|---|---|---|
| tight | **`tralo_reseed` (pure RNG)** | 3357 | **+89** |
| tight | `tralo_head` | 3362 | +11 |
| tight | `tralo_uniform` | 3534 | **-43** |
| tight | `tralo` | 3647 | **-189** |
| loose | `alm` | 1749 | **+255** |
| loose | `tralo` | 1662 | **+221** |
| loose | **`tralo_reseed` (pure RNG)** | 1576 | **+167** |
| loose | `tralo_uniform` | 1683 | **+148** |

🔴 **THE SWAP COUNT IS NOISE.** A pure RNG reseed moves **3357** items where
the constraint arms move 3362-3647. The constraint relocates no more of the
selection than re-rolling the seed does. This is the count-level result
(75-95 RMS constrained vs 83-95 from a reseed) now confirmed at the level of
WHICH ITEMS are chosen, which is the level that decides the metric.

🔴 **`tralo_uniform` IS BELOW THE RNG FLOOR IN BOTH REGIMES**: -43 against
+89 at tight, +148 against +167 at loose. `tralo` clears the floor at loose
(+221) and is far below at tight (-189). **`alm` is the only arm clearly above
the floor anywhere** (+255, loose).

⚠️ **AND IT REFUTES THE ORDER-PRESERVATION PREMISE.** `tralo_uniform`'s whole
design claim is that a uniform gradient in log-odds is a pure BIAS SHIFT and so
cannot reorder. If that held, it would swap ~0 items against its own twin. It
swaps **3534**. The cause is almost certainly 2(prm-grad): **`prm.grad` is not
the delivery mechanism, Adam is.** A gradient uniform across items does not stay
uniform after `sqrt(v)` per-parameter scaling and momentum mixing with CE --
exactly as `ortho_project` delivered 0.0% of its promised CE-neutrality in
16/16 conditions. The uniform count is an intervention on `prm.grad`, and this
project has already measured that such interventions mostly do not survive.

⇒ **VERIFY `tralo_uniform` AT THE WEIGHT-DELTA LEVEL** before any further
claim rests on its order-preservation story. `scripts/ortho_survival.py` is the
tool and it needs no artefact.

⚠️ **WHAT THIS DOES NOT OVERTURN.** `tralo_uniform` still beats `clip` on AP
and AUROC 25/29 (2(z7)), and that contrast is against a DIFFERENT model at
equal compute, which is a different question from this one. Both are true: it
is better than the post-hoc bar, and its item-level movement against its own
twin does not clear the RNG floor. Quote whichever question is being asked, and
never one as evidence for the other.

⚠️ **LIMITS OF THIS TABLE.** Pooled item counts over cells and seeds, not a
paired significance test, and `tralo_reseed` is ONE alternative RNG draw per
seed rather than the floor's distribution -- so its +89 is a sample, not a mean.
The robust part is the MAGNITUDE agreement (3357 vs 3362-3647), which does not
depend on the draw.

### (z10) 🟢🟢 THE CONSTRAINT DOES RESHAPE THE SELECTION -- AND WHY IT INVERTS

Measured 2026-08-31 with `scripts/boundary_probe.py`, 288 cell-seed-class
comparisons, 12,362 individual item swaps. **This is the mechanism behind
2(z8)'s reversal, and it is not a correlation.**

Both arms deploy EXACTLY K (2(z9)), so the count is uninformative by
construction and the only question is WHICH K items. With `evicted` = clipper's
selection minus the arm's, and `admitted` = the reverse (always equal in size):

| regime | arm | swapped | prec ADMITTED | prec EVICTED | **net items** | median depth |
|---|---|---|---|---|---|---|
| tight | `tralo_uniform` | 3649 | 0.923 | **0.996** | **-265** | 85 |
| tight | `tralo` | 3710 | 0.885 | **0.996** | **-411** | 89 |
| loose | `tralo_uniform` | 1599 | 0.860 | 0.846 | **+22** | 84 |
| loose | `tralo` | 1669 | 0.893 | 0.836 | **+95** | 84 |
| loose | `alm` | 1735 | 0.915 | 0.840 | **+129** | 90 |

✅ **THE CONSTRAINT IS NOT INERT AND NOT COSMETIC.** It reaches a median
**84-90 ranks BELOW the clipper's cut** in every condition, and some crossings
come from 400+ ranks down. It really does pull in items the clipper rejected.

🔑 **THE INVERSION HAS ONE CAUSE: HOW GOOD THE CLIPPER'S SET ALREADY IS.**
At tight caps the clipper's selected items are **99.6% correct**, so there is
nothing left to win and every swap trades a true positive for something worse.
At loose caps the clipper's set is only **~84% correct**, so there is real room
and the admitted items are BETTER than those they displace.

⇒ So the reversal is NOT a property of the cap. The damage appears only when
the clipper had nothing left to give; the benefit only when its selection was
diluted enough to improve on. This also explains the ORDERING inside each
regime: at loose the arm that reaches hardest wins (`alm` +129 > `tralo` +95 >
`tralo_uniform` +22), and at tight the same aggression is what costs
(`tralo` -411 worse than `tralo_uniform` -265).

🛑 **AND IT CONNECTS TO THE CEILING RESULT.** 2(headroom) measured the prize
from `clip` to a PERFECT allocator at 0.0-1.0 items in 4 of 6 tight cells,
because ccP is already 0.9954. This is the same fact seen from the other side,
now with the swap-level receipt: at tight K/n the top-K is nearly all true
positives, so no method can win there, whatever its loss shape.

⚠️ **Quote `net` beside the swap count, NEVER the swap count alone.** A pure
RNG reseed moves 63 items for a net of +0.38. A large swap count is what noise
looks like.

### (z9) ✅ THE BUDGET IS EQUALIZED, BOTH IN THE SCORER AND AS DEPLOYED -- RECEIPT

The corpus mistake was that cc-F1 was **partly a budget measurement**:
`corr(budget, d ccF1)` was **+0.81** on `hounie`, and matching the budget cut
TraLO's head-to-heads 3-4x. The hinge ablation had the same defect from the
other side -- the hinge arm emitted **16.3% more predictions in 24/24 pairs**,
so part of its +3.23 pp was free fill. Verified 2026-08-31 that neither can
happen in the current pipeline. Both halves are checked because they are
different mechanisms.

**1. THE SCORER.** `full_panel.panel()` re-derives its own allocation with
`equalize_multi`, which walks every (item, capped class) pair in descending
probability and assigns while the class has global AND local room. Empirical
proof rather than the comment: `items_per_001` is derived from
`(eq == c).sum() + (y == c).sum()`, so it is constant across arms if and only
if every arm emitted the same count. Across **all 44 cell groups** in the live
corpus it is identical for every arm, **max spread 0.0000000000**.

**2. AS DEPLOYED.** `final_predictions.csv`, dom1 / MobileNetV3 / L80_G95,
seed 1, all 16 arms. K = 296 for class 2 and 364 for class 7:

| arm | deployed c2 | deployed c7 | RAW c2/c7 before the allocator |
|---|---|---|---|
| alm | 296 | 364 | **391**/442  (over -> clipped DOWN) |
| ~~fioretto~~ ⛔ **DEAD ARM** | ~~296~~ | ~~364~~ | ~~387/430~~ |
| tralo | 296 | 364 | 346/471 |
| clip / lp / cb_lp | 296 | 364 | 318/507 |
| tralo_reseed | 296 | 364 | 304/520 |
| la_lp | 296 | 364 | 297/530 |
| ~~hounie~~ ⛔ **DEAD ARM** | ~~296~~ | ~~364~~ | ~~288/437  (UNDER -> filled UP)~~ |
| tralo_uniform | 296 | 364 | **281**/546  (UNDER -> filled UP) |
| the four nulls | 296 | 364 | 278/550 |

⇒ **16 of 16 arms deploy an identical budget** while their raw counts span
278-391 on class 2 and 430-550 on class 7. The fill works in BOTH directions:
an arm over budget is clipped down, and an arm that finishes SHORT is topped up
to exactly K by the same allocator. `posthoc_adjustment` phase 2 runs with
`force_exact=True`, and `heuristic/train.py` pass 1 walks every (item, capped
class) pair, which is why undershoot is filled rather than left short.

⇒ So no ccF1, macroF1, uncF1 or acc number in the live corpus can be a budget
measurement, and `raw_over_K` is the only column that reads the pre-allocator
count. AP and AUROC are allocation-free and never touched the budget at all.

⚠️ **This is a property of the SCORER and the PIPELINE, not of the arms.** It
is exactly why `full_panel` is allocator-blind (2(x1)) and why two arms sharing
a warm-up read `+0.0000` on every budget-equalized metric. An allocator
comparison must use `final_predictions.csv`, never the panel.

### (z8) 🟢 THE COUNT-FUNCTION REVERSAL -- GATED, DEDUPLICATED, AND WHAT IT IS NOT

Measured 2026-08-31 from the 372-cell table, then RE-DERIVED after running the
integrity gates, which changed the numbers. Both versions are kept here because
the first was quoted before the gates were run and that is the error to learn.

**THE GATES, run on all six source campaigns.**

| gate | result |
|---|---|
| dose | **100%** on every trained arm, and `tralo`/`tralo_uniform` ATTEMPT the same steps in every campaign (1044, 696, 348, 232), so the contrast is dose-matched |
| `check_parity` | green: 30 optimizer epochs every arm, matched lr / lr_constraint / dropout / batch / pretrained |
| `code_version` | ONE per campaign, none split. Three distinct commits ACROSS campaigns, but every delta is computed WITHIN one campaign, so no delta crosses a commit |
| saturation | warm-up acc 0.961-0.964, constraint epochs add +0.034-0.036 -> **NOT the saturated signature**. The live regime |
| terminal collapse | none. Final acc 0.9951-0.9995 on every arm in every campaign |
| inert flag | `tralo` vs `tralo_uniform` raw md5s **differ in every cell checked**. The `soft_count_mode` flag is LIVE, not a fifth inert one |

🐛 **DUPLICATION FOUND, AND IT CHANGED THE RESULT.** `dom1` and `loose1`
hold **byte-identical runs** on the cells they share -- md5 of
`final_predictions_raw.csv` matches on 8 of 8 checked (MobileNetV2 and V3 x
L80_G95 and L90_G95 x both arms). Counting both inflated the loose-side n from
13 to 17. Deduplicated:

| cap | K (cls 2, n=370) | K/n | dAP | dAUROC | AP wins |
|---|---|---|---|---|---|
| L20_G50 | 74 | 0.20 | +0.0579 | +0.0139 | 4/4 |
| L30_G50 | 111 | 0.30 | +0.0842 | +0.0206 | 4/4 |
| L50_G30 | 111 *(global binds)* | 0.30 | +0.1110 | +0.0270 | 4/4 |
| **L60_G95** | **222** | **0.60** | **never run** | | |
| L80_G95 | 296 | 0.80 | -0.0137 | -0.0010 | 1/5 |
| L95_G80 | 296 *(global binds)* | 0.80 | -0.0164 | -0.0020 | 1/3 |
| L90_G95 | 333 | 0.90 | -0.0231 | -0.0036 | **0/5** |

**TIGHT 12/12 positive, mean +0.0844. LOOSE 2/13 positive, mean -0.0179.**

⚠️ **A CORRECTED SIGNIFICANCE CLAIM.** `L90_G95` was written here as
"0/7, p=0.016". After dedup it is **0/5, p=0.0625**, which does NOT reach 0.05.
No individual cap level is significant.

🛑🛑 **AND THE CELL IS NOT THE INDEPENDENT UNIT (2(z)).** The three cap
tags within a (model, seed) SHARE ONE WARM-UP, so 12 tight "cells" are 4
backbones x 3 correlated replicates, not 12 draws. On the correct unit the
tight result is **4 of 4 backbones, exact sign floor p = 2/2^4 = 0.125**, and
the loose result is 5 units. **Neither reaches p < 0.05 on the honest unit.**
What is defensible is a UNANIMOUS DIRECTION with a large effect, not a
significant one. This is the same trap that evaporated 8 of 9 dom1 sweeps.

⚠️ **ONE CELL FLIPS WITH THE NUMERIC REGIME.** RegNetY400MF at L80_G95 reads
**+0.0074 under fp16 (`dom1b`)** and **-0.0283 under bf16 (`loose1`)** -- same
backbone, same cap, opposite sign. It is one of the two positive loose cells.
Do not treat the loose side as cleanly unanimous.

🔑 **WHAT `log_health` SHOWS, AND IT REFRAMES THE MECHANISM.** The caps are
**never satisfied during training at ANY cap level** (`satisfied 0/N` in every
campaign). Satisfaction is not the variable. The DISTANCE to K is:

* **LOOSE:** `tralo` ends class 2 at **396 against K=352** and class 7 at
  **425 against K=433** -- at or already UNDER budget. There is nearly no work
  to do, so `sum`'s targeted push does no damage.
* **TIGHT:** class 2 ends at **370 against K=185**, twice the budget, never
  converging. The constraint must move the count enormously, and that is where
  `sum`'s `p(1-p)` weighting reorders destructively while `uniform` cannot.

So the reversal is not "uniform is better at tight caps" as a property of the
cap. It is that the DAMAGE only materialises when the constraint has real work
to do. ⚠️ Note also that the NULL's count slope is steeper than the
constraint's (-0.70/ep against -0.33/ep at tight), so CE drift moves the count
more than the constraint does -- no count trajectory is attributable without
its twin.

✅ **The crossover is still bracketed at 0.30 < K/n < 0.80 with nothing run
inside it**, and `vitdom2`'s `L60_G95` (K/n = 0.60) remains the right first
interior probe.

⚠️ **A SCOPE EFFECT MAY BE HIDING HERE AND IS NOT A RESULT.** `L30_G50` and
`L50_G30` impose the same K=111 through different scopes and read +0.0842 vs
+0.1110; the same-K pair at the loose end agrees closely (-0.0137 vs -0.0164).
Four and three cells. Well inside the noise. Recorded only because the same-K
pairs are the cheapest scope test available and nobody has used them.

🛑 **THIS REMAINS THE ARGUMENT FOR THE CUT-WINDOWED COUNT.** `tralo_margin`
is implemented and passes `smoke_arms --matrix` and has never been run
anywhere. But WHERE to window depends on where the crossover is, so it waits on
`L60_G95`.

### (z7) 🛑🛑🛑 `tralo` DOES NOT BEAT THE CLIPPER. `tralo_uniform` DOES.

**Measured 2026-08-31 across ALL 372 live cells, 1,340 runs, 4 backbones, both
cap regimes (`scripts/cell_table.py`). This is the first all-backbone
all-regime head-to-head and it settles the count-function question.**

`tralo` (the shipped `sum` count) vs `clip`, 44 paired cells:

| metric | wins/n | mean d | p | verdict |
|---|---|---|---|---|
| ccF1 | 22/44 | +0.0027 | 1.000 | **TIE** |
| macroF1 | 23/44 | -0.0005 | 0.880 | tie |
| AP | 21/44 | **-0.0182** | 0.880 | **LOSS** |
| AUROC | 20/44 | **-0.0048** | 0.652 | **LOSS** |
| acc | 21/44 | -0.0021 | 0.880 | LOSS |

⇒ **the basic assumption is FALSE as stated.** Pooled over the regimes
`tralo` does not beat the post-hoc bar on any metric.

**It is entirely a REGIME split, and the tight half is a rout:**

| | ccF1 | AP | AUROC |
|---|---|---|---|
| TIGHT (21 cells) | **2/21, p=0.0002** | **0/21, p<1e-4** | **0/21, p<1e-4** |
| LOOSE (23 cells) | 20/23, p=0.0005 | 21/23, p=0.0001 | 20/23, p=0.0005 |

`tralo` loses AP in **0 of 21** tight cells and wins in **21 of 23** loose ones.
The same arm, the same code, opposite verdicts at three-star significance.

**`tralo_uniform` vs `clip`, 29 paired cells -- THIS is the arm that wins:**

| metric | wins/n | mean d | p |
|---|---|---|---|
| macroF1 | 22/29 | +0.0059 | **0.0081** |
| uncF1 | 22/29 | +0.0071 | **0.0081** |
| AP | **25/29** | +0.0163 | **0.0001** |
| AUROC | **25/29** | +0.0047 | **0.0001** |
| ccF1 | 17/29 | +0.0021 | 0.458 (tie) |
| acc | 13/29 | +0.0013 | 0.711 (tie) |

⇒ **`tralo_uniform` beats the clipper on both RANKING metrics and on both
uncapped-class metrics, and ties on ccF1 and acc. It never loses.** AP and
AUROC are the only two metrics that can change a top-K set, so this is the
family in which a win is possible at all.

🔑 **AND IT SURVIVES THE NOISE FLOOR, WHERE `tralo` DOES NOT.**

| contrast | ccF1 | AUROC |
|---|---|---|
| `tralo` - own null | 28/44, p=0.096 | 22/44, p=1.000 |
| **RNG reseed** - same null | **29/44, p=0.049** | 20/44, p=0.652 |
| `tralo_uniform` - own null | 21/29, p=0.024 | **23/29, p=0.0023** |

A pure RNG reseed wins ccF1 in **more** cells (29/44) than `tralo` does
(28/44). So **TraLO's ccF1 gain over its own twin is at or below the reseed
floor and is not attributable.** But the floor does NOT move AUROC (20/44,
p=0.65) while `tralo_uniform` does (23/29, p=0.0023) -- so **the uniform arm's
RANKING gain is the one attributable effect in the corpus.**

✅ **The collateral damage is gone too.** `tralo` - null costs uncF1 in 12/44
cells (p=0.0037) and acc in 13/44 (p=0.0096). `tralo_uniform` - null is 14/29
on both, a clean tie. The uniform count removes the damage at 4-backbone scale,
confirming 2(u2) beyond the single campaign it was found on.

⚠️ **The one thing `sum` still wins is LOOSE caps**, head to head against
uniform: ccF1 13/17 (p=0.049), AP 15/17 (p=0.0023), AUROC 15/17. At TIGHT the
same contrast is AP **0/12** and AUROC **0/12** (p=0.0005) the other way. Both
directions are three-star. The count function is regime-dependent and neither
mode is right everywhere -- which is the argument for the CUT-WINDOWED count
(`tralo_margin`), never run anywhere.

**The rival duals, for scale** (15 cells, no ViTB16): `alm` beats `clip` on
ccF1 12/15 (p=0.035) and AP 12/15 (p=0.035). `tralo` vs `alm` head to head is
9/15 on every metric, p=0.61 -- **a tie, on CNNs only.** Both survive: `alm` is
at 29.00 steps and the `clip` contrast is at equal dose.

⛔ **THIS ALSO SAID "`fioretto` and `hounie` are weaker and both LOSE uncF1"
UNTIL 2026-09-04.** Both are dead arms in all 15 of these cells, so that half
is struck -- not reversed, but no longer sayable from this corpus.

⚠️⚠️ **AND THE DOMINANCE HALF IS NOT SETTLED, BECAUSE IT HAS ONLY EVER
BEEN MEASURED IN `uniform`'s WORST REGIME.** Checked 2026-08-31, and it
qualifies everything above:

| contrast | cells | AP | verdict |
|---|---|---|---|
| `tralo_uniform` - `alm` | **9, ALL LOOSE** | **1/9, p=0.039** | **LOSS** |
| ~~`tralo_uniform` - `hounie`~~ ⛔ **DEAD ARM 2026-09-04** | ~~9, ALL LOOSE~~ | ~~1/9, p=0.039~~ | ~~LOSS~~ |
| `tralo` - `alm` | 15, all loose | 9/15, p=0.61 | tie |
| ~~`tralo` - `hounie`~~ ⛔ **DEAD ARM 2026-09-04** | ~~15, all loose~~ | ~~11/15, p=0.12~~ | ~~direction only~~ |

**Every cell in which `tralo_uniform` has ever met a rival dual is a LOOSE
cap** (dom1, dom1b and equaldose1 are `L80_G95 / L90_G95 / L95_G80` and nothing
else). Loose is exactly where `sum` beats `uniform` head to head, AP 15/17
p=0.0023. So `tralo_uniform` has been compared against the duals **only in the
regime its own count function is known to lose**, and the p=0.039 is
confounded with that, not a verdict on the arm.

⇒ **Two different claims, two different evidence bases, and they must be kept
apart:**

* **vs the CLIPPER: decided.** 29 cells spanning BOTH regimes and all four
  backbones, `tralo_uniform` wins AP and AUROC 25/29 (p=0.0001) and never
  loses a metric. This is the claim FRAMEWORK 2(z7) rests on.
* **vs the RIVAL DUALS: OPEN, and currently pointing the wrong way.** 9 loose
  cells, no tight, no mid, no ViTB16. `tralo` looks better there and
  `tralo_uniform` looks worse, and the regime confound explains the sign.

🛑 So the consolidation is a decision about the DESIGN SPACE (stop adding
count variants, `tralo_uniform` is the default), NOT yet a decision about which
arm carries the dominance claim. `vitdom2` carries BOTH `tralo_uniform` and
`tralo` across tight, MID and loose against all three duals for exactly this
reason: it is the first campaign in which the regime and the rival vary
independently. Do not retire `tralo` until it lands.

🛑 **CONSOLIDATION DECISION, 2026-08-31: `tralo_uniform` IS TraLO.** It is
the only variant that beats the clipper without ever losing, and the only one
whose gain clears its own reseed floor. `tralo` (sum) is retained as the
loose-cap comparison and as the ablation that shows what the count function is
worth. Everything else in the family is a guardrail or is retired: `tralo_st`,
`tralo_ortho`, `tralo_head` and `tralo_coin` are controls that answered their
question, and `tralo_margin` is the ONE remaining design step.

### (z5) 🔴🔴 dom1b: THE RANKING-METRIC LEAD DOES NOT REPRODUCE ON RegNetY400MF

**Scored 2026-08-30, 192/192** (`check_parity` OK on one commit, no inert arm,
**0 of 136 runs show a terminal collapse**, `constraint_fp32: true`).

⛔ **"all gates green ... dose 100% on all five trained arms" IS STRUCK
2026-09-04.** `dom1b` is PARTIALLY QUARANTINED and its `fioretto`/`hounie` run
at **28.00** steps against 29.00 (2(z40)); the 100% was computed within each
arm and cannot see that.

| metric | `tralo` rank, dom1 | rank, dom1b | vs its OWN reseed floor |
|---|---|---|---|
| ccF1 | **1 of 3** (+0.0141) | **1 of 3** (+0.0058) | 2.49x the floor ✅ |
| AP | **1 of 3** (+0.0371) | **2 of 3**, behind `alm` (+0.0314 vs +0.0458) | **0.97x -- BELOW the floor** ❌ |
| AUROC | **1 of 3** (+0.0106) | **2 of 3**, behind `alm` (+0.0044 vs +0.0069) | **0.77x -- BELOW the floor** ❌ |

⛔ **THE RANK COLUMN READ "rank of 5", WITH dom1b AP **4** AND AUROC **3**,
UNTIL 2026-09-04.** Two of those five arms are dead here, so a rank out of five
is not available; the surviving field is `tralo` / `alm` / `tralo_uniform`.

✅ **THE CONCLUSION IS UNCHANGED AND IS CARRIED ENTIRELY BY `alm`, WHICH IS
LIVE.** `the ccF1 lead reproduces; the ranking lead does not.` `alm` still
takes both ranking metrics on RegNet (+0.0458 vs +0.0314 AP, +0.0069 vs +0.0044
AUROC), and on both `tralo`'s gain over its own lambda=0 twin is smaller than a
pure RNG reseed buys -- a `vs_null` / `vs_reseed` reading that touches no dead
arm. **The ranking channel is the only one a top-K allocator can see**, so this
is the channel that mattered. Only the DENOMINATOR was wrong, not the finding.

⚠️ **BACKBONE AND NUMERIC REGIME CHANGED TOGETHER.** dom1 ran Blackwell
bfloat16 with no GradScaler; dom1b ran Quadro float16 + GradScaler. Same
commit, same 100% dose, same `constraint_fp32`. So dom1b **cannot** be pooled
with dom1, and a dom1-vs-dom1b difference cannot be assigned to the backbone or
to the regime alone. Write it as *"not reproduced on RegNetY400MF under a
different numeric regime"*, never as *"backbone-specific"*.

🛑 **NOTHING IN dom1b IS SIGNIFICANT AND NOTHING COULD BE.** One backbone x
4 seeds = **4 warm-up units**, exact sign floor **p = 0.125**; at the 3 cap
tags it is p = 0.25. Every RESOLUTION block reads UNDERPOWERED -- `tralo` vs
its null needs **~24 seeds per cell** against 4 present. Every tie here is an
absence of measurement, not a null.

**The item decomposition, which is the useful part.** `tralo` - `clip` =
**+8.44 items**, of which **+4.06 is compute** (the lambda=0 twin at equal
epochs) and **+1.76 more is RNG** (reseed - null), leaving **+2.62 items for
the constraint term** -- against a within-cell paired seed sd of 6.56-7.52
items, i.e. **0.35-0.40 sd**.

**macroF1 / uncF1.** All 15 arm x cell combinations are negative on macroF1
(`tralo` -0.0081, 0/3; uncF1 -0.0127, 0/3). ⚠️ **But the reseed floor is MORE
negative** (-0.0098 / -0.0138), so on RegNet the macro damage is **inside the
RNG floor and is not attributable to the constraint**. Against `clip`, `tralo`
reads macroF1 +0.0078 while **`tralo_null` at equal compute reads +0.0159** --
the lambda=0 twin BEATS the constrained arm by +0.0081. The macroF1 gain over
the clipper is compute, and the constraint subtracts from it. Third backbone
to say so.

🐛 **`base_model_id` COLLIDES ACROSS CAMPAIGNS** -- identical in 24/24 shared
(arm, seed) keys between dom1b and loose1. It does not collide in fact, for two
independent reasons: `get_cache_path` roots the cache per WORKTREE, and
`load_from_cache` refuses a cache trained under a different AMP regime. The
weight md5s differ and 0 of 48 predictions match. **But the key alone would not
have protected a same-worktree, same-AMP relaunch.** Do not rely on
`base_model_id` to prove two campaigns trained different models; md5 the
weights.

### (z6) ⚠️ "FIRST OF FIVE, 6/6 CELLS" IS A POOLED MEAN, NOT A PER-CELL SWEEP

Corrected 2026-08-30. dom1's headline was quoted here and to Roei as `tralo`
first of five on ccF1/AP/AUROC "6/6 cells each". That conflates two things:

* **`tralo` is first on the POOLED MEAN delta** for all three metrics. True.
* **"6/6 cells" is the sign test of `tralo` vs its own null** -- positive in
  6 of 6 cells. Also true, and a different statement.
* **On absolute per-cell RANK among the five trained arms, dom1's `tralo`
  takes only 2/6 firsts on ccF1, 2/6 on AP and 3/6 on AUROC.** dom1b's takes
  2/3, 0/3, 0/3.

⇒ "first of five in 6/6 cells" was never measured and must not be written.
Say which of the three statements is meant, every time.

🐛 **AND `full_panel` COMPUTED `uncF1` WITHOUT EVER PRINTING IT.** It was in
the frame from line ~413 and absent from `GROUPS`, so the standing rule "read
uncF1 beside macroF1" could not be satisfied from the tool's own output and
every uncapped-damage number had to be recomputed by hand. Fixed 2026-08-30:
`uncF1` is now in the printed group and in `EQ_RESOLUTION`. A metric that is
computed and not printed is worse than a missing one -- it looks covered.

### (z2) 🔴🔴🔴 THE MANUSCRIPT CLAIMS NOTHING ON iwildcam

**Audited 2026-08-30. `grep -ril iwildcam docs/paper/` returns ZERO files.**
Not the `.tex` sources, not `tables/`, not the figure generators, not
`corpus_final.csv`. The manuscript names DermMNIST (31), OctMNIST (58),
TissueMNIST (18) and HAM10000 (11), at **warm-up 50**, with **no lambda=0
twins**.

Meanwhile `data/` on every server worktree holds **only iwildcam**, all three
MedMNIST datasets are removed from disk, and every campaign run since
2026-08-21 is iwildcam at **warm-up 1**.

⇒ **the paper and the evidence base share no dataset, no warm-up regime and no
controls.** Every iwildcam result -- `dom1`, `loose1`, `uniform1`, `vitu1`,
`loosevit1`, `iwc4` -- currently has **no destination**, and every paper claim
rests on data that is quarantined or deleted. This is the largest single fact
about the project's state and it contains every other gap below.

It is not, by itself, an argument for either direction. The two coherent
resolutions are: rewrite the empirical section around iwildcam (in which case
the corpus tables go, and they cannot be rebuilt), or restore a MedMNIST
dataset that survives the leakage audit (octmnist keeps MedMNIST's official
split and was CLEAN; dermmnist leaked 38.7%). **Deciding this is Roei's call
and it should be made before the next campaign, not after.**

### (z3) 🛑 TRALO GETS 29 CONSTRAINT STEPS, FIORETTO AND HOUNIE GET 28

**Measured 2026-08-30 on `dom1`, 24 runs per arm, and verified at the GRADIENT
level -- this is a real dose difference, not an accounting one.**

| arm | attempted/run | lambda at epoch 1 | logged grad norm, epoch 1 |
|---|---|---|---|
| `tralo` | **29** | **0.06** | **3.09** -- a real step |
| `alm` | **29** | mu0 > 0 | **6426.97** -- a real step |
| `fioretto` | **28** | 0 | **0.0** -- no step |
| `hounie` | **28** | 0 | **0.0** -- no step |

All four are configured `constraint_epochs: 29`, and **`dose_landed` printed
100.0% for all four**, because that figure is `applied / attempted` WITHIN an
arm and is structurally blind to a cross-arm gap. Fixed 2026-08-30: it now
prints a CROSS-ARM ATTEMPTS PER RUN block.

**The cause.** The subgradient duals guard their step on `has_work` -- spelled
`has_active` in `hounie_rcl/train.py:181`, which is also the one dual whose
zero lambda init is hard-coded in Python with no `protocol.yml` knob -- (is any
lambda > 0) and perform the dual update at the END of the epoch, so their first
constraint epoch does nothing. TraLO guards on `has_constraint` and initialises
lambda to **0.06**.

⚠️ **This is faithful to the published algorithms** -- `lambda^0 = 0` is what
subgradient dual ascent specifies -- so it is a property of the METHODS, not a
handicap this harness imposes. Do not "fix" it by hacking a baseline. But it is
a **1-in-29 = 3.4% dose advantage to TraLO in every head-to-head ever run
here**, it comes from a hyperparameter WE chose, and a reviewer will ask
whether the win is the method or the head start.

⇒ **State it in every dominance claim**, and run the clean test: a `tralo` arm
with `lambda_init = 0`, which takes 28 effective steps like the rivals. If the
win survives that, it is not the head start.

### (z4) COVERAGE, AUDITED -- what has never been measured

Audited 2026-08-30 over 2,095 configs; 1,816 completed in live campaigns.

| hole | what it undermines |
|---|---|
| **ViTB16 x any rival dual = 0 runs anywhere** | the dominance claim on the pre-registered headline backbone |
| `alm` / `lp` at TIGHT exist only in `iwc1` (now quarantined); `alm_null` at TIGHT is 0 runs everywhere | any tight-cap dual comparison |
| `tralo_st`, `tralo_margin`, `tralo_coin`, `tralo_ortho` never run anywhere | the whole count-function 2x2 |
| **MID caps (L55-L70) on iwildcam: 0 runs** | the tight/loose reversal is measured only at its endpoints |
| `tralo_head` never at LOOSE, never on ViTB16 | -- |
| `tralo_bounded` is a compiled Table 1 column whose code is DELETED | that table row is unreproducible |

Two more identity findings, both by md5:
* **`cb_lp` is byte-identical to `lp`** in 24/24 `dom1` and 10/10 `dom1b` cells,
  RAW and DEPLOYED. (2(x1) recorded it as identical to `clip`; it is identical
  to `lp`, which is the same conclusion about the arm and a different sentence.)
* **`dom1`'s four lambda=0 nulls are byte-identical to each other**, so its 16
  arms yield only 10 byte-distinct raw prediction sets per cell. A future
  campaign needs ONE null, not four -- which is 3 arms x 4 seeds x n_caps of
  free GPU time.

### (z) THE INDEPENDENT UNIT IS (model, seed), NOT THE CELL -- 8 of 9 dom1 significances evaporate

**Measured 2026-08-30 on `dom1`.** A lambda=0 twin's raw predictions are
**byte-identical across cap tags** (md5, 4/4 seeds, both backbones) -- correct by
construction, since at lambda=0 the cap touches only the allocator. So the three
cap levels within one (model, seed) share ONE control, and counting them as three
independent cells triple-counts the control.

The correct independent unit is **(model, seed) = 8**, with the caps as
correlated replicates. `full_panel` states this itself in its parity block and it
was not being read.

| result on dom1 | as 6 cells | at n=8 | verdict |
|---|---|---|---|
| RAW capped, tralo-null | p=0.031 | 7/8, p=0.0703 | evaporates |
| ALLOC capped, tralo-null | p=0.031 | 7/8, p=0.0703 | evaporates |
| ALLOC c2 / c7, tralo-null | p=0.031 | 7/8, p=0.0703 | evaporates |
| RAW capped, tralo-reseed | p=0.031 | 7/8, p=0.0703 | evaporates |
| LP-minus-topK allocator cost | p=0.031 | 2/8, p=0.2891 | evaporates |
| **ALLOC class 4, tralo-null, -0.0411** | p=0.031 | **0/8, p=0.0078** | **SURVIVES** |

⚠️ Class 3 is also 0/8 (p=0.0078) at a mean of **-0.0005** -- perfect sign
consistency at numerically zero magnitude. Sign without magnitude is not an
effect; quote both.

🛑 **Every "6/6 cells" ever reported on a 3-cap campaign must be restated this
way.** The claim in 3(0) that dom1's sweeps were bare-but-real was too generous.

### (z1) THE RAW CAPPED GAIN IS REAL, AND THE ALLOCATOR KEEPS 35% OF IT

Two corrections to how dom1 was read, both measured 2026-08-30.

**The raw gain is not a budget artifact, but it is smaller than argmax says.**
`final_predictions_raw.csv` is a plain argmax and is NOT budget-equalized, so the
first reading (+0.0436 capped ccF1, 6/6) mixed quality with count. Re-scored with
both arms taking top-K at the SAME K: **+0.0363, 5/6, p=0.219**. So 83% is
quality -- corroborated directly by **+30 true positives at an identical K=380**,
which no arithmetic can manufacture -- but the sweep does not survive, and at the
correct n=8 nothing here is callable.

**The constraint does NOT simply push counts down.** It suppresses class 7 by
**-31.9 items** (0/6) and *inflates* class 2 by **+19.4** (5/6): it redistributes
softmax mass rather than shrinking it. The run configs corroborate this
independently (`reordering.class_2.soft_before 337.0 -> soft_after 356.6`).
⇒ never describe the constraint as "reducing the count" without naming the class.

**What the allocator costs is the RULE, not the budget.** At three matched
budgets: a global top-K at the allocator's own K keeps **+0.0344**, the LP keeps
**+0.0121**. The tighter budget costs 0.0019 (5%); the LP allocation rule costs
0.0223 (**65%**). The LP imposes the same per-group budgets on both arms -- 7 of
14 per-group ceilings are K=0 on iwildcam -- which forces the two arms' deployed
outputs toward each other and absorbs most of the ranking difference.
⚠️ At n=8 this contrast is 2/8, p=0.289: the magnitude is large, the sign
consistency is not established.

**The uncapped damage is in the MODEL, not the allocator.** It is fully present
pre-allocator (RAW -0.0107) and is **44% larger there than deployed** (-0.0074),
so allocation cannot be its source. It is concentrated in exactly two classes
(c1 -0.0330, c4 -0.0411) and nearly cancelled by c6 (+0.0234). Only c4 has an
allocator-attributable component: ~7.7 evicted items/run land there.

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

#### 🆕 A SECOND CLASS, found 2026-08-25: **ABSENT DATA THAT READS AS A VALUE**

The rows above are all one class -- a knob that does nothing. This is the mirror:
a quantity that is *missing* but is rendered, printed or averaged as though it
were *measured*. Two instances, both fixed, both gated, both found by asking
"what would this show me if the data were simply not there?"

| site | absent value became | why it was dangerous | state |
|---|---|---|---|
| `docs/paper/scripts/make_deployment_fig.py` | a `NaN` bar height, from `reindex` | **`ax.bar` draws a `NaN` height and a `0.00` height to byte-identical pixels** (measured: both PNGs 236 bytes). That figure's headline claim is that the post-hoc clippers sit at ~0.00 native satisfaction -- so a `(backbone, method)` cell that had vanished from the corpus would have drawn as an empty bar and read as **evidence FOR the claim**. `reindex` also silently DROPS any corpus key the hardcoded orders omit. | ✅ `_require_full_grid` raises instead of drawing, and prints what it excludes. The dot overlay's `except KeyError: pts = []` -- the same swallow one layer down -- is removed. Gated by `test_the_deployment_figure_REFUSES_a_bar_it_has_no_data_for`. |
| `scripts/full_panel.py` `macroP`/`macroR`/`macroF1` | an F1 of `0.0` for a class with no true instances, **and a denominator that moved with the arm** | With no explicit `labels=`, sklearn macro-averages over `unique(y_true) \| unique(y_pred)`. An arm emitting a class absent from `y_true` is divided by one MORE class than an arm that does not -- measured, same truth: **0.8289 vs 1.0000**. This project reads nothing but arm-minus-arm differences on exactly this metric. | ✅ pinned to `present = sorted(set(y))`, matching `score_scan.py` and `paired_seeds.py`, which already did it. Gated by `test_the_macro_denominator_is_the_DATA_not_the_arm_s_predictions`. |

⚠️ **The second one was NOT reachable on iwildcam and was fixed anyway.** All 8
classes appear in the `oodslice` test split (counts 720/480/370/180/210/367/160/456),
so the union never varies and the pin is a **verified no-op** -- 200/200 random
prediction vectors give bit-identical macro-F1 before and after, and no committed
number moves. It was reachable only if a future slice lost a class, or the moment
anyone scored **per group** -- and the per-group structure makes that acute:
camera 130's test rows are class 7 **only**, camera 516's are class 3 **only**,
and cameras 53/306/410 hold only classes 2 and 7. A per-camera macro-F1 would
have hit this on the first run.

🔑 **The generalisation, worth more than either fix:** this project's gates all ask
*"is this flag live?"*. Neither of these was a dead flag -- both were live code
doing the wrong thing with **missing input**. The question that finds them is
**"what does this render when the data is absent, and is that distinguishable
from a real measurement?"** Where the two are indistinguishable, absence must
raise, not draw.

#### 🆕 A THIRD CLASS, found 2026-08-25: **THE TOOL DIES MID-REPORT**

| site | what happened | state |
|---|---|---|
| `scripts/ortho_survival.py`, `scripts/scope_probe.py` (x3), `scripts/dataset_screen.py`, `scripts/prep_iwildcam.py` | a `print` containing an emoji raises `UnicodeEncodeError` on a **cp1252 console** -- the Windows default -- and the script **exits 1 mid-report**, so everything already printed reads as the complete output | ✅ printed strings ASCII-ised (`!!`, `->`, `=>`); docstrings, comments and every `.md` keep their emoji. Gated by `test_no_script_CRASHES_when_it_prints_its_own_conclusion` (AST over `print`/`SystemExit`/`sys.exit`) |

🔑 **Why this is not cosmetic.** It was found because `ortho_survival` died
between its table and the caveat that qualifies it -- the reader would have got
the number without the warning. And `scope_probe`'s crash sits inside the
**`PROBE CANNOT RESOLVE THIS`** branch: it would fail exactly when its job is to
report that it cannot answer, which is the one message this project most needs
to survive. These run clean on the servers (Linux, UTF-8); the bug is invisible
there and fatal locally.

#### 🆕 A FOURTH CLASS, found 2026-08-25: **THE INSTRUMENT DROPS DATA WITHOUT SAYING SO**

The third class was a tool that dies loudly in the wrong place. This is the
opposite and worse: a tool that survives by discarding an input and then prints
a number over the smaller set, with the header still describing the larger one.
`except ...: pass` is the whole mechanism. Found by asking the class question of
2(e) on the probes: *what does this report when an input is missing?*

| site | what was silently dropped | why it mattered | state |
|---|---|---|---|
| `scripts/straddle_probe.py` | a `_null` twin that fails to load, via `except SystemExit: pass` | the **BASELINE** block could be built from fewer runs than the **TREATED** block printed directly below it, while its header says they are the same cells. `report()` also took a run count and **ignored** it, so nothing anywhere stated the coverage | ✅ skip is printed and counted; `report()` prints its own `n_runs`; the baseline is labelled with `n_base`, not the treatment's `n_ok`; a divergence prints "these blocks DO NOT COVER THE SAME RUNS" |
| `scripts/full_panel.py` `_treatment_weight_keys` | every key derived from `protocol.yml`, falling back to the floor | the floor is **exactly** the hardcoded list this function's own docstring records as a bug -- `fioretto_null`, `hounie_null` and `alm_null` then fall through to the treated-arm branch. A scorer quietly regressing to a known defect is worse than one that crashes | ✅ warns on stderr, naming the three arms that will be misread |
| `scripts/check_parity.py` | the protocol's full `warmup_identity_keys` list | the gate narrows to **one** key and still prints **PARITY OK** | ✅ warns that the gate is now weaker than PARITY OK implies |
| `scripts/variance_probe.py` | any run whose metric will not parse | this is the **NOISE FLOOR**, which every effect in this project is judged against (0.0358 macro-F1 = 21x the effect it was measuring). A spread over a silently smaller set understates it | ✅ prints how many runs were dropped per metric, and says so explicitly when fewer than 2 remain |

**Two more of the same class in `family_split`, found in the same sweep:**

* `won = sum(1 for c in percell if np.mean(percell[c]) > 0)` counted a **NaN**
  cell as a LOSS, because `nan > 0` is False. `full_panel` returns NaN for
  `uncF1` with no capped classes, for `ConfGap` when every item is correct, and
  for `AP`/`AUROC` in degenerate cells -- so "2 won, 3 unmeasurable, 4 lost" was
  printed as **`2/9`**. ✅ unmeasurable cells are now excluded from the
  denominator and counted separately in the line.
* `matched()` dropped every incomplete cell-seed **silently**. Its own docstring
  records that unmatched pooling once compared `clip` on 7 cells against a
  treatment on 6. ✅ it now prints how many were dropped and which arm was
  missing -- "16 matched" reads very differently when 18 existed than when 200
  did.

✅ **Four other silent swallows were examined and KEPT**, each with a recorded
reason: feature-detecting an optional torch API (`bisect_determinism`), the
error writer that must not raise while recording a failure
(`src/utils/error_handler`), an optional `config.json` for a log diagnostic
(`log_health`), and a fallback to an equivalent source (`hp_liveness`).

🔑 **THE SWEEP IS COMPLETE, AND IT FOUND ONE PATTERN THAT PREDICTS THE BUG.**
All eleven generators under `docs/paper/scripts/` were audited on 2026-08-25.
They divide cleanly:

| pattern | generators | outcome |
|---|---|---|
| **asserts the seed count, refuses otherwise** -- `assert len(s) == 4`, `assert (n_per_cell == 4).all()` | `make_main_table`, `make_graft_table` | ✅ **clean by construction.** A thin cell raises; it cannot reach the table |
| **`.dropna()` / `reindex` and carry on** | `make_deployment_fig`, `make_backbone_tables`, `make_granular_tables` | ⚠️ **all three carried a defect** -- an empty bar indistinguishable from 0.00, a discarded cap level indistinguishable from one never run, and a one-seed column beside four-seed neighbours |
| reads one file, or no data at all | `make_convergence_fig`, `make_datasets_fig`, `make_figs`, `make_loss_shape_fig`, `make_octmnist_fig` | ✅ nothing to drop; `make_octmnist_fig` was measured and its `dropna` removes **0** rows across 3 backbones x 9 cap tags |

⇒ **the rule for the next generator: assert the shape you expect, do not repair
it.** `assert len(s) == 4` is three words and it made two of these files immune
to a class that hit all three of their neighbours. Every defect above came from
a function that quietly accepted a smaller input and returned a number anyway.

Gated by `test_no_scorer_or_gate_DROPS_DATA_WITHOUT_SAYING_SO`, an AST walk over
`scripts/`, `docs/paper/scripts/` and `src/` with `SILENT_SWALLOW_ALLOWED` as
the exemption list -- **and the test also fails on a STALE exemption**, because
an allowlist entry that outlives its code silently re-permits the bug if the
code returns.

🟢 **AND THE TRAINING PIPELINE ITSELF CAME BACK CLEAN.** `src/` and `main.py`
were audited on the same question 2026-08-25. Exactly one defect --
`lp_fallback_*` above -- and it lives in the *recording* layer. Every
aggregation point inside the pipeline was already guarded, and several carry the
history in their own comments:

* `src/training/metrics.py` skips empty ECE bins explicitly rather than taking
  `mean()` of nothing;
* `src/utils/posthoc_adjustment.py` writes both meta keys on **all three** of
  `targeted_correction`'s return paths -- the gap is upstream, in the arms that
  never call it;
* `src/training/logging.py` does materialise a missing counts dict as all
  zeros, **but its only consumer knows**: `scripts/log_health.py` excludes
  warm-up rows ("their counts are zero, which registers as trivially
  satisfied"), treats an all-blank column as *not logged*, and identifies a
  post-hoc arm by a finite `Limit_Class` rather than by column presence --
  having twice got that wrong before. Gated by
  `test_log_health_does_not_cry_wolf_on_a_warm_up_row_or_a_posthoc_arm`.

🔑 **The distribution is the finding.** All ten instances of this class sit in
the **analysis and presentation** layer -- scorers, probes, figure and table
generators -- and none in the training path. That is what repeated auditing
looks like from the outside: the pipeline has been hardened for a year and the
tools that read it have not. **Audit the reader, not the writer.**

#### 🆕 A FIFTH CLASS, found 2026-08-25: **THE STAGED ARTEFACT NOBODY PARSES**

`docs/launch_uniform.sh` had carried, since it was written, an arm list that
bash resolves to something other than what it reads as:

```text
--arms tralo tralo_uniform tralo_ortho tralo_head tralo_null tralo_reseed \n           clip focal_clip \
```

That `\n` is **not a newline**. A backslash inside an unquoted bash word escapes
the next character, so the shell passes a bare argument `n`. `gen_campaign`'s
`choices=` rejects it, exit 2, and `set -euo pipefail` kills the script. The
campaign dies **at launch** -- after a GPU has been found, the worktree taken and
the pin checked out -- for a defect that was in the file the whole time.

🔑 **The class is not "a typo". It is that a launch script is the only executable
artefact in this repository that nothing ever parsed.** `src/`, `configs/` and
`scripts/` are all imported by 584 tests. `main.py` runs every campaign.
`docs/*.sh` were prose to every tool in the repo and code to exactly one reader:
the server, once, under time pressure. Two of them existed; one was broken.

⚠️ **AND FAILING LOUDLY WAS LUCK.** Measured by dropping each token of that
line in turn and reading what `gen_campaign` actually does:

| token lost | generator | outcome |
|---|---|---|
| `clip`, `focal_clip` | auto-re-added (`mandatory_arms`) | **harmless** |
| `tralo_reseed` | REFUSED, exit 1 | **caught** |
| `tralo`, `tralo_uniform`, `tralo_head` | exit 0, 216 runs | silent |
| **`tralo_null`** | **exit 0, 216 runs** | 🛑 **silent and fatal** |

Losing the twin prints `*** NO ZERO-DOSE CONTROL for: ...` and **exits 0**. In a
launch script that warning scrolls past inside the generator's own output, `set
-e` does nothing about it, and the dispatcher starts 45 seconds later. The
campaign would run to completion and be **unreadable**: every contrast in it is
seed-paired against the twin, so `family_split` finds no null and `full_panel
--control tralo_null` has no control. 216 runs, unattributable -- which is
section 3b's defect (`corpus_final.csv` has zero `_null` arms) reproduced from
scratch by a stray backslash.

✅ Gated by `test_a_staged_launch_script_NAMES_ONLY_ARMS_THAT_EXIST`, which
parses every `docs/*.sh` the way bash does (`shlex`, posix, after removing
backslash-NEWLINE only) and checks two things: every arm named EXISTS in
`configs/protocol.yml`, and every **trained** arm named has its `_null` sibling
named beside it. Shown to FAIL on all three breaks -- the original `
`, a
mangled-away `tralo_null`, and a duplicated arm.

🛑 **AND THE SAME SWEEP FOUND THE GUARD BOTH SCRIPTS USE IS BLIND TO THE
THING IT GUARDS AGAINST.** Both shipped:

```text
pgrep -u "$(whoami)" -f "envs/optloss/bin/python main.py"
```

`main.py` runs every experiment as a subprocess --
`[sys.executable, '-u', '-m', RUNNER_MODULE, config]` with
`RUNNER_MODULE = 'src.experiments.runner'` (`main.py:121`) -- and **that command
line contains no `main.py`**. So a dispatcher that was killed while a run was in
flight leaves a live runner this guard cannot see, and the script reports a
clear host and starts a second dispatcher into the same tree. That is verbatim
the operational failure `CLAUDE.md` already records ("a killed dispatcher
leaving three runners alive writing into a directory a fresh dispatcher had
claimed"): **the guard written to prevent it could not detect it.** Both scripts
now count dispatcher AND runner processes, `sort -u` so one PID matching both
patterns counts once; verified across all four states (orphaned runner alone,
dispatcher alone, same PID twice, idle host). Gated by
`test_a_launch_script_CANNOT_SEE_A_LIVE_RUN_by_looking_for_main_py`, which reads
`RUNNER_MODULE` out of `main.py` by AST so renaming it makes the gate demand the
new name rather than pass on the old one.

🛑 **AND THE SWEEP DID NOT STOP AT THE SCRIPTS -- IT REACHED THE TOOLS THEY
NAME.** `launch_uniform.sh`'s read-order lists
`python -m scripts.family_split --campaign results/uniform1` as step 5. That
command **cannot run on that campaign**: `family_split` defaults to
`--families tralo fioretto hounie`, and `uniform1` has neither dual family, so
it exits with *"No cell-seed carries all of ..."* and prints nothing.

Underneath that was a real instrument defect. `family_split` derived each
family's twin as `fam + "_null"`. `uniform1`'s `tralo_uniform` and `tralo_head`
share `tralo_null` via `null_sibling` (protocol.yml, because at lambda = 0 they
are the same run), so concatenation invented `tralo_uniform_null` -- an arm that
exists nowhere -- and the tool refused a campaign whose twin was sitting in it.

⚠️ **The obvious fix is the opposite bug, and I wrote it before catching
it.** Resolving everything through `null_sibling` looks cleaner, but
protocol.yml points `fioretto` and `hounie` at `tralo_null` **too**, so it would
stop reading xfam1's *dedicated* `fioretto_null` / `hounie_null` arms -- turning
the byte-identity positive control that section 2(s) rests on from a measurement
into a tautology, silently. The rule is therefore **dedicated null if the
campaign ran one, shared twin otherwise**, and the tool now prints which rule
fired per family and says out loud when the identity check is vacuous rather
than passed. Gated by
`test_family_split_resolves_a_twin_the_way_the_CAMPAIGN_ran_it`, shown to FAIL on
both wrong fixes.

📌 Two more in the same file, both found by asking "what does this actually
resolve to": `launch_margin1.sh` told the operator to run
`git push origin cleanup/consolidate-pipeline` and asked for permission on the
grounds that it publishes -- but that branch is not the publish target
(`headroom/small-cnn` is), so the push would advance a branch nothing reads and
the `git pull --ff-only` after it would still fetch nothing. Everything the
script claimed was unpushed is in fact already on the remote. And its
"if it moves, extend with octmnist" escape route points at a dataset 2(n)
removed, so the independence it promises cannot be bought.

🔑 **THE GENERALISATION, and it is the same one as the fourth class turned
around.** The fourth class was *audit the reader, not the writer*. This one is
**an artefact that only one machine ever executes is the artefact least likely to
have been executed.** Anything in this repo that is run once, elsewhere, by hand
-- launch scripts, deploy commands, the `ssh` one-liners in operational docs --
is in that category. `bash -n` and a `shlex` parse are the cheapest gate this
project has, and neither had ever been pointed at them.

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
six restored baselines, six new gate scripts, and 584 tests. **Do not quote a line count as a
quality measure** -- it has only gone UP since the purge while the repository got
strictly more correct, and every per-component figure written here has gone stale
within days. Measure it if you need it: `git ls-files '*.py' | xargs wc -l`.

What is actually load-bearing is that every one of those lines is reachable and every knob is
read: `audit_config` (no orphan hyperparameters), `smoke_arms` (every arm runs end to end; caps verified for the arms that emit predictions directly, and for the trained arms under `--matrix`),
`verify_caps` (the caps bind on the real slices), `check_parity` (equal compute, shared knobs,
no cross-objective warm-up sharing), and `pytest tests` (584 tests, ~200 s, no dataset needed).

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

🛑 **RE-OPENED AND RE-CLOSED 2026-09-01, ON A STRONGER GROUND.** The
measurement above was taken on **dermmnist**, which is REMOVED and whose test
set leaks 38.7% of itself, and it was POOLED across cells. Re-run on
`dom1` (iwildcam, 384 runs, `--dump`), diffusion is NOT a null: `+2.01` items
pooled, 232/384 runs, and per cell it is `-0.65 / -0.18 / +1.28` on
MobileNetV2 against `+3.79 / +4.73 / +3.09` on MobileNetV3. The controls are
clean and large in the right direction (shuffled graph `-13.11`, shuffled
features `-16.31`), so this is geometry, not re-normalisation.

⛔ **AND IT IS STILL NOT A DIRECTION, FOR A REASON THE OLD READING COULD NOT
SEE: THE PROBE IS POST-HOC AND SCORES EVERY ARM AGAINST ITS OWN UNDIFFUSED
SCORES.** A gain available to every arm raises the BASELINE and moves no
arm-vs-arm delta. Measured per arm, 24 runs each, `d items`:

| arm | gain | arm | gain |
|---|---|---|---|
| `tralo_reseed` | **+3.51** | `cb_lp` / `clip` / `lp` | +1.37 |
| `alm_null` / `fioretto_null` / `hounie_null` / `tralo_null` | **+2.91** | `tralo_uniform` | +1.34 |
| `focal_clip` / `focal_lp` | +2.11 | `la_lp` | +1.17 |
| `hounie` +1.84, `tralo` **+1.83**, `alm` +1.43 | | `fioretto` | +1.10 |

**The UNTREATED arms gain the MOST.** `tralo_null` collects +2.91 against
`tralo`'s +1.83, so diffusing everything would move the treated-minus-null
contrast **-1.08 items AGAINST TraLO**. Per cell that difference is
`+1.95 / +0.68 / +0.16` (MobileNetV2) and `-3.80 / -6.38 / +0.94`
(MobileNetV3): **4 of 6 cells positive, sign p = 0.34, a coin**, and the two
largest effects are both negative. ⇒ The geometry is real and it belongs to
the BASELINE. **DO NOT RUN a GPU campaign for it** -- the instruction stands,
now for the right reason.

🔑 **A FREE AUDIT FELL OUT OF THE SAME TABLE.** The probe reads only
features and probabilities, and it reproduces the known byte-identities
exactly: the four `*_null` arms all read **+2.91**, and `cb_lp` = `clip` =
`lp` = **+1.37**. That is 2(x1) (`class_balanced` is inert on iwildcam) and
the shared lambda=0 model, recovered by an instrument that knows nothing about
either. A per-arm column disagreeing with a known identity is a defect signal.

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

⚠️ **SCOPE QUALIFIER ADDED 2026-08-25, and it narrows the SUPPORT without
changing the CONCLUSION.** This section's evidence is
`lp_fallback_used = False` with `0` candidates, stated as holding "on all 50
completed runs". **It is not measured on all of them.** Six arms --
`clip`, `focal_clip`, `lp`, `focal_lp`, `cb_lp`, `la_lp` -- reach the runner
through methodologies that set `skip_targeted_correction=True`, so
`src/pipeline/eval.py` leaves `posthoc_meta = {}` and
`src/experiments/runner.py` fills it with `.get('lp_fallback_used', False)` and
`.get('lp_fallback_candidates', 0)`. **Both defaults are values that mean
something else**, and `clip` + `focal_clip` are in every campaign by CLAUDE.md
rule 2.

⇒ read the evidence as **"on every run that RAN the allocator"**. The
conclusion stands, because it rests on the trained arms where the field is
genuinely measured; what was wrong was the claimed breadth.

🔑 **This is the SIBLING of the `flag_live` defect fixed the same day** -- there,
the gate called six post-hoc arms INERT because the harness runs neither the
warm-up nor the allocator. **Same root cause: the post-hoc arms do not traverse
the pipeline path the field describes, so a field read across all arms mixes
measurement with default.** Ask it of any other per-run field before quoting one
across arms. Gated by
`test_the_lp_fallback_fields_are_a_DEFAULT_for_the_post_hoc_arms`, which walks
the whole chain from the methodologies through `eval.py` to `runner.py` rather
than trusting any single line.


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
(`prep_octmnist.py`, deleted by `61e34c0a`), so the groups are i.i.d. draws from one
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

#### 🔴🔴🔴 REPLICATED AND **CALLED** AT 9 CELLS -- `results/iwc3`, read 2026-08-25

`iwc1` measured this on 2 cells, where the exact sign-test floor is p = 0.25 and
nothing can clear BH. `iwc3` is the same question at **9 cells**: iwildcam x
{MobileNetV2, MobileNetV3, RegNetY400MF} x {L20_G50, L30_G50, L50_G30}, arms
`clip` / `focal_clip` / `tralo` / `tralo_null` / `tralo_reseed`, 4 seeds = 180
runs, complete, **zero error logs, one `code_version`**. At 9 cells the floor is
p = 0.0039, so a 0-of-9 sweep is callable and a BH-corrected verdict exists.

`tralo` against **its own lambda = 0 twin** -- same warm-up, same allocator,
same seed, the constraint the only difference:

| metric | tralo_null | tralo | delta | cells won | BH q | verdict |
|---|---|---|---|---|---|---|
| AP | 0.9443 | 0.9049 | **-0.0394** | **0/9** | 0.0072 | *** LOSS |
| AUROC | 0.9889 | 0.9795 | **-0.0094** | **0/9** | 0.0072 | *** LOSS |
| ECE | 0.1653 | 0.1907 | -0.0254 | 0/9 | 0.0072 | *** LOSS |
| Brier | 0.3692 | 0.4168 | -0.0476 | 0/9 | 0.0072 | *** LOSS |
| NLL | 1.2053 | 1.4598 | -0.2545 | 0/9 | 0.0072 | *** LOSS |
| ConfGap | 0.1065 | 0.0928 | -0.0136 | 1/8 | 0.0143 | *** LOSS |
| ccF1 | 0.4175 | 0.4177 | +0.0002 | 5/3 | 0.5312 | tie (**0.1 items**) |
| macroF1 | 0.6100 | 0.6012 | -0.0087 | 1/8 | 0.0107 | *** LOSS |
| acc | 0.6147 | 0.6054 | -0.0093 | 0/9 | 0.0072 | *** LOSS |

🔑 **AND THE NOISE FLOOR SPLITS THE TABLE IN TWO, WHICH IS THE WHOLE POINT OF
CARRYING IT.** `tralo_reseed` is the same null with the RNG stream perturbed and
nothing else. Against the twin it reads:

    AP +0.0030 tie   AUROC +0.0005 tie   Brier +0.0041 tie   NLL +0.0019 tie
    macroF1 -0.0104 *** LOSS 0/9         acc -0.0070 *** LOSS 0/9

* **macroF1 and acc are NOT attributable to the constraint.** A pure reseed
  costs MORE macro-F1 (-0.0104) than the constraint does (-0.0087), in the same
  0-of-9 sweep. Anyone quoting `tralo`'s macro-F1 loss here is quoting the
  seed.
* **AP, AUROC, Brier and NLL ARE attributable.** The reseed moves them 0.0005
  to 0.0041 and ties; the constraint moves them 0.0094 to 0.2545 and sweeps
  0/9. That is **2x to 130x the floor**, in the one channel a top-K allocator
  can read.
* **ccF1 is a tie worth 0.1 items**, against a paired seed sd of 2.11 items.
  At 80% power that prices detection at **~3495 seeds per cell**
  (`7.85*(2.11/0.1)^2`), not the ~152 this line used to print -- 152 is the
  price of a **0.48**-item effect, so the old figure understated the cost
  **23x** and the conclusion only gets stronger. The honest report is a
  stated MDE, not a null.
  ⚠️ The "headroom of 1.9-9.9" formerly quoted here is a **dermmnist**
  number on a removed, 38.7%-leaking dataset, and is superseded even for
  dermmnist by section 4's corrected 2-18. It is not an iwildcam quantity;
  see 2(z32).

**So the finding is not "TraLO is noisy". It is that the constraint pays a
measurable, repeatable price in the representation and buys nothing back in the
allocation** -- and the metric the manuscript headlines is precisely the one
that cannot tell the two apart.

⚠️ **This run also lost 328 of 1044 constraint steps (68.6% landed).** See (u):
the damage above was done at roughly two thirds of the intended dose, which
makes it a LOWER bound, not an overstatement.

#### What the OPTIMISATION did, from `log_health` -- 132 readable logs

The metrics above are the outcome. This is the process, and it rules out the
two ways a table like that gets believed for the wrong reason.

* **No terminal collapse.** Final training accuracy is 0.9982-0.9989 across
  all five arms, with no run off its own trajectory. The trap in
  `project_terminal_collapse_trap` -- one control ending 0.9934 -> 0.9116 and
  reversing a headline -- is not present here.
* **The regime is LIVE, not the saturated one.** Warm-up ends at 0.963 median
  and the constraint epochs add **+0.036**, against the saturated signature's
  `acc >= 0.93 AND |gain| <= 0.005`. This is not warm-up 50 in disguise.
* **The constraint is never satisfied: 0 of 1073 epochs**, for `tralo` and for
  its twin alike.

🔑 **AND THE COUNT TRAJECTORY SAYS THE SAME THING THE METRICS DO, IN ITEMS.**
Over all 29 constraint epochs, mean capped-class count, first epoch to last:

| arm | class 2 (K=185) | class 7 (K=228) | slope /epoch |
|---|---|---|---|
| `tralo` | 367 -> 363 (**-4**) | 458 -> 461 (**+3**) | -0.38 / +0.26 |
| `tralo_null` | 367 -> 368 (+1) | 459 -> 465 (+6) | +0.16 / -0.07 |
| `tralo_reseed` | 393 -> 371 (**-22**) | 436 -> 459 (**+23**) | **-0.79 / +0.80** |

**Perturbing the RNG stream moves the capped count five to eight times more
than the constraint does**, in the same cells, over the same epochs, against
violations of 182 and 230 items. That is section (13) measured again on
iwildcam at 9 cells, and it is measured per class rather than as an RMS.

🔴 **THE STARVATION SIGNATURE IS PRESENT AND `log_health` NAMES IT.** Per
(group, class) scope, `tralo`'s WORST violation -- `g53/class7`, over by 148 --
*rises* at **+0.69/epoch**, while the already-satisfied `g218/class2`, over by
0, sits at -0.02. The penalty's gradient is non-monotone in the violation and
the scopes compete for one unit-norm clip: 2(a2), on this dataset, at this cap
sweep, in the campaign whose metrics are quoted above.

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

Launch: `docs/launch_uniform.sh` (9 cells, 7 arms, 4 seeds = 252 runs; 9 cells
is deliberate -- sign-test floor 0.00391 against BH 0.00455, so unlike iwc1/iwc2
it can return a CALLABLE verdict). Read with `order_probe --evictions` FIRST.

### (s) 🔴🔴🔴 EVERY DUAL'S MARGIN IS THE 29 EPOCHS, AND THE FAMILY ORDERING IS COLLATERAL DAMAGE

`results/xfam1`, dsisco02, **complete at 324 of 324 runs, zero crashed**,
scored 2026-08-25. iwildcam x {MobileNetV2, MobileNetV3, RegNetY400MF} x
{L20_G50, L30_G50, L50_G30} = 9 cells, **36 matched cell-seeds** (4 seeds in
every cell), nine arms including **a lambda=0 twin for every dual family** -- the thing the 7,574-row corpus does not have in a single row
(see the corpus audit in section 1), and without which no published number in
this project could be attributed to a constraint rather than to compute.

Read with `python -m scripts.family_split --campaign results/xfam1`.

🛑 **`full_panel` REFUSES `xfam1` -- ITS `run_code_version` IS SPLIT, AND THE
SPLIT IS BENIGN BUT REAL.** Audited 2026-08-28. The GENERATOR stamped one
commit on all 324 runs; the RUNNER stamped **142 at `9b89ce26d6bb` and 182 at
`9b89ce26d6bb-dirty`**. That is the pre-2026-08-24 whole-tree `git_version`
defect -- it diffed the entire tree, so deploying a scorer flipped the stamp
(see the `git_version` scoping fix, now `TRAINING_PATHS = (src, configs,
main.py)` and verified deployed on both hosts).

**Four independent checks say the split cannot bias an arm contrast:**

| check | result |
|---|---|
| aliased with ARM? | **no** -- every one of the 9 arms splits ~16 clean / ~20 dirty |
| `(model, cap, seed)` groups internally pure? | **35 of 36** -- one straddles (`RegNetY400MF`, `L20_G50`, `seed_2`) |
| dose per half | **100.0% both** -- 1360/1360 clean, 1700/1700 dirty |
| `data_fingerprint` | **identical**, `6b836adf59ec7d56` |

Per cell the split is **2 clean / 2 dirty in 6 of 9 cells** (RegNet is 2/3,
1/3, 1/3). So every arm sees the same seeds on the same versions, and since a
`(cell, seed)` group is internally pure, every PAIRED contrast is
version-consistent. ⇒ **a balanced nuisance factor: it inflates variance, it
cannot bias `arm - control`.**

⚠️ **Do NOT weaken the refusal to score it.** Splitting the campaign by
version leaves **2 seeds per cell** and makes 2 of 9 cells unpairable, so each
half is hopeless on its own; and the gate is right as a default policy. The
100%-dose row above matters because the commits immediately after this pin
include *"the probability clamp was a no-op in every dtype, and it cost the
live campaign 96.6% of its dose"* -- that defect did **not** touch either half
of `xfam1`, which is the direct evidence the dirt was not on the training path.

🔑 **`results/dom1` SUPERSEDES `xfam1` on exactly this question** and is why it
was built: 9 cells, 16 arms, all four duals with their own nulls, both scopes,
**one clean `code_version`** (verified: all 384 configs stamp `1d92117363d2`,
no `-dirty`). Quote `xfam1` as a prior, `dom1` as the result.

🔑 **THE POSITIVE CONTROL IS EXACT, AND IT COMES FIRST.** At lambda = 0 the
dual family is irrelevant: same cached warm-up, same allocator, same seed, no
constraint gradient. So `tralo_null`, `fioretto_null` and `hounie_null` must be
the SAME RUN. They are -- **byte-identical raw predictions in 36 of 36
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

✅ **RE-READ AT THE FULL 324 RUNS, 2026-08-25** -- the campaign finished during
a multi-hour SSH outage. **36 matched cell-seeds over the same 9 cells** (4 seeds
everywhere, against 16 cell-seeds before), nulls byte-identical in **36 of 36**,
zero crashed runs. These are the numbers to quote; the 141-run column is kept
beside them because two claims moved.

| metric | compute (all three) | `tralo` | `fioretto` | `hounie` | reseed floor | *(compute @141)* |
|---|---|---|---|---|---|---|
| macroF1 | +0.0059 | -0.0048 | **+0.0005** | -0.0021 | -0.0028 | *+0.0145* |
| uncF1 | +0.0083 | **-0.0062** | **+0.0009** | **-0.0022** | -0.0041 | *+0.0194* |
| ccF1 | -0.0014 | -0.0004 | -0.0008 | -0.0018 | +0.0010 | *-0.0003* |
| AP | **+0.0143** | **-0.0809** | **-0.1405** | **-0.1567** | -0.0016 | *-0.0060* |
| AUROC | **+0.0016** | -0.0179 | -0.0555 | -0.0560 | +0.0016 | *-0.0047* |
| ECE | **+0.0052** | -0.0328 | -0.0504 | -0.0549 | -0.0064 | *-0.0006* |
| Brier | **+0.0101** | -0.0613 | -0.1036 | -0.1079 | -0.0137 | *-0.0020* |
| NLL | **+0.0135** | -0.3517 | -0.5788 | -0.5977 | -0.0616 | *-0.0477* |

🔄 **THE COMPUTE TERM FLIPPED SIGN ON FIVE OF EIGHT METRICS.** At 141 runs the
29 extra epochs read as helping macro-F1 and *hurting* AP, AUROC and all three
calibration metrics. At 324 they help on **every metric**, and the macro-F1 gain
halves (+0.0145 -> +0.0059). The 141-run signs were a half-filled campaign, not
a finding -- which is what that read's own caveat said, and this is the receipt.

Signs are oriented so **+ is better** on every row, including the three where
lower is better natively. `ccF1` converts at **5.2 items per 0.01** here.

🛑 **22 OF THE 24 CONSTRAINT TERMS ARE NEGATIVE, AND THE OTHER TWO ARE ZERO.**
⚠️ At 141 runs this said "not one of the 24 is positive". That is **no longer
literally true**: `fioretto`'s macroF1 (+0.0005) and uncF1 (+0.0009) came out
positive at 4 seeds. Neither is a win. Both win **4 of 9 cells** -- under half --
and both sit **inside the reseed floor**, which moves macroF1 -0.0028 and uncF1
-0.0041 on RNG alone. So the honest statement is *indistinguishable from zero*,
and the finding survives in substance while losing its absolute form: **no dual
family's constraint produces a positive contribution on any metric**, and every
method's entire margin over `clip` is the 29 epochs that every trained arm gets
and the post-hoc clipper does not. This is section 3's "regime beats method"
measured directly instead of inferred.

🔑 **THE PUBLISHED ORDERING IS A DAMAGE RANKING WEARING A BENEFIT RANKING'S
CLOTHES.** On macro-F1 the totals are `fioretto` +0.0063 > `hounie` +0.0038 >
`tralo` +0.0011, which is exactly the manuscript's ordering. The compute term
is identical across the three, so that ordering is `0.0059` minus the damage --
i.e. it ranks the families by how little each one spoils a gain none of them
produced. **TraLO is not "improving less". It is subtracting more.**

⚠️ **AND `fioretto`'s ADVANTAGE SITS BELOW THE NOISE FLOOR.** `tralo_reseed`
-- the same null with one RNG draw perturbed and nothing else -- moves macroF1
**-0.0028**, and `fioretto`'s whole constraint term is **+0.0005**, a fifth of
that. So "fioretto's constraint is gentle on macro-F1" is not distinguishable
from "fioretto's constraint does nothing to macro-F1", and the metric the paper
headlines cannot separate the two. It wins **4 of 9 cells**, which is the
coin-flip cell count.

#### The ordering REVERSES on the only channel an allocator can see

A top-K allocator reads the ranking and nothing else; a calibration move
provably leaves every top-K set untouched (section 2(j)). On AP the constraint
terms are `tralo` **-0.0809** against `fioretto` -0.1405 and `hounie` -0.1567
-- TraLO damages the ranking **1.74x and 1.94x less**. On AUROC it is -0.0179
against -0.0555 and -0.0560: **3.10x and 3.13x less**. The reversal narrowed at
4 seeds (it read 2.0x/2.2x and 4.5x/4.6x at 141 runs) and it did not close.

**TraLO is the gentlest of the three on the representation and the harshest on
the composite, and nobody was scoring the channel where it wins.** That is not
a rescue -- all three are still negative, and negative is the finding -- but it
says the shortfall is not in the constraint machinery TraLO was designed
around.

#### Where the difference actually lives

`ccF1` and `uncF1` split macro-F1 into the classes the constraint names and the
six it does not. The capped-class terms are **-0.0004 / -0.0008 / -0.0018** --
**0.2 to 0.9 items**, under one item each, near-identical, and all three inside
each other's noise. The uncapped terms are **-0.0062 / +0.0009 / -0.0022**, a
range **5.1x wider** than the capped one and spanning zero.

🔑 **All three dual families do the same negligible thing to the classes the
constraint is about, and differ five-fold in what they do to the classes it
never mentions.** The entire cross-family story is collateral damage.

✅ **The split is arithmetic, and it was checked rather than assumed.**
`C * macroF1 == m * ccF1 + (C - m) * uncF1` holds to 1e-9 on **56 of 56**
stored runs from `evidence/`, so the `uncF1` column is exactly the complement
of `ccF1` within the composite and neither leaks a capped class nor counts an
absent one. Gated by
`test_uncF1_is_exactly_the_classes_the_constraint_never_names`, whose negative
control is the obvious wrong label list -- the macro over ALL classes -- which
must break the identity.

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

⛔⛔ **AND A SECOND, INDEPENDENT REASON `ovr` IS CLOSED: IT IS NOT
GAUGE-INVARIANT.** `z` and `z + c` describe the same model -- softmax is
invariant to a per-item additive shift, CE never penalises one, and nothing in
training pins the absolute logit scale. `S_c = sum_i softmax(z)_ic` is a
function of `softmax(z)` alone and is therefore invariant by construction.
`S_c = sum_i sigmoid(z_ic)` reads the absolute logits and is not. Measured on a
stored run across four gauges (`log p`; `log p` with the row max at 0;
`log p + 5`; `log p - 5`), the push on a `p > 0.99` capped item:

| count | log p | max->0 | +5 | -5 |
|---|---|---|---|---|
| `sum` | 2.361e-04 | 2.361e-04 | 2.361e-04 | 2.361e-04 |
| `ovr` | 4.518e-02 | 4.409e-02 | **1.955e-03** | 4.882e-02 |

**23x from a shift that changes no prediction.** A count whose dose drifts with
a quantity the objective does not control is the invisible-dose failure this
project has hit three times already (`constraint_grad_mode` across arms,
`cut_temp` across seeds, `hounie` at 1% of its intended dose). Gated by
`test_a_count_must_be_INVARIANT_to_the_logit_gauge`, with `ovr` kept only as
the negative control that proves the check can fire.

⚠️ **SO READ THE NEXT TABLE'S `ovr` COLUMN AS GAUGE-BOUND.** Its `sum` column
is invariant and is the result.

#### 2(a2) QUANTIFIED: where the shipped count's push actually goes

🛑 **READ THE NEXT THREE TABLES WITH THIS CAVEAT.** They are measured on the
128 stored runs in `evidence/`, which are `mcbar` + `multiclass` -- **dermmnist
and octmnist**, both REMOVED under 2(n), and dermmnist 38.7% test-leaked. That
is acceptable for what is being measured and not for more:

* **The algebra is dataset-independent** and holds anywhere -- gauge invariance,
  the monotonicity that makes the cross-term unable to reorder, the fact that
  `sum`'s per-item slope is `p(1-p)`. Those are properties of the objective.
* **The NUMBERS are distribution-dependent.** Which confidence band holds most
  items, how far the 102x collapse actually bites, and what fraction of the
  excess is reachable all depend on the score distribution, and iwildcam's is
  not dermmnist's. ⚠️ **Re-run `collateral_probe` on `results/xfam1` before
  quoting any figure below as an iwildcam quantity.**
* Leakage does not threaten these specifically -- no label enters the
  computation; the probe reads probabilities and budgets only. It is listed
  because the runs are otherwise unusable and someone will ask.


Unit-normalised push on the capped logit, by how confident that item is
(56 stored runs). 2(a2) established that the penalty's gradient vanishes on the
worst violations; this is that statement in numbers:

| `p` on the capped class | `sum` | `ovr` (gauge-bound) |
|---|---|---|
| 0.00 - 0.50 | 1.263e-03 | 1.093e-03 |
| 0.50 - 0.90 | **3.826e-02** | 4.466e-02 |
| 0.90 - 0.99 | 8.530e-03 | 4.620e-02 |
| **0.99 - 1.00** | **3.739e-04** | 4.623e-02 |

**`sum`'s push at `p > 0.99` is 102x smaller than its own peak** -- and a
`p > 0.99` capped prediction is exactly the violation the cap most needs
removed. The constraint is strongest on the items it least needs to move.

Consequence, measured with `--feasibility` (each run stepped until it sheds its
OWN excess, mean 207.6 items): at `eta = 4096`, where the logits have moved
tens of units and which is orders above anything training delivers, `sum`
leaves a **residual excess of 100.4 items and reaches feasibility in 25 of
56 runs**. The rest is immovable along that direction at any dose.

🟢 **AND `uniform` IS THE GAUGE-INVARIANT VERSION OF THE FIX `ovr` COULD NOT
BE.** The log-odds coordinate `u_c = log(p_c/(1-p_c))` is a function of
`softmax(z)` alone, so `uniform` passes the gauge test `ovr` fails --
`max |g(z) - g(z+5)| = 1.7e-15` over 16 runs, machine precision. Splitting each
unit-normalised gradient into the CAPPED column (the work) and the uncapped
ones (where the norm budget leaks):

| `p` on the capped class | `sum` capped | `sum` uncapped | `uniform` capped | `uniform` uncapped |
|---|---|---|---|---|
| 0.00 - 0.50 | 1.263e-03 | 1.385e-03 | 4.844e-03 | 3.160e-03 |
| 0.50 - 0.90 | **3.826e-02** | 9.959e-03 | 7.294e-03 | 3.307e-03 |
| 0.90 - 0.99 | 8.530e-03 | 2.145e-03 | 1.028e-02 | 2.456e-03 |
| **0.99 - 1.00** | **3.739e-04** | 9.352e-05 | **8.849e-03** | 2.301e-03 |

**`uniform` is flat (4.8e-03 to 1.0e-02 across the whole range) and delivers
23.7x more push than `sum` on `p > 0.99` violations -- and unlike `ovr`'s 123x,
this number survives a gauge shift, so it is a property of the method.** The
off-diagonal concern does not materialise either: `uniform`'s `1/(1-p_c)`
off-diagonal stays at 2.3e-03 on the confident items rather than exploding.

⚠️ **BUT `uniform` IS A GENERALIST AND `sum` IS A SPECIALIST, AND ON TOTAL
COUNT REDUCTION THE SPECIALIST WINS.** In the 0.50-0.90 band `sum` delivers 5x
more push than `uniform`, and that is where most items sit -- which is why the
`--feasibility` read has `sum` reaching feasibility in 25 of 56 runs against
`uniform`'s 11, with residuals 100.4 and 125.7. Flattening the gradient does
not make the constraint stronger. It makes it *differently aimed*.

🔑 **AND THAT IS 2(r)'s FIX WITH A MECHANISM UNDER IT.** 2(r) measured that the
constraint evicts items that are true positives 68.8% of the time while
admitting ones that are true positives 30.1% of the time, with the cut at
`p = 0.536` and evicted items averaging `p = 0.788`. That is the 0.50-0.90
band -- exactly where `sum` puts its 100x peak, and consistent with section 1's
independent measurement that `sum` lands 29.4% of its gradient on the 2% of
items nearest the boundary. `uniform` removes that concentration by
construction. So `results/uniform1` is not testing "a stronger constraint", it
is testing **whether spreading the same step off the boundary stops the
eviction** -- and its own `--feasibility` row predicts it will enforce LESS.

✅ **AND IT IS NOT THE SEE-SAW -- that was tested and REFUTED.** The obvious
explanation is 2(a)'s cross-term pushing one capped class up as another goes
down. It does not happen: **no capped class rose in 56 of 56 runs** under
`sum`. All of them fall, and simply not far enough -- e.g. counts
`[92, 180, 184]` against `K = [52, 110, 112]` land at `[56, 115, 106]`, still
over on two classes. The residual is the confident items, not an exchange.
(`ovr` is the one that shows a rise, in 5 of 56, because suppressing only the
class it targets lets a sibling capped class take the vacated items.)

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
* **The callable subset is smaller than the table, and at 4 seeds it GREW.**
  **All fifteen** ranking/calibration contrasts -- AP, AUROC, ECE, Brier and
  NLL for each of the three families -- now lose in **0 of 9 cells**, sign-test
  p = 0.0039, clearing BH at 0.00455. At 141 runs `tralo`'s AP was 1/8 cells,
  q = 0.060, "directional, not called"; at 324 it is 0/9 and **called**.
  The three F1 contrasts are the ones that stay uncallable: macroF1 4/9, 4/9,
  5/9; ccF1 5/9, 3/9, 2/9; uncF1 5/9, 4/9, 5/9 -- coin flips, and UNDERPOWERED
  at the seed level besides.
* ✅ **The re-read at 4 seeds is DONE (2026-08-25).** This bullet used to say
  "16 matched cell-seeds is 2 seeds in 6 cells and 1 in 3, re-read before
  quoting". The campaign finished, the table above is the 36-cell-seed one, and
  the two claims that moved are flagged inline. Nothing here is provisional on
  seed count any more; it is provisional on **one dataset**.

#### Relation to the manuscript -- it CORROBORATES, then goes further

`docs/paper/main_edited_by_roei.tex` already concedes the host-vs-ingredient
half of this. Its graft experiment adds TraLO's optimizer reset and undershoot
hinge to both dual baselines and reports that this recovers a median 86% and
92% of TraLO's margin over them, concluding in its own words that once the
recipe is grafted **the three methods are statistically interchangeable**, and
that "the two portable ingredients, not the host, carry the constrained-class
advantage". 2(s) reaches the same place from the other side -- a lambda = 0
twin instead of a component graft -- and agrees.

🛑 **WHAT 2(s) ADDS IS NOT IN THE PAPER AND MUST NOT BE PUT THERE YET.** The
paper says the ingredients carry the advantage. 2(s) says the ingredients carry
**nothing**: the compute term is the whole margin and 22 of the 24 constraint
terms are negative while the other two sit inside the reseed floor. That is a
strictly stronger claim, it cannot be made from the corpus (no nulls in 7,574
rows), and it rests on 4 seeds in 9 cells of **one dataset**. Before it touches
the manuscript it needs, at minimum:

* ✅ `xfam1` complete at 4 seeds in all 9 cells -- **done 2026-08-25, 324/324**;
* the macroF1 and ccF1 contrasts to clear their own resolution -- at 4 seeds
  they did NOT: 2-5 cells of 9, which is a coin flip, so the honest report for
  those three rows is a stated MDE and not a null. The five ranking and
  calibration rows DID clear, at 0/9 for every family;
* a second dataset, or the generality claim stays a direction. Section 2(n)
  rules the other three structurally incapable, so this is a real constraint on
  what the finding can ever say, not a scheduling problem.

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

⛔ **CORRECTION 2026-08-25: THERE *IS* AN OFFLINE PRICE, AND IT IS DAMNING.**
This paragraph used to read "there is no offline price for this one" -- on the
argument that a projection acts on parameter gradients during training, which no
stored artefact records. That is true of the *gradients* and irrelevant to the
*guarantee*, which is algebra and can be checked without a single run.
`python -m scripts.ortho_survival`.

🔑 **THE PREMISE, CHECKED RATHER THAN ASSUMED.** All of this is about the Adam
path. Under `constraint_step_rule: sgd` the step is `p -= lr * g` -- no
momentum, no preconditioner -- and the projection WOULD be delivered in full.
Every trained TraLO arm resolves to **`shared`**
(`constraint_phase.constraint_step_rule`, no block overrides it), so the verdict
applies to what actually ships. Gated inside
`test_ortho_project_s_GUARANTEE_DOES_NOT_REACH_THE_WEIGHTS`, which fails if any
of `tralo` / `tralo_ortho` / `tralo_head` ever resolves to `sgd`.

**What the guarantee is.** `project_out` sets `<g_con, r> = 0` on the raw
gradient. A step `-lr*u` changes the CE loss by `-lr*<grad_CE, u>` to first
order, so that zero is a claim of **CE-neutrality**: enforcing the cap neither
helps nor undoes CE progress. That is the entire rationale for the arm.

**Where it dies.** The step that lands is not `g_con`. It is Adam's
`m/sqrt(v)`, and two things there are untouched by the projection:

1. **Momentum.** `<m_new, r> = b1*<m_CE, r> + (1-b1)*<g_perp, r>`. The
   projection zeroes only the second term. The stale CE momentum rides at full
   weight `b1 = 0.9`.
2. **Diagonal preconditioning.** `<g, r> = 0` does **not** imply
   `<g/sqrt(v), r> = 0`. A coordinate-wise rescale is not an isometry, so
   orthogonality installed before Adam is not orthogonality after it.

**The bound, assumption-free.** After the clip the constraint gradient has norm
exactly 1.0 (2(a3): the clip delivers 1.000 against raw norms of 2,560-12,400),
so its share of the momentum vector is
`(1-b1)*1.0 / (b1*|m_CE| + (1-b1)*1.0)` = **7.4%** at the measured
`|m_CE| = 1.394`. The projection can only ever act on that share; the other
**92.6% is stale CE momentum**, which points along the CE direction by
construction.

**The measurement.** Norms are the real per-epoch values from
an `adam_contamination.py` audit on octmnist L50 -- a script that has NEVER been committed to this repository, see the caveat below. The one quantity not
measured is the coordinate-wise spread of `v`, so it is swept four orders of
magnitude and the whole curve is reported. Projected vs unprojected share one
RNG draw, so the difference is the projection and not Monte-Carlo noise:

| | flat `v` | spread 1 | spread 2 | spread 3 |
|---|---|---|---|---|
| `\|cos(update, CE)\|` | 0.9993 | 0.8420 | 0.6518 | 0.5360 |
| **removed by the projection** | **0.0%** | **0.0%** | **0.0%** | **0.0%** |

**0.0% in all 12 conditions, across three epochs.** And that is the arm's *best
case*: it assumes the reference IS the CE momentum, whereas
`ortho_ref = snapshot_grads(model)` captures **one minibatch's gradient**. Swept
over `rho = cos(ref, m_CE)` in {1.0, 0.5, 0.2, 0.05} the removal is 0.01% and
below -- every realistic reference buys less than the best case, and the best
case buys nothing.

⚠️ **AND THE SCRIPT THAT PRODUCED THOSE NORMS IS NOT IN THIS TREE.**
`git log --all -- '*adam_contamination*'` is empty; the audit that measured
`|m_CE|`, `|g_con|` and `|sqrt(v)|` lived outside this repository. So the
verdict is stated so that **it does not depend on them.** The removable share
has a closed form, `removed = (1-b1)/(b1*ratio + (1-b1))` with
`ratio = |m_CE|/|g_con|`:

| `\|m_CE\|/\|g_con\|` | 0.01 | **0.111** | 0.5 | **1.4** | **3.0** | 10.0 |
|---|---|---|---|---|---|---|
| max removable | 91.7% | **50.0%** | 18.2% | **7.4%** | **3.6%** | 1.1% |

0.111 is `(1-b1)/b1`, the break-even. The measured pair sits at **1.4-3.0**,
more than an order of magnitude clear of it. ⇒ the conclusion fails only if the
CE momentum were roughly **ten times smaller** than the constraint gradient,
which would be a different pipeline. And note the two numbers are consistent,
not contradictory: **7.4% is the ceiling on what could be removed from the
momentum, and 0.0% is what survives `sqrt(v)` to reach the weights.**

⚠️ **The probe's own controls decide whether the 0.0% is a measurement or
silence,** and they pass: with momentum off and a flat preconditioner the
projection removes **>99%** (so the instrument can see the effect); momentum
alone carries the CE direction back to **cos = 0.9993**; and a spread
preconditioner alone breaks orthogonality on its own. `--self-test` gates all
four.

🔑 **AND THE INTERVENTION THAT *DOES* ACT ON THE 92.6% IS ALREADY REJECTED.**
A dedicated constraint Adam -- `separate_constraint_optimizer` -- removes the
stale momentum outright, and it sits in 2(f)'s table at **AP -0.0938,
p = 0.0006**. ⚠️ Read that as *directional and confounded*, not as a clean
verdict: it moved the arm **8,900x further**, so dose and mechanism are not
separated there (section 1b-pre(6)). What it does establish is that this axis
has been intervened on once, hard, and it went badly.

⇒ **The honest entry for `ortho` is no longer OPEN. It is: the flag does not do
what its name says.** It is not inert -- `project_out` really does change
`prm.grad`, and `flag_live` would show differing predictions -- but the *reason*
2(t) reopened it, that a projection acts on the mechanism 2(s) found, is void.
Whatever `tralo_ortho` measures, it is not "the constraint no longer undoes CE
progress", because the delivered step's CE inner product is unchanged to 0.0%.

#### The same channel compresses a change to the COUNT FUNCTION

`tralo` vs `tralo_uniform` differ in `g`. On the step where they first diverge
they share `m_CE`, so the update difference is `(1-b1)*(g'-g)/sqrt(v)` against a
total dominated by `b1*m_CE/sqrt(v)` -- the same 7.4% channel:

| `cos(g, g')` | 0.99 | 0.90 | 0.50 | 0.00 | **-1.00** |
|---|---|---|---|---|---|
| angle between the count gradients | 8.1 deg | 25.8 deg | 60 deg | 90 deg | **180 deg** |
| **angle between the DELIVERED updates** | 0.64 deg | 2.0 deg | 4.6 deg | 6.4 deg | **9.1 deg** |

**Two count functions pointing in exactly OPPOSITE directions deliver updates
9.1 degrees apart** -- a ~20x angular compression, monotone across the range.

🛑 **READ THIS AS A POWER CONSIDERATION, NOT AS A PREDICTED NULL.** It is
per-step geometry. A consistent difference **compounds over the 29 constraint
steps**, and 1b-pre(6) is direct evidence that compounding *can* separate arms
whose per-step contrast is small -- `linear` and a coin have non-overlapping
distributions at `L50_G30`. Converting this geometry into an outcome claim is
exactly the error retracted in 1b-pre(6) on the same day this was measured, and
it is not to be repeated here.

**What it does license:** `scripts/flag_live` (md5 across arms, CLAUDE.md
rule 3) remains the gate, and it is *more* load-bearing than usual for the
staged campaign -- an arm whose count change is compressed 20x per step is an
arm whose predictions could plausibly come back identical. It also means the
`uniform` vs `sum` contrast is being read through a channel that carries a small
fraction of it, so an underpowered result there should be attributed to the
channel before it is attributed to the idea.

🛑 **CONSEQUENCE FOR THE STAGED CAMPAIGN.** `tralo_ortho` is one of the eight
arms in `docs/launch_uniform.sh` (36 of its 288 runs). Its stated purpose is
now void, so those runs should be **reallocated to seeds on the arms that do
have a live rationale** -- `tralo_uniform`, which is gauge-invariant and whose
mechanism 2(s) measured, and `tralo_head`, which confines the constraint by
*parameter set* rather than by gradient direction and is therefore untouched by
this argument. Note the asymmetry, and note it PRECISELY, because the obvious version of it is
wrong. It is **not** true that "a zeroed coordinate stays zero through `m` and
`v`" -- measured with real `torch.optim.Adam`, a coordinate whose gradient is
set to 0.0 still moves at **90.4%** of an unmasked coordinate's step, because
`m <- 0.9*m + 0.1*0` decays but does not vanish. The ratio converges to
`b1 = 0.9` from below as the CE phase lengthens -- 0.670 after 1 CE step, 0.819
after 3, **0.904 after the 126 the trainer actually runs** -- so a longer CE
phase makes the mask LESS effective, not more. **`head_only` does not freeze
the backbone.** What it does deliver is that **no constraint information**
reaches the backbone: the residual drift is pure CE momentum, which the
`lambda = 0` twin carries too, so it is common-mode in the only contrast that
matters. Read that arm as *"the constraint sees only the head"*, never as
*"the backbone is frozen"*.

⇒ the two arms fail differently, and only one fails fatally. `ortho`'s
guarantee is **quantitatively destroyed** -- 0.0% of the promised CE-neutrality
is delivered. `head_only`'s guarantee **survives in the form that matters**,
with a residual that cancels against its own control. **`tralo_head` is the
surviving member of the pair.**

**If anyone still wants the CE-neutrality the projection promises**, it has to
be applied to the DELIVERED update rather than the raw gradient -- capture `w`
before and after `optimizer.step()`, project `dw` off `r`, re-apply. That is a
different intervention with a different cost, and it is **not priced**; do not
treat this paragraph as an endorsement of it.

If it is run anyway, the bar is unchanged:

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

🛑 **AND IT SHIPS WITH ITS OWN CONTROL, `tralo_head`, WHICH IS NOT OPTIONAL.**
An `ortho` null read alone means either "the projection is too weak" or "the
backbone was never the culprit" -- opposite conclusions from one table, and
exactly the shape of null this project has misread before. `tralo_head`
confines the constraint gradient to the classifier head so the backbone cannot
move under it at all, testing 2(s)'s hypothesis outright instead of trying to
fix it. CE still trains the whole model, so the +0.0145 compute term is intact.
Read together they separate:

| `tralo_head` | `tralo_ortho` | conclusion |
|---|---|---|
| recovers | null | real effect, wrong tool -- keep the mechanism, change the instrument |
| recovers | recovers | the backbone is confirmed as the channel |
| null | null | **the backbone hypothesis is dead.** The damage is intrinsic to training under a count penalty, and the post-hoc clipper is the honest recommendation |

The head is identified BY SHAPE -- the single `nn.Linear` emitting `n_classes`
logits -- because the four backbones name it `classifier` / `fc` / `heads` and
a name list breaks on the fifth. It REFUSES on ambiguity rather than confining
the constraint to an arbitrary layer while still logging `head_only: true`.
Verified on all four: exactly weight + bias, 10,248 / 10,248 / 3,528 / 6,152
parameters.

⚠️ **The masking runs BEFORE the norm bound**, so the arm differs from its
control in the constraint's SUPPORT and not its dose -- but the whole step then
lands on **0.09-0.46%** of the parameters in this campaign's three backbones,
two to three orders more per-parameter movement than `tralo` delivers there.
Holding the per-parameter step instead would shrink the arm's total step below
its control's and fail `check_parity`. The convention is total norm; the
consequence is stated so it is read with the result.

✅ **BOTH ARMS ARE PRE-GATED, AND THE RECEIPTS ARE FROM A REAL RUN, 2026-08-24.**
Every gate in `CLAUDE.md` runs on synthetic tensors on CPU, so all of it was
executed without a dataset and without the server:

| gate | result |
|---|---|
| `audit_config` | every emitted key has a reader, every reader a key |
| `smoke_arms` | `tralo_ortho` and `tralo_head` run end to end |
| `smoke_arms --matrix` | caps hold in **12 of 12** combinations -- {1, 2} capped classes x {L30_G30, L50_G30}, both arms |
| `flag_live` | **"Every flag is connected on every binding seed"**, exit 0. Distinct md5s from `tralo` and from each other |
| `check_parity` | 288 runs, 8 arms x 36, one commit |
| `pytest tests` | 366, including four new gates each shown to FAIL when broken |

✅ **AND A FULL LIVENESS SWEEP OF THE TRAINED ARMS, 2026-08-25: NO FIFTH
INERT FLAG.** All eleven trained treatment arms -- `tralo`, `tralo_st`,
`tralo_margin`, `tralo_uniform`, `tralo_ortho`, `tralo_head`, `tralo_coin`,
`fioretto`, `hounie`, `alm`, `select` -- produce **distinct md5s on every
binding seed**. Section 2(e)'s inert-flag catalogue gains no entry.

🛑 **BUT THE SWEEP FOUND A DEFECT IN THE GATE ITSELF, and it was pointed at the
healthiest arms in the protocol.** Run over the six POST-HOC arms, `flag_live`
reported `clip`, `focal_clip`, `lp`, `focal_lp`, `cb_lp` and `la_lp` all
bit-identical and printed *"do not launch a campaign on it"* -- about the two
bars every campaign here is scored against and four of the nine methodologies
the paper claims.

**They are not inert; the harness cannot see them.** It calls
`TRAIN_FNS[methodology]` directly and therefore runs neither phase a post-hoc
arm's treatment lives in:

* the **WARM-UP** -- `warmup_loss` (focal / class_balanced / logit_adjust) is
  read in `make_ce_criterion`, reached only from `run_warmup`, which only
  `src/experiments/runner.py` calls. Verified by AST: `warmup_loss` has exactly
  one reader in the whole of `src/`, and it is `src/pipeline/warmup.py`.
* the **ALLOCATOR** -- `clip` vs `lp` differ in how the budget is filled, which
  is downstream of the model `flag_live` hashes.

So a post-hoc arm comes back identical however live it is. `flag_live` now
**refuses** them, names both reasons, and points at what does cover them
(`smoke_arms` for running and caps; a campaign's own `final_predictions_raw.csv`
md5s, which are written after both phases). A gate that condemns the healthy is
worse than no gate -- this project has already had a correct `iwc1` nearly
thrown out by a claim of the same shape. Gated by
`test_flag_live_REFUSES_post_hoc_arms_instead_of_calling_them_inert`, which
also pins the one-reader claim so the refusal can be narrowed the day a
methodology reads `warmup_loss` directly.

🛑 **`flag_live` is the one that matters here.** `ortho_project`'s 8-run
predecessor left zero prediction files, so nothing about it could be audited
after the fact, and an inert flag passes `audit_config` (the key has a reader)
AND `smoke_arms` (the arm does not crash) while producing a full campaign of
meaningless numbers -- which has happened four times in this project. The md5
check is the only gate that separates a live treatment from a renamed control,
and both arms clear it.

⚠️ Read nothing else off that harness. Random labels, a 4-layer net at chance
accuracy, one dose: the same four arms gave capped counts 31/35/32/12, then
0/0/0/0, then 120/120/119/119 across three seeds. **Connectedness survives it;
ordering does not.**

Until that campaign exists, the honest entry for `ortho` is **OPEN**, and this
section replaces its listing among the rejected.

### (u) 🔴🔴🔴 A SIXTH DEFECT CLASS: THE TREATMENT THAT REPORTS `completed` AND NEVER LANDED

2(e) catalogues five ways a knob can be dead. This is a sixth, and it is worse,
because the arm is live, the flag is live, the config is right, the run
finishes, and the **dose** is missing.

**MEASURED 2026-08-25, on the live campaign, before it was 2% done:**

    campaign   arm             steps landed / attempted   soft_count_mode
    uniform1   tralo             29 / 29     100.0%       sum
    uniform1   tralo_head        29 / 29     100.0%       sum
    uniform1   tralo_uniform      1 / 29     **3.4%**     uniform
    iwc3       tralo            716 / 1044    68.6%       sum
    xfam1      tralo / fioretto / hounie      100.0%      sum

`results/uniform1` exists to test exactly one thing -- `soft_count_mode:
uniform` -- and **that arm was running at 3.4% of its dose while every other
arm ran at 100%**. A campaign in that state does not measure a loss shape. It
measures a dose, wearing a loss shape's clothes.

#### The cause, and it is one line

`uniform_grad_count` builds the log-odds `u = log p - log1p(-p)`, guarded by

    p = proba.clamp(EPSILON, 1.0 - EPSILON)          # EPSILON = 1e-8

**That upper bound is a no-op in every dtype the pipeline uses.** float32's own
epsilon is 1.19e-7, float16's is 9.8e-4, bfloat16's is 7.8e-3 -- so `1 - 1e-8`
rounds to exactly 1.0 on the cast into the tensor. Then `log1p(-p)` is -inf,
`u` is +inf, the straight-through term `w * (u - u.detach())` is `inf - inf` =
NaN, `finish_constraint_step` drops the update, and the run writes `status:
completed`. The lower bound is equally dead in float16, where 1e-8 is below the
smallest subnormal and rounds to 0.

`sum` is untouched because `p(1-p)` never takes a logarithm -- which is why the
three `sum` arms landed 100% in the same campaign and the contrast looked
healthy from every angle except this one.

🔑 **WHY READING THE EXPRESSION NEVER FOUND IT.** Python evaluates `1.0 -
EPSILON` in float64, where it *is* representable (0.99999999). The bound only
dies on the cast. The line is correct in the interpreter and wrong in the
tensor, and no amount of staring at it helps.

#### The fix, and what it does NOT fix

`clamp_probability` in `src/utils/constants.py` takes its epsilon from the
tensor's own dtype, so `1 - eps` is by construction representable wherever the
tensor lives. Both call sites now use it (`uniform_grad_count`, and the
rejected `select` arm's per-item CE, which had the identical bug).

⚠️ **The clamp stops the NaN. It does not give the arm its resolution back.**
At bfloat16 the log-odds now saturate at **+-4.85**; at float32 the same
quantity reaches **+-15.9**. An arm whose count is defined in log-odds must run
under `constraint_fp32: true` or its behaviour is a function of which GPU it
landed on. `uniform1` is relaunched with it set.

✅ Gated by `test_a_probability_clamp_SURVIVES_THE_DTYPE_IT_ACTUALLY_RUNS_IN`,
which asserts finiteness of value AND gradient in float16, bfloat16 and
float32, and AST-scans `src/` for any hand-written `clamp(EPSILON, 1 -
EPSILON)`. Negative control: it FAILS on the shipped code with
`uniform_grad_count returned a non-finite VALUE in torch.float16`, on exactly
the saturated probabilities iwildcam produces.

#### 🔴 AND IT IS NOT ONE ARM. **EVERY FP16 CAMPAIGN RAN AT ~70% DOSE.**

Audited 2026-08-25 over every campaign on the server that records the counts,
5,364 attempted steps:

| campaign | AMP dtype | landed | runs with a lost step |
|---|---|---|---|
| `iwc1` | float16 | **68.8%** | 24 / 32 |
| `iwc2` | float16 | **74.6%** | 8 / 8 |
| `iwc3` | float16 | **68.6%** | **36 / 36** |
| `xfam1` | bfloat16 | **100.0%** | 0 / 108 |

The split is exactly the AMP dtype, and the mechanism is not the clamp bug
above: **FP16 + `GradScaler` SKIPS an optimizer step whose gradient overflows,**
and BF16, which has float32's exponent range, does not. dsisco01 is the FP16
host and dsisco02 the BF16 one, so *which server a campaign landed on set its
constraint dose*, at roughly a third of the phase.

🔑 **Read the SIGN of that before deciding what it costs.** Less dose means
less constraint, and every iwc* finding is that the constraint DAMAGES the
representation. So those numbers are **lower bounds** -- iwc3's 0-of-9 sweep
was produced by two thirds of the intended pressure. The confound does not
threaten the negative results; it would threaten a positive one.

⚠️ **It DOES forbid one thing: comparing an arm across hosts.** Within a
campaign every arm shares the dtype, so the internal contrasts are sound.
Across campaigns they are not, and 2(s)'s cross-family table is BF16 while
2(p-post)'s is FP16.

The provenance was there the whole time: `results.runtime` records `amp_dtype`,
`amp_enabled`, the CUDA version and the device on every run since the field
was added. Nothing read it. This is the same shape as every other defect in
2(e) -- the record existed, the reader did not.

#### 🔍 THE CLASS IS AUDITED TO ITS EDGE, not one instance at a time

Two dead guards were found by accident. The rest of the surface was then
enumerated by AST -- **63 sites in `src/` that take a logarithm, a square root,
or divide by something non-constant** -- and triaged with a reason each, not a
count:

| site | why it is safe |
|---|---|
| `constraint_step._randomize_direction` | `if total > 0` |
| `constraint_step.project_out` | `if nrm <= 0.0: return 0.0` |
| `constraint_step` normalize rescale | only reached when `raw > 0` |
| `hounie_rcl` group means | `max(1, group_sizes[g])` |
| `transductive_loss._penalty` | `scale = K if K >= 1 else 1.0` |
| `reordering` log-odds | eps 1e-6 in **float64**, where `1 - eps` IS representable |
| `LogitAdjustedLoss` | `clamp(min=1e-12)` on a **float32** buffer |
| `select` risk denominators | `+ EPSILON` on a float32 sum that is >= 0 |
| the remaining ~40 | `pathlib` `/`, not division |

**Enumeration is not verification**, so the paths are now EXERCISED by
`test_no_numerical_guard_in_the_TRAINING_PATH_is_a_no_op` with the inputs that
would break them -- a class with no training instances, a zero-norm reference,
an all-zero gradient, K = 0, a saturated softmax -- in all three dtypes.

Negative controls: deleting `project_out`'s zero-reference guard fails it with
`ZeroDivisionError` (the zero case is a Python float division, so it raises
rather than returning nan -- a failure nothing else was checking for), and
deleting `LogitAdjustedLoss`'s clamp fails it on a zero-prior class.

⚠️ **One honest limit, written into the gate.** Not every guard here is
load-bearing for FINITENESS. Replacing `scale = K if K >= 1 else 1.0` with
`float(K)` does NOT fail it: at K = 0 the `+ EPSILON` keeps the quotient finite,
it only makes it enormous. That guard protects the SCALE, which is 2(a2)'s
subject. A gate claiming to cover both would be lying about one.

⇒ **The dead-guard class is CLOSED at two instances**, both fixed, with the
remaining 61 sites checked rather than assumed.

#### ✅ THE FIX IS CONFIRMED ON THE RELAUNCH, 2026-08-25

`results/uniform1` was stopped at 4 of 252 runs, regenerated at the clamp fix
with `--constraint-fp32`, and relaunched. On the first completed run of each
trained arm:

    tralo           29 / 29   100.0%   amp=bfloat16
    tralo_head      29 / 29   100.0%   amp=bfloat16
    tralo_uniform   29 / 29   100.0%   amp=bfloat16      (was 1 / 29)

So the campaign is now measuring the LOSS SHAPE, which is what it was built
to measure, rather than a numerical failure in one of the three arms.

`scripts/dose_landed.py` is the reader, and it is the one command to run
FIRST on a live campaign: no predictions, no pairing, seconds at 1% done. It
names which of the two causes it is looking at -- one arm low is the loss
shape, every arm low is the host -- because the two need different fixes and
only one of them is recoverable by relaunching the same campaign.

#### 🔁 THE SAME DEFECT AT THE OTHER END, found by looking for it

`EPSILON` guards a divisor in `window_temp` as `clamp(t, min=EPSILON)`. In
float16 the smallest SUBNORMAL is 5.96e-8, so 1e-8 rounds to exactly 0 and the
floor becomes `clamp(min=0)`. `margin_window` then computes
`sigmoid(margin / 0)`, which is **NaN at margin 0** -- the items AT the
decision boundary, which are the entire point of a margin window.

Measured 2026-08-25, `sigmoid(m / clamp(0, min=EPSILON))`:

| dtype | clamped temp | result at margin 0 |
|---|---|---|
| float16 | **0.0** | **NaN** |
| bfloat16 | 1.00e-8 | 0.5 |
| float32 | 1.00e-8 | 0.5 |

`tralo_margin` is not in any live campaign, but `docs/launch_margin1.sh`
stages it, and `margin1`'s host would decide whether it NaNs. Fixed by
`clamp_denominator`, whose floor is `finfo(dtype).tiny` (the smallest NORMAL,
not `.eps` -- this bounds a divisor, not a distance from 1). float32 and
bfloat16 are unchanged at 1e-8; float16 becomes 6.10e-5.

⚠️ **SCOPE, stated so the boundary is a decision and not an oversight.** The
gate covers CLAMPS only. `x / (s + EPSILON)` has the same rounding, but
whether it is safe depends on `s` -- in the two live cases `s >= 1` -- so
flagging the additive form would be noise, not a finding.

#### 📌 PRE-REGISTERED: `results/iwc4`, written 2026-08-25 BEFORE the launch

`docs/launch_iwc4.sh`. iwc3's exact design -- 9 cells, `tralo` / `tralo_null` /
`tralo_reseed` / `clip` / `focal_clip`, 4 seeds, 180 runs -- with **one knob
different**, `--constraint-fp32`, on **dsisco01 on purpose**. The BF16 host
already lands 100%, so running it there would answer the first question below
and silently skip the second.

1. **Does the 0-of-9 sweep survive at full dose?** iwc3 is a LOWER bound (less
   dose, less constraint, and the finding is that the constraint damages), so
   the prediction is *at least as negative*. **FALSIFIED IF** `tralo` loses AP
   in fewer than 8 of 9 cells, or by a margin smaller than iwc3's -0.0394 --
   in which case 2(p-post) is a DOSE result and is retracted as a constraint
   result.
2. **Does `constraint_fp32` remove the FP16 loss?** Read straight off the
   `CONSTRAINT DOSE` block on the FIRST completed run. If `tralo` is not at
   100%, the campaign is measuring the scaler and must be stopped, not scored.

The answer to (2) decides how every archived FP16 number is read, and it costs
nothing extra to ask -- which is the whole reason this campaign is on the FP16
host rather than the free BF16 one.

✅ **QUESTION (2) IS ANSWERED, on iwc4's FIRST trained run, 2026-08-25:**

    tralo   29 / 29   100.0%   amp=float16      (iwc3, same host, no flag: 68.6%)

`--constraint-fp32` removes the FP16 GradScaler loss **entirely**, on the host
that produced it. So the ~31% shortfall in `iwc1`, `iwc2` and `iwc3` is a
FIXABLE setting and not a property of the hardware, and every number from those
three is a **lower bound on constraint effect that can now be lifted**. It cost
one run to learn, against 292 runs of not knowing.

✅✅ **QUESTION (1) IS ANSWERED TOO, AND THE PRE-REGISTRATION HELD.**
Read at 106 of 180 runs, 9 cells x 3 complete seeds, 2026-08-26. The condition
fixed before launch was: **`tralo` at 100% dose loses AP in at least 8 of 9
cells, by a margin no smaller than iwc3's -0.0394.** Measured:

| | iwc3, 68.6% dose, FP16 | **iwc4, 100% dose, FP16 + fp32** | uniform1, 100%, BF16 |
|---|---|---|---|
| AP | -0.0394 | **-0.0585  8 of 9  \*\*\* q=0.0123** | -0.0754  9 of 9 |
| AUROC | -0.0094 | **-0.0136  7 of 9  \*\*\*** | -0.0172  9 of 9 |
| NLL | +0.2545 | **+0.4923  9 of 9  \*\*\*** | +0.3206  9 of 9 |

⇒ **The damage is 1.5 to 1.9x LARGER at full dose, on two different hosts.**
iwc3 was a lower bound exactly as 2(u) predicted, the sweep survives, and
2(p-post) is CONFIRMED rather than retracted. Note the cross-host replication
was free: uniform1's `tralo` arm ran the same three backbones and the same
three caps on the BF16 host, so it is an independent replicate of iwc4 and it
agrees in sign, in size and in cell count.

#### 🔑 WHAT IS ATTRIBUTABLE AND WHAT IS THE SEED -- FINAL, 180/180, 4 seeds

⚠️ **CORRECTION, 2026-08-27.** Read at 106/180 with 3 seeds, `tralo` and
`tralo_reseed` both showed macroF1 -0.0156 and this section said the macro-F1
damage **was** the reseed floor, 1.00x, to four decimals. **At the full 4
seeds that is wrong**: the ratio is 1.51x, and the metric that is actually
indistinguishable from a reseed is **macroP**, not macroF1. The four-decimal
agreement was a three-seed coincidence. Never read an attribution ratio off a
partial campaign -- the floor moves more than the treatment does.

| iwc4 final, vs `tralo_null` | `tralo` | `tralo_reseed` (RNG only) | ratio |
|---|---|---|---|
| AP | -0.0572 \*\*\* 0/9 | +0.0030 tie | **19.1x -- ATTRIBUTABLE** |
| AUROC | -0.0146 \*\*\* 0/9 | +0.0005 tie | **29.2x -- ATTRIBUTABLE** |
| NLL | +0.3297 \*\*\* 0/9 | +0.0019 tie | ATTRIBUTABLE |
| Brier | +0.0632 \*\*\* 0/9 | +0.0041 tie | ATTRIBUTABLE |
| **macroP** | **-0.0136 \*\*\* 0/9** | **-0.0135 \*\*\* 0/9** | **1.01x -- NOT ATTRIBUTABLE** |
| macroF1 | -0.0157 \*\*\* 0/9 | -0.0104 \*\*\* 0/9 | 1.51x -- two thirds is seed |
| macroR | -0.0102 1/8 | -0.0061 2/7 | 1.67x |
| acc | -0.0121 \*\*\* 0/9 | -0.0070 \*\*\* 0/9 | 1.73x |
| ccF1 | -0.0007 tie | +0.0011 tie | 0.64x -- the RESEED moves it more |

🛑 **The warning survives the correction and is arguably worse stated
correctly.** Perturbing the RNG stream and changing nothing else costs
-0.0104 macro-F1 **in 9 of 9 cells** -- two thirds of the entire constraint
effect, and systematic rather than noisy. macro-F1 is carried by the UNCAPPED
classes and **macro-F1 is what the paper headlines** (2(a)). A macro-F1 delta
of this size quoted without its `tralo_reseed` twin beside it is mostly
measuring the seed. On macroP it is measuring nothing else at all.

✅ The ranking channel is the opposite and it got STRONGER at 4 seeds: the
reseed floor on AP and AUROC settled to a tie, so the ratios rose from 5.8x
and 7.2x at three seeds to **19.1x and 29.2x**.

The falsification condition above is left exactly as it was written before
launch, and is not to be edited now that it has been met. The point of
fixing it in advance was that it COULD have come back positive, in which
case the damage would have been a dosing artefact and 2(p-post) would be
retracted here instead.

⚠️ **`--constraint-fp32` CHANGES TWO THINGS, and the pre-registration owns**
**that.** It raises the step COUNT (68.6% -> 100% is the prediction) *and* it
runs the constraint pass at float32 rather than float16, so the step DIRECTION
also stops carrying fp16 rounding. The two are not separable in this campaign.
They are not equal in size, though: under `constraint_grad_mode: clip` the
delivered step is `min(raw, K)` and 2(a3) measured that the clip binds at
exactly 1.000 against raw norms of 2,560-12,400, so MAGNITUDE is void and only
direction and count are live. A precision change perturbs direction at the
level of rounding; the dose change is a third of the phase. **Report it as a
dose result with a precision confound, never as a clean dose result.**

🛑 **THE GENERAL RULE, and it is cheap to apply.** `full_panel` already prints
`CONSTRAINT DOSE -- steps that LANDED, against steps attempted` and already
refuses to let two arms at different landing rates be compared silently. **Read
that block FIRST, on the first completed runs, not at the end.** It cost 4 runs
here. At `iwc3` it would have cost 180 and did.

### (v) 🛑🛑🛑 AT EVERY CAP THIS PROTOCOL SWEEPS, THE WHOLE PRIZE IS BELOW THE SEED NOISE

`python -m scripts.headroom results/iwc3`, 2026-08-25, 9 cells, 4 seeds, control
`clip`. This is the gap to a PERFECT RANKING -- not to a better method, and not
to a better allocator, which is already optimal given these probabilities.

| cap | class | n | K | ceiling `2K/(K+n)` | achieved | **headroom in ITEMS** |
|---|---|---|---|---|---|---|
| L20_G50 | 2 | 370 | 74 | 0.3333 | 0.3333 | **-0.0** |
| L20_G50 | 7 | 456 | 92 | 0.3358 | 0.3333 | **0.7** |
| L30_G50 | 2 | 370 | 111 | 0.4615 | 0.4615 | **0.0** |
| L30_G50 | 7 | 456 | 137 | 0.4621 | 0.4587 | **1.0** |
| L50_G30 | 2 | 370 | 111 | 0.4615 | 0.4615 | **0.0** |
| L50_G30 | 7 | 456 | 137 | 0.4621 | 0.4621 | **-0.0** |

**The entire prize is 0.0 to 1.0 items, and in FOUR of the six it is exactly
zero.** iwc3's own paired seed sd is **2.11 items**. The prize is smaller than
the noise, on every cell, by construction rather than by bad luck.

#### WHY -- and it is arithmetic, not a property of any method

When you may emit only `K` predictions for a class with `n` true instances, the
best possible cc-F1 is `2K/(K+n)`: precision 1, recall `K/n`. On iwildcam `K` is
**74 to 137 against n = 370 to 456**, so the budget is 16-30% of the true
positives, and the top-K set is drawn from a pool of positives four to six times
its own size. The model fills it with correct items already -- `ccP` is **0.9954**
against its lambda=0 twin -- so `achieved` equals `ceiling` to four decimals.

🔑 **There is nothing left for a ranking to win.** Every score-pushing arm this
project has built or proposed -- `rank`, `rankpair`, `budget_margin`, the
pairwise hinges, `select` -- is trying to reorder items inside a set that is
already all-correct. That is why they tie, and it is NOT evidence about the
idea. It is the ceiling.

#### THE SCREEN, so this is asked BEFORE the next dataset is downloaded

`scripts/ceiling_screen.py` prices a candidate from LABELS AND THE CAP POLICY
alone -- no images, no model, no GPU -- because `K` is computable from them and
the bound is arithmetic:

```
python -m scripts.ceiling_screen data/iwildcam/oodslice \
    --caps L20_G50 L30_G50 L50_G30 --classes 2 7
```

| cap | class | n | K | K/n | ceiling | prize | vs noise |
|---|---|---|---|---|---|---|---|
| L20_G50 | 2 | 370 | 74 | 20.0% | 0.3333 | 0.34 | 0.16x |
| L20_G50 | 7 | 456 | 92 | 20.2% | 0.3358 | 0.42 | 0.20x |
| L30_G50 | 2 | 370 | 111 | 30.0% | 0.4615 | 0.51 | 0.24x |
| L30_G50 | 7 | 456 | 137 | 30.0% | 0.4621 | 0.63 | 0.30x |
| L50_G30 | 2 | 370 | 111 | 30.0% | 0.4615 | 0.51 | 0.24x |
| L50_G30 | 7 | 456 | 137 | 30.0% | 0.4621 | 0.63 | 0.30x |

It reproduces `headroom`'s K and ceiling **exactly** -- 74 / 92 / 111 / 137 and
0.3333 / 0.3358 / 0.4615 / 0.4621 -- with no model and no predictions, which is
an independent check on both tools. It also names the binding scope per row, so
the inert-global bug that cost 30x cannot come back silently: at `L50_G30` the
GLOBAL binds (111 against a local sum of 185) and at `L20_G50` the LOCAL does
(74 against a global of 185).

It can say yes. At `p = 0.95` a `K = 300` budget prices at 15 items and reads
**WORTH RUNNING**; a screen that only ever refuses decides nothing, and that is
in its `--self-test`.

⚠️ **`dataset_screen` and `ceiling_screen` are INDEPENDENT and a candidate needs
BOTH.** The first asks whether the counts carry information the training set
lacks; the second asks whether there is anything to win with it. iwildcam passes
the first at **+3131 items, z = 97.4** and fails the second at **0.34-0.63**.

#### ⚠️ CORRECTION TO THIS SECTION, made the same day it was written

The heading says the ceiling is *already reached*, and that is true **only at
the cap levels this protocol sweeps**. Measured off the same 36 `clip` runs, at
budgets the protocol has never run, the prize is NOT zero -- and neither is the
noise. Both are in items; `prize = (1 - p@K) * K` is the gap to a perfect
ranking and `seed sd` is the within-cell sd of TP@K across the 4 seeds:

| K/n | class 2 p@K | prize | seed sd | prize/sd | class 7 prize | seed sd | prize/sd |
|---|---|---|---|---|---|---|---|
| **20%** | 0.9944 | 0.42 | 0.80 | **0.52x** | 0.00 | 0.00 | -- |
| **30%** | 0.9895 | 1.17 | 1.96 | **0.60x** | 0.00 | 0.00 | -- |
| 40% | 0.9854 | 2.17 | 3.66 | 0.59x | 0.17 | 0.41 | 0.41x |
| **50%** | 0.9779 | 4.08 | 6.20 | **0.66x** | 0.42 | 0.73 | 0.57x |
| 60% | 0.9688 | 6.92 | 8.36 | 0.83x | 1.50 | 2.39 | 0.63x |
| 70% | 0.9556 | 11.50 | 10.44 | **1.10x** | 3.58 | 4.42 | 0.81x |
| 80% | 0.9392 | 18.00 | 11.52 | **1.56x** | 11.08 | 7.79 | 1.42x |
| 90% | 0.9104 | 29.83 | 13.45 | **2.22x** | 24.58 | 12.91 | 1.90x |

(bold rows are `L20`, `L30`, `L50` -- the only levels the protocol runs.)

🔑 **THE PRIZE AND THE NOISE GROW TOGETHER, and the ratio is what matters.**
A looser cap does buy headroom -- 0.42 items at 20% becomes 29.83 at 90% -- but
it cuts deeper into the contested middle and the allocation noise rises with
it, which is 2(i) measured on this dataset. `prize/sd` is **0.41 to 0.66 at
every cap the protocol sweeps**, crosses 1.0 only above K/n ~ 70%, and reaches
2.2x at 90%.

⇒ **Three statements, in decreasing strength, and only the first two are
safe:**

1. **At L20 / L30 / L50 the entire gap to a PERFECT ranking is smaller than the
   seed noise.** A method capturing 100% of it would still not be detectable at
   4 seeds. This is the closure that matters, because it is the region every
   campaign in this project has run.
2. **The ratio is monotone in K/n and the protocol has never gone above 0.5.**
   If a looser cap is ever run, price it here first -- and re-measure the sd
   there rather than reusing 2.11, which is a `L20`-`L50` number.
3. ~~Nothing can ever be won on iwildcam~~ is NOT established. What is
   established is that at K/n <= 0.5 nothing can be MEASURED. At K/n = 0.9 a
   method capturing half the prize would sit at ~1.1x the seed sd, which is
   still not enough at 4 seeds -- but that is a power statement, not a null.

⚠️ And a looser cap costs the research question something real: at K/n = 0.9 the
budget admits 90% of the true positives, so the constraint barely constrains.
**Say what the cap is buying before proposing to loosen it.**

### 2(w2b) 🛑 **THE FACTORIAL GATE DECIDES, NOT THE NET COLUMN**
### **-- 21 candidates screened, and stage 1 picks the wrong winner**

⛔⛔ **CORRECTED 2026-09-01 -- THE TWO TOP-RANKED ROWS OF THE TABLE BELOW WERE
NEVER MEASURED, AND ONE OF THEM WAS THIS GATE'S OWN POSITIVE CONTROL.** See
2(w2c). `factorial_control` split `location` on `--sep` and read tokens `[0]`
and `[-1]`. When the separator is ABSENT both are the whole string, `f0 == f1`,
every unseen group kept `p_glob`, and the additive arm WAS the global arm -- so
`survives` came out ~100% by arithmetic, with the 0.1-0.2% scatter supplied by
the null draw. **8 of the 21 candidates rake ZERO groups**, `iwildcam` and every
`fmow` slice among them, because a camera and a country are ATOMIC. The claim
below that iwildcam "supplies" the positive control is therefore void: an atomic
dataset cannot supply one, and this gate had none until `--self-test` grew a
synthetic slice on which raking is exact. **The ranking RULE survives; the
ranking does not** -- an atomic group has no survival number to rank. The
FACTORIAL rows reproduce to the decimal, so every conclusion resting on one of
those stands. Read the corrected table in 2(w2c), not this one.

Audited 2026-08-28. **21 candidate slices were already staged in `~/_cand` on
dsisco02** (fmow x6, ISIC x7, BCN x3, ISIC-archive x2, Fitzpatrick x2,
DomainNet) and had never been screened as a set. `dataset_screen` runs on all
of them in minutes -- labels and metadata only, no images, no GPU.

`scripts.factorial_control` then credits the model with **interpolating a
product group's two factors**. An ATOMIC group must return **~100%**: there is
nothing to interpolate. That is the positive control, and `iwildcam` supplies
it.

| slice | modality / group | NET | LOCAL | GLOBAL | unseen | **survives** |
|---|---|---|---|---|---|---|
| **`iwildcam/oodslice`** | camera trap / camera | +3133 | +3531 | +994 | 7 | **100.1%** |
| **`fmow_s1`** | **satellite / COUNTRY** | +2767 | +2817 | +712 | 10 | **100.1%** |
| `fmow_country` | satellite / country | +2969 | +3100 | +1087 | 10 | (same family) |
| `isicarch_instsite` | dermoscopy / inst x site | +2010 | +2169 | +572 | 7 | 86.3% |
| `bcn_s2` | dermoscopy / BCN | +2031 | +2029 | **+91** | 8 | **50.4%** |
| `isic_ssa` | dermoscopy | +1751 | +1811 | +360 | 10 | **30.8%** |
| `isic_siteage` | dermoscopy / site x age | +2169 | +2204 | +464 | 7 | **17.6%** |
| `domainnet` | -- | +57 (z=1.2) | +365 | +308 | 2 | ⛔ DEAD |
| `isic_src` | 1 group | **-141** | +5576 | +5581 | 1 | ⛔ DEAD |

🔑 **STAGE 1 PICKS THE WRONG WINNER, AND THIS IS THE POINT.** `bcn_s2` has the
cleanest differential in the whole inventory -- its global excess is **+91**
against a local **+2029**, i.e. **4%**, better than iwildcam's 28%. Then the
factorial gate **halves it**. And `isic_siteage`, the second-best on stage 1,
**collapses +2169 -> +380**: "site x age" is a PRODUCT whose factors both
appear in training, so the model interpolates. ⇒ **Rank candidates on
`factorial_control` SURVIVAL, never on the NET or the NET/GLOBAL ratio.**

🛑 **Two more NET-column traps in the same run**, both invisible in the local
number alone. `isic_src` reads LOCAL **+5576** at z=92.5 -- the largest local
figure in the inventory -- and GLOBAL **+5581**, so its entire local signal IS
one global shift and NET is **-141**. `isicarch_inst` is LOCAL +5796 against
GLOBAL +5203 (90%). **A huge LOCAL with no NET is the dermmnist failure mode
(2(n)), and it recurs.**

⇒ **THE THREE-DATASET PLAN**, across three genuinely different modalities:
`iwildcam` (camera trap, RUNNABLE), **`fmow`** (satellite by country, the only
clean candidate, 100.1%), and `isicarch_instsite` (dermoscopy, 86.3%, caveated).

🎯 **The fmow ask is ~21,100 images** (17,670 train + 3,442 test), because it
is a SLICE -- a few GB, not the 3.5 TB corpus. That makes it a real decision.
⚠️ Both gates are still **necessary only**: dermmnist scored +65 at z=2.9 and
nulled anyway. Stage 2 is `scope_probe --calibrate` and needs a trained model.


### 2(w2c) ⛔ **THE FACTORIAL GATE COULD NOT FAIL, AND ITS RANKING MIXED**
### **A MEASUREMENT WITH AN ARITHMETIC IDENTITY -- 21 candidates re-run**

Found and fixed 2026-09-01. Two defects in `scripts/factorial_control`, both
pushing `survives` toward a reassuring 100%, which is the direction that keeps
a bad candidate alive.

**1. THE PERCENTAGE WAS REACHABLE WITHOUT RAKING A SINGLE GROUP.**
`s.str.split(sep)[0]` and `[-1]` both return the WHOLE string when `sep` does
not occur, so `f0 == f1`, the `f0 != f1` guard sent every unseen group to
`q = p_glob`, and `units_add` was element-wise EQUAL to `units_glob`. `obs` was
then bit-identical between the arms and only the null draw moved the number.
Measured on a synthetic atomic slice, **three different separators return the
same 99.6%** (net_glob +393 vs net_add +392) with 0 of 6 groups factorised. So
a genuinely atomic group and a WRONG `--sep` were indistinguishable, and both
read as PASSED. ✅ Fixed: `raked` is counted and printed, and `raked = 0`
prints **NOT A CONTROL** instead of a figure. The two arms now share their null
draws, so an un-raked slice returns *exactly* equal nets rather than a
plausible 99.6.

**2. THE RATIO WAS DILUTED BY THE SEEN GROUPS.** `survives` spanned the whole
slice, but the arms differ only on the UNSEEN units -- every seen group
contributes identically to both -- so the figure was dragged toward 100% in
proportion to the seen share. On a slice built so that raking is EXACTLY right,
6 seeds per row:

| unseen share of test | shipped ratio | unseen-only ratio |
|---|---|---|
| 65.2% | 87.1% | 81.4% |
| 20.0% | 47.5% | **19.6%** |
| 7.0% | 76.0% | **21.2%** |
| 2.4% | 91.9% | **26.1%** |

The right-hand column is flat, as it must be; the shipped column was reading the
unseen SHARE. ✅ `survives` is now the unseen-only ratio and the diluted figure
is printed beside it, labelled. 🔑 **No published candidate number moved**,
because every `~/_cand` slice is built 100% unseen -- which is exactly why this
needed a test and not a note.

**THE CORRECTED TABLE.** All 21 candidates, re-run on CPU in minutes:

| slice | group | NET | unseen | raked | **survives** |
|---|---|---|---|---|---|
| `isic_siteage` | site x age | +2168 | 7 | 7/7 | **17.6%** |
| `isic_srcage` | src x age | +1533 | 7 | 7/7 | **30.7%** |
| `isic_ssa` | src x site x age | +1749 | 10 | 10/10 | **31.0%** ⚠️ 3 factors, 2 used |
| `bcn_s2` | site x age | +2033 | 8 | 8/8 | **50.4%** |
| `bcn_s1` | site x age | +1567 | 9 | 9/9 | 68.7% |
| `isic_srcsite` | src x site | +1123 | 12 | 12/12 | 72.4% |
| `isic_bcn` | site x age | +1704 | 10 | 10/10 | 83.6% |
| `isicarch_instsite` | inst x site | +2012 | 7 | 7/7 | **86.3%** |
| `fitz_atlasfst` | atlas x FST | +369 | 6 | **2/6** | 90.5% ⚠️ 4 groups ungated |
| `bcn_s3` | site x age | +1596 | 10 | 10/10 | **128.2%** ⚠️ see below |
| `iwildcam/oodslice` | **camera** | +3130 | 7 | **0/7** | ⛔ NOT A CONTROL |
| `fmow_s1` | **country** | +2766 | 10 | **0/10** | ⛔ NOT A CONTROL |
| `fmow_country` | country | +2968 | 10 | **0/10** | ⛔ NOT A CONTROL |
| `fmow_check` | country | +2968 | 10 | 0/10 | ⛔ NOT A CONTROL |
| `fmow_s2` / `fmow_s3` / `fmow_country_wide` | country | +2309 / +2401 / +2188 | 11 / 9 / 12 | 0 | ⛔ NOT A CONTROL |
| `isicarch_inst` | institution | +2557 | 4 | 0/4 | ⛔ NOT A CONTROL |
| `isic_site` | site | +1484 | 2 | 0/2 | ⛔ NOT A CONTROL |
| `fitz_skintype` | FST | +180 | 2 | 0/2 | ⛔ NOT A CONTROL |
| `domainnet` | domain | +57 (z=1.2) | 2 | 0/2 | ⛔ DEAD on stage 1 |
| `isic_src` | 1 group | **-144** | 1 | 0/1 | ⛔ DEAD on stage 1 |

🔑 **WHAT ACTUALLY CHANGES.** The factorial rows reproduce the old table to
the decimal (`isicarch_instsite` 86.3, `bcn_s2` 50.4, `isic_siteage` 17.6,
`isic_ssa` 30.8 -> 31.0), so the `isic_siteage` collapse and the rule "rank on
SURVIVAL, never on NET" both stand. What falls is the TOP of the ranking:
`iwildcam` and `fmow` were never on the same axis as the rest. **fmow remains
the clean second dataset** -- a country is atomic, so there are no factors to
interpolate and 2(n)'s baseline is sound there -- but the ground is "the gate
does not apply", NOT "it scored 100.1%". Quote it that way.

⚠️ **THREE ROWS CARRY A CAVEAT THE OLD TABLE COULD NOT SHOW.**

- `isic_ssa` is a **THREE**-factor group (`BCN|anterior torso|60s`) scored as
  two: only the FIRST and LAST tokens are used, so `site` is silently dropped
  and its 31.0% credits the model with interpolating src x age only. The tool
  now says so per slice.
- `fitz_atlasfst` rakes **2 of 6** groups; the other four have a factor level
  that is itself unseen, so two thirds of its 90.5% is the ungated baseline.
- `bcn_s3` reads **128.2%** -- raking WORSE than the global prior. The old
  docstring blamed "raking is noisy on a small training set". It is the SHIFT:
  `net_expect` applies the GLOBAL test-vs-train label shift to both baselines,
  and at a 100% unseen share that shift is computed largely FROM these very
  groups, so a raked baseline is corrected twice. Measured -- the raking
  estimate sits **0.014** from the truth in L1 and **0.126** after the shift,
  while the global baseline is IMPROVED by it. **Every `~/_cand` slice is 100%
  unseen**, so this caveat covers the whole table: the item counts are safer
  than the ratio. It is `dataset_screen`'s own definition of NET, so it is not
  fixable inside this tool -- the tool now prints the unseen share and warns
  above 50%.

✅ GATED IN BOTH DIRECTIONS, in `--self-test` and in two regression tests
(`test_the_factorial_gate_cannot_report_a_pass_it_did_not_measure`,
`test_the_factorial_gate_reports_the_undiluted_ratio`): a synthetic slice whose
held-out cell IS the product of the observed training marginals must MEASURE
(19.7% over 6 seeds), and the same slice under a wrong separator must REFUSE.

⛔ **THE OTHER HALF OF THE SAME GATE HAD THE SAME SHAPE: `ceiling_screen`
KILLED A CANDIDATE ON iwildcam's RANKING QUALITY.** Fixed 2026-09-01. Its
`p@K` and seed sd come from one measured curve -- iwc3, 36 `clip` runs -- and
the docstring said loudly that it does not transfer, while the VERDICT column
went on printing `*** PRIZE BELOW THE NOISE` for datasets whose own curve
nobody has measured. iwildcam sits at **p@K 0.9948-0.9972**; 2(w2) prices fmow
at **p@K <= 0.92**, a bar iwildcam's numbers can neither pass nor test. A
prose warning beside a kill verdict is a kill verdict. ✅ Now: a BORROWED
calibration prints `UNPRICED HERE, needs p@K <= 0.8856` per cell -- the number
to go and measure -- refuses to count any cell as worth running, says
`NOTHING WAS DECIDED`, and exits **3**, not 1, so no caller can read it as
dead. `--native-calibration` (inferred for any path under `iwildcam`) restores
the deciding behaviour, and passing a measured `--ccp/--noise` does too.

⚠️ **AND ITS CLAMP WAS SILENT.** `calibrated` returns the nearest ENDPOINT
outside K/n 0.20-0.90 as though it were interpolated -- and the per-class caps
now in the protocol run to **K/n = 1.00** (`L80-100_G95`), so the clamp fires
in the live campaigns, not in a corner. It now returns a third value saying so
and the row prints `!! K/n 1.00 is OUTSIDE the measured 0.20-0.90`. Both are
gated in `--self-test` and in
`test_the_ceiling_screen_cannot_kill_a_dataset_it_never_measured` /
`test_the_ceiling_screen_says_when_its_curve_was_extrapolated`, each in both
directions -- iwildcam itself must still reach a verdict.

🔑 **SO THE fmow DECISION NEEDS ONE MEASUREMENT, AND ONLY ONE.** Both
gates on the second dataset were returning iwildcam's answer: the factorial
gate by arithmetic identity, the ceiling screen by borrowed calibration.
Neither now claims anything about fmow. What is still true is that fmow's
group (country) is ATOMIC, so 2(n)'s baseline is sound there, and its stage-1
NET is +2766 at z=80.4. The open number is **fmow's own p@K at the cap**, which
needs a finished unconstrained run on fmow -- i.e. the images.


### 2(z24b) 🛑 **ALL FOUR WINDOWS RE-MEASURED PER SEED -- THE**
### **STRICT ONES ARE HALF THE WIDTH, AND `taskwin2` RUNS ON THE EDGE**

2(z24) found that `task_window` applied `MIN_FORCED` to the MEAN unconstrained
count and fixed the TOOL. `configs/task_windows.yml` was never regenerated, so
the file every launch is gated against still carried the mean-based ranges.
Re-measured 2026-09-01 on the same reference runs, reading `binds n/N`:

| backbone | class 2 STRICT (4/4) | class 7 STRICT (4/4) | old (mean) c2 | old c7 |
|---|---|---|---|---|
| **ViTB16** | **0.60-0.70** | **0.90** | 0.60-0.90 | 0.90-1.00 |
| **MobileNetV3** | **0.70** | **0.90** | 0.70-0.90 | 0.90-1.00 |
| **MobileNetV2** | **0.80** | **0.80** | 0.80-1.00 | 0.80-0.90 |
| **RegNetY400MF** | **0.60-0.80** | **0.80-0.90** | 0.60-0.90 | 0.80-1.00 |

Everything the old file called a task above those ranges is PARTIAL: e.g.
MobileNetV3 class 2 binds in **3 of 4** seeds at 0.80 and 0.90 and **1 of 4**
at 1.00; class 7 binds **2 of 4** at 1.00.

✅ **PARTIAL IS RECORDED, NOT REFUSED.** A slack seed has
`relu(hard-K) = 0` on the class total, so it dilutes toward zero -- a positive
measured in a partial cell is CONSERVATIVE. But its effective n is below its
seed count, so a NULL there is weaker evidence than a null in a strict cell,
and the two must never be quoted the same way. The yml now carries `class:`
(strict) and `partial:` per backbone; `classify` returns **task / partial /
unmeasured / non_task**; `gen_campaign` takes strict silently and partial with
a printed label naming the ratio. Refusing partial outright would leave
MobileNetV2 with exactly ONE legal cap and MobileNetV3 with one -- too narrow
to run an experiment in.

🛑 **AND A FOURTH STATUS WAS NEEDED, BECAUSE ONE LIVE CAP IS IN A GAP
NOBODY MEASURED.** The windows come off a 0.1 grid. `L80-100_G95` puts
MobileNetV3 class 7 at **K/n = 0.950** -- exactly halfway between the strict
0.90 and the partial 1.00, and **ten times** the 0.005 snapping tolerance from
either. Calling that `non_task` claims a measurement nobody took; calling it
`task` is worse. It now reports **`unmeasured`**, with the nearest measured
fraction printed beside it.

🔑 **WHAT THIS SAYS ABOUT THE TWO CAMPAIGNS IN FLIGHT:**

| campaign | cap | verdict |
|---|---|---|
| `taskwin2` (MobileNetV3) | `L70-90_G95` | ⛔ **`no_strict_band`** -- read ✅ STRICT here on 2026-09-01, re-measured EMPTY 2026-09-02 (see the banner below) |
| `taskwin2` (MobileNetV3) | `L80-100_G95` | ⚠️ class 2 **PARTIAL 3/4**, class 7 **UNMEASURED** at 0.950 |
| `vittask1` (ViTB16) | `L60-90_G95` | ✅ **STRICT**, 4/4 both classes |
| `vittask1` (ViTB16) | `L70-90_G95` | ✅ **STRICT**, 4/4 both classes |

So **`vittask1` is clean on both cells** -- and it is the first ViTB16 campaign
that is, which retires "ViTB16 has zero strict task cells at any cap ever run"
(true of L20/L30/L50; `vittask1` is the campaign that fixes it). ⛔ **`taskwin2`
is clean on NEITHER half -- this read "clean on half" and that is WITHDRAWN,
2026-09-02.** Its `L70-90_G95` half was re-measured `no_strict_band` under the
per-group prize (the row above), so it does NOT carry the arm-vs-arm claim; the
`L80-100_G95` half is a second reading whose cell was never characterised at
its own K/n, and it must be labelled that way wherever it is quoted.

⛔ **AND THAT COSTS A UNIT, NOT ONLY A CELL.** `taskwin2` / MobileNetV3 is
ledger unit **C1** in `scripts/paper_rows.MEASURED_UNITS`, and with both cells
non-task it contributes **ZERO** verified `task` cells. So the headline is
**4/4 units, sign p=0.0625** over the licensed set, and
**3/3 units, sign p=0.125** restricted to units that carry a verified `task`
cell -- every sign unchanged, C1 being also the dissent on the reseed contrast.
`scripts/paper_rows.py` computes and prints "UNITS CARRYING AT LEAST ONE
VERIFIED `task` CELL: N of M"; read the restriction there. 2(z26-CORRECTED).

🔑 **ONE MORE MEASURED CHANGE.** The two classes' windows no longer
differ on EVERY backbone: **MobileNetV2's strict windows COINCIDE at
0.80/0.80**, so it is the one backbone where the plain single-fraction form
`L80_G95` expresses a valid experiment. The per-class form is still required on
the other three. Gated both ways in `configs/task_cells.py --self-test`.

### 2(z25) 🛑 **THE PROBES RETURNED A PLAUSIBLE DEFAULT INSTEAD OF**
### **REFUSING -- four sites, one defect class, and the FIFTH inert flag**

Audited 2026-09-01. The four offline probes decide which directions get a GPU
campaign, so a probe that answers when it cannot measure is more expensive than
one that crashes. All four failures below are the SAME shape: a code path
returns something readable, every gate stays green, and the output is
indistinguishable from a real measurement.

| tool | what it returned | what was true |
|---|---|---|
| `factorial_control` | `survives 100.1%`, PASSED | 0 of 7 groups factorised (2(w2c)) |
| `factorial_control` | ratio over the whole slice | the arms differ only on the unseen units |
| `ceiling_screen` | `*** PRIZE BELOW THE NOISE` | p@K and sd were another dataset's |
| `ceiling_screen` | interpolated p at K/n = 1.00 | the curve stops at 0.90; that is the endpoint |
| `straddle_probe` | `reachable = 0` at the measured delta | the ARM was byte-identical to its null |
| `graph_probe` | a complete report, no file | `--dump` was declared and never read |
| `order_probe` | `=> TIE ... INDISTINGUISHABLE from a reseed` | ZERO points differed -- nothing was compared |
| `tralo` + `ortho_project` | an unprojected constraint step | the CE reference was non-finite; no trace in the log |

⛔ **THE INERT-FLAG COUNT IS FIVE, AND THE FIFTH ONE IS MINE.** CLAUDE.md
rule 3 counts four (`rho_step`, and `base_loss`/`focal_alpha`/`focal_gamma` in
`arm_joint`). `graph_probe --dump` is the fifth: it parsed, the probe ran to
completion over 384 runs, printed its entire report, exited 0, and wrote
nothing. It took two full 15-minute runs to notice, because the first empty
result read as a missing `/tmp` file. `audit_config` covers config KEYS;
**nothing covered argparse DESTINATIONS**, which fail the same way and just as
silently. ✅ `test_every_probe_flag_is_actually_read` now walks the AST of all
six probes and requires every `--flag` to be read as `args.<dest>` somewhere in
its module. Negative control: run against `graph_probe.py` at the parent
commit it reports `['dump']`.

🛑 **`straddle_probe` WOULD HAVE CLOSED A DIRECTION ON AN INERT ARM.** Its
delta ladder is anchored on `q95` of `|p_treated - p_null|`, so an arm
byte-identical to its own twin gives delta = 0, `reachable = 0` in every band,
and the report reads *"the constraint as configured cannot collect the oracle
gap however it is tuned"* -- a physics claim about the cut, produced by an arm
that never ran. This is not hypothetical here: `cb_lp`'s raw predictions are
byte-identical to `clip`'s in 24 of 24. ✅ Inert twins are now named per pair
and excluded from the aggregate; if EVERY pair is inert the probe prints
`NOTHING WAS MEASURED` and exits **3**, not 0.

🛑 **TWO MORE OF THE SAME SHAPE, FOUND WITH THE TABLE ABOVE AS THE
SEARCH PATTERN.** `order_probe.verdict` branched on `if not n_g or p_g >=
alpha`, so "the effect is a coin flip" and "not one point differs" printed the
SAME verdict -- and the TIE branch goes on to explain the mechanism (a
monotone map on the logit channel cannot reorder), which reads as a CONFIRMED
account for a run that compared nothing. `n_g` counts points where the arm and
its reseed differ AT ALL, so zero is an inert arm or an empty glob. ✅ Now
`NOTHING TO TEST`, with the mechanism paragraph suppressed, and both liveness
directions gated -- a genuine 48-point coin flip must still read TIE.

🛑 And in the TRAINER: `snapshot_grads` returns `None` whenever any CE
gradient is non-finite, which is routine on the FP16 path, and
`finish_constraint_step` then takes an **UNPROJECTED** step. The arm keeps its
name, writes `status: completed`, and nothing said which epochs got the
treatment -- the same shape as the dose defect `dose_landed` exists for. ✅
Counted and warned per epoch. (Latent today: no live arm sets
`ortho_project`, and `ortho_survival` measures that the projection delivers
**0.0%** of its promised CE-neutrality anyway.)

✅ **AND THREE PROBES THAT MADE DIRECTION-CLOSING CLAIMS HAD NO
`--self-test` AT ALL.** `test_EVERY_script_offering_a_self_test_actually_PASSES_it`
DISCOVERS them rather than enumerating, so each is gated the moment it lands.
15 of the 45 scripts now carry one.

- **`graph_probe`** -- 2(g) is a NULL, and a null is worth nothing unless the
  instrument could have said otherwise. The gate builds the case diffusion is
  supposed to win (positives clustered in feature space, scores noisy) and
  requires a real gain there -- **+13 of 105 items** -- then requires the
  shuffled-feature control to take it away (**-1**). It also writes a dump and
  reads it back, because `--dump` was inert for its whole first life.
- **`dataset_screen`** -- it CLOSED octmnist and tissuemnist and OPENED fmow
  and terra. Gated both ways on synthetic slices carrying the SAME global
  shift: groups built from an INDEX must read **DEAD** (z=-1.7, beside the
  real octmnist -0.4 and tissuemnist -1.9) and groups with their own
  prevalences must clear stage 1. Plus the `nan`-z branch, which used to
  UPGRADE the verdict.
- **`scope_probe`** -- it closed the local-cap direction on "pinning the split
  costs -0.86 items while wrong-shape controls cost 5.3-5.5", and that is
  legal ONLY if the control differs in SHAPE and not in DOSE. Nothing checked
  that `_permute_ceilings` preserves the budget. It does; now gated, with a
  negative control that moves one ceiling by ONE and must be caught, plus the
  identity permutation as a no-op and `_splits` enumerating every composition
  exactly once.

⚠️ Writing that last gate reproduced the defect class inside the gate itself:
a `for ... else` printed its PASS line **unconditionally**, because `else`
fires whenever the loop does not `break` -- so it would have printed PASS
directly beside its own FAIL. Fixed with an explicit flag. The pattern is
worth recognising: `for/else` in a checker is almost always this bug.

✅ **THE GENERAL RULE THIS SECTION EXISTS FOR.** A probe may return a number
or refuse; it may not return a number that means "I could not measure". Every
refusal added here is gated in BOTH directions -- the refusal must fire on the
broken input AND the tool must still decide on the good one -- because a gate
that always refuses closes as many directions as one that never does.


### 2(w2) 🟢 **TWO MORE DATASETS PASS THE SCREEN -- the one-dataset era ends**
screened 2026-08-28, labels and metadata only, no images and no GPU

2(w1) shows no second iwildcam SLICE exists, so replication needs a new
DATASET. Two candidates were prepared meta-only and screened beside the
incumbent. **Both pass stage 1, and both have UNSEEN test groups**, which is
the strongest form 2(n) asks for -- training carries no prior for them at all,
so the cap is the only source of information about their composition.

| dataset | group | NET items | z | unseen | n_test | cls | imbal | rarest |
|---|---|---|---|---|---|---|---|---|
| `iwildcam/oodslice` | camera | +3133 | 96.3 | 7 | 2943 | 8 | 4.5x | 160 |
| **`fmow/oodslice`** | **country** | **+2969** | **79.7** | **10** | 3442 | 8 | 1.7x | 320 |
| **`terra/oodslice`** | camera | **+2546** | **75.8** | 5 | 2985 | 8 | 2.5x | 249 |
| dermmnist/slice_1 | synth | +65 | 2.9 | **0** | -- | -- | -- | -- |
| octmnist / tissuemnist | `index % 3` | -7 / -56 | -0.4 / -1.9 | **0** | -- | -- | -- | -- |

Reproduce, in minutes, on CPU:

```bash
python -m scripts.prep_fmow --meta-only --cache <cache> --out <dir>/fmow/oodslice
curl -sL -o cct.json.zip https://lilawildlife.blob.core.windows.net/lila-wildlife/caltechcameratraps/labels/caltech_camera_traps.json.zip
python -m scripts.prep_iwildcam --annotations <cct>.json --out <dir>/terra/oodslice --meta-only
python -m scripts.dataset_screen <dir>/terra/oodslice <dir>/fmow/oodslice data/iwildcam/oodslice
```

🔑 **`fmow` is the scientifically valuable one and `terra` is the cheap one.**
`terra` is Caltech Camera Traps: same sensor, same COCO-CameraTraps format,
same held-out-camera group -- so it reads through `prep_iwildcam` unchanged,
and that similarity is also its weakness. It replicates on nearly the same
structure. `fmow` is satellite imagery grouped by COUNTRY, the first candidate
that is not a camera trap, so it is the one that can say whether 2(w) is a
property of the method or of wildlife photography.

#### ⚠️ WHAT THIS DOES **NOT** SAY, and both limits are load-bearing

**1. Stage 1 is necessary, not sufficient, and this project has been burned by
exactly that.** dermmnist scored +65 items at z=2.9 -- a real per-group shift
-- and still nulled in 2(m), where feeding a model the TRUE per-group counts
moved 6 items. Information existing is not the same as it being convertible
into ORDERING. Stage 2 is `scope_probe --calibrate` and needs a trained model.

**2. The PRIZE is unmeasured on both, and `ceiling_screen`'s verdict for them
is borrowed, not measured.** Run on `fmow` it prints PRIZE BELOW THE NOISE at
every cap -- but its `p@K` and `sd` columns are an **iwildcam calibration
that the tool itself says does not transfer**. The K structure is comparable
(K = 64-165 at K/n 20-30%), so the verdict is really -- IF fmow's ranking were
as good as iwildcam's, the prize would be equally hopeless.

🔑 **That is the number to go and get, and it is a single threshold:**

| fmow cap | class | K | p@K needed for prize = 1.0x sd | for 2.0x |
|---|---|---|---|---|
| L20_G50 | 2 | 109 | 0.9564 | **0.9128** |
| L30_G50 | 2 | 165 | 0.9612 | **0.9224** |
| L30_G50 | 7 | 96 | 0.9339 | **0.8677** |

**iwildcam's MEASURED p@K at these K/n is 0.9948 to 0.9972**, which is why its
prize is 0.4-4 items. fmow needs only `p@K <= ~0.92` at L30 to clear twice the
noise. Satellite imagery over eight confusable land-use classes is a much
harder ranking problem than eight African species, so this is plausible --
**and it is unmeasured**. It is the first thing to measure there, it needs one
trained model, and it decides whether fmow is the dataset on which this method
can finally be shown to win or lose. ⚠️ The `sd` moves with the prize (2(v)),
so re-measure BOTH with `scripts.paired_noise` rather than assuming iwildcam's.

⚠️ `prep_*` warns that a meta-only NET is an UPPER bound on the delivered
slice, since shards can fail to download -- good enough to REJECT a candidate,
never to accept a borderline one. At z = 76 and 80 neither is borderline.


### 2(w1) ⛔ **THERE IS NO SECOND iwildcam SLICE. THE SHIPPED ONE IS THE ONLY**
**VIABLE DRAW** -- measured 2026-08-27, labels and metadata only, no GPU

`oodslice` holds out 7 cameras and trains on 143. Those 143 look like a free
pool of alternative holdouts -- a SECOND disjoint slice with a different group
structure, at no download and no image cost, which is the only replication of
2(w) available without a new dataset. **It does not exist.** The capped
species live in cameras that contain nothing else:

| | class 2 (impala) | class 7 (cattle) |
|---|---|---|
| instances in train | 2500 | 2500 |
| cameras containing it, of 143 | 52 | 49 |
| share held by camera **501** alone | **55%** | **33%** |
| share held by its top 7 cameras | 74% | 55% |

Nine cameras hold at least 20 of BOTH capped classes. **All nine contain zero
images of any other class** -- camera 501 is 2201 images of pure impala and
cattle. Widen it to -- at least 30 other-class images AND at least 15 of a
capped class -- and the answer is **0 cameras of 143**. 77 of the 143 are
single-class outright.

⇒ **Any holdout with enough capped instances to constrain is ~100% capped**,
which deletes the six uncapped classes from test -- and macro-F1, the metric
the paper headlines, is carried by exactly those. The shipped slice is the
one draw that is not degenerate: 2943 test images, 370 + 456 capped, **2117
other, all 8 classes present**.

#### 🛑 AND THE SAME NUMBERS BOUND EVERY RESULT IN THIS PROJECT

In the shipped test slice the capped classes appear in **3 of 7 cameras
(class 2) and 4 of 7 (class 7)**, and within those one camera carries 43%
(160/370) and 50% (229/456). That independently reproduces the documented
-- 7 of 14 per-group ceilings are K=0 -- from the other direction: 4 zero
cells for class 2 plus 3 for class 7 is exactly 7.

So the LOCAL scope, which 2(n) says is what makes iwildcam usable at all, is
carried by **three or four cameras**, not seven. Quote that whenever a
per-group result is described as holding across groups.

⇒ **Replication of 2(w) needs a new DATASET, not a new slice.** The cheap
screen still applies and needs no images: any COCO-CameraTraps annotation
file (Terra Incognita / CCT) can go through
`prep_iwildcam --meta-only` then `dataset_screen`, and 2(n) already gives the
triage rule -- a group built from an index, a randomisation or a balanced
assay design is dead by construction.


### 2(w3) 🟢🟢 **THE CONSTRAINT HELPS AT LOOSE CAPS -- the first attributable**
**positive effect in the project** -- `results/loose1`, 144 runs, complete 2026-08-28

6 cells (3 backbones x L80_G95, L90_G95), 4 seeds, both trained arms at
**696/696 steps = 100.0%**. Paired against each arm's own lambda=0 twin:

| vs `tralo_null` | AP | AUROC | ccF1 |
|---|---|---|---|
| **`tralo`** | **+0.0253  5/1** | **+0.0075  6/0** | +0.0120  6/0 |
| `tralo_reseed` (RNG floor) | -0.0016 tie | +0.0016 tie | **+0.0088  6/0** |
| `tralo_uniform` | +0.0005 tie | +0.0038 tie | +0.0077  6/0 |

🛑 **READ THE RESEED ROW FIRST, because it eats most of one column.** A pure
RNG reseed also produces a 6-of-6 ccF1 -- win -- of +0.0088. So `tralo`'s
+0.0120 is **1.36x the floor** and `tralo_uniform`'s +0.0077 is **BELOW** it.
**The ccF1 gain here is mostly the seed.** A 6/0 sweep is not evidence when
the null arm also sweeps 6/0.

✅ **What survives its control is the RANKING.** AP +0.0253 and AUROC +0.0075
against a reseed floor that TIES on both -- roughly 16x and 4.7x. That is the
first constraint effect in this project that is positive AND attributable, and
it is a ranking effect, not an allocation one.

🔢 **THE ccF1 COLUMN IN ITEMS** -- measured 2026-08-28 with `full_panel
--control tralo_null`, i.e. through the LP allocator that actually ran. These
are the numbers to quote:

| vs `tralo_null` | items | paired seed sd | seeds needed at 80% |
|---|---|---|---|
| **`tralo`** | **+9.24** | 10.37 | ~10 |
| `tralo_reseed` (RNG floor) | +6.71 | 9.49 | ~16 |
| `tralo_uniform` | +5.91 | 10.82 | ~26 |
| `clip` | +5.22 | 10.85 | ~34 |
| `focal_clip` | +3.63 | 10.55 | ~67 |

⇒ the constraint's **attributable share is 9.24 - 6.71 = +2.53 items**, and
`tralo` beats `clip` by **+4.02 items**, of which a pure reseed alone supplies
1.49. 🛑 **Every one of these is UNDERPOWERED at the 4 seeds the protocol
runs** -- the within-cell paired sd is ~10 items against effects of 3-9. The
signs are right; the magnitudes are not resolvable inside a cell.

🔑 **Which is exactly why `dom1` is designed on CELL-LEVEL SIGN CONSISTENCY**
(the pre-registration asks for 6 of 9 cells), not on within-cell magnitude.
Nine cells give a sign-test floor of 0.0039, which is reachable; within-cell
power at these sds is not.

⛔ **Do NOT quote `order_probe --evictions`' +16.50 items for this campaign.**
It scores a GLOBAL top-K rather than the LP allocator and is 6.5x too large.
See 2(w4).

⚠️ 6 cells, so every verdict reads -- win (not after BH) --: the exact Wilcoxon
floor is 0.031 and BH over 11 metrics needs 0.0045. `gen_campaign` states 9
cells as the minimum for a `***`, which is why `dom1` is 9.

#### 🔑🔑 THE REGIME REVERSAL: THE TWO COUNT FUNCTIONS SUIT OPPOSITE CAPS

| AP vs own null | tight caps L20-L50 | loose caps L80-L90 |
|---|---|---|
| **`tralo`** (`sum`, weight `p(1-p)`) | **-0.0572 to -0.0933, 0 of 12 cells** | **+0.0253, 5/1** |
| **`tralo_uniform`** (flat log-odds) | **+0.0030 to +0.0087, tie** | **+0.0005, tie** |
| `tralo_reseed` | -0.0016 to -0.0142 | -0.0016 tie |

**Each count wins exactly where the other fails.** The fix of 2(w) is not a
strict improvement -- it is a fix FOR THE TIGHT-CAP REGIME, and at loose caps
the original count is the better one by 50x.

🔑 **THE MECHANISM, and it closes 2(t) and 4 together.** `sum`'s per-item
weight `p(1-p)` is maximal at `p = 0.5`, i.e. AT THE DECISION BOUNDARY. Where
the boundary sits relative to the cut is what changes with the cap:

* **tight cap**: the hard count is ~368 against `K = 74`, so the boundary is at
  item 368 and the cut is at 74 -- **buried deep inside the class**. `sum` puts
  its largest push ~294 items away from the cut, on items it should not touch,
  and 2(t) measured the result: it evicts at `p ~ 0.79` and admits at `p ~ 0.25`.
* **loose cap**: the count is 368 against `K = 333`, so boundary and cut are 35
  items apart and `p(1-p)` pushes almost exactly where the decision is made.

That is the same quantity as the work-to-prize ratio: 294 evictions for a
0.42-item prize at L20 (**700x**), 35 for a 29.8-item prize at L90 (**1.2x**).
The flat count cannot exploit the loose regime because it declines to
concentrate anywhere -- which is exactly why it is safe in the tight one.

⇒ **The design lesson for TraLO: the count function should concentrate where
an item can actually FLIP, not at `p = 0.5`.**

🛑🛑 **CORRECTION 2026-08-28 -- the sentence that stood here was wrong, and it
had already been used to queue an arm.** It read: *"the count should
concentrate at the CUT (rank K), not at the decision boundary (p = 0.5) ...
`soft_count_mode: margin` with `cut_window_items` windows the gradient around
the cut by construction."* Both halves are wrong.

`margins()` in `src/losses/transductive_loss.py` is `m_ic = p_ic -
max_{c' != c} p_ic'`, and its own docstring says **"per-item distance to the
DECISION BOUNDARY"**. There are **three** distinct points here and the old
sentence collapsed two of them and mislabelled a third:

| target | sits at | what it is |
|---|---|---|
| `p(1-p)` peak | `p_c = 0.5` | the SHIPPED count |
| **decision boundary**, `m = 0` | `p_c = 0.20` in the case measured below | **what `margin` actually windows** |
| the **cut**, rank K | a *ranking position* | what was claimed; nothing implements it |

Measured directly (8 classes, `scripts` not needed -- pure algebra on
`margins`): item A with `p_0 = 0.50` and runner-up `0.30` gets **1.56x** the
`p(1-p)` weight of item B with `p_0 = 0.20` and runner-up `0.20`. **B is the
one that can flip**, and only the margin window ranks it first. So `p = 0.5` is
*not* the decision boundary, and `cut_window_items` is **misnamed** -- it sets
a **boundary** window, not a cut window.

✅ **`margin` survives the correction, on a stronger reason.**
`scripts/order_probe`'s docstring carries it: every penalty this project ships
has the form `f(sum_i p_ic)`, whose per-item logit gradient is
`f'(S) * p_ic(1 - p_ic)` -- a function of `p_ic` **alone**, hence a monotone
map on the logit channel, and **a monotone map cannot move an item across
another**. It moves the cut; it cannot re-rank. `margin` reads the whole row,
so two items with equal `p_ic` but different runners-up get different
gradients. It is the **only** arm in the family that can reorder through the
direct channel.

⚠️ **But that argument bounds the DIRECT channel only** -- 2(w4) measures both
shipped arms reordering more than a reseed via the shared weights. `margin`
makes reordering **targeted**, not possible.

🎯 **`tralo_margin` and `tralo_st` are fully built, protocol-registered,
null-sibling-tested and gated in `gen_campaign` -- and have NEVER RUN.** Zero
run directories across all **fourteen** server worktrees, recounted 2026-09-01 (2(z22)); the 2026-08-28 pass saw only nine and undercounted.
`tralo_st` is the decomposition arm: `tralo` = (soft value, `p(1-p)`
placement), `tralo_st` = the value fixed only, `tralo_margin` = both. **They
are the next arms to run.**


### 2(x) ⛔⛔⛔ **RETRACTED 2026-09-04. THIS SECTION WAS HEADED "TraLO LEADS ALL**
### **FOUR RIVAL DUALS AT LOOSE CAPS -- `results/dom1`, 384 runs, complete**
### **2026-08-29. The first dominance evidence in the project." TWO OF THE FOUR**
### **RIVALS ARE DEAD ARMS, SO THE CLAIM IS NOT WEAKENED -- IT IS UNANSWERABLE.**

🛑 **THE OLD HEADER IS KEPT VISIBLE BECAUSE A SILENTLY-FIXED ERROR IS A LESSON
LOST.** `fioretto` and `hounie` ran at **28.00** attempted constraint steps per
run against `tralo` / `alm` / `tralo_uniform`'s **29.00** (2(z40)), so both are
`dead_arms` in `dom1` and every contrast touching either is not comparable. Of
the four rival duals exactly ONE -- `alm`, at 29.00 -- survives. **"Leads all
four" is not a claim that got smaller: it is a comparison with two of its four
terms removed.** What actually survives is in 2(z43), and it is a different and
much smaller claim.

**THE CLEANEST CAMPAIGN THIS PROJECT HAS RUN.** iwildcam x {MobileNetV2,
MobileNetV3} x {L80_G95, L90_G95, L95_G80} = 6 cells, 4 seeds, **16 arms**:
five trained methodologies, **a lambda=0 twin for each of the four dual
families**, the RNG floor, and four post-hoc bars. Two integrity facts first,
because they are what make the numbers readable:

⛔⛔ **THIS PARAGRAPH IS THE LOAD-BEARING FALSE SENTENCE OF THE WHOLE FILE.
IT STOOD UNTIL 2026-09-04 AND READ:**

> 🔑 **DOSE IS 100.0% FOR ALL FIVE TRAINED ARMS** -- `tralo` 696/696, `alm`
> 696/696, `fioretto` 672/672, `hounie` **672/672**, `tralo_uniform` 696/696.
> `hounie` is the arm that ran at **1% dose** under `constraint_grad_mode: clip`
> (2(g)). `normalize` + `--constraint-fp32` removed the ~20x dual dose
> asymmetry entirely. **This is the first campaign in which the four duals were
> ever given a comparable dose**, and no dual-vs-dual number before it is safe.

🛑 **672 = 24 x 28. 696 = 24 x 29.** The six numbers printed side by side as
PROOF of parity ARE the defect. Four of them are 24 runs x 29 steps and two are
24 runs x **28**. The gap was on the page, in the same sentence, spelled out in
the denominators -- and it read as parity because both were rendered as a
PERCENTAGE. **A percentage computed WITHIN an arm cannot see a gap BETWEEN
arms.** `672/672` and `696/696` are both exactly 100.0%, and that is the entire
failure. `dose_landed`'s `attempted/run` column is the one that shows it, which
is why CLAUDE.md now says to read that column and never the percentage.

⛔ **AND THE LAST CLAUSE IS EXACTLY INVERTED.** `dom1` is not "the first
campaign in which the four duals were ever given a comparable dose". It is the
first campaign in which their dose was WRITTEN DOWN per arm, which is how the
gap was eventually found. The first campaign that actually gives them a
comparable dose is `vitdual2` (`fioretto` 29.00, `hounie` 29.00, `alm` 29.00),
and on 2026-09-04 it stands at **32 of 88 runs complete**.

✅ **RE-VERIFIED FROM THE RAW LOGS, 2026-09-04.** Counted nonzero `grad_norm`
epochs in every `training_log.csv` across all 792 runs of `dom1` + `dom1b` +
`equaldose1`: `min == max` on every arm, no variance anywhere. The gap is the
design, not a flake. `alm` is 29.00 in all three campaigns and is NOT a dead
arm; every `*_null` and `*_reseed` arm is 0.00 and is therefore untouched,
which is why they are correctly absent from `dead_arms`.

🔑 **THE POSITIVE CONTROL PASSES EXACTLY**: `tralo_null`, `fioretto_null`,
`hounie_null` and `alm_null` are **byte-identical in 24 of 24 cell-seeds**. At
lambda = 0 the four families ARE one run, and they md5 as one run. So the
"compute" term is shared *exactly* across families, and every difference in the
constraint term is attributable to the method rather than to the 29 epochs.

#### THE RESULT, `full_panel --control clip` (the scorer of record), 6 cells

| arm | ccF1 | AP | AUROC | macroF1 |
|---|---|---|---|---|
| **`tralo`** | **+0.0080 (1.86x floor) 6/6** | **+0.0462 (2.37x) 6/6** | **+0.0111 (2.18x) 6/6** | -0.0023 |
| `alm` | +0.0075 (1.74x) 6/6 | +0.0426 (2.18x) | +0.0087 (1.71x) | -0.0028 |
| `tralo_reseed` **(RNG FLOOR)** | **+0.0043 6/6** | **+0.0195 6/6** | **+0.0051 6/6** | -0.0011 |
| ~~`hounie`~~ ⛔ **DEAD ARM, 28.00 steps** | ~~+0.0042 (0.98x -- BELOW)~~ | ~~+0.0419 (2.15x)~~ | ~~+0.0090 (1.76x)~~ | ~~+0.0007~~ |
| ~~`fioretto`~~ ⛔ **DEAD ARM, 28.00 steps** | ~~+0.0040 (0.93x -- BELOW)~~ | ~~+0.0359 (1.84x)~~ | ~~+0.0066 (1.29x)~~ | ~~-0.0030~~ |
| `tralo_uniform` | +0.0034 (**0.79x -- BELOW**) | +0.0226 (1.16x) | +0.0075 (1.47x) | -0.0044 |
| `la_lp` | +0.0023 (0.53x) | +0.0144 (0.74x) | +0.0042 (0.82x) | -0.0069 |
| `focal_clip` | **-0.0025** | +0.0070 (0.36x) | +0.0047 (0.92x) | -0.0063 |

⛔ **THIS READ "TraLO is #1 of five on ccF1, AP and AUROC -- every metric a
top-K allocator can see -- and it sweeps 6 of 6 cells on all three" UNTIL
2026-09-04. THE DENOMINATOR IS THREE, NOT FIVE.** Two of the five trained arms,
`fioretto` and `hounie`, are dead arms here, so the ranked field is `tralo`,
`alm`, `tralo_uniform`. **TraLO is #1 of three** -- and the paragraph directly
below already said what that is worth: `tralo` +0.0080 against `alm` +0.0075 is
**0.38 items, under the one-item line, so on ccF1 they are TIED at the top.**
With the dead rows struck, the surviving ccF1 reading of this table is a TIE
WITH `alm`, not a sweep of a field of five. The AP/AUROC margin over `alm`
(+0.0462 vs +0.0426; +0.0111 vs +0.0087) is the only part of "dominance" left
standing, and 2(z43) prices it as-deployed.

🛑 **READ THE FLOOR ROW.** A pure RNG reseed also sweeps 6/6 at +0.0043 ccF1.
So **`tralo_uniform` does NOT clear it on ccF1** -- its 6/6 sweep is what doing
nothing produces. Only `tralo` (1.86x) and `alm` (1.74x) survive their own
floor, and **they are 0.38 items apart, which is under the one-item line, so on
ccF1 they are TIED at the top.** The separation is on AP/AUROC, where `tralo`
leads outright.

⚠️ **This paragraph also read "`hounie`, `fioretto` and `tralo_uniform` do NOT
clear it" until 2026-09-04.** That was a claim ABOUT two dead arms, so it goes
with them: it is not that the finding reversed, it is that it can no longer be
made here at all. `tralo_uniform`'s failure is unaffected and stands.

#### WHERE TraLO LOSES, and it must be said

⚠️ **macroF1 -- the metric the manuscript headlines -- is NEGATIVE for `tralo`
(-0.0023) and it loses to `clip`.** `hounie` is the ONLY arm with a positive
macroF1 (+0.0007). On NLL the ordering reverses outright: `tralo` **-0.0590**
against `hounie` **+0.0445** (5/6). macroF1 is carried by the UNCAPPED classes
and `uncF1` is negative for every trained arm, so **the constraint still buys
capped-class accuracy with uncapped-class damage**. The 2(u)/2(w) damage
finding is NOT overturned; it is confined to a different metric family.

⚠️ **6 cells cannot reach `***`**: the exact Wilcoxon floor is 0.031 and BH over
11 metrics needs 0.0045. `results/dom1b` (RegNetY400MF, 192 runs, same pin
`1d921173`, launched 2026-08-29) brings it to **9 cells**, where the floor is
0.0039 and a `***` becomes reachable. Score them together:
`full_panel --campaign results/dom1 results/dom1b --control clip`.

⛔ **THE PRE-REGISTRATION IS NOT MERELY UNMET, IT IS UNANSWERABLE (2026-09-04).**
It asked for `tralo` to beat each of `fioretto`, `hounie` and `alm` on ccF1 in
>= 6 of 9 cells. **Two of its three targets are dead arms**, so two thirds of
the pre-registered test cannot be run on this corpus at any effect size. On the
one target that survives, `alm`, `tralo` does not beat it -- they tie. Until
2026-09-04 this paragraph read "The pre-registration is NOT met as written ...
Against `alm` it does not -- they tie", which was the right verdict for the
wrong reason: it treated a 1-of-3 result as a partial failure when 2 of the 3
were never measurable. `vitdual2` is the only campaign that can re-pose the
full pre-registration.

🔑 **AND THE COMPUTE STORY INVERTS HERE.** 2(r) measured "the win is compute,
not method" -- every trained arm beat the clipper on the 29 extra epochs while
TraLO's own part was +0.15 pp. In `dom1` the ccF1 **compute term is NEGATIVE
(-0.0061 for every family, identically, because the nulls are one run)** and the
**constraint term is +0.0141 for `tralo`**. At loose caps the constraint is
doing the work and the extra compute is a cost. That is consistent with 2(w3)
and is the regime this result lives in -- **all three caps here are LOOSE**.


### 2(x2) ⛔⛔ **`logit_adjust` IS INERT TOO -- AND md5 CANNOT SEE IT**

Found 2026-09-01 by the stage-1 gate agent, verified here against the shipped
`src/losses/imbalanced_losses.py`. **`la_lp` is not a baseline on iwildcam. It
is a reseed of `lp`.**

`LogitAdjustedLoss.forward` is `F.cross_entropy(logits + tau * log_prior, t)`.
iwildcam's TRAIN set is exactly 2500/class, so `prior` is uniform and
`log_prior` is the CONSTANT vector `log(1/8) = -2.0794` in every coordinate.
`log_softmax` is shift-invariant, so adding a constant to every logit changes
the objective **not at all**. Same arithmetic as 2(x1), different mechanism.

Measured on the real criterion at batch 64:

| arm | loss vs CE | max abs grad diff | gradients BITWISE equal? | md5 of raw preds |
|---|---|---|---|---|
| `class_balanced` | 0.0 | **0.0** | **YES** | identical to `clip`, 24/24 |
| `logit_adjust` | 0.0 | **9.3e-10** | **NO** | **DIFFERS from `clip`, 24/24** |

🛑🛑 **THIS IS THE NEW FAILURE MODE, AND IT IS THE INVERSE OF RULE 3.** Rule 3
says hash the raw predictions, because an inert flag leaves them identical.
`logit_adjust` is inert and its predictions DIFFER -- the `-2.0794` shifts
float rounding by ~1e-9 per step, and 30 epochs compound that into a genuinely
different model. So:

> **md5 DIVERGENCE IS NOT EVIDENCE OF A LIVE MECHANISM.** Identical predictions
> prove inertness; different predictions prove nothing at all. 2(x1)'s md5
> table lists `la_lp` as "different model" and that reading was wrong.

The only instrument that catches this is the one used above: **evaluate the
criterion against plain CE on the real training prior and compare the
GRADIENTS**, which is what `tests/gates/test_g1_data.py` gate 4 now does. Its
liveness control is the same criterion on a 4.5x prior, where the loss moves
3.8e-2 -- so the tool can tell inert from live.

**CONSEQUENCE FOR THE PAPER.** Of the nine claimed methodologies, on the only
runnable dataset **two of the three imbalanced recipes are non-baselines**:
`class_balanced` exactly, `logit_adjust` mechanistically. `focal` survives --
it reweights by `(1-p)^gamma` per EXAMPLE and never reads the prior, so it is
live at any class balance. Any `la_lp` number already published is `lp` plus
RNG noise and must be relabelled or dropped, not defended.

✅ **NOTHING PUBLISHED IS AFFECTED, AND THAT IS MEASURED.** All 120
`class_balanced` / `logit_adjust` rows in `docs/paper/data/manifest/experiments.csv`
sit on `dermmnist` (40), `octmnist` (40) and `tissuemnist` (40) -- **zero on
iwildcam** -- and on those three the mechanism is LIVE, which is measured and
not assumed: paired against `danits_lp` on the same (dataset, model, cap, seed),
`class_balanced` moves cc-F1 by up to **0.0811** over 36 pairs and
`logit_adjust` by up to **0.0721**. (9 and 4 of those 36 are exactly identical,
but so are 6 of `focal`'s -- which never reads the prior -- so that is the LP
allocator returning the same top-K, not inertness.) `imbalanced_baselines.csv` is likewise 312 rows of
dermmnist/octmnist/tissuemnist only. No number needs relabelling; the finding
scopes to the ONE runnable dataset, where it means the two arms cannot be run
as baselines at all.

✅ **AND THE GENERATOR NOW REFUSES THEM.** `configs.gen_campaign.prior_arm_gate`
measures `max/min` over the TRAINING class counts (`train_meta.csv`, else
`train_labels.npy`) and refuses `cb_lp` / `la_lp` below `BALANCE_TOL = 1.05`.
It is a measurement, not a hardcoded dataset name, so it will refuse the next
balanced slice too. Absent labels print `THE PRIOR-ARM GATE DID NOT RUN` --
unknown is not balanced. `--allow-inert-baseline` overrides and says what it
let through. Gated in `--self-test` in all three directions plus the
measurement itself (iwildcam TRAIN = 1.0000 exactly, so a run that read the
4.5x TEST figure by mistake would fail the gate).

⚠️ **AND THE RULE GENERALISES.** Before claiming any loss variant is live,
`assert not torch.equal(grad_variant, grad_ce)` is the WRONG test. The right
one is `max|grad_variant - grad_ce| > fp_noise_floor`, with the floor measured
on the same shapes. At 9.3e-10 against a gradient of order 1e-1, `logit_adjust`
is eight orders of magnitude inside the noise.

---

### 2(x1) ⛔ **`class_balanced` IS INERT ON iwildcam -- and the headline table**
### **CANNOT SEE AN ALLOCATOR AT ALL**

🛑 **CORRECTION, SAME DAY.** This section first read *"three of the nine
methodologies are the same arm"*, from an md5 over `final_predictions_raw.csv`
that found `lp == clip`, `focal_lp == focal_clip` and `cb_lp == clip` at 24/24.
**Two of those three were a misreading of which file to hash**, and
`full_panel`'s own comment says so at the hashing site: *"the ALLOCATED
predictions can differ while the raw ones are identical, which is exactly
`clip` vs `lp` and is NOT an inert flag."*

| pair | RAW same | **ALLOCATED same** | what it means |
|---|---|---|---|
| `lp` **vs** `clip` | 24/24 | **1/24** | same warm-up MODEL, allocator genuinely differs |
| `focal_lp` **vs** `focal_clip` | 24/24 | **1/24** | same |
| **`cb_lp` vs `clip`** | **24/24** | 1/24 | **same model from a DIFFERENT recipe -- the real finding** |
| `la_lp` vs `clip` | 0/24 | 0/24 | different model |

`lp` and `clip` **share `base_model_id` by construction** -- they are one
warm-up with two allocators -- so identical raw predictions is what they are
supposed to produce and says nothing. Hash `final_predictions.csv` to compare
allocators; hash `final_predictions_raw.csv` to compare models.

#### ✅ WHAT SURVIVES: `class_balanced` is inert, and it is not a code bug

`cb_lp` carries a **different** `base_model_id` (`7e92e1b76bc5` vs
`067715022594`), so it genuinely retrained under its own cache key -- and
landed on **byte-identical raw predictions**. The reason is arithmetic:
`data/iwildcam/oodslice/train_labels.npy` is **exactly 2500 images in each of
the 8 classes, imbalance 1.0x**. The class-balanced weight
`(1-beta)/(1-beta^n_c)` normalised to mean 1 is then **exactly 1.0 for every
class** (verified `max|w-1| = 0.0e+00` at beta 0.9999 and 0.999), and weighted
CE with unit weights *is* plain CE.

🔑 **THE 4.5x IMBALANCE QUOTED FOR iwildcam IS THE TEST SET. TRAIN IS 1.0x.**
⇒ **any baseline whose mechanism reads the TRAINING prior is inert here.** That
is `class_balanced` outright. Check `np.bincount(train_labels)` before claiming
any imbalance recipe as a comparison.

⚠️ **No audit this project owns catches this.** The AST audit passes -- all four
recipe keys have readers at `src/losses/imbalanced_losses.py:85-92`. The cache
key differs, so no collision is flagged. `audit_config` and `check_parity` are
green. **Only hashing the raw predictions finds it**, and it is inert by DATA,
not by code.

#### 🛑 AND THE ONE THAT MATTERS FOR THE PAPER: THE PANEL IS ALLOCATOR-BLIND

`full_panel` re-derives its OWN allocation from the raw probabilities for every
arm -- `eq = equalize(P, g, G, L, cls)` -- so that arms are compared at an equal
budget. That is correct and deliberate, and it has a consequence nobody wrote
down: **two arms sharing a warm-up model score `+0.0000` on every
budget-equalized metric no matter how differently they allocate.** `lp` vs
`clip` reads `+0.0000 p=1.000` on ccF1, AP, AUROC and macroF1 while their
deployed predictions differ in **23 of 24** cell-seeds.

⇒ **The headline table cannot evaluate an allocator, by construction.** Any
claim comparing `danits_lp` (Shifman-LP) against `heuristic` needs the
AS-DEPLOYED numbers, not the equalized panel. On the as-deployed ccF1 with
budgets matched (emitted counts identical), `tralo` beats `lp` by **+0.0046 in
4/4 local cells and +0.0082 in 2/2 global cells**.

#### THE SCOPE SPLIT -- `dom1` carries both, and they agree

As-deployed ccF1, `tralo` minus rival, **emitted counts equal so this is not
free fill**:

| vs | LOCAL-binding (4 cells) | GLOBAL-binding (2 cells) |
|---|---|---|
| `clip` | **+0.0037, 3/4** | **+0.0082, 2/2** |
| `lp` | **+0.0046, 4/4** | **+0.0082, 2/2** |
| `fioretto` | +0.0011, 2/4 | +0.0046, 2/2 |
| `tralo_uniform` | +0.0034, 3/4 | +0.0052, 2/2 |
| `tralo_reseed` (floor) | +0.0010, 3/4 | +0.0063, 2/2 |
| `hounie` | +0.0019, 3/4 | +0.0002, 1/2 |
| **`alm`** | **-0.0025, 1/4** | **-0.0008, 1/2** |

✅ **TraLO beats both clippers in BOTH scopes** -- which is the form the thesis
claim needs. ⚠️ **It loses to `alm` in both**, by well under one item. ⚠️ 2
cells is a p-floor of 0.50: the GLOBAL column is a DIRECTION, never a
significance claim.

### 2(w4) 🔬🔬 **`order_probe` HAD NO SIGNIFICANCE GATE -- and the band/global**
### **DISSOCIATION it was hiding**

🛑 **The probe was calling coin flips.** Its verdict branched on the bare
pooled mean of `rho_arm - rho_reseed` with **no test at all**, so a mean of
`-0.0076` at a **27/48** split printed *"the constraint reordered MORE than a
reseed. The order-preservation argument does NOT hold here."*

The tell is the second arm it fired on. **`tralo_uniform` read 26/48, p=0.66,
and got the same verdict** -- and `tralo_uniform`'s per-item gradient is
constant in log-odds, so on the direct channel it is a pure bias shift that
**cannot reorder**; `configs/protocol.yml` says exactly that at its
definition. That arm is this probe's built-in **negative control**, and the
probe failed it.

✅ Fixed: an exact two-sided binomial `sign_test` (verified against
`scipy.binomtest` for all k in 0..48), the verdict split out of `main()` so it
runs on a synthetic split with no campaign, and points-needed-at-80%-power
printed on every tie so "no effect" and "not enough points" stay separable.
The gate drives both real splits and a 40/48 live control; reverting the fix
makes it fail.

#### The dissociation the gate exposed

Global rho is taken over the whole capped class and is dominated by the easy
mass. The **band** is ranks K/2..2K, where the cut actually falls. The probe
printed both and branched on neither correctly -- and they come apart:

| campaign (caps) | arm | GLOBAL | BAND |
|---|---|---|---|
| `vitu1` ViTB16 (tight) | `tralo_uniform` | **-0.1325, 24/24, q=0.007** | **+0.0427, 5/24 OTHER way, q=0.025** |
| `iwc4` MNv2/3 (tight) | `tralo` | **-0.0355, 50/72, q=0.007** | -0.0172, 46/72, q=0.067 |
| `uniform1` MNx3 (tight) | `tralo_uniform` | **-0.0312, 48/72, q=0.028** | +0.0154, 31/72, ns |
| `uniform1` MNx3 (tight) | `tralo` | -0.0100, 41/72, ns | -0.0454, 45/72, q=0.103 |
| `loose1` MNx3 (**LOOSE**) | `tralo`, `tralo_uniform` | **ns** (p=0.47, 0.67) | **ns** (p=1.00, 0.47) |

BH over **all 14** sign tests run; only the **q < 0.05** rows are callable.

🔑 **`tralo_uniform` reorders the EASY MASS and PROTECTS THE CUT.** On ViTB16
it reorders more than a reseed in **24 of 24** points globally while reordering
the contested band **LESS** than pure RNG noise in 19 of 24. That is the
mechanism for 2(w), and it **resolves the paradox recorded there** -- the tau
0.4371 reading said `tralo_uniform` "reorders 5.6x more than `tralo` and still
improves AP". It reorders more *overall* and less *where it matters*.

⚠️ **`tralo`'s band effect does NOT survive BH** (q=0.067 and 0.103 on two
independent campaigns). Same sign twice, suggestive, **not callable**.

⚠️ **The monotone-map argument bounds the DIRECT channel only.** Both arms do
reorder more than a reseed at tight caps, because the shared weights are a
second channel. State which channel before quoting the argument.

🟢 **At LOOSE caps nothing reorders at all** -- all four tests tie. Which is
consistent with 2(w3): that is the regime where the constraint helps, and it
helps by moving the **cut**, not the ranking.

#### `--evictions` overstates by 6.5x -- it is not the allocator that ran

It reported *"+16.50 items per cell attributable; the constraint's swaps are
BETTER than a reseed's"*. Two independent defects, both now fixed:

1. **NO POWER.** It branched on `d_net` against a bare `+/-1.0` items and
   printed no noise at all. The within-cell paired seed sd is **18.11 items**,
   larger than the effect. It now prints a RESOLUTION block: **~10 seeds per
   cell needed, UNDERPOWERED** at the 4 the protocol runs.
2. **WRONG ALLOCATOR.** Its sets are `argsort(-p)[:K]` on the raw class column
   -- a **GLOBAL top-K**. The allocator that actually ran is LP/greedy under
   per-group ceilings, and **7 of 14 iwildcam local ceilings are K=0**, so it
   cannot take the global top-K and does not. `full_panel --control
   tralo_null` scored the same campaign at `tralo` **+9.24** items against
   `tralo_reseed` **+6.71**, i.e. **+2.53 attributable**.

⇒ **Read `order_probe` for WHICH items moved and WHY. Quote `full_panel` for
HOW MANY.** Both fixes are gated in `tests/test_baseline_fidelity.py`.


### 2(w0) ⛔ **THE WHOLE UNREAD-CAMPAIGN BACKLOG IS DEAD, AND IT IS ONE REASON**

Audited 2026-08-26. Eight campaigns sat on the server either complete-and-never-
read or part-finished, and they were carried as a standing to-do for weeks:

| campaign | runs | why it was kept |
|---|---|---|
| `dosefix` | 32/32 | complete, never scored |
| `dualbar2` | 88/88 | complete; the only 4-dual set with a null PER FAMILY |
| `selectrun` | 32/32 | complete |
| `mc29` | 13/14 | one run short |
| `vit_diag` | 18/49 | part-finished |
| `mnv3bar` | 17/80 | part-finished |
| `vit_ceskip` | 1/48 | barely started |
| `mc_sgd` | 0/32 | generated, never started |

**Every one of them is `dermmnist`.** Not a single iwildcam run among them. So
all eight close the same way and none of it is a judgement call:

* they **cannot be read** -- the dermmnist test set is leaked, 38.7% overall and
  67.3% of melanoma (2(o));
* they **cannot be finished** -- the dataset is removed from `data/` and
  unrunnable, not merely discouraged (2(n));
* `vit_ceskip` could not be finished even with the data, because it sweeps
  `enable_ce_skip`, a key DELETED from the pipeline.

⇒ **Delete them from the backlog and do not re-derive this.** The check that
settles it is one command and it is the FIRST one to run on any campaign whose
provenance is not immediately obvious:

```bash
ls -d results/<root>/*/*/ | awk -F/ '{print $(NF-1)}' | sort -u   # the DATASET
```

🔑 The general lesson, and it has now cost this project twice: **an unread
result is not automatically a pending result.** Check what dataset and what
code version produced it BEFORE scheduling time to read it -- a campaign on a
withdrawn dataset is history, and scoring it would only manufacture numbers
that cannot be quoted.


### 2(w) 🟢 **THE FIRST ARM THAT TAKES A FULL DOSE AND DOES NOT DAMAGE THE
RANKING** -- `results/uniform1`, 252 runs, complete 2026-08-26

Every previous section here reports a cost. This one reports a fix, and it is
the direct answer to 2(t): the constraint evicts the CORRECT items because the
`sum` count carries a `p(1-p)` gradient that is largest exactly where the model
is most confident. `tralo_uniform` replaces that count with a straight-through
log-odds count whose per-item weight is FLAT. Pre-registered prediction: the
ranking damage disappears and the arm ties its own null. **It did.**

9 cells (3 backbones x 3 caps), 4 seeds, 7 arms, one campaign, one commit. All
three trained arms landed **1044 of 1044** constraint steps (100.0%, bfloat16),
and `full_panel` cleared house rule 3: every arm pair differs on at least one
cell-seed.

| vs its OWN lambda=0 twin | `tralo` | **`tralo_uniform`** | `tralo_head` | `tralo_reseed` (RNG floor) |
|---|---|---|---|---|
| AP | **-0.0754  0/9  \*\*\* LOSS** | **+0.0030  6/3  tie** | +0.0026 tie | -0.0016 tie |
| AUROC | **-0.0172  0/9  \*\*\* LOSS** | **+0.0027  6/3  tie** | +0.0024 tie | +0.0016 tie |
| ECE | +0.0320  0/9  \*\*\* LOSS | +0.0126  loss, not after BH | +0.0050 tie | +0.0064 tie |
| Brier | +0.0570  0/9  \*\*\* LOSS | +0.0177  tie | +0.0101 tie | +0.0137 tie |
| NLL | +0.3206  0/9  \*\*\* LOSS | +0.2398  1/8  \*\*\* LOSS | -0.0047 tie | +0.0616 tie |
| ConfGap | -0.0186  0/9  \*\*\* LOSS | -0.0191  0/9  \*\*\* LOSS | +0.0077  9/0  \*\*\* WIN | -0.0092 tie |
| ccF1 | +0.0003 tie | +0.0010 tie | +0.0002 tie | +0.0010 tie |
| macroF1 | -0.0057 tie | +0.0012 tie | (tie) | -0.0028 tie |

**AP goes from a clean 0-of-9 sweep at BH q = 0.0072 to a 6/3 tie, sign
flipped, at the same dose in the same campaign.** AUROC does the same. The
residual `tralo_uniform` deltas are the size of the RNG-only reseed floor, which
is what -- the constraint became free -- actually means.

#### 🔑 THE TIE IS NOT CHEAP: IT STILL ENFORCES. And `tralo_head` does not.

A tie against the null is worthless if the arm simply stopped constraining, so
read it beside how far each arm pulled the raw count toward its budget. This is
a **liveness check, not a metric** -- house rule 5 stands, `raw_over_K` ranks
nothing -- but -- did the treatment do anything -- is exactly what it can answer.

| arm | count pull | x the RNG floor | AP damage |
|---|---|---|---|
| `tralo` | -0.0733 | 7.5x | **-0.0754 \*\*\* LOSS** |
| **`tralo_uniform`** | **-0.0409** | **4.2x** | **+0.0030 tie** |
| `tralo_head` | -0.0170 | 1.7x | +0.0026 tie |
| `tralo_reseed` | -0.0098 | 1.0x | -0.0016 tie |
| `clip` / `focal_clip` | -0.0055 / -0.0000 | 0.6x / 0.0x | -- |

⇒ **`tralo_uniform` keeps 56% of `tralo`s enforcement and pays none of its
ranking cost.** `tralo_head` ties everything because it barely constrains at
all -- 1.7x the RNG floor -- so ITS tie carries no information, and this is the
outcome-level confirmation of 2(u): masking `prm.grad` does not freeze the
backbone, it only starves the constraint. **`head_only` is not the fix.**

#### 🔑 THE CONTROL THAT SETTLES IT: EACH ARM AGAINST **ITS OWN BACKBONE'S** RNG FLOOR

A campaign-wide mean can hide a backbone. `tralo_reseed` gives a per-backbone
noise floor for free -- it is the same null with the RNG stream perturbed, and
because the cap is not in its loss its delta is IDENTICAL across all three cap
levels (span 0.0000, visible in the table and a useful self-check). Score each
arm against the floor of the backbone it ran on:

| AP delta vs own null | L20_G50 | L30_G50 | L50_G30 | span | that backbone's RNG floor |
|---|---|---|---|---|---|
| `tralo` MobileNetV2 | -0.0853 | -0.1347 | -0.0525 | 0.0822 | +0.0002 |
| `tralo` MobileNetV3 | -0.0452 | -0.0633 | -0.0903 | 0.0451 | +0.0207 |
| `tralo` RegNetY400MF | -0.0500 | -0.0478 | -0.1099 | 0.0621 | -0.0255 |
| **`tralo_uniform` MobileNetV2** | **+0.0236** | **+0.0309** | **+0.0243** | **0.0073** | +0.0002 |
| **`tralo_uniform` MobileNetV3** | **+0.0004** | **+0.0058** | **+0.0173** | **0.0169** | +0.0207 |
| **`tralo_uniform` RegNetY400MF** | **-0.0355** | **-0.0197** | **-0.0199** | **0.0158** | -0.0255 |

⚠️ **CORRECTION 2026-08-27.** This first read -- `tralo` below its backbone's
floor in 9 of 9, `tralo_uniform` at or above in 9 of 9 -- and the second half
is **wrong**. MobileNetV3's reseed draw is **+0.0207**, i.e. that particular
RNG perturbation IMPROVED AP, so `tralo_uniform`'s +0.0004 / +0.0058 / +0.0173
all sit BELOW it. Counted properly it is 5 of 9, not 9 of 9.

🔑 **The test itself is the mistake, and it is worth naming.** A single
`tralo_reseed` draw is ONE realization, not an sd, so a SIGNED comparison
against it is not a threshold -- on MobileNetV2 the draw is +0.0002 and every
arm 'clears' it; on MobileNetV3 it is +0.0207 and almost nothing does. Use the
reseed for the MAGNITUDE of the noise, or use `paired_noise` for a real sd.
Do not use one draw as a per-cell bar.

**What the cells actually say, counting signs, across all four backbones:**

| | cells | `tralo` negative | `tralo_uniform` positive | mean `tralo` | mean `tralo_uniform` | mean reseed |
|---|---|---|---|---|---|---|
| uniform1 (3 MobileNet-class) | 9 | **9 of 9** | 6 of 9 | -0.0754 | +0.0030 | -0.0016 |
| vitu1 (ViTB16) | 3 | **3 of 3** | 2 of 3 | -0.0933 | +0.0087 | -0.0142 |
| **combined** | **12** | **12 of 12** | **8 of 12** | | | |

⇒ **`tralo` damages AP in 12 of 12 cells across four backbones, and
`tralo_uniform` does not damage it in any campaign-level reading.** That is
the claim the data supports. The stronger per-cell version does not survive.

ViTB16 per cap, with the flat reseed line beside it (span 0 by construction,
because the cap is not in the reseed's loss -- a useful self-check):

| ViTB16 AP vs own null | L20_G50 | L30_G50 | L50_G30 | span |
|---|---|---|---|---|
| `tralo` | -0.0481 | -0.0834 | **-0.1484** | 0.1003 |
| **`tralo_uniform`** | **+0.0145** | **-0.0096** | **+0.0213** | **0.0309** |
| `tralo_reseed` | -0.0142 | -0.0142 | -0.0142 | 0.0000 |

⚠️ `L50_G30` is the GLOBAL-bound cell: local 50% but global 30%, so its
binding budget equals `L30_G50`'s and only the SCOPE differs. `tralo` is 1.8x
worse there on the same budget, which is a scope effect and not a dose one.
#### 🔑 AND THE DAMAGE NO LONGER TRACKS CONSTRAINT PRESSURE

The three cap levels apply different pressure, so the SPAN across them prices
how the arm behaves as the constraint pushes harder. `tralo` swings 0.045 to
0.082 of AP across caps; **`tralo_uniform` swings 0.007 to 0.017, a 4 to 11x
tighter band**, and does not trend toward the tight cap. Its freedom is
a property of the count function, not of a lucky operating point.

⚠️ **THIS DOES NOT LICENSE A DOSE SWEEP, and the reason is already measured.**
The obvious follow-up -- if it is free, buy more of it -- is void twice over.
Magnitude is not a lever: under `constraint_grad_mode: clip` the step is exactly
`lr*clip` whatever lambda says, so a lambda sweep would be a FIFTH inert flag
(house rule 3). And step COUNT cannot rise without breaking equal compute
against the clippers, which rule 2 forbids and which warm-up 50 already closed
from the other side. **The live test is a fourth BACKBONE, not a bigger dose.**


#### ⚠️ WHAT SURVIVES IS CALIBRATION-ONLY, WHICH 2(j) SAYS CANNOT COST AN ALLOCATION

`tralo_uniform` still loses NLL (+0.2398) and ConfGap (-0.0191) against its
null on the MobileNets, both at \*\*\*. Both are CALIBRATION. 2(j) proves a
monotone rescale leaves every top-K set untouched, so that damage is confined
to the one channel that provably buys and costs no items. On ViTB16 even that
goes away: NLL -0.0070 (tie) and Brier -0.0279 (3/0, BETTER than the null).

#### 🛑 CORRECTION 2026-08-27: THE MECHANISM IS **NOT** -- SCALE MOVES, ORDER DOES NOT --

This section first explained the fix as a pure gauge shift: `tralo_uniform`
carries the largest bias shift of any arm while holding its rank correlation
with the warm-up model at the null's value. That reading came from the
MobileNets alone, and **`vitu1` refutes it as a general mechanism.**

| tau vs warm-up (lower = MORE reordering) | null | reseed | `tralo` | `tralo_uniform` |
|---|---|---|---|---|
| uniform1, 3 MobileNet-class backbones | 0.5229 | 0.5418 | 0.5004 (d -0.023) | **0.5203 (d -0.003)** |
| **vitu1, ViTB16** | 0.5019 | 0.5148 | 0.4904 (d -0.012) | **0.4371 (d -0.065)** |

On ViTB16 `tralo_uniform` reorders **5.6x MORE than `tralo` does**, and its AP
is +0.0087 while `tralo`'s is -0.0933. So it is not preserving the ranking.

⇒ **The correct statement is weaker and more interesting: `tralo_uniform`
does not damage the ranking, and NOT because it leaves the ranking alone.**
Whatever reordering the flat count induces is benign or mildly helpful, while
the `p(1-p)` count's is harmful. The gauge-shift story survives only as a
description of the MobileNet cells, and must not be quoted as the mechanism.

✅ **What IS consistent across both campaigns is the bias shift**, and it is
the liveness evidence: relative to its own null, `tralo_uniform` moves the
logit scale 8.2x the RNG floor on the MobileNets and **26.4x on ViTB16**, in
both cases MORE than `tralo` (4.6x and 6.1x). The arm is emphatically not
inert on either.

⚠️ **AND `raw_over_K` IS NOT USABLE AS LIVENESS ON ViTB16.** On the MobileNets
the post-hoc arms sat at 0.0-0.6x the floor, so an arm at 4.2x was clearly
enforcing. On ViTB16 `clip` reads 5.7x and `focal_clip` 6.3x -- the same as
`tralo_uniform`'s 6.3x -- because these are DIFFERENT MODELS with different raw
counts, not arms with different enforcement. Use the bias shift there, and
say which of the two you mean.
#### ⛔ AND IT STILL WINS NOTHING. Say both halves or the result is a lie.

`tralo_uniform` beats `clip` on AP by +0.0173 (7/2, lean win) and on macroF1 by
+0.0071. **Neither is a constraint win.** `tralo_null` -- same compute, lambda=0
-- already beats `clip` by +0.0143 AP and +0.0059 macroF1. Subtract and the
constraints own contribution is +0.0030 AP and +0.0012 macroF1: the tie above.
This is 2(r) again, unchanged -- **the win is compute, not method** -- and the
fix does not alter it.

On the budget-equalized metrics it is a tie at 0.50 items needing ~101 seeds per
cell. That was never in doubt: 2(v) prices the WHOLE prize at these caps at
0.04 to 0.09x the paired seed noise, so no method could have shown a win here.

🔑 **The honest one-line claim, and it is a real one:** on iwildcam at
warm-up 1, **the TraLO constraint can now be applied at full dose for free.**
It was previously not free -- it cost AP 0.0754, about 2 to 150x the entire
prize, spent backwards. Free is not profitable, and this campaign cannot show
profitable at K/n = 16-30%. But every count-constraint direction this project
has left was blocked behind -- the constraint damages the representation -- and
that blocker is now measured, understood and removed.


#### 🛑 THIRD PASS, AND THE LAST: THE PAIRED NOISE IS **LARGER**, NOT SMALLER

The ratios above use `sd(TP@K)` for ONE arm across seeds. Every comparison in
this project is seed-PAIRED against the arm's own lambda=0 twin, and pairing
normally shrinks the noise -- so the obvious next question is whether that
makes a loose cap affordable. **It does the opposite.** Measured on the same
iwc3 predictions, all three in the same per-class TP items:

| K/n | K | prize | unpaired sd | **reseed sd** | **treated sd** | prize/reseed | prize/treated |
|---|---|---|---|---|---|---|---|
| 20% | 74 | 0.42 | 0.80 | **6.17** | **7.59** | 0.07x | **0.05x** |
| 30% | 111 | 1.17 | 1.96 | **8.32** | **9.73** | 0.14x | **0.12x** |
| 50% | 185 | 4.08 | 6.20 | **10.61** | **16.67** | 0.38x | **0.24x** |
| 70% | 259 | 11.50 | 10.44 | **15.30** | **23.74** | 0.75x | **0.48x** |
| 80% | 296 | 18.00 | 11.52 | **19.79** | **28.42** | 0.91x | **0.63x** |
| 90% | 333 | 29.83 | 13.45 | **23.35** | **29.07** | 1.28x | **1.03x** |

(class 2; class 7 is the same shape, 0.00x to 0.92x. `reseed sd` is
sd(`tralo_reseed` - `tralo_null`), RNG stream only. `treated sd` is
sd(`tralo` - `tralo_null`), the contrast actually run.)

🔑 **WHY PAIRING FAILS HERE, and it is structural.** `tralo` and `tralo_null`
share ONE warm-up epoch and then train 29 more apart. They are not two
readings of one model, they are two models. Pairing cancels almost nothing and
adds the variance of a second training -- so the paired sd is 7.6 to 29.1
items where the unpaired one is 0.8 to 13.5.

⇒ **The RNG-only floor alone matches or exceeds the ENTIRE prize at every cap
level, on both capped classes.** `prize/reseed` is 0.07x at L20 and reaches
1.28x only at K/n = 0.9. That is the answer to "why did ~20 arms tie": the
design's noise floor is larger than its prize everywhere in the region it can
run, and no loss function moves either number.

#### 🛑 THE SEED COLUMN: THIS IS CLOSED BY THE **CAP CHOICE**, NOT BY PHYSICS

A ratio below 1.0 reads as -- shut, everywhere -- and that reading is wrong.
Convert it to the seeds per cell needed at 80% power to detect the WHOLE prize
and the picture separates into two different verdicts (class 2, iwc3):

| K/n | prize | treated sd | **seeds per cell** | verdict |
|---|---|---|---|---|
| 20% (`L20`, protocol) | 0.42 | 7.59 | **2607** | hopeless |
| 30% (`L30`/`L50`, protocol) | 1.17 | 9.73 | **546** | hopeless |
| 50% | 4.08 | 16.67 | **131** | hopeless |
| 70% | 11.50 | 23.74 | **33** | expensive |
| 80% | 18.00 | 28.42 | **20** | affordable |
| **90%** | **29.83** | **29.07** | **7-8** | **~2x the protocol's 4, and affordable** |

🔑 **At K/n = 0.9, SEVEN TO EIGHT seeds per cell would resolve the entire
prize, against the four this protocol already runs.** (7 from the unrounded
inputs, 8 recomputed from the prize and sd as printed above -- the point is the
order of magnitude, and it is the only row where that order is single digits.) So the reason nothing has been measurable
here is the CAP LEVEL, not the method, not the dataset and not the noise. The
three caps the protocol sweeps sit at 16-30% of n, which is the far end of the
hopeless column. That is a design choice, and it was never priced.

⚠️ **AND THE CATCH, WHICH IS REAL AND MUST BE SAID WITH IT.** At K/n = 0.9 the
budget admits 90% of the true positives, so the cap barely constrains: a win
there is a win in a regime where the constraint is nearly vacuous, and it is
NOT evidence for the tight-cap regime the paper is about. The two facts
together are the honest statement:

* where the constraint BINDS, no method can be shown to work at any affordable
  seed count;
* where a method could be shown to work, the constraint barely binds.

That is a property of the EXPERIMENT, not of TraLO, and it is the single most
useful thing this section knows. Quote the seed count with the K/n beside it,
always -- and remember a method capturing HALF the prize costs 4x the seeds
(28 per cell at K/n = 0.9, still affordable; 2184 at L20, still not).


✅ **THE RECEIPT.** `python -m scripts.paired_noise --campaign results/iwc3`
reproduces every column of the table above from the stored predictions, and
`--self-test` gates it. Its liveness case is the one that matters: it builds
two arms differing by a constant per cell and requires the paired sd to come
back at exactly 0 against a large unpaired one. **Without that, -- pairing grows
the noise -- would be unfalsifiable** -- a tool that can only ever report a big
paired sd measures nothing.

⚠️ **THREE NOISE NUMBERS EXIST AND THEY ARE NOT INTERCHANGEABLE.** This
section quoted two of them before getting to the right one, so state which is
meant, every time:

* **unpaired** `sd(TP@K)` for one arm -- what an absolute quality claim faces;
* **reseed-paired** -- the RNG-only floor, and the honest bar for any arm;
* **treated-paired** -- what the contrast you are actually running faces;
* `full_panel`'s **2.11 items** is a FOURTH thing: the paired sd of `d ccF1`
  MACRO-averaged over both capped classes and converted through `(K+n)/2`. It
  is not comparable to the per-class TP items above and must not be
  substituted for them.

✅ **WHAT THIS DOES NOT TOUCH: the DAMAGE is still callable.** 2(p-post)'s
AP -0.0394 at 0 of 9 cells is a large effect measured against its own noise,
not a prize. `uniform1` and `iwc4` both measure damage, not prize, and both
are correctly aimed. The ceiling on what they can conclude is a TIE with the
null -- which is what `uniform1` pre-registered, in those words.

#### What this leaves, stated plainly

* **On the CAPPED classes, the best any method can do here is TIE.** That is not
  pessimism; it is `achieved == ceiling` in four of six.
* **On the UNCAPPED classes there is only downside**, because every trained arm
  shares a backbone with them: 2(s) measures `uncF1` at -0.0062 for `tralo`, and
  the whole cross-family ordering as collateral damage there.
* ⇒ **The only method that can come out non-negative on iwildcam is one that
  provably does not touch the uncapped classes -- and its best case is a tie.**
  That is the honest frame for `tralo_uniform` and `tralo_head` in `uniform1`:
  their pre-registered claim is *the constraint becomes free*, and 2(v) says
  free IS the ceiling, so the campaign is correctly aimed. It is not aimed at a
  win, and no campaign on this dataset can be.
* ⚠️ **This does NOT generalise off iwildcam.** The ceiling is set by `K/n`. A
  dataset where the budget is a large fraction of the true positives has a real
  prize; iwildcam does not, because 2(n) selected it for per-group label SHIFT,
  which is a different property. **Any claim that the method cannot work must
  say `K/n` before it says anything else.**

### 2(z27) 🛑🛑🛑 **THE INDEPENDENT UNIT IS (backbone, HOST). A NEW CAMPAIGN BUYS NOTHING**

Measured 2026-09-01: md5 of `final_predictions_raw.csv` for EVERY `tralo_null`
on iwildcam across all 14 worktrees, grouped by (backbone, seed).

**There are EXACTLY TWO distinct null models per (backbone, seed), however many
campaigns exist.** Nine MobileNetV3 campaigns share two models. And the two
groups are the two HOSTS:

| group | GPU | `amp_dtype` | `grad_scaler` | campaigns |
|---|---|---|---|---|
| **a** | NVIDIA RTX PRO 6000 (dsisco02) | `bfloat16` | False | `dom1` `loose1` `uniform1` `xfam1` |
| **B** | Quadro RTX 6000 (dsisco01) | `float16` | True | `equaldose1` `iwc1` `iwc3` `iwc4` `taskwin2` |

🛑 **`base_model_id` IS IDENTICAL ACROSS BOTH GROUPS** --
`MobileNetV3_iwildcam_f598484ecba1` for all five campaigns checked -- so the id
CANNOT separate them and only the md5 can. The warm-up is genuinely shared; the
divergence is the numerics of the 29 lambda=0 epochs. This is the mirror of the
usual cache trap: there the id collides and the model is reused; here the id
collides and the model is NOT the same.

**This confirms the unit map of 2(z26) exactly**, and explains it:

| unit | backbone | host | campaigns |
|---|---|---|---|
| A1 | MobileNetV2 | a / dsisco02 | `dom1` = `loose1` byte-identically |
| A2 | MobileNetV2 | B / dsisco01 | `equaldose1` |
| B1 | RegNetY400MF | B / dsisco01 | `dom1b` |
| B2 | RegNetY400MF | a / dsisco02 | `loose1` |

🔑 **THE OPERATIONAL RULE, AND IT DECIDES WHAT TO RUN NEXT:**

> **A new campaign on an already-used (backbone, host) buys NO independent
> unit.** There are 4 backbones x 2 hosts = **8 possible units** on iwildcam,
> and four are spent. Another MobileNetV2 or RegNetY400MF campaign on either
> host adds cells and adds nothing to the sign test.

So `taskwin2` (MobileNetV3 x dsisco01) and `vittask1` (ViTB16 x dsisco01) are
units 5 and 6 **because they are new BACKBONES**, not because they are new
campaigns -- and units 7 and 8 are the same two backbones on dsisco02, which is
the cheapest remaining evidence in the project. No new design, no new knob.

⚠️ **AND SAY WHAT THE AXIS IS, EVERY TIME.** These units are independent
MODELS. They are not independent datasets, splits or tasks: all of them sit on
one iwildcam slice. A sign test over them supports **"the sign is stable across
backbones and numerics"**, never "across datasets". The second dataset question
(2(w2), 2(w2c)) is a different and still-open one.

---

### 2(z26-CORRECTED) 🛑 **THE UNIT TABLE BELOW CARRIED A STALE-RECIPE UNIT**

Corrected 2026-09-02. The reading below was computed over five units, one of
which -- `B2 = loose1 / RegNetY400MF` -- ran `constraint_grad_mode: clip`, not
the current `normalize`. It is a **different method**, and it was the single
unit that dissented on all three contrasts. `loose1` is archived.

**On the current recipe only (4 units, `docs/COVERAGE.md`):**

| contrast | was (5 units, mixed recipe) | now (4 units, one recipe) | restricted to units with a verified `task` cell |
|---|---|---|---|
| `tralo` vs `clip` | 4/5, p=0.188 | **4/4, p=0.0625** | **3/3, p=0.125** |
| `tralo` vs its own null | 5/5, p=0.031 | 4/4, p=0.0625 | 3/3, p=0.125 |
| `tralo` vs `tralo_reseed` | 3/5, p=0.500 | **3/4, p=0.3125** | **3/3, p=0.125** |
| #1 of four duals | 3/6 cells | 3/6 cells | -- |

⛔ **THE FOURTH COLUMN WAS ADDED 2026-09-04, AND THE THIRD WENT STALE THE DAY
AFTER IT WAS WRITTEN.** Unit **C1** (`taskwin2` / MobileNetV3) contributes
**ZERO** verified `task` cells: `configs.task_cells.classify` returns
`no_strict_band` for its `L70-90_G95` -- MobileNetV3 class 2's strict band was
re-measured EMPTY on 2026-09-02 under the per-group prize -- and `unmeasured`
for its `L80-100_G95`, where c7 sits at K/n 0.950. A measured-empty band and an
unlooked-at fraction are different things, and neither is `non_task`. Restricted
to units that carry a verified `task` cell the tally is **3**, and a sign test
over 3 floors at 0.125. Every SIGN is unchanged, and C1 was the unit DISSENTING
on the reseed row, so the restricted corpus is CLEANER and LESS significant at
once. `scripts/paper_rows.py` computes this and prints "UNITS CARRYING AT LEAST
ONE VERIFIED `task` CELL: N of M" -- take it from there, not from this table.

⚠️ **Removing `loose1` costs no MobileNetV2 data**: its `tralo` there is
byte-identical to `dom1`'s in 4/4 seeds despite a different `grad_mode` and a
different `code_version`, because `clip` scales by `min(raw_norm, 1.0)` and IS
`normalize` wherever the raw norm is >= 1. The two modes coincide exactly in
that regime. Measured 2026-09-02, not assumed.

🛑 **THE GENERAL LESSON, AND IT IS THE THIRD TIME.** A corpus assembled by
campaign NAME rather than by RECIPE mixes methods. Five distinct TraLO
configurations existed across 277 completed `tralo` runs; only 106 were the
current one. Group by `(constraint_fp32, constraint_grad_mode, code_version)`
BEFORE counting anything.

**The paragraphs below are retained for the per-cell arithmetic and the power
tables, which are unaffected. Read their unit counts as the pre-correction
ones.**

---

### 2(z26) 🛑🛑🛑 **BROKEN TO PAPER-LEVEL ITEMS, ONE ROW IN 158 RESOLVES**

`scripts/paper_rows.py` (added 2026-09-01, `--self-test` gates it) emits one
line per (cell, contrast) and averages NOTHING over cells. Run against the 234
cells of `dom1` + `dom1b` + `loose1` + `equaldose1` it produces 393 rows, and
the reading is uncomfortable and load-bearing:

| | |
|---|---|
| rows in a **strict task** cell | 158 |
| of those, **resolved at 2 sd** | **1** |
| the one that resolves | A1 / MobileNetV2 / `L95_G80` / `tralo` vs `clip`, +9.85 items |

⚠️ **THE `items` FIGURES ARE APPROXIMATE, AND SIGNS ARE WHAT THEY SUPPORT.**
`full_panel` macro-averages cc-F1 over BOTH capped classes, and class 2 and
class 7 have different `(K+n)`. So the macro delta has no single quantum and
no single `(K+n)/2` converts it exactly; `items_from_f1` is exact only PER
CLASS. Every sign, ordering and order of magnitude below stands; a two-decimal
items figure does not. Found by the stage-6 gate agent 2026-09-01.

**No other contrast in this corpus separates from its own seed noise in its own
cell.** Every other number we quote is a SIGN, not a measurement, and the sd
used here is a LOWER BOUND (it assumes the two arms are independent; they are
two models sharing one warm-up).
⛔ **THAT PARENTHETICAL USED TO SAY "measured at 6-12x, FRAMEWORK 2(v)" AND
IT WAS ALGEBRAICALLY IMPOSSIBLE.** For ANY correlation
`sd(A-B) <= sa + sb <= sqrt(2)*sqrt(sa^2 + sb^2)`, so a quadrature sd can
understate the truth by at most **41%**, and positive correlation makes it an
OVER-statement instead. 2(v)'s 0.80-vs-7.59 compares the paired difference sd
to **ONE ARM's** sd, which the quadrature already contains. `paired_noise`'s
own self-test has said `about sqrt(2)` all along. The printed seeds-needed
figures are therefore accurate to within a factor of two, and the repo has
been UNDER-claiming its own power. See 2(z32).

**So the entire evidence base is sign consistency over the FOUR independent
units**, not over the eight cells and certainly not over the 158 rows:

| contrast | units positive | sign p | per-unit items |
|---|---|---|---|
| `tralo` vs its own **null** | **4/4** | **0.0625** | A1 +11.61/+13.23 · A2 +1.71/+3.80 · B1 +4.38/+4.60 · B2 +1.62 |
| `tralo` vs `clip` | 3/4 | 0.3125 | A1 +5.77/+9.85 · A2 +2.84/+4.48 · B1 +6.40/+7.98 · **B2 -1.80** |
| `tralo` vs `tralo_reseed` | 3/4 | 0.3125 | A1 +3.89/+5.98 · A2 +6.54/+7.32 · B1 +1.62/+2.42 · **B2 -1.74** |

🔑 **`p=0.0625` IS THE FLOOR AT FOUR UNITS.** A 4/4 sign test cannot go below
it, so no amount of agreement in this corpus reaches 0.05. **The bar is not
crossed by finding a bigger effect, only by adding a fifth independent unit.**
That, not another knob, is what `taskwin2` and `vittask1` buy.
⛔ **`taskwin2` DID NOT BUY IT** -- both its cells are non-task
(`no_strict_band` / `unmeasured`), so its unit C1 carries no verified `task`
cell at all. 2(z26-CORRECTED).

⛔ **`B2` (`loose1` / RegNetY400MF / `L80_G95`) DISSENTS ON ALL THREE
CONTRASTS**, -1.80 / +1.62 / -1.74. One unit in four is negative against both
the clipper and the RNG floor. It is not noise-shaped -- it is consistent
within itself -- and it must appear in any table that shows the other three.

🔑 **THE TABLE PROVES ITS OWN INDEPENDENCE CLAIM.** `dom1` and `loose1` at
MobileNetV2 `L80_G95` print IDENTICAL rows (+5.77 / +13.23 / +3.89) because
they are one model byte-for-byte. Anyone reading the cell list as eight
replicates would have doubled that unit and reported p=0.0039.

**POWER, per cell, lower bound, for `tralo` vs `clip`:**

| unit | backbone | cap | items | sd >= | seeds needed >= | seeds run |
|---|---|---|---|---|---|---|
| A1 | MobileNetV2 | `L95_G80` | +9.85 | 4.80 | **2** | 4 |
| B1 | RegNetY400MF | `L80_G95` | +6.40 | 7.97 | 13 | 4 |
| B1 | RegNetY400MF | `L95_G80` | +7.98 | 11.07 | 16 | 4 |
| A1 | MobileNetV2 | `L80_G95` | +5.77 | 8.67 | 18 | 4 |
| A2 | MobileNetV2 | `L95_G80` | +4.48 | 7.72 | 24 | 4 |
| A2 | MobileNetV2 | `L80_G95` | +2.84 | 6.15 | 37 | 4 |
| B2 | RegNetY400MF | `L80_G95` | -1.80 | 7.54 | 139 | 4 |

The one cell that resolves is the one needing 2 seeds, and it is the highest
`K/n` cell in the table. That is `paired_noise`'s curve showing up in the
results rather than in a screen: **seeds needed falls as `K/n` rises** (2607 at
L20, 546 at L30/L50, 7 at K/n=0.9). The design is under-powered by 3-9x at
`L80`, and roughly correctly powered at `L95`.

⛔ **DOMINANCE OVER THE RIVAL DUALS IS NOT SHOWN.** `tralo` is #1 of the four
(`tralo`/`alm`/`fioretto`/`hounie`) in **3 of 6** strict task cells. The
"leads all four duals" reading of `dom1` came from a cell list that included
`L90_G95`, which the per-seed re-measurement reclassified as PARTIAL
(2(z24b)). At the strict bar the claim is a coin flip.

### 🎯 THE COROLLARY: TWO STAGED CAMPAIGNS CROSS p<0.05, AND NOTHING ELSE DOES

At `n` unanimous units the one-sided sign test is exactly `0.5^n`:

| units | p | |
|---|---|---|
| **3** (today, restricted to units carrying a verified `task` cell) | **0.125** | the honest floor -- see the `taskwin2` correction below |
| 4 (today, every unit the ledger licenses) | 0.0625 | above the bar, and it CANNOT go lower |
| **5** | **0.03125** | **below** |
| 6 | 0.01562 | below |

Verified against `configs.task_cells.classify` on 2026-09-01 -- and the
`taskwin2` rows RE-VERIFIED 2026-09-04, where the first one FELL:

| campaign | backbone | cap | status | buys |
|---|---|---|---|---|
| `taskwin2` | MobileNetV3 | `L70-90_G95` | ⛔ **`no_strict_band`** (read **task** on 2026-09-01) | **NOTHING** -- unit 5 never arrived |
| `taskwin2` | MobileNetV3 | `L80-100_G95` | `unmeasured` | nothing -- c7 sits at K/n 0.950 |
| `vittask1` | ViTB16 | `L60-90_G95` | **task** | **unit 6**, and it is the HEADLINE backbone |
| `vittask1` | ViTB16 | `L70-90_G95` | **task** | same unit (one campaign, one warm-up) |

⛔ **THE FIRST ROW WAS TRUE ON 2026-09-01 AND FALSE ON 2026-09-02, AND IT IS
THE ROW THE TALLY RESTED ON.** The cap screen behind the 09-01 reading counted
the PRIZE over a GLOBAL top-K while every allocator here is per-group;
re-measured with the per-group prize, MobileNetV3 class 2's strict band is
**EMPTY** on the dsisco01 model `taskwin2` uses -- at every 0.1-grid fraction
either the cap binds 4/4 and the local prize is under the 3.0-item floor, or
the prize clears the floor and the cap has gone slack in some seed. So
`taskwin2` / MobileNetV3 -- ledger unit **C1** -- contributes **ZERO** verified
`task` cells, and the deciding experiment is `vittask1` alone.

🔑 **`no_strict_band` IS A MEASUREMENT; `unmeasured` IS AN ABSENCE; NEITHER
IS `non_task`.** C1's other cell, `L80-100_G95`, is the opposite failure: c7
sits at K/n 0.950, a fraction nobody has looked at. Do not collapse the three.

⚠️ **CONSEQUENCE FOR THE TALLY, AND BOTH NUMBERS BELONG IN ANY WRITE-UP:**
**4/4 units, sign p=0.0625** over the units the ledger licenses;
**3/3 units, sign p=0.125** once restricted to units carrying a verified
`task` cell. The SIGNS do not change -- dropping C1 flips nothing, and it
removes the one unit that was FAILING the `vs tralo_reseed` contrast, so the
corpus gets CLEANER and LESS significant at once. `scripts/paper_rows.py`
computes the restriction itself and prints "UNITS CARRYING AT LEAST ONE
VERIFIED `task` CELL: N of M"; read it there rather than re-deriving a number
that has now gone stale twice.

Both are already staged, single `code_version`, `constraint_fp32: true`,
warm-up 1 / constraint 29, six arms including `tralo_null` and
`tralo_reseed`. `taskwin2` is at 39/48 with dose **203/203 and 174/174**.

🔑 **So the deciding experiment is not a new idea -- it is finishing the two
campaigns already on the disk.** No knob, no loss variant and no extra cap
level moves the headline p below 0.05; only a fifth and sixth independent unit
does. Anything that delays those two campaigns costs the result directly.

⚠️ **And it can go the other way.** Unit 5 disagreeing takes 4/5 to p=0.1875 --
WORSE than today. The two campaigns are the test, not a formality.

✅ **WHAT MAY BE WRITTEN.** "In every independent unit measured, the constraint
moves cc-F1 in TraLO's favour relative to its own lambda=0 twin (4/4 units,
sign p=0.0625, +1.6 to +13.2 items)", stated beside the unit count, the
dissenting unit on the other two contrasts, and the fact that one cell in 158
separates from its own noise.
⚠️ **AND THE RESTRICTED TALLY IN THE SAME BREATH (2026-09-04):** unit C1
(`taskwin2` / MobileNetV3) carries no verified `task` cell, so over the units
that do carry one it is
**3/3 units, sign p=0.125**. Every sign is unchanged. Take the restriction from
`scripts/paper_rows.py`, which prints it; do not re-derive it.
⛔ **WHAT MAY NOT.** Any per-cell effect size quoted as a measurement, any
count of CELLS used as a count of replicates, and any dominance claim.

---

---

### 2(z28) 🛑🛑🛑 **`fioretto_alm` AND `fioretto_ldf` ARE ONE METHOD UNDER `normalize`, AND TraLO's 83-DEGREE DIRECTION DIFFERENCE CHANGES NOTHING**

Measured 2026-09-02, `scripts/dual_cone_probe.py` (192 stored model states,
RegNetY400MF x iwildcam, dom1b, head-exact) and `scripts/arm_identity_check.py`
(24 paired cell-seeds, as-deployed predictions). Both carry `--self-test` with
liveness controls in both directions.

**THE ALGEBRA.** Every trained arm builds its constraint objective from the same
per-constraint soft counts `S_j = sum_{i in G_j} p_ic`:

```
fioretto_ldf   L = sum_j  lambda_j * S_j
fioretto_alm   L = sum_j (lambda_j + mu_t * relu(r_j)) * S_j
hounie_rcl     L = sum_j (lambda_j / N_j) * S_j
tralo          L = sum_j  lambda_j * pen(S_j)
```

Differentiate and every one is `sum_j c_j * g_j` with `c_j >= 0` and
`g_j = dS_j/dtheta`. So all four live in the cone generated by `{g_j}` and can
differ ONLY in the direction of their weight vector inside it, because
`finish_constraint_step` under `mode="normalize"` rescales the result to exactly
`constraint_grad_clip` -- it scales UP when the raw norm is below the bound, not
merely down -- so the magnitude is discarded outright.

**AND AT A FIXED STATE TWO OF THE ARMS BUILD THE SAME WEIGHT VECTOR.** Hold the
model still and `r_j` is fixed, so LDF accumulates
`lambda_j = T * step * relu(r_j)`, ALM accumulates `lambda_j = T * eta * relu(r_j)`
(its `max(0, .)` never binds where `relu(r_j) > 0`) and then adds
`mu_T * relu(r_j)`. Both are `(a positive scalar) * relu(r_j)`. Same direction.
The shipped constants make this exact rather than approximate:
`alm_eta = fioretto_step_size = 0.005` and both `lambda_init = 0`.

⛔ **SO THE "cos = 1.0000 IN 192 OF 192 STATES" I ORIGINALLY REPORTED HERE IS
NOT A MEASUREMENT. IT IS THE ALGEBRA ABOVE, RESTATED.** `dual_cone_probe`
replays the 29 dual updates against ONE frozen residual vector, and under a
constant `r` the identity is forced -- the probe could not have returned
anything else, at any state, on any dataset. The `--self-test` liveness control
does not rescue it either: it proves the instrument can separate SOME arms, not
that it could ever have separated these two. **Retracted as evidence.**

🛑 **AND THE TWO ARMS ARE NOT BOUNDED CLOSE IN GENERAL -- THEY CAN BE
ORTHOGONAL.** Searching over residual TRAJECTORIES rather than fixed states
(random + hill-climb, shipped constants, T = 29) reaches **90.000 degrees**,
with the two weight vectors on DISJOINT supports. The construction is exactly
the one the fixed-state replay cannot express: constraint A violated early then
deeply slack, so LDF's positive-part accumulation HOLDS `lambda_A` while ALM's
raw-residual projection DECAYS it to 0; constraint B violated only at the last
epoch, so ALM's `mu_T * relu(r_B)` fires immediately while LDF's `lambda_B` is
still 0. Milder histories give milder gaps: one violation then permanent slack
is 2.0 degrees, a mid-phase switch to slack is 11 degrees, and "all violated
throughout, magnitudes drifting" is exactly 0.

🔑 **SO THE REAL DISCRIMINATOR IS HISTORY, AND IT NEEDS TWO CONSTRAINTS.**
They can differ only when at least two constraints have DIFFERENT violation
histories. With one constraint every weight vector is a positive multiple of
one basis vector and `normalize` erases the difference outright.

⛔⛔ **AND THE AS-DEPLOYED COMPARISON THAT CARRIED THE CLAIM IS ITSELF A
DEAD-ARM CONTRAST, 2026-09-04.** It read:

> ✅ **WHAT ACTUALLY CARRIES THE CLAIM IS THE AS-DEPLOYED COMPARISON**, which
> is independent of the probe: `|alm - fioretto|` is a median **2.5 items**
> over 24 paired cell-seeds, **0.83x** the RNG floor.

`fioretto` runs at 28.00 steps in every campaign this was measured on (`dom1`,
`dom1b`, `equaldose1`), so `|alm - fioretto|` is a comparison between one arm
at 29 steps and one at 28. **The two arms may genuinely be one method, but this
corpus can no longer be used to say so** -- an observed near-identity between
arms at different dose is not evidence of method identity, and the direction of
the confound is unknown. **UNVERIFIED, and `vitdual2` is what would settle it**:
it runs `fioretto` and `alm` both at 29.00.

| contrast (cos of the constraint-gradient direction, epoch 29) | min | median | max |
|---|---|---|---|
| `fioretto_alm` vs `fioretto_ldf` | **+1.0000** | **+1.0000** | **+1.0000** |
| `fioretto_*` vs `hounie_rcl` | -0.4363 | +0.9548 | +0.9988 |
| `fioretto_*` vs `tralo` | -0.8586 | **+0.1095** | +0.9533 |
| `hounie_rcl` vs `tralo` | -0.8798 | +0.1295 | +0.9408 |

⛔ **"SO THE PAPER'S 'FOUR DUALS' IS EFFECTIVELY THREE ON THIS CORPUS" IS
WITHDRAWN 2026-09-04, AND IT IS WORSE THAN IT LOOKS.** The argument ran:
`fioretto_alm` did not separate from `fioretto_ldf` in anything measured here,
their deployed outputs sit at 0.83x the RNG floor, so any dominance claim
counting them as two rivals counts one comparison twice. **The deployed half of
that argument needs `fioretto`, which is a dead arm** (see the block above).

🛑 **AND THE CONCLUSION IS OVERTAKEN ANYWAY: on this corpus the paper's "four
duals" is not three, it is TWO.** `fioretto` and `hounie` are both dead in all
15 dual-carrying cells, leaving `tralo` and `alm`. The reduction is now a
quarantine fact, not a geometric one, and it is much larger. See 2(z43).

**🔑 THE PART THAT MATTERS MORE.** TraLO's direction really IS different: median
cosine +0.1095 against the duals, i.e. **83 degrees apart, and >60 degrees in
124 of 192 states.** It can even be anti-aligned (cos -0.86). And it makes no
difference to the output. On the deployed predictions, captured true positives
per capped class, paired within (model, cap, seed), 24 pairs each:

```
FLOOR  |tralo_null - tralo_reseed|   median 3.0     <- RNG only, the corrected floor
TEST   |alm - tralo|                 median 3.0     the ONE surviving dual pair
       |clip - tralo|                median 3.5     equal dose, survives
--- STRUCK 2026-09-04: every line below has a DEAD ARM on at least one side ---
       |alm - fioretto|              median 2.5     0.83x the floor
       |fioretto - hounie|           median 2.0
       |fioretto - tralo|            median 2.5
       |hounie - tralo|              median 3.5
```

**Every trained-arm contrast sits at or below the RNG floor.** That is already
2(z11)'s finding, but this supplies the mechanism and makes it much stronger:
it is not that the methods are secretly the same. Two of them are, and the
others are not -- TraLO's gradient points 83 degrees away and the deployed
prediction set does not notice. **A constraint direction can be rotated most of
a right angle with no measurable effect on what is emitted.** That is the
sharpest statement of the structural null this project has, and it is evidence
FOR section 4's account rather than against it.

**⚠️ WHY TraLO's DIRECTION DIFFERS, and it is the known defect.** TraLO's
weight is `lambda_j * pen'(S_j)`. The ratchet is ADDITIVE and gated on the HARD
count (`lambda += 0.05` while `hard_c > limit_c`), so after 29 epochs every
violated constraint carries the same `lambda ~= 1.46` and the differentiation is
entirely `pen'`. The bounded shape's `pen'` decays with the violation
(2(a2)), so TraLO puts its mass on the LEAST violated scopes while every
classical dual puts it on the most violated. Measured: at dom1b/L90 seed 1
⛔ **DEAD-ARM READING, 2026-09-04: the state below is a `fioretto` state at
28.00 steps.** The ALGEBRA of the weight profile is unaffected (it is read at a
fixed model state, not across a training budget), but any comparison of the
REALISED profiles is a 29-vs-28 comparison. Treat the geometry as sound and the
arm-vs-arm reading as UNVERIFIED until `vitdual2`.
Fioretto puts 0.394 of its weight on `global/c7/K=433` while TraLO puts 0.293 on
`g306/c7/K=30`. The 83 degrees is that inversion.

**⚠️ AND THE CONSTRAINT CONE IS NOT ACUTE.** `gamma` (the minimum pairwise
cosine among the generators) is **-0.99 to -1.00 in every one of the 192
states**, and the worst pair is ALWAYS the same group at the two different
capped classes: `g130/c2` vs `g130/c7`, `g306/c2` vs `g306/c7`. Within a group
the two capped classes hold most of the mass, so suppressing one promotes the
other and their gradients are near-antiparallel. The penalty is being asked to
push two coupled quantities down at once and the two pushes cancel. NOTE this
does NOT contradict the 2026-08-21 "the capped classes do not compete" result:
that measured TOP-K SET OVERLAP, on dermmnist, which is removed. This measures
GRADIENT interaction, on iwildcam. Different quantity, different dataset.

**SCOPE, stated because it bounds the claim.** Head-exact only: it is computed
from `dS_j/dW = sum_i p_ic(delta_ck - p_ik) f_i`, which needs no head weights
but DOES need the stored features to be the head's input. That holds for
RegNetY400MF (`fc` is one Linear) and ViTB16 (`heads.head`), and **fails for
MobileNetV2/V3**, whose classifier is `Linear(960,1280) -> Hardswish ->
Linear(1280,8)` -- the stored 960-d features are the BACKBONE output, not the
head input, and gradients computed from them would be fiction. `dual_cone_probe`
refuses those runs by fitting the head and checking `max|p_hat - p|` in
PROBABILITY space (7.2e-06 on RegNet, and the check fires on MobileNetV3). The
weight trajectory is simulated with the model held FIXED, which is the
CHARITABLE direction for separation -- no shared drift pulls the arms together.
The real-run check (`arm_identity_check`) has no such approximation and agrees.

### 2(z29) 🛑🛑🛑 **A COIN FLIP OF THE SAME NORM IS INDISTINGUISHABLE FROM THE CONSTRAINT. THE DIRECTION CARRIES NOTHING**

Campaign `coin1`, run 2026-09-02/03 on dsisco01 GPU 1. RegNetY400MF x iwildcam,
caps `L70_G95` and `L80_G95` (both strictly inside the measured task windows for
BOTH capped classes: class 2 [0.70,0.80], class 7 [0.60,0.90]), 6 arms x 4
seeds x 2 caps = 48 runs, 0 failed. Recipe: `constraint_fp32: True` +
`constraint_grad_mode: normalize`.

**EQUAL DOSE, AND IT IS THE POINT.** `tralo` landed 232/232 constraint steps and
`tralo_coin` landed 232/232. `tralo_coin` is `tralo` with
`constraint_random_direction: true`, which replaces the constraint gradient with
a RANDOM vector rescaled to the SAME delivered norm
(`src/training/constraint_step.py::_randomize_direction`, seeded from a private
generator so it draws nothing from the global RNG and the two arms' dropout
masks and batch order stay identical). Same dose, same schedule, same
everything. **Only the information in the direction differs.**

Captured true positives per capped class, AS DEPLOYED (`final_predictions.csv`,
not the allocator-blind panel), paired within (model, cap, seed), 16 points:

| contrast | median items | ratio to the RNG floor |
|---|---|---|
| FLOOR `\|tralo_null - tralo_reseed\|` | 2.0 | 1.00x by definition |
| **`\|tralo - tralo_coin\|`** | **2.0** | **1.00x** |
| `\|tralo - tralo_null\|` | 2.0 | 1.00x |
| `\|tralo - clip\|` | 3.0 | 1.50x |

**A COIN FLIP IS AS GOOD AS THE PENALTY.** The constraint gradient's DIRECTION
is worth exactly nothing: perturb the model by the same amount in a direction
chosen at random and the deployed prediction set moves as much as it does under
the real penalty, which is as much as it moves under no penalty at all, which is
as much as it moves under a pure reseed.

🔑 **THIS WAS PRE-REGISTERED, WHICH IS WHY IT COUNTS.** It was predicted from
2(z28)'s geometry BEFORE `coin1` was generated. 2(z28) measured that `tralo`
sits a median 83 degrees away from every classical dual and that this changed
nothing; the obvious next question was whether ANY direction changes anything,
and the answer was written down as a prediction first. It is not a null found by
looking.

⚠️ **WHAT IT DOES NOT SAY.** It does not say the constraint phase is inert:
`|tralo - clip|` is 1.50x the floor, so the trained arms DO differ from the
post-hoc clipper. That difference survives when the direction is randomised, so
it is attributable to the REGIME (an extra 29 CE epochs under a fresh Adam plus
29 unit-norm perturbations), not to the constraint. That is 3(0) "the win is
compute, not method", now with the mechanism isolated by a control rather than
inferred.

⚠️ **SCOPE.** One backbone, one dataset, 8 cells, 16 paired points, medians of
small integers so the resolution is coarse. `coin2` (MobileNetV2, the only other
backbone whose two classes have OVERLAPPING strict windows, [0.70,0.80]) is the
replication and is running. **ViTB16 -- the a-priori headline backbone -- has
NO strict task window for either capped class**, so this experiment cannot be
run there at all as the protocol currently defines a task.

⚠️ **AND THE COIN IS A CONTROL, NOT A METHOD.** Do not read this as "use a
random direction". It bounds what the penalty could ever have delivered through
this channel; it does not license anything.

### 2(z30) 🛑🛑🛑 **THE PUBLISHED CORPUS'S HOUNIE BASELINE WAS CRIPPLED THREE WAYS AT ONCE, AND THE METHODS SECTION STILL DESCRIBES THAT ERA**

Baseline-fidelity audit, 2026-09-03. The manuscript side is verified here
directly; the paper side is corroborated by two independent sources
(`configs/protocol.yml` citing arXiv:2306.02426 App. F, and a reviewer that
downloaded and grepped the arXiv LaTeX source).

**THIS IS ABOUT THE PUBLISHED CORPUS, NOT THE CURRENT CODE.** There are two
regimes and they get opposite verdicts. `corpus_final.csv` (warm-up 50, three
MedMNIST datasets, `constraint_grad_mode: clip`) is the one at issue. The
current recipe fixed all three defects between 2026-08-20 and 2026-08-23.

**(a) THE RATES.** `docs/paper/main_edited_by_roei.tex:629-630` states, in the
paper of record, "dual and relaxation steps $0.01$ (Hounie-RCL) on all three
datasets". The paper's own values are eta_lambda = eta_u = **0.1** with
h(u) = ||u||^2, i.e. **alpha = 1**; the corpus ran `hounie_alpha: 10.0`.
`configs/protocol.yml` now says outright that 0.01 "appears NOWHERE in the paper
as a rate".

**(b) THE MECHANISM WAS PROVABLY INERT.** The resilient relaxation's fixed point
is `u* = lambda / (2 alpha)`. A 10x smaller lambda against a 10x larger alpha
puts u* ~100x under the paper's, and the repo's own measurement closes it:
sweeping `alpha` over 200x emitted **bit-identical predictions**
(`configs/protocol.yml`). The thing that makes Hounie-RCL *resilient* did
nothing in any published row.

**(c) AND ITS STEP NEVER REACHED THE CLIP.** Under the corpus-era `clip` mode,
hounie's raw constraint-gradient norm ran 0.005 to 0.1105 against a clip of 1.0
that bound on 0 of 29 epochs, while tralo and fioretto each delivered a
unit-norm step. A ~20x delivered-dose gap
(`src/training/constraint_step.py`, measured on `results/vit_diag`).
⛔ The manuscript's mechanism paragraph
(`main_edited_by_roei.tex:1497-1505`) claims every method "takes a single
norm-clipped constraint step per epoch" and that "the update is independent of
lambda". **Both are false for hounie in the corpus that sentence describes.**
The step-fairness sweep cited as insurance (`:1459-1466`) ran inside the same
crippled regime, so it could not have found the faithful method.

**AND THE METHODS SECTION DESCRIBES A PIPELINE THAT NO LONGER EXISTS.** Verified
line by line in the paper of record: warm-up 50 with a "$300$-epoch budget"
(`:584`) against today's warm-up 1 + 29; "ratchet step $0.002$ with hinge weight
$\\beta$" (`:631`, `:1679-1694`) when `lambda_step` is 0.05, a 25x change, and
**the undershoot hinge is DELETED from the pipeline entirely**. Nothing in the
current headline campaigns matches the methods section as written.

**ALSO: `focal_alpha` IS MATHEMATICALLY DEAD AND THE PAPER PRESENTS IT AS LIVE.**
`src/losses/imbalanced_losses.py` multiplies the WHOLE per-sample loss by a
scalar alpha. Lin et al.'s alpha is the CLASS-dependent balancing factor
`alpha_t`. A global scalar is cancelled by Adam's scale invariance, and the repo
measured it: a 10,000x alpha change gives argmax agreement 1.0000. The
manuscript writes the loss as `-alpha (1-p_y)^gamma log p_y` citing
lin2017focal. So `focal` is a GAMMA-ONLY focal baseline described as an
(alpha, gamma) one, and the imbalanced-baselines table was measured on the three
REMOVED datasets, which are exactly where a real `alpha_t` would have been live.
✅ The plumbing is otherwise clean: the keys are read on the live warm-up path
and are in `warmup_identity_keys`, so `focal_clip` is genuinely gamma-focal
trained and the historical "focal_clip is a second clip" cache defect is NOT
present.

**ALSO: THE PAPER'S PRINTED ALM UPDATE IS NOT THE ONE THE CODE RUNS.**
`:601-604` prints `lambda <- max(0, lambda + eta r) + mu_t max(0, r)` as the
multiplier update. The code stores only `lambda <- max(0, lambda + eta r)` and
rebuilds `mu_t r^+` from the CURRENT iterate at use time, precisely because
storing it compounds it. The code is the more faithful ALM; the printed formula,
read literally, is the compounding variant.

🔑 **THE CONSEQUENCE, AND IT IS THE WHOLE POINT.** No claim of the form "TraLO
beats X" should survive from `corpus_final.csv` into the revised manuscript. The
recipe corpus is the only defensible basis, and it is SMALLER than this
sentence used to say. ⛔ **It listed `dom1` / `dom1b` / `equaldose1` /
`taskwin2` / `vittask1` / `coin1` until 2026-09-04. `vittask1` is WHOLLY
UNSCORABLE** (`scorable=False`, both cells non-task, 2(z42)) **and three of the
remaining five -- `dom1`, `dom1b`, `equaldose1` -- are PARTIAL**, so any
"TraLO beats X" claim drawn from them may not have `fioretto` or `hounie` on
either side (2(z40), 2(z43)). What is left for a rival comparison is `alm`, and
nothing else. The honest caveats are then 2(z28)
and 2(z29): under `normalize` the rivals' own dose mechanisms are cancelled, so
what is being compared is each method's constraint-WEIGHT PROFILE at one imposed
dose, and the direction has been measured to carry nothing.

⚠️ **AND NO SCALAR KNOB HAS BEEN SWEPT ON THE CURRENT CORPUS, ON EITHER SIDE.**
FRAMEWORK 1a already records `constraint_grad_clip` "that sweep has never run",
`lambda_step` / `lambda_global` / `lambda_local` "one mention each, no sweep",
`initial_rho` / `rho_target` "no sweep", `fioretto_step_size` "no sweep". So the
comparison is not tuned-against-untuned. It is untuned-against-untuned, with
TraLO's values the frozen residue of long iteration on this same dataset and the
rivals' the frozen imports. Say exactly that. Two mitigations are real: the
headline backbone was fixed a priori on 2026-08-20, and both harness fixes
(`normalize`, `constraint_fp32`) run in the BASELINES' favour.
✅ And `fioretto_step_size` is now known to be INERT under `normalize` anyway:
`lambda_i = step * sum_t viol_i(t)` scales every multiplier uniformly, so it
cannot change the direction, and the direction is all `normalize` keeps.
Fioretto-LDF has zero live hyperparameters on this corpus. It cannot be
strawmanned by tuning, and it cannot be tuned.

### 2(z33) 🛑🛑🛑 **THE PAPER'S HEADLINE p IS BELOW ITS OWN FLOOR:
SIX CELLS ARE THREE WARM-UP MODELS, AND THE CACHE KEY PROVES IT**

Found and fixed 2026-09-03 in `docs/paper/main_edited_by_roei.tex`. The
internal doctrine has said since 2026-09-01 that two cap levels in one campaign
share a warm-up (`scripts/paper_rows.py`: "EIGHT cells are FOUR units"). It was
never propagated to the text, and the text says the opposite in as many words:

> "Claims spanning several cells are tested on cell means (**cells are the
> independent units**; the four seeds recur across cells)"

**THE MECHANISM, VERIFIED AT THE CACHE KEY.**
`configs/gen_campaign.compute_base_model_id` hashes
`{model_name, dataset_mode, data_dir, num_classes}` plus
`protocol.yml: warmup_identity_keys`:

```
lr  dropout  batch_size  warmup_epochs  pretrained  class_weighted_ce
seed  warmup_loss  focal_alpha  focal_gamma  cb_beta  logit_adjust_tau
```

🔑 **THE CAP IS NOT IN THAT LIST.** So `L30_G30` and `L40_G40` at the same
(backbone, seed) resolve to the SAME `base_model_id`, load the SAME cached
warm-up, and differ only in the 29 constraint epochs that follow. They are two
constrained runs off one model, not two experiments.

**AND THE CORPUS HAS EXACTLY THAT SHAPE.** From
`docs/paper/data/corpus/corpus_final.csv`, the six tight-cap cells are
`{RegNetY400MF, MobileNetV3, ViTB16} x {L30_G30, L40_G40} x seeds 1-4`.
MobileNetV2 carries `L30_G30` but no `L40_G40`, which is precisely why the
headline excludes it. **Six cells, three warm-up models per seed.**

| statistic as printed | unit assumed | unit available | floor |
|---|---|---|---|
| six-cell sign test `p=0.031` | 6 | **3** | 0.125 one-sided |
| `t`-test on six cell means `p=0.013` | 6 | **3** | n/a, inadmissible |
| BH cross-check "per load-bearing component" | 6 | **3** | inherits both |

A one-sided sign test over three unanimous units floors at `0.5^3 = 0.125`, so
**`p=0.031` is not attainable from this design at any effect size** -- the same
arithmetic the paper already applies correctly WITHIN a cell ("a four-pair
Wilcoxon cannot fall below `p=0.125`"). The defect is that the floor doctrine
was applied to seeds and not to cells.

✅ **FIXED IN THE PAPER OF RECORD**, four sites, additions in blue: the
methods sentence now defines the unit as the (backbone, seed) warm-up and says
why; the two headline p-values are restated at `p=0.125` over three groups with
the six per-cell gaps kept as descriptive; the `t`-test is explicitly demoted to
a consistency summary rather than evidence. `pdflatex` clean, 0 errors.
⚠️ `docs/paper/main.tex` is the professor's file and is NOT edited; it still
carries the old numbers, as do `main_rev.tex` and `main_clean.tex`. Anyone
quoting from those three is quoting the defect.

🔑 **WHAT DOES NOT CHANGE.** The effect itself replicates: on fresh seeds
5-10 the same six cells give a mean cc-F1 gap of +0.0369, all six positive,
against +0.037 on the original four, and ten reruns are bit-identical. **The
finding is stable; the p-value attached to it was not admissible.** Those are
different criticisms and only the second is fixed here. The remaining ones are
2(z30) (the rivals are mis-configured in the published rows) and the budget
component of raw cc-F1.

### 2(z34) 🛑🛑 **THE `vs_null` EFFECT IS THE NULL MOVING, NOT TraLO.
TraLO'S ABSOLUTE LEVEL IS 10x MORE STABLE ACROSS CAMPAIGNS THAN ITS OWN
λ=0 TWIN**

Measured 2026-09-03 from `cells_5units.csv`, on-recipe campaigns only
(`dom1` `dom1b` `equaldose1` `taskwin2`; `loose1` excluded, it runs
`grad_mode: clip`). Hold `(backbone, cap)` fixed and ask how much each ARM's
absolute cc-F1 moves between campaigns, converted to items with that cell's own
scale:

| arm | median cross-campaign spread |
|---|---|
| `tralo` | **0.63 items** |
| `tralo_reseed` | 2.21 items |
| `clip` | 3.36 items |
| **`tralo_null`** | **6.60 items** |

The individual cells make it plainer than the median does. MobileNetV2:

| cell | `tralo` | `tralo_null` | `clip` |
|---|---|---|---|
| dom1 / L80_G95 | 0.8773 | **0.8595** | 0.8695 |
| equaldose1 / L80_G95 | 0.8774 | **0.8751** | 0.8736 |
| dom1 / L90_G95 | 0.9275 | **0.9132** | 0.9201 |
| equaldose1 / L90_G95 | 0.9271 | **0.9279** | 0.9248 |

🔑 **TraLO lands on the SAME NUMBER TO FOUR DECIMALS across two campaigns on
two hosts; the null moves by up to 11.6 items.** So `dom1`'s "+12 items versus
its own null" is the null being bad there, not TraLO being good; `equaldose1`'s
"+1.6" is the null being fine. At `equaldose1`/MobileNetV2/`L90_G95` the null
**beats** TraLO (0.9279 vs 0.9271).

**AND THIS IS WHY THE HOST CLUSTERING IS ONLY IN ONE CONTRAST.** Per-campaign
mean `tralo` effect, in items:

| contrast | dsisco02 (`dom1`) | dsisco01 (`dom1b`, `equaldose1`, `taskwin2`) | clusters? |
|---|---|---|---|
| `vs_null` | +12.03, +9.41 | +4.38, +1.63, +4.98, +3.11 | ✅ **no overlap, 2-7x** |
| `vs_clip` | +7.16, +5.02 | +8.44, +3.06, +8.51, +9.09 | ⛔ no -- dsisco01 holds 3 of the 4 largest |
| `vs_reseed` | +4.76, +0.93 | +2.62, +6.04, +2.99, +1.50 | ⛔ no |

So "the effect size clusters by host" is true of `vs_null` and **false of
`vs_clip`, which is the headline contrast.** The clustering lives in the
denominator arm, exactly as the stability table predicts.

⚠️ **WHAT THIS DOES AND DOES NOT SAY.** It does NOT refute the sign: `tralo`
still beats its null in 4/4 units, and a sign test does not read magnitudes.
What it kills is any quotation of a `vs_null` MAGNITUDE as "the size of the
constraint's effect" -- that number is set by how the untreated twin happened
to land, and it varies 11.6 items across campaigns while TraLO varies 0.6.
Beside 2(z29) (a same-norm coin flip is indistinguishable from the penalty)
the reading is consistent and unflattering: **the constraint phase pins the
model to a stable operating point, and neither the phase's DIRECTION nor its
magnitude-versus-null is evidence that the count information did the work.**

⚠️ **CONFOUNDED, AND SAY SO EVERY TIME.** "Campaign" here bundles host, dose
and cap set; `equaldose1` exists precisely to equalise dose, so it differs from
`dom1` by design as well as by machine. n = 6 `(backbone, cap)` combinations.
The A/B that separates host from the rest is queued and unrun.

### 2(z35) 🛑🛑🛑 **`normalize` DELETES THE RIVALS' HYPERPARAMETERS.
`fioretto_ldf` HAS NONE LEFT AT ALL, AND TraLO'S ONLY STRUCTURAL DIFFERENCE IS
THAT IT IS NOT POSITIVELY HOMOGENEOUS**

Measured 2026-09-03, pure algebra plus simulation against the shipped rules in
`scripts/dual_cone_probe.arm_weights`. No GPU, no artefacts. This is the
mathematical answer to "in what sense are these four different methods".

**THE MECHANISM.** `finish_constraint_step` under `mode="normalize"` rescales
the constraint gradient to exactly `constraint_grad_clip`, so **only the
DIRECTION of the weight vector `c` is ever delivered**. And every dual's rule
for `c` is built from linear maps and `max(0, .)`, both **positively
homogeneous**: `max(0, a*x) = a*max(0, x)` for `a > 0`. Scale the residual and
the whole weight trajectory scales with it, direction untouched. TraLO's
`lambda * pen'(S)` is not homogeneous, because `pen'` saturates.

Scale the residual by `c`, hold `K` and the group sizes, read the cosine
against `c = 1`:

| c | `fioretto_ldf` | `fioretto_alm` | `hounie_rcl` | `tralo` |
|---|---|---|---|---|
| 0.25 | 1.000000000 | 1.000000000 | 1.000000000 | 1.000000000 |
| 1.00 | 1.000000000 | 1.000000000 | 1.000000000 | **0.7161** |
| 4.00 | 1.000000000 | 1.000000000 | 1.000000000 | **0.4527** |
| 16.0 | 1.000000000 | 1.000000000 | 1.000000000 | **0.0649** |

🔑 **SO THE DUALS ARE SCALE-FREE AND TraLO IS NOT.** Every dual weights the
constraints by the residual's DIRECTION alone; how badly a cap is violated in
absolute items cannot reweight them. TraLO's saturation reads the depth. **That
is the one structural difference between TraLO and the entire rival family**,
and it is a clean thing to claim -- far cleaner than "a better dual".

**AND IT MAKES THE RIVALS' KNOBS INERT.** At a fixed state, over 300 random
constraint sets, every swept rival hyperparameter gives `cos = 1.000000000`:

| arm | knob | swept over | min cos | verdict |
|---|---|---|---|---|
| `fioretto_ldf` | `fioretto_step_size` | 0.0005 - 5.0 | 1.000000000 | DIRECTION-INERT |
| `fioretto_alm` | `alm_eta` | 0.0005 - 5.0 | 1.000000000 | DIRECTION-INERT |
| `fioretto_alm` | `alm_mu_step` | 0.001 - 10.0 | 1.000000000 | DIRECTION-INERT |
| `hounie_rcl` | `hounie_alpha` | 0.01 - 1000 | 1.000000000 | DIRECTION-INERT |
| `hounie_rcl` | `hounie_eta_lambda` | 0.001 - 1.0 | 1.000000000 | DIRECTION-INERT |
| `tralo` | `tralo_rho_target` | 1 - 10000 | **0.9658** | LIVE |
| `tralo` | `tralo_lambda_step` | 0.005 - 0.5 | 0.99999956 | live, barely |

⚠️ **BUT A REAL RUN HAS A VARYING RESIDUAL, AND THAT SPLITS THE RIVALS INTO
TWO GROUPS.** Repeating over 400 random 29-step trajectories where the
constraints move differently:

| contrast | min cos | median | max |
|---|---|---|---|
| **`ldf`: step 0.0005 vs 5.0** | **1.000000** | **1.000000** | **1.000000** |
| `alm`: eta 0.0005 vs 5.0 | 0.2478 | 0.9654 | 1.0 |
| `alm`: mu_step 0.001 vs 10.0 | 0.3402 | 0.9812 | 1.0 |
| `hounie`: alpha 0.01 vs 1000 | 0.0018 | 0.9833 | 1.0 |
| `hounie`: eta_lambda 0.001 vs 1.0 | 0.0256 | 0.9925 | 1.0 |

🔑🔑 **`fioretto_ldf` IS HYPERPARAMETER-FREE UNDER THIS RECIPE, FULL STOP.**
Its only knob multiplies the entire accumulation, `lambda_T = step * sum_s
relu(r_s)`, so it factors out of the direction at every state and along every
trajectory -- min = median = max = 1.000000. **A 10,000x change delivers a
bit-identical update.** Any criticism of the form "the LDF baseline was
mis-tuned" is void: there is nothing to tune.

`alm` and `hounie` keep live knobs only because each multiplies **two** terms
whose RATIO survives normalisation (`eta` against `mu_t`; `eta_lambda` against
the relaxation `u`). Their median effect is a ~10-degree rotation, occasionally
much more.

⛔ **WHAT THIS DOES TO 2(z30).** That section blames the published corpus's
hounie on a 10x-wrong `alpha` and calls the resilience "provably inert". The
inertness is right but the reason given is too narrow: **`alpha` is
direction-inert at ANY value at a fixed state**, and along a trajectory a
100,000x change moves the direction by a median of 10 degrees. So the corpus's
mis-set `alpha` is a real fidelity defect and a SMALL one at the level the
pipeline delivers. The larger 2(z30) findings (the ~20x delivered-dose gap
under `clip`, and the methods section describing a deleted pipeline) are
untouched and remain the serious ones.

⚠️ **SCOPE.** Everything here is about the DIRECTION delivered by
`mode="normalize"`, which is the recipe. Under `mode="clip"` magnitude survives
below the bound and every verdict above can change -- which is exactly why
`clip` and `normalize` are different methods, not variants.

### 2(z36) 🛑🛑🛑 **HALF THE LOCAL CONSTRAINTS ON iwildcam CARRY
0.016% OF THE PULL. A `K == 0` CEILING IS PAST ITS GRADIENT PEAK ONCE THE GROUP
EXPECTS ONE ITEM, AND REAL GROUPS EXPECT HUNDREDS**

Derived 2026-09-03 from `src/losses/transductive_loss._penalty`, verified
against `torch.autograd` to **7.3e-12** relative (the analytic derivative used
by `scripts/dual_cone_probe` is exact once the shipped `EPSILON` is carried; the
3e-8 seen without it IS the epsilon). No GPU, no artefacts.

**THE SHAPE.** `pen(E) = E/(E+scale) + rho * e^2/(1+e^2)`, `e = E/scale`,
`scale = max(K, 1)`. Its derivative is **non-monotone in the violation** once
`rho >~ 1`: it peaks at **57.5% over budget** and decays for anything worse.
The shipped ramp is `rho: 0.5 -> 100` over 29 epochs, so the peak sits at 1.0%
over at epoch 0 and at 57.4-57.5% from epoch ~1 onward. Reproduced
independently: `g(58% over) / g(8x over)` = **167.1x** at `rho = 100`, matching
the figure in the source comment to one decimal.

🔑 **NOW APPLY IT AT `K == 0`, WHICH IS HALF THE iwildcam LOCAL SET.** There
`scale = 1`, so the peak pull lands at a **soft count of 0.5755**, and:

| group's soft count | d pen / d soft | share of the peak |
|---|---|---|
| 0.575 | 63.13 | 100% |
| 1 | 48.53 | 76.9% |
| 5 | 1.456 | 2.31% |
| 25 | 1.38e-02 | 0.022% |
| 100 | 2.91e-04 | 0.00046% |
| 400 | 9.24e-06 | **0.000015%** |

A soft count is `sum_i p_ic` over every item in the group. iwildcam's test
cameras hold hundreds of items each, so a `K == 0` group's soft count is tens
to hundreds **at initialisation and forever**. It is never anywhere near 0.58.
**Every `K == 0` ceiling is permanently on the far decaying tail, by
construction, and no dose or schedule moves it.**

**WHAT THAT DOES TO THE DELIVERED DIRECTION.** A representative 14-ceiling local
set for one capped class, seven at `K == 0` (the measured iwildcam split) and
seven binding at 1.05x-1.57x, at the end of the rho ramp:

| ceilings | share of the total constraint pull |
|---|---|
| the **seven `K == 0`** | **0.0162%** |
| the seven real budgets | 99.9838% |

Largest-to-smallest weight across one set: **191,315x**. And since
`mode="normalize"` rescales the SUM, the `K == 0` terms do not merely get less
pull -- they round out of the delivered direction entirely.

⛔ **SO THIS LINE IN `CLAUDE.md` AND 2(n) IS TRUE OF THE ALLOCATOR AND FALSE OF
THE OBJECTIVE:** "7 of 14 per-group ceilings are K=0 ... so the LOCAL scope
constrains the output at every cap level". The allocator *is* bound by a zero
ceiling -- it must emit nothing there, and that is real. **The TRAINED model is
not.** The constraint phase optimises against the seven non-zero ceilings and,
to four decimal places, nothing else. Say which of the two you mean.

⚠️ **AND `transductive_loss._penalty`'s OWN DOCSTRING OVERSTATES IT.** It says
`K == 0` "contributes a permanent, non-vanishing gradient pushing `p_ic` down
in that group". Non-vanishing is literally true and practically empty: 9.2e-06
against a sibling constraint's 1.58, under a normaliser that only keeps the
direction of the sum.

🔑 **THIS IS THE SAME DEFECT 2a2 FOUND, AT A SCALE 20,000x LARGER.** 2a2
measured the deepest violator being starved at ~30x spread on dermmnist and
called the bounded shape backwards. On iwildcam the spread is 191,315x, because
`K == 0` forces `scale = 1` and a count of hundreds is then a violation of
hundreds-fold. `linear` and `squared` exist in the code precisely because of
this, and the default stays `rational_bounded` -- it is the manuscript's Eq. 4,
and changing it would silently reinterpret every stored result. **That decision
should now be made deliberately rather than by inertia**, and the cheapest
honest version is to report that the local scope is, in the objective, a
seven-constraint problem and not a fourteen-constraint one.

⚠️ **SCOPE.** The 14-ceiling set above uses realistic but ILLUSTRATIVE soft
counts -- the server has been unreachable since this was derived, so the real
per-group soft counts are not read here. What needs no data is the structural
part: `scale = 1` at `K == 0`, the peak at a soft count of 0.5755, and the
decay beyond it. Re-measure the shares on a real run before quoting 0.0162%.

### 2(z37) 🛑🛑 **THE PAPER REPORTED A 67.3%-LEAKED CAPPED CLASS AND
NEVER SAID SO. NOW IT DOES**

Found 2026-09-03 while sweeping the manuscript for the unit defect. A grep of
`main_edited_by_roei.tex` for `leak` / `duplicat` / `lesion_id` / `overlap`
returned **nothing**, while the paper reports DermMNIST throughout.

2(o) measured this on 2026-08-19 and it was never propagated: `slice_1`, "the
one every derm result uses", has **776 of 2003 test images (38.7%)** sharing a
`lesion_id` with a TRAIN image, and **150 of 223 melanoma (67.3%)**. Melanoma
is the CONSTRAINED class, so the paper's central metric on that dataset is
measured on a two-thirds-leaked class. The cause is `create_slices.py` pooling
train+val+test and re-splitting 80/20 at the IMAGE level, which undoes exactly
the protection `download_data.py` fetches DermaMNIST-C to get.

**THE THREE DATASETS GET THREE DIFFERENT VERDICTS AND THE PAPER NOW STATES ALL
THREE:**

| dataset | status | consequence |
|---|---|---|
| DermMNIST | leaked, **measured** 38.7% / 67.3% melanoma | absolute numbers inflated by an unknown amount |
| TissueMNIST | identical pooling, same base seed, **unauditable** (no per-instance id survives) | must be assumed the same |
| **OctMNIST** | official MedMNIST test split kept whole | **unaffected, and the headline lives here** |

✅ **WHAT SURVIVES, AND IT IS NOT NOTHING.** Every method and baseline is
trained and scored on the SAME split, seed by seed, so the leak is common to
both arms of every contrast. **The paired comparisons -- which are the claims
the paper makes -- stand.** What cannot be quoted is any DermMNIST or
TissueMNIST number as achievable performance. That is 2(o)'s own verdict,
preserved verbatim rather than restated.

✅ Disclosed in `main_edited_by_roei.tex` §Limitations as "Same-lesion
leakage in the DermMNIST and TissueMNIST splits", in blue, pdflatex clean.
Gated: `tests/test_lessons_learned` now fails if the paper mentions DermMNIST
without carrying the disclosure. ⚠️ `main.tex` (the professor's file),
`main_rev.tex` and `main_clean.tex` do NOT carry it.

🔑 **AND ONE THING 2(z30) GOT WRONG, RECORDED SO IT IS NOT "FIXED" BY
MISTAKE.** 2(z30) reads the methods section's warm-up 50, 300-epoch budget and
`ratchet step 0.002` as "describing a pipeline that no longer exists". The
paper reports **MedMNIST only** -- there is no iwildcam result in it -- so those
settings are the CORRECT description of the runs it presents. The genuine
fidelity defects in 2(z30) are the hounie rate misstatement and the mechanism
paragraph's "every method takes a single norm-clipped constraint step", which
is false for hounie under the corpus-era `clip` mode. Do not "modernise" the
methods section to today's recipe; that would make it describe experiments the
paper does not report.

### 2(z43) 🛑🛑🛑 **THE HEAD-TO-HEAD RECOUNT: WITH THE DEAD ARMS DROPPED, TraLO IS #1 IN ZERO OF 15 CELLS. ALL FOUR OF ITS #1 CALLS WERE MANUFACTURED BY A DEAD ARM**

Run 2026-09-04, read-only, over `dom1` + `dom1b` + `equaldose1` -- the only
clean-recipe campaigns carrying rival duals. `deployed_h2h --control clip`
names a #1 only when the items spread exceeds that cell's RNG floor. I then
applied **the tool's own printed rule** to the survivor set, with `fioretto`
and `hounie` removed.

```
#1 named BEFORE (all arms)          : 8 of 15   tralo 4,  alm 2,  fioretto 2
#1 named AFTER  (dead arms dropped) : 2 of 15   alm 2,    TRALO 0
```

| cell | task status | #1, all arms | #1, dead arms dropped | survivor spread vs floor |
|---|---|---|---|---|
| `dom1`/MobileNetV2/L80_G95 | **task** | `tralo` | ⛔ **REFUSED** | 0.00 vs 3.5 |
| `dom1`/MobileNetV2/L90_G95 | partial | REFUSED | REFUSED | 0.50 vs 3.0 |
| `dom1`/MobileNetV2/L95_G80 | **task** | REFUSED | REFUSED | 2.25 vs 7.5 |
| `dom1`/MobileNetV3/L80_G95 | partial | `alm` | ✅ `alm` | 7.25 vs 7.0 |
| `dom1`/MobileNetV3/L90_G95 | partial | REFUSED | REFUSED | 0.75 vs 5.0 |
| `dom1`/MobileNetV3/L95_G80 | partial | REFUSED | REFUSED | 0.75 vs 9.5 |
| `dom1b`/RegNetY400MF/L80_G95 | **task** | `tralo` | ⛔ **REFUSED** | 2.25 vs 4.5 |
| `dom1b`/RegNetY400MF/L90_G95 | partial | REFUSED | REFUSED | 2.00 vs 10.0 |
| `dom1b`/RegNetY400MF/L95_G80 | **task** | `fioretto` | ✅ **`alm`** | 5.25 vs 4.0 |
| `equaldose1`/MobileNetV2/L80_G95 | **task** | `tralo` | ⛔ **REFUSED** | 0.25 vs 6.0 |
| `equaldose1`/MobileNetV2/L90_G95 | partial | REFUSED | REFUSED | 1.00 vs 9.0 |
| `equaldose1`/MobileNetV2/L95_G80 | **task** | `tralo` | ⛔ **REFUSED** | 3.50 vs 5.0 |
| `equaldose1`/MobileNetV3/L80_G95 | partial | REFUSED | REFUSED | 2.25 vs 5.0 |
| `equaldose1`/MobileNetV3/L90_G95 | partial | `fioretto` | ⛔ REFUSED | 1.00 vs 3.0 |
| `equaldose1`/MobileNetV3/L95_G80 | partial | `alm` | ⛔ REFUSED | 2.25 vs 7.0 |

🛑 **ALL FOUR OF TraLO'S #1 CALLS WERE IN VERIFIED `task` CELLS, AND ALL FOUR
COLLAPSE TO REFUSED.** Task status is from `configs.task_cells.classify` on
today's `configs/task_windows.yml`. **Not one of the four was produced by a
lead over `alm`.** In every case the #1 was named because a DEAD ARM sat far
enough below TraLO to stretch the cell's max-minus-min spread past the RNG
floor: `dom1`/MNv2/L80_G95 was named on `fioretto` and `hounie` at -0.75 while
`tralo` and `alm` are **exactly tied at +4.25**; `dom1b`/RegNet/L80_G95 on
`hounie` at -5.75; `equaldose1`/MNv2/L80_G95 on `hounie` at -10.25;
`equaldose1`/MNv2/L95_G80 on `fioretto` at -1.50. **The dead arms were not
TraLO's competition, they were TraLO's measuring stick, and the stick is what
made the gap look wide enough to name.**

🔑 **AND THIS IS THE FAIREST READING IN THE TABLE, NOT THE HARSHEST.** 2(z39)
established that a max-minus-min RANGE over `k` arms grows like
`sd*sqrt(2 ln k)` and so certifies pure noise as differentiated at ~2.7x when
it is compared against a TWO-arm floor. **With exactly two survivors the range
IS the pairwise difference, so that inflation factor is 1.0 and no correction
applies.** Dropping the dead arms does not merely shrink the field; it removes
the one statistical artefact that was inflating every #1 call in this table.

⚠️ **THE SURVIVING FIELD IS TWO ARMS, EVERYWHERE.** In the entire clean-recipe
corpus, once `fioretto` and `hounie` are dropped, **no campaign compares TraLO
against more than one rival dual.** Every "four duals" sentence in this repo is
now a `tralo`-vs-`alm` sentence. The only campaign that can restore a four-way
comparison is `vitdual2` (all four at 29.00 steps, verified), at **32 of 88
runs complete on 2026-09-04**.

#### HOW MUCH OF THE CORPUS THIS TOUCHES

| unit | total | involves a dead arm | survives |
|---|---|---|---|
| runs | 792 | **144 (18.2%)** | 648 |
| cells | 15 | **0 lost entirely** | **15** |
| (cell, arm) rows vs `clip` | 183 | 36 (19.7%) | 147 |
| (cell, arm, contrast) `paper_rows`-style | 423 | **108 (25.5%)** | 315 |
| all (cell, arm-pair) rows | 1,296 | 387 (29.9%) | 909 |

**`equaldose1` is hit hardest: 3 dead arms of 7 emitted, 42.9% of its paper
rows.** Read the second row too: **no cell is lost.** Every one of the 15 keeps
its `tralo` vs `clip` / vs `_null` / vs `reseed` contrasts intact, which is
exactly why the marker is PARTIAL and not blanket.

#### ✅ WHAT IS UNTOUCHED, AND IT MUST BE SAID AS LOUDLY AS THE LOSSES

🟢 **THE PROJECT'S HEADLINE LEDGER SURVIVES THIS ENTIRELY.**
`scripts/paper_rows.CONTRASTS` is exactly three things -- `vs_clip`, `vs_null`
(resolved per FAMILY) and `vs_reseed` -- and **not one of them touches a dead
arm**. So all of the following stand unchanged:

* `tralo` vs `clip`: **4/4 units, sign p = 0.0625**
* `tralo` vs its own `_null`: **4/4 units, p = 0.0625**
* `tralo` vs `tralo_reseed`: **3/4 units**
* restricted to units carrying a verified `task` cell: **3/3, p = 0.125**

🟢 **AND THE PAPER OF RECORD IS ENTIRELY UNAFFECTED.**
`docs/paper/main_edited_by_roei.tex` and its whole `data/` tree are built on a
DISJOINT generation of experiments. Verified 2026-09-04:
`grep -ciE "iwildcam|dom1|equaldose|vitdual|taskwin|vittask"` on the paper
returns **0**; the same grep over all of `docs/paper/data/` matches **0 files**;
and `corpus_final.csv` holds 7,574 rows on `{dermmnist, tissuemnist, octmnist,
aider, eurosat}` with **no iwildcam** and no quarantined campaign in its
`campaign` column. **Every Fioretto / Hounie / ALM number in the manuscript
survives this quarantine.** That corpus has its own separate and already
documented problems -- warm-up 50, the 38.7%-leaked derm test set, removed
datasets -- and none of them are this quarantine's business.

#### ⚠️ UNVERIFIED, WITH WHAT WOULD SETTLE EACH

1. **The 6-vs-8 tally discrepancy.** `docs/COVERAGE.md` and `docs/MISSION.md`
   record "19 cells: #1 NAMED in 6, REFUSED in 13 ... of the 6 named: alm 2,
   tralo 2, fioretto 2". The root set reproduces EXACTLY
   (`dom1`+`dom1b`+`equaldose1`+`taskwin2`+`vittask1` = 19 cells, and both the
   jackknife count 10 and the items/ccF1-disagree count 5 match to the
   integer), but the tally does not: I measure **NAMED in 8, REFUSED in 11; of
   the 8 named, tralo 4, alm 2, fioretto 2**. **Cause not established.** I ran
   the SERVER's copy of `scripts/deployed_h2h.py`, which differs by md5 from
   the local one; the likely explanation is 2(z39)'s pairwise-spread correction
   having landed in the local copy only, which would name FEWER cells.
   **What would settle it:** run the current local `deployed_h2h.py` against
   the same three roots. The recount above is unaffected either way -- at k=2
   the correction factor is 1.0 -- but the per-cell RNG floors it reuses are
   the server scorer's.
2. **`boundary_probe`'s `alm` rows and the graft figures** name no campaign in
   the text. `alm` exists only in the four dual campaigns, so those rows are
   probably `dom1` and probably survive -- but "probably" is not a source.
   **What would settle it:** the `--campaign` argument those runs were given.
3. **Which of the three cap tags is the `partial` cell in unit B1.** All nine
   (model, cap) cells were classified 2026-09-04 -- MobileNetV2
   task/partial/task, MobileNetV3 partial/partial/partial, RegNetY400MF
   task/partial/task -- but this was not cross-checked against
   `docs/COVERAGE.md`'s per-unit "2 task + 1 partial" line for A1/A2/B1.
   **What would settle it:** run `paper_rows` and read its `cell_status`
   column per unit.

### 2(z42) 🛑🛑 **TWO CAMPAIGNS WERE MECHANICALLY PERFECT AND MEASURED NOTHING -- 265 RUNS, AND NO HEALTH CHECK COULD EVER HAVE SEEN IT**

`uniform1` (252 runs) and `vittask1` (13) pass every instrument this project
owns. Parity green. Zero terminal collapse. Zero non-finite values. One code
version each. On-recipe. `uniform1` landed **1044 / 1044** constraint steps and
`vittask1` 29/29. Nothing went wrong.

**And every one of their cells sits outside the measured task window**, so both
campaigns measured the absence of a question:

| campaign | cells | verdict |
|---|---|---|
| `uniform1` | **9 of 9** | 6 `non_task` (MNv2, RegNetY400MF: both classes outside every band) + 3 `no_strict_band` (MNv3). Caps are `L20_G50` / `L30_G50` / `L50_G30` -- exactly the regime 2(z16) closed |
| `vittask1` | **2 of 2** | `non_task`. Class 2 at K/n 0.600 and 0.700 against ViTB16's measured strict band [0.80, 0.90] |

🔑 **THIS IS A DIFFERENT DEFECT CLASS FROM EVERY OTHER ENTRY IN SECTION 2, AND
IT IS THE ONE THE GATES ARE STRUCTURALLY BLIND TO.** Dose, parity, collapse,
recipe, inert flags, quarantine -- every one of those asks *did the experiment
run correctly*. This one asks *was there a question*, and a run that poses no
question runs perfectly. `full_panel` printed a complete, plausible panel for
both.

✅ **FIXED WHERE IT CANNOT GO STALE.** `scripts.quarantine.gate()` -- the one
refusal all seven scorers now call -- CLASSIFIES the cells it is about to
score and announces every one that poses no question:

```
!! ALL 2 OF 2 CELLS DO NOT POSE THE CAP QUESTION: results/vittask1
     ViTB16         iwildcam       L60-90_G95     non_task
     ViTB16         iwildcam       L70-90_G95     non_task
```

Hand-listing the two campaigns we happen to know about catches those two and
goes stale on the next one; classifying whatever is in front of the scorer
cannot. `vittask1` was ALSO found **stalled** -- 34 pending, no dispatcher --
and those 34 were dropped rather than resumed.

⚠️ **AND THE FIVE NEGATIVES ARE STILL NOT ONE THING.** The banner prints the
status verbatim and says so. `non_task` is a measured statement that the cap
chosen sits outside a band that exists; `no_strict_band` is a measured
statement that no cap could pose a question on that class at all;
`unmeasured` is a K/n nobody has looked at; `no_window` is an unmeasured
backbone; `no_data` is a missing slice. Collapsing any into `non_task` is
2(z25)'s inversion. Three places had drifted apart and now share **one list**,
`quarantine.NOT_A_TASK`:

* `paper_rows` checked a hand-written `("non_task", "unmeasured")` and was
  therefore **SILENT on `no_strict_band`** -- the status of
  `taskwin2`/`L70-90_G95`, the only cell independent unit C1 contributes. The
  tool whose entire job is saying what may be WRITTEN printed that row with no
  warning;
* `full_panel` had no branch for `partial` or `unmeasured`, so both printed as
  **"slice not on this machine"** -- a missing-instrument message on a cell
  that was measured. Three of the four licensed units carry a `partial` cell;
* `configs/task_cells.classify`'s own docstring said **"SIX statuses, and the
  FOUR negatives"**, having never been updated for the seventh it gained on
  2026-09-02. There are seven and five.

🛑 **A CHECKOUT WITH NO INSTRUMENT ANNOUNCES `UNVERIFIABLE`, NEVER SILENCE.**
A campaign worktree is PINNED, and `configs/task_cells.py` and
`configs/task_windows.yml` exist in exactly ONE of the five checkouts that hold
campaigns. Reporting those campaigns as non-task would blame them for version
skew; reporting them as fine would be worse. This is the same third outcome
`run_campaign` already names.

**7/7 MUTATIONS CAUGHT** (`tests/gates/test_g6_results.py`): the banner
suppressed; the five negatives narrowed to `non_task` alone; a missing
instrument returning `{}` instead of `None`; the path walk off by one so zero
cells are read; the announcer never CALLED from `gate()`; `uniform1` losing its
marker; and `some cells are dead` printing identically to `NO cell is usable`.
The two controls that first read MISSED were the ones that only ever ran
through a monkeypatch -- the UNVERIFIABLE path and the `gate()` wiring -- which
is the recurring lesson that a control which never touches the real code is not
a control.

### 2(z41) 🔑 **THE RNG FLOOR RESTED ON FOUR NUMBERS, AND MORE `_reseed` ARMS COULD NEVER HAVE FIXED IT -- `rng_reseed` IS NOW A DRAW COUNT**

Every arm-vs-arm claim this project has ever made was judged against a noise
floor estimated from **four observations**: one `tralo_null` / `tralo_reseed`
pair at four seeds. The order-statistic confidence interval of a 4-sample
median is the **entire sample range**, so the floor is known to roughly a
factor of two, and 2(z39)'s verdict -- 36 of 38 cells UNDER-POWERED rather than
NOT DIFFERENTIATED -- is driven as much by the floor's imprecision as by the
arms' similarity. `sensitivity_screen` refuses to decide below `MIN_FLOOR_OBS
= 8` for exactly this reason.

⛔ **THE OBVIOUS FIX IS ARITHMETICALLY EMPTY, AND I PUBLISHED IT BEFORE
CHECKING.** Commit `d9f0844a` and a CLAUDE.md line claimed that swapping the
three duplicate `<family>_null` arms for `<family>_reseed` twins would take the
floor from 4 observations to 16 **at zero GPU cost**, since those runs are
already being paid for. That is false. At `lambda = 0` every family's objective
is **plain CE** -- the dual multiplier never leaves zero, no constraint gradient
is ever formed, and the arm name is the only thing that differs. So
`alm_reseed` would be **byte-identical** to `tralo_reseed`, and sixteen
"observations" would be four observations printed four times. The corpus
already demonstrates this: the `_null` arms of all four families are one model.

✅ **WHAT ACTUALLY ADDS OBSERVATIONS IS A DISTINCT RNG *STREAM*.** `rng_reseed`
was a boolean meaning "take one draw from the global generator before
training", which perturbs dropout and shuffling and nothing else. It is now a
**draw count**:

```
rng_reseed: false / absent  ->  0 draws   (tralo_null)
rng_reseed: true            ->  1 draw    (tralo_reseed -- UNCHANGED, forever)
rng_reseed: 2               ->  2 draws   (tralo_reseed2, new)
```

`true` is pinned at exactly one draw and must stay there: every `tralo_reseed`
run in the corpus was produced that way, and redefining it would silently move
the published floor and make those runs irreproducible. Anything that is
neither a bool nor a non-negative int **RAISES** -- `bool("2")` is `True`, which
is precisely how a reseed control stops reseeding while its name still promises
otherwise (the same mechanism as the sixth inert flag, 2(z39)).

Three lambda=0 variants give **C(3,2) = 3 pairs per seed** instead of one, so a
4-seed campaign estimates its floor from **12 observations, not 4**, and clears
`MIN_FLOOR_OBS` with margin. The cost is 4 runs per campaign, the cheapest arm
there is (no constraint phase at all).

🛑 **AND THE FLOOR IS NOT A DETAIL -- IT IS THE DENOMINATOR OF EVERY VERDICT.**
`deployed_h2h` refuses to name a #1 in 13 of 19 cells because the arm-vs-arm
spread sits at 1.00x the floor; `paired_noise` prices seeds-needed off it;
`MIN_PRIZE = 3.0`, which sets every task window in `configs/task_windows.yml`
and therefore which caps `gen_campaign` will even emit, was itself derived from
it. A floor known to a factor of two propagates into all three.

**5/5 MUTATIONS CAUGHT** (`tests/test_baseline_fidelity.py`): `true` silently
stopping the reseed; `true` becoming two draws; a string being coerced rather
than raising; the draw count being ignored so `reseed2` collapses onto
`reseed`; and `reseed2` being given `reseed`'s stream. The gate asserts four
things at once -- the three arms are pairwise different by md5, all three still
attempt **zero** constraint steps (a reseed control that trains against the cap
is a treated arm wearing a control's name, and the floor would absorb the very
effect it exists to measure), `true` is still one draw, and a bad type raises
-- plus a liveness check that the three arms resolve to three distinct draw
counts, so the md5 assertion cannot be vacuous.

⚠️ **THIS IS FOR THE NEXT CAMPAIGN, NOT THE RUNNING ONES.** `src/` and
`configs/` are frozen while `vitdual2` and `vitcoin1` are live; `code_version`
is a git hash and deploying this would split both campaigns. The existing
corpus keeps its 4-observation floor and every claim built on it keeps the
2(z39) caveat.

### 2(z40) 🛑🛑 **THE 29-vs-28 DOSE GAP IS IN `dom1`, `dom1b` AND `equaldose1` TOO -- 792 RUNS, AND FIVE OF SEVEN SCORERS IGNORED THE QUARANTINE**

`vitdual1` was quarantined for running `fioretto` and `hounie` at 28.00
attempted constraint steps against `tralo`'s 29.00. `scripts.dose_landed` on
the rest of the corpus, 2026-09-04:

| campaign | alm | tralo | fioretto | hounie | other |
|---|---|---|---|---|---|
| `dom1` (384) | 29.00 | 29.00 | **28.00** | **28.00** | `tralo_uniform` 29.00 |
| `dom1b` (192) | 29.00 | 29.00 | **28.00** | **28.00** | `tralo_uniform` 29.00 |
| `equaldose1` (216) | 29.00 | 29.00 | **28.00** | **28.00** | **`tralo_lam0` 28.00** |

**The campaign named for equal dose does not have it**, and it has an extra
offender: `tralo_lam0` is a lambda=0 arm that still gates its backward on a
multiplier, so it loses epoch 0 exactly as the duals do.

✅ **MARKED PARTIALLY, NOT WHOLLY, AND THE DISTINCTION IS THE POINT.** A
blanket marker would delete the evidence behind the headline in order to
describe a defect touching two arms: `tralo` vs `clip` / `focal_clip` / `lp` /
`alm` / `tralo_uniform` / its own `_null` is at EQUAL dose in all three, and
these campaigns carry three of the independent units. So `scripts.quarantine`
grew a third state -- `scorable=True` WITH `dead_arms` -- and the scorers drop
contrasts touching a dead arm while scoring everything else. `scorable=True`
with NO dead arms is now itself a self-test failure: it is a registry row that
does nothing.

🛑 **AND THE AUDIT FOUND THE MARKER REACHED ALMOST NOTHING.** Of seven
scorers, `full_panel` and `cell_table` each carried a PRIVATE COPY of the
refusal, and **`deployed_h2h`, `paper_rows`, `score_scan`, `paired_noise` and
`sensitivity_screen` checked nothing at all**. `paper_rows` is the tool whose
entire job is saying what may be WRITTEN, and it was ungated for a structural
reason: it reads a `cell_table` CSV and has no campaign path to walk, so it now
gates by campaign NAME and DROPS rows for dead arms. There is now ONE
`quarantine.gate()` and all seven call it. Verified end to end on the server:
all six path-based scorers exit 1 on `vitdual1`, including on a SUBDIRECTORY of
it, while `dom1` proceeds with its dead arms announced.

⚠️ **THE REGISTRY IS THE SOURCE OF TRUTH, NOT THE MARKER FILE.**
`QUARANTINE.json` is only its on-disk copy, written by `--apply --execute` on
ONE host, while scoring happens in fourteen worktrees and on a laptop with no
`results/` at all. `_marker_at` already fell back to the registry; that
fallback was undocumented, and someone (me) added a SECOND copy of it to
`is_quarantined` on 2026-09-04 without noticing. **The mutation test is what
found the duplicate** -- removing one copy changed no behaviour, so the
mutation read as MISSED rather than CAUGHT. Now documented once and gated.

✅ **GATED, 7/7 MUTATIONS CAUGHT** (`tests/gates/test_g6_results.py`):
every scorer must CALL the gate and not merely import it (AST, never grep, with
a negative control built by stripping the call from real source); a partial
marker must not hard-block but must still return its dead arms; `scorable=False`
must still be absolute; an UNREGISTERED campaign must still read clean; and the
cross-arm asymmetry must print for a 29-vs-28 shape, print for a ONE-step gap,
and print NOTHING at equal dose.

⛔ `scripts/dose_landed.cross_arm_attempts` used to argue in its own
docstring that the gap was "a property of the METHODS rather than a handicap
this harness imposes" and only had to be STATED. That reasoning is what let
four campaigns run on it. Corrected.

### 2(z39) 🛑🛑🛑 **NOT ONE CELL IN THE CORPUS COULD HAVE SEPARATED TWO METHODS -- AND THE SPREAD STATISTIC WAS INFLATED BY ARM COUNT**

`scripts/sensitivity_screen` over `dom1` + `dom1b` + `equaldose1` + `taskwin2` +
`vittask1` on 2026-09-04. **38 cells, ~850 runs: SENSITIVE 0, UNDER-POWERED 36,
SATURATED 2.**

🔑 **THE TIE IS NOT WHAT SATURATION ALONE PREDICTS, AND SAYING SO
PRECISELY MATTERS.** The models DO saturate globally: `dom1`/MobileNetV2 train
accuracy runs **0.9595** at warm-up exit to **0.9992** at the end, so CE keeps
sharpening through all 29 constraint epochs, and the contestable band is a
median **108 items of 2943 -- 93.6% sit at p > 0.99 or p < 0.01**. But at loose
caps the CUT is not in saturated territory: **p@cut is 0.41-0.65 on most
cells**, and the one cell where it is high (0.9943, `dom1`/MobileNetV3 class 7)
has its DECISION BOUNDARY wide open at `p(1-p) = 0.248`. That cell is a
CUT-PLACEMENT result, not a frozen model -- exactly the distinction section 4
insists on, and the screen now prints both numbers so it cannot be collapsed.

⛔ **WHAT ACTUALLY STOPS THEM IS ARITHMETIC.** The typical arm-PAIR
difference is **2-5 deployed TP items**; the RNG floor in the same cell is
**1.0-10.5**. Same size. This reproduces `deployed_h2h`'s "ratio 1.00x"
automatically, per cell, with the verdict spelled out.

🛑 **DEFECT 1: A `max - min` RANGE IS NOT COMPARABLE TO A TWO-ARM
FLOOR.** A range over k arms grows like `sd*sqrt(2 ln k)` -- ~3.1*sd at k=10 --
against `E|X-Y| = 1.13*sd` for the two-arm floor, so **`range >= floor`
certifies PURE NOISE as differentiated at ~2.7x** before any method does
anything. Measured two independent ways on the same 50 cells: raw `range/floor`
reads a healthy median **2.51**, and the SAME cells read **0.97** once each
range is divided by `E[range of n]`; an sd-based estimator agrees at **0.94**,
with the ratio at or below 1.0 in 26/50 and 30/50 cells respectively. The raw
ratio was an artefact of how many arms each campaign happened to carry.
⚠️ **`deployed_h2h` still uses the range form** (`spread = order_tp[0] -
order_tp[-1]`), and the direction of the error there is anti-conservative: an
inflated spread makes it REFUSE less often, so it can name a #1 on noise. Not
changed here, because it is a working tool and this is a separate fix.

🛑 **DEFECT 2: THE FLOOR ITSELF RESTS ON FOUR OBSERVATIONS.** Every
campaign carries exactly ONE `_null`/`_reseed` pair at 4 seeds, so the noise
every comparison is judged against is a median of four numbers whose
order-statistic CI is the whole sample range. Comparing a well-estimated median
against a badly-estimated one is how a noise cell passes. `MIN_FLOOR_OBS = 8`;
below it the screen returns UNDER-POWERED naming the floor, not the spread.

⛔ **AND I FIRST GOT THE FIX WRONG.** I claimed swapping `alm_null` /
`fioretto_null` / `hounie_null` for `<fam>_reseed` twins would take the floor
from 4 observations to 16 at zero cost. **It would not.** `tralo_reseed` is
`tralo_null` plus the single key `rng_reseed: True`; the `_null` arms are
byte-identical because `lambda = 0` makes them all plain CE (2944); so an
`alm_reseed` is plain CE plus that same key and is byte-identical to
`tralo_reseed`. **Reseed FAMILIES buy nothing.** What buys observations:

| route | observations | extra runs | needs |
|---|---|---|---|
| a third lambda=0 variant (`tralo_reseed2`, distinct offset) | 3 pairs x 4 = **12** | **8** | `rng_reseed` becomes an offset, not a boolean |
| seeds 5-8 on the existing pair | 1 pair x 8 = **8** | **16** | nothing |

The per-observation price differs 4x. Say which is being bought.

⚠️ **DE-SATURATING IS NOT THE INDICATED FIX, AND 2(j) IS THE REASON.**
Post-hoc allocation is optimal for expected TP GIVEN the probabilities, and that
optimality is distribution-free, so a worse model raises the headroom for `clip`
by the same amount it raises it for a trained arm. **A bigger prize is not a
bigger GAP.** Any de-saturation experiment must pre-register why it moves the
gap. `--pretrained {true,false}` now exists in `gen_campaign` to make that pilot
expressible (it is in `warmup_identity_keys`, so the two regimes cannot share a
cached warm-up), but it is a PILOT knob, not a protocol change.

⛔ **THE SIXTH INERT FLAG WAS MINE, AND IT WAS THAT FLAG.** The first version
carried `type=lambda v: v.lower()`, so `--pretrained false` arrived as the
STRING `"false"` -- and `bool("false")` is True. It emitted **48 configs at
`pretrained: True` while every gate passed**, because the gate called
`build_hyperparams` with a real bool and never went through argparse. Caught by
a dry run, not by a test. `test_g3_model` now drives the parser end to end and
`_pretrained` RAISES on anything argparse did not validate. Mutation-tested 6/6
across both gates.

✅ **GATED.** `tests/gates/test_g5_trainlog.py` covers the four verdicts with
a liveness case and the range-vs-pairwise arithmetic as a simulation that cannot
rot; `tests/gates/test_g3_model.py` covers the pretraining override and the
cache split. The screen runs in `run_campaign --step firstrun` as ADVISORY --
required=True would block every campaign this project currently knows how to
run, which is itself the finding.

### 2(z38) 🛑🛑🛑 **A CAP LEVEL IS NOT A SEED: THE ViTB16 WINDOW
WAS MEASURED OFF AN INFLATED COUNT, AND TWO OF `vitdual1`'S THREE CAPS POSE NO
QUESTION** (2026-09-03)

`vitdual1` held THREE completed `tralo_null` runs on ViTB16 and they are TWO
models. A `_null` arm is `lambda = 0`, so it carries no constraint term and its
RAW predictions cannot depend on the cap; only the ALLOCATION downstream can.
Measured: `L60-90_G95/tralo_null/seed_1` and `L70-90_G95/tralo_null/seed_1` are
byte-identical, md5 **3701265ff7c3e9f2**, one `base_model_id`
`ViTB16_iwildcam_9ef746e8e9e5`. Reading the run count would have written a
"3 seed" window from 2 observations, and `binds n/N` -- the quantity the
one-seed entry it replaces was rejected for being unable to establish -- is
computed directly from that count.

🔑 **This is `dom1`/`loose1` (2(z26)) ONE LEVEL DEEPER.** There two
campaigns shared a WARM-UP; here two cap levels share the ENTIRE 30-epoch
model. The general rule: **md5 the reference arm before counting seeds**, and
note this is md5 used in its VALID direction (identical proves identity; rule 3
in CLAUDE.md).

**THE WINDOW, from the 2 distinct models** (dsisco01/fp16, per-group prize,
MIN_PRIZE 3.0), replacing the single vittask1 seed:

| class | n | unconstrained | strict window | LOCerr 0.80 / 0.90 | binds |
|---|---|---|---|---|---|
| 2 | 370 | 362 | **[0.80, 0.90]** | 3.5 / 6.0 | 2/2 |
| 7 | 456 | 472 | **[0.80, 0.90]** | 4.5 / 8.0 | 2/2 |

⛔ **SO TWO OF THE THREE CAPS `vitdual1` WAS RUNNING MEASURE NOTHING.** The
old provisional band was [0.70, 0.90] from ONE seed and it was too WIDE. At
K/n 0.70 class 2 *binds* 2/2 (forced 85..120) but its prize is **2.5 items**
against the 3.0 floor, and against a ~4-item RNG floor: the whole question
there is smaller than the noise. At 0.60 it is 1.5. Both were dropped, their
COMPLETED runs kept as receipts (22 at L60-90, 11 at L70-90; a completed run is
never deleted), and the campaign now runs **`L80-80_G95` + `L90-90_G95`, both
in-window on BOTH classes**, 88 pending, one `code_version` `6658ef8cbc59`, one
recipe. `configs.task_cells.classify` confirms it independently: L60-90 and
L70-90 `non_task`, L80-80 and L90-90 `task`.

⚠️ **AND THE GENERATOR'S WINDOW GATE NEVER RAN ON THIS CAMPAIGN.** The
worktree is pinned at `6658ef8cbc59`, which PREDATES `configs/task_windows.yml`
and `--allow-nontask` -- the file does not exist there and the flag is not in
its `--help`. A pinned campaign tree silently carries a pinned GATE, so
"gen_campaign would have refused it" is not available as a defence for anything
generated in a worktree. Check the gate exists before relying on it.

🛑 **AND IT IS WIDER THAN THE NULLS: THE POST-HOC CLIPPERS COLLAPSE
TOO, AND HARDER.** `clip` and `focal_clip` are warm-up 30 / constraint 0, so
they never see a constraint at ALL and the cap enters only their allocator.
`scripts.paired_noise` on `vitdual1` reports, on real data:

| arm | runs | distinct models |
|---|---|---|
| `clip` | 4 | **2** (seed_1 identical across **THREE** cap levels) |
| `tralo_null` | 3 | **2** |
| `tralo_reseed` | 3 | **2** |
| `tralo` | 3 | **3** ← trained, and it does NOT collapse |

🔑 **SO A SECOND CAP LEVEL BUYS ZERO EXTRA CONTROL MODELS.** It adds
independent observations for the TRAINED arms only. Every clipper and every
`_null` / `_reseed` at the new level is the same network re-allocated -- which
is exactly why it is cheap (the warm-up is cached; `base_model_id` omits the
cap, 2(z33)) and exactly why it must not be counted as a replicate. `tralo`
staying 3/3 is the negative control on real data: the detector does not fire
where it should not.

⚠️ **THE OPEN QUESTION THIS RAISES, AND IT IS NOT CLOSED HERE.**
`paired_noise` on ViTB16 puts the `reseed` floor for class 2 at **6.36 items
at K/n 0.80 and 7.78 at 0.90**, while `task_window`'s LOCerr prize at the same
points is 3.5 and 6.0 and `MIN_PRIZE` is a global 3.0. If those were the same
scale the cap would fail its own floor. **THEY ARE NOT THE SAME SCALE** --
LOCerr counts errors inside the per-group LOCAL budget, `paired_noise` sweeps
its own K from labels -- and 2(v) records four different noise numbers here
differing up to 12x. So this is a QUESTION, not a refutation: it needs the two
put on one scale before either moves. Ticketed; do not restate the comparison
as a result. What can be said now: **the floor is backbone-dependent and 3.0 is
a corpus-wide median**, so a per-backbone `MIN_PRIZE` is the thing to derive.

🛑 **AND THE FOUR-DUAL HEAD-TO-HEAD WAS NOT AT EQUAL DOSE. FIXED, AND THE
CAMPAIGN WAS DISCARDED AND RELAUNCHED RATHER THAN CAVEATED.**
`scripts.dose_landed` on `vitdual1` read:

| arm | landed / attempted | attempted per run |
|---|---|---|
| `alm` | 116 / 116 = **100.0%** | **29.00** |
| `tralo` | 87 / 87 = **100.0%** | **29.00** |
| `fioretto` | 84 / 84 = **100.0%** | **28.00** |
| `hounie` | 84 / 84 = **100.0%** | **28.00** |

Every arm lands 100% of what it attempts, so **no step is being lost** and
`--constraint-fp32` is doing its job (2(u)). The DENOMINATORS differ.
`steps_attempted` counts epochs that reached `finish_constraint_step`, and
`fioretto_ldf` / `hounie_rcl` reach it only when their weighted constraint loss
is strictly positive. Both initialise their multipliers at **exactly zero**
(`fioretto_lambda_init: 0.0`; hounie's `u_g = 0.0`), so on epoch 1 that loss is
identically 0, no backward runs, and they take 28 of 29.

✅ **THIS IS THE METHOD, NOT A DEFECT, AND `alm` IS THE CONTROL THAT PROVES
IT.** A Lagrangian dual started at `lambda = 0` genuinely applies no constraint
force before its first dual update. `fioretto_alm` starts at `lambda = 0` too
and still attempts 29, because its augmented term carries `mu * violation^2`,
which is nonzero at `lambda = 0` -- so the cause is the MULTIPLIER, not the
dual family. TraLO's penalty coefficient is fixed and live from epoch 1.

⛔ **AND "IT IS THE METHOD" IS NOT A LICENCE TO SHIP IT.** 29 / 29 / 28 / 28
is a 3.4% gap in the ONLY phase this comparison is about. It sits under
`full_panel`'s 5-point refusal, so scoring would have proceeded and the number
would have been quoted. That is precisely the failure mode: an arm-vs-arm claim
resting on unequal compute, with every gate green. **The campaign was killed
and relaunched, not annotated.**

✅ **THE FIX IS AN ORDERING, NOT A HYPERPARAMETER.** Both arms ran
`CE -> counts -> violations -> PRIMAL step -> dual update`; the dual block now
runs BEFORE the primal gate:

    CE -> counts -> violations -> DUAL update -> PRIMAL step

Same violations (they are computed on the pre-step model either way), same step
size, `lambda_0 = 0` untouched, no new knob, and no change to the lambda/u
recursion -- for hounie Steps 3 and 4 moved together, so Step 4 still reads the
lambda Step 3 wrote (the deliberate Gauss-Seidel its own comment documents).
Both orders of the alternating primal/dual scheme are conventional, so neither
baseline is made less faithful. `fioretto_alm` is deliberately UNCHANGED: it
always attempted 29, and leaving it alone keeps the control that identified the
multiplier as the cause.

✅ **GATED TWICE, BOTH MUTATION-TESTED.**
`tests/gates/test_g4_grid.py::test_every_trained_arm_ATTEMPTS_every_constraint_epoch`
runs every arm end to end on the smoke harness and asserts
`constraint_steps_attempted == constraint_epochs`; against the pre-fix code it
reports `fioretto attempted 1 ... expected 2`. Its NEGATIVE CONTROL is that the
`lambda=0` twins must still attempt **zero** -- a gate demanding a step from
every arm would pass a null that had started taking them, destroying the only
baseline that isolates the constraint. Lesson 29 guards the ORDERING in source,
which is what a later tidy-up would silently undo.

✅ **FOUR FIXES, EACH GATED.**
* `scripts/task_window` now DEDUPES byte-identical references, prints
  `N run(s) -> M distinct model(s)` naming the pair, and no longer heads its
  output with the glob size as though it were the reference count. Self-tested
  in both directions: two identical runs collapse to one, two differing runs
  (in probabilities OR in hard count alone) stay two.
* `configs.task_cells.classify` CRASHED on an empty `partial` band --
  `partial_w.get(c, (None, None))` covers the ABSENT row, but a row written
  `2: []` returns `[]` and unpacking it raised `ValueError`. The empty STRICT
  band was already handled; this is the same measurement one field over, and
  ViTB16 is the live case (at K/n 1.00 class 2 binds 0/2, class 7 1/2). Gated
  in `tests/gates/test_g4_grid.py` with the pre-fix expression as the negative
  control, and mutation-tested: reverting the fix fails the gate.
* The 4-seed fixture in `task_window --self-test` shared ONE probability array
  across its four "seeds", so the new dedupe correctly collapsed it to two.
  Real seeds never coincide in float; the fixture now perturbs each by 1e-6.
* `scripts/paired_noise` now prints `N (M distinct)` per arm and NAMES the
  collapsing runs, because a sd pooled over cells that hold one model
  double-counts it and biases every prize/noise ratio in that table
  OPTIMISTIC. Self-tested in both directions and mutation-tested: forcing the
  census to report the raw count fails the gate.

### 2(z32) 🛑🛑 **THE ONE ROW THAT "RESOLVES" IS BELOW THE CHANCE
EXPECTATION, AND THE `sd` GLOSS THE WHOLE REPO USES TO DISCOUNT ITS OWN POWER
IS ALGEBRAICALLY IMPOSSIBLE**

Statistics audit 2026-09-03. Every number below was recomputed here from the
source or from `paper_rows.csv` / `cells_5units.csv` directly; nothing is
relayed.

**(a) "1 OF 158 STRICT-TASK ROWS CLEARS 2 sd" IS 4.4 ROWS SHORT OF CHANCE, NOT
A SURVIVOR.** `paper_rows.py` marks a row `resolved` when `|d| >= 2*sd`, where
`d` is a difference of 4-seed MEANS and `sd` is a **per-seed** sd. In t units
that is `t = d/(sd/sqrt(4)) = 2d/sd >= 4` on **df = 3** (every strict row in
the corpus has `n_seeds = 4`, checked). So under the global null:

| df | P(abs(t) >= 4) | expected of 158 |
|---|---|---|
| 3 (**the actual design**) | 0.0280 | **4.43** |
| 6 | 0.0071 | 1.12 |

Reproduced from the dump: 393 rows, 158 strict-task, **exactly 1 resolved**
(`dom1`/MobileNetV2/`L95_G80`, `tralo` vs `clip`, +9.85 items, sd 4.80).
One observed against 4.43 expected. ⛔ **So the honest sentence is "0 of 158
resolve beyond chance", and 2(z26)'s reading of that row as the power curve
"showing up in the results" is selection narrated as confirmation.** The bar is
being cleared LESS often than noise alone would clear it.
🔑 **AND THE TWO LARGEST RESOLVED EFFECTS IN THE WHOLE CORPUS BELONG TO A
RIVAL.** Over all 393 rows exactly 3 resolve; the other two are `alm` at
`equaldose1`/MobileNetV3/`L95_G80` (**+11.80** vs `clip`, **+10.51** vs its
reseed), both larger than TraLO's one. They sit in a `non_task` cell, which is
why they are not headline -- but any sentence of the form "only TraLO resolves"
is false, and it was never written the other way round.

**(b) THE "sd IS A LOWER BOUND, MEASURED AT 6-12x" GLOSS CANNOT BE TRUE OF THE
QUANTITY IT DESCRIBES.** `paper_rows` builds `sd = sqrt(sa^2 + sb^2)`, the
rho = 0 quadrature, and its comment told every reader that the true noise runs
6-12x higher so `seeds_needed` is a lower bound. For ANY correlation

```
sd(A - B) = sqrt(sa^2 + sb^2 - 2*rho*sa*sb) <= sa + sb <= sqrt(2)*sqrt(sa^2 + sb^2)
```

so the worst possible **UNDER**-statement is **41%**, and positive correlation
-- which is what sharing a warm-up produces -- makes it an **OVER**-statement.
Checked numerically over 200,000 random `(sa, sb, rho)`: max ratio
**1.414000** against `sqrt(2) = 1.414214`.

🔑 **WHERE THE 6-12x CAME FROM.** 2(v)'s "0.80 unpaired vs 7.59 treated"
compares the paired-difference sd to **ONE ARM's** sd -- `paired_noise`'s own
docstring defines `unpaired` as "sd of one arm's TP@K across seeds". The
quadrature already contains the treated arm's inflated variance in `sa`, so the
ratio does not transfer. `paired_noise`'s **own self-test has said "LARGER than
either unpaired sd, by about sqrt(2)" all along**; the correct fact was in the
repo, one file away, the whole time.

✅ **AND IT DOES NOT HAPPEN IN THE DATA EITHER.** `sd(treated)/sd(null)` per
cell over the clean corpus: **73 pairs, median 0.78, range 0.13-2.65, and ZERO
above 6x.** The iwc3 variance injection that motivated the gloss is a property
of that campaign's regime, not of the design.

⛔ **NET DIRECTION: THIS REPO HAS BEEN UNDER-CLAIMING ITS OWN POWER.** The
printed `seeds_needed` figures are right to within a factor of two. Corrected
at the definition site and in the three documents that repeated it.

**(c) ONE PRINTED POWER FIGURE WAS 23x OFF.** Section 2's iwc3 note priced a
0.1-item effect against a 2.11-item seed sd at "~152 seeds per cell".
`7.85*(2.11/0.1)^2 = 3495`. 152 is the price of a **0.48**-item effect. The
conclusion is unchanged in direction and stronger in degree; the figure is
corrected in place.

**(d) THE "1.9-9.9 ITEMS" EFFECT SPACE IS A dermmnist NUMBER AND IT IS PRINTED
ON EVERY iwildcam RUN.** Definition site is `full_panel._items_scale`, whose
docstring said "Measured on dermmnist" and was quoted everywhere without the
qualifier -- including in `CLAUDE.md`'s own resolution argument. dermmnist is
removed and leaks 38.7% of its test set, the constant predates iwildcam
entirely, and section 4 already supersedes it **even for dermmnist** with a
corrected 2-18. Caveated at the definition site so the caveat travels with the
print. The per-cell scale the function computes is the number to quote.

🛑 **WHAT THIS DOES NOT CHANGE.** (a) is about per-CELL resolution, which
was already reported as unresolved; the headline rests on SIGN consistency over
units, and 2(z26) plus the unit-ledger gate govern that. (b) moves the power
accounting toward the repo, not away from it. Neither rescues a per-cell
effect: 4 seeds at df = 3 is a t >= 4 bar, and nothing in the corpus clears it
at a rate distinguishable from noise.

#### 2(z30) RESOLUTION, 2026-09-03

✅ **(a) THE RATE, FIXED.** `main_edited_by_roei.tex` now states the resilience
weight (`alpha = 10`, previously absent from the paper entirely) beside the
`0.01` rates, and carries a footnote saying plainly that both depart from
\citet{hounie2023resilient} -- who take `eta_lambda = eta_u = 0.1` with
`h(u) = ||u||^2`, i.e. `alpha = 1` -- that our rates are 10x smaller and our
alpha 10x larger, that the two compound through the fixed point
`u* = lambda/(2 alpha)`, and that the step-fairness sweep does not cover the
alpha axis. Reporting the departure is the honest move; silently "correcting"
the text to the source's values would misdescribe the runs.

✅ **(c) THE MECHANISM CLAIM, NARROWED.** "every method takes a single
norm-clipped constraint step per epoch" now reads "both arms shown here", and
the `lambda`-independence sentence carries the measurement that breaks it:
hounie's raw constraint-gradient norm ran 0.005-0.1105 against a clip of 1.0,
so it was never rescaled and its magnitude, hence its multiplier, passed
straight through. The invariance covers the figure's two arms, not the baseline
set. `pdflatex` clean, 0 errors.
⚠️ `\footnote` inside `\caption` is a LaTeX error -- the note had to be
inlined into the caption body. Worth knowing before the next such edit.

⛔ **AND THE "STALE METHODS SECTION" PART OF 2(z30) IS WITHDRAWN.** It read the
warm-up 50, the "$300$-epoch budget" and `ratchet step 0.002 with hinge weight
beta` as describing a pipeline that no longer exists. **The paper reports
MedMNIST only** -- there is no iwildcam result anywhere in it -- so those are
the CORRECT description of the runs it presents. Modernising them to today's
warm-up 1 + 29 and `lambda_step 0.05` would make the methods section describe
experiments the paper does not report. See 2(z37).

✅ **(d) `focal_alpha`, FIXED, AND MEASURED MORE PRECISELY THAN BEFORE.**
`FocalLoss` computes `(alpha * (1-p_t)**gamma * ce).mean()`, so `alpha`
multiplies the WHOLE objective; the imbalanced arms are LP-clipped, so this
loss is their entire training budget and there is nothing else for it to be
relative to. Lin et al.'s `alpha` is `alpha_t`, CLASS-dependent, and does not
factor out. Re-measured on a real Adam run, 200 steps, everything else held:

| alpha | argmax agreement vs alpha=1 | max abs weight delta |
|---|---|---|
| 0.25 (**the shipped value**) | 0.9961 | 3.24e-02 |
| 1 | 1.0000 | 0 |
| 25 | **1.0000** | 2.17e-03 |
| 2500 | **1.0000** | 1.31e-03 |
| 10000 | **1.0000** | 2.17e-03 |

🔑 **Two regimes, and the earlier note flattened them.** Above ~1 the
invariance is exact -- a 10,000x change is bit-equivalent, reproducing 2(z30)'s
figure. The shipped 0.25 differs from that limit by 0.4% of predictions, and
that gap is **Adam's `eps`**, not a mechanism: at small alpha the gradients are
no longer large against `eps = 1e-8`. So `alpha = 0.25` is not a tuned setting,
it is a value sitting just inside the epsilon regime.
✅ CONTROL: `gamma` IS live, moving the weights 0.26-0.30, two orders more.
So `focal` remains a legitimate baseline -- it reweights per EXAMPLE and never
reads the prior -- but it is **gamma-only focal**, and the paper now says so in
a footnote rather than restating `alpha=0.25` as if it were tuned.

### 2(z31) 🛑🛑 **THE ITEMS SCALE INVENTS ITEMS THAT DO NOT EXIST, THE QUANTUM RULE IS FALSE ON THE HEADLINE METRIC, AND TWO LP RUNS PLAY A DIFFERENT GAME**

Allocator and metric audit, 2026-09-03. Scan basis: all 12 live worktrees,
1,223 runs with `final_predictions.csv`, 2,446 (run, capped-class) pairs.
Arithmetic re-verified independently here.

**(a) A NET-ZERO REALLOCATION REPORTS +1.06 ITEMS.** `full_panel` converts
`d ccF1` to items with ONE scale, `sum_c (K_c + n_c)/2`. But ccF1 is
MACRO-AVERAGED over two capped classes whose `(K+n)` differ, so the conversion
is exact only if the delta splits proportionally to `(K_c+n_c)`, which it never
does. Measured on `dom1`/`L90_G95` (class 2: K=333 n=370 -> 703; class 7:
K=411 n=456 -> 867; panel scale 785):

| what actually happened | reported by the formula |
|---|---|
| +1 true item on class 2 | **1.117** items (+11.7%) |
| +1 true item on class 7 | **0.905** items (-9.5%) |
| **5 items traded class 7 -> class 2, NET ZERO** | **+1.06 items** |

So at the one-item scale this project works at, **the SIGN of a reported effect
can be an artefact of which class moved.** Quote items PER CLASS.

**(b) THE `2/(K+n)` DIVISIBILITY RULE IS FALSE FOR THE PRINTED ccF1.** It is
correct for ONE class at exact fill. Macro-averaged, the lattice is
TWO-DIMENSIONAL: `d ccF1 = a/703 + b/867` for integers a, b. One class-2 item
moves ccF1 by `1/703`, which is **0.5583** of the `2/785` quantum the rule
predicts -- a half-quantum move, routine and legitimate. `gcd(703,867)=1`, so
the achievable spacing is as fine as `1/609501` and the test is near-vacuous.
It has been used as an arithmetic-bug detector. It cannot be one here.
✅ Corrected in CLAUDE.md rule 2.

**(c) GREEDY IS NOT PROVABLY OPTIMAL, BUT IT MEASURES OPTIMAL HERE.** With two
capped classes competing for items the problem is a matroid-intersection /
transportation problem and greedy carries only the generic 1/2 guarantee.
A counterexample was built and RUN through the repo's own
`apply_allocation_heuristic`: 2 items, classes {0,1} capped at K=1,
`P = [[.60,.39,.01],[.55,.01,.44]]`, truth `[1,0]`. Greedy takes `[0,1]` for
capped sum-p 0.61 and **TP 0**; the feasible swap `[1,0]` gives 0.94 and
**TP 2**. Same emitted counts, both feasible.
✅ **AND IT DOES NOT HAPPEN ON THIS CORPUS.** Against an exact-fill
transportation LP (scipy HiGHS, integral vertices asserted) over 13 real
run-instances spanning 3 cells, 2 backbones and 3 cap shapes:
**`d sum-p = 0.000` and `d TP = 0` in 13 of 13.**
🔑 So `scripts/headroom.py`'s claim that the allocator "is already optimal given
these probabilities" is TRUE AS A MEASUREMENT and FALSE AS A THEOREM (it is a
theorem only for ONE capped class, where the constraint is a single laminar
matroid; the protocol always caps two). Cite it as measured. The `2K/(K+n)`
ceiling itself is unaffected: `F1 = 2TP/(M+n) <= 2M/(M+n) <= 2K/(K+n)` for any
allocator emitting `M <= K`, so headroom numbers stand.

**(d) THE LP ARMS BYPASS THE EXACT-FILL DOCTRINE, AND IT COSTS 12 TRUE ITEMS.**
Of 2,446 (run, class) pairs, 2,444 emit exactly `K_eff`, 0 over-emit, and **2
under-emit by 14** -- `lp` and `cb_lp` at `dom1`/MobileNetV2/`L90_G95`/seed_1,
class 2, 319 against `K_eff = 333`. They are ONE event: the two runs' raw
predictions share an md5 (`class_balanced` is bitwise-inert on iwildcam), and
the LP is deterministic. The whole gap is camera 410 (130 emitted against local
K=144); the other six groups sit exactly on budget.
**It is not a bug and not an infeasibility** -- a feasible exact fill exists,
and the LP, minimising expected 0-1 error under `<= K`, correctly declines the
last 14 slots because assigning them strictly increases expected cost.
⛔ **But every other arm is FORCED to exactly K** by
`targeted_correction(force_exact=True)`, which exists precisely "so cross-method
comparisons are apples-to-apples". `danits_lp` and the imbalanced arms opt out
via `skip_targeted_correction=True`. The budget `K = round(0.9 * n_true)` is
label-informed side information the problem grants; force-filling spends it and
the LP leaves 14 slots of it unspent. Measured cost: clip's 15 extra picks in
camera 410 average `p_2 = 0.036` yet **13 of 15 are true impalas**, so clip
captures `TP_2 = 319` against lp's 307, and deployed ccF1 is clip 0.9151 vs lp
0.9069 -- **from byte-identical probabilities**. Part of every published
`lp`-vs-`clip` gap is unequal budget SPEND, not allocator quality.
🛑 **NOTHING IN THE PIPELINE CHECKED UNDER-EMISSION, AND THE METRIC PAYS FOR
IT.** `verify_allocation` and the eval-time raise both test `count > limit`
only, so 319 against `K_eff = 333` logged as "OK". `full_panel` cannot see it
(it re-derives an exact-K allocation) and `deployed_h2h` silently set
`K := emitted`, which is right per arm and compares arms at unequal spend.
⚠️ **The harm runs in the flattering direction**: at FIXED TP, spending less
RAISES cc-F1, because `2TP/(K+n)` shrinks the denominator for the items
forfeited. Measured on this cell, 2*307/(319+370) = **0.8911** against
2*307/(333+370) = **0.8734**, so the arm is paid 0.0177 for declining 14 slots.

✅ **CLOSED 2026-09-03, by ANNOTATION rather than by force-fill.**
`scripts.deployed_h2h.spend_audit` compares emitted counts arm-vs-arm at a
fixed seed and prints an `!! UNEQUAL SPEND` block naming the under-spending
arms and the shortfall; the per-cell rows carry `unequal_spend` and the summary
counts the cells. It needs **no labels and no budget re-derivation** -- every
arm in a cell faces the same `K_eff`, so a difference in emitted counts IS the
finding. Gated as stage-6 GATE 10 with three negative controls (equal spend,
a single arm, and the over-emission direction) and mutation-tested: forcing
`spread` to 0 turns it red.
⛔ **The force_exact route was CONSIDERED AND DECLINED.** An
equality-constrained LP keeps the same totally-unimodular structure and would
stay integral, so it is buildable -- but it changes the METHOD, and every
published `lp` number was produced by the current one. Changing the allocator
to fix a reporting defect trades a known, now-annotated bias for a corpus-wide
re-run. Revisit only if a deployed `lp` gap ever becomes load-bearing.

**✅ WHAT IS CLEAN.** The LP's constraint matrix is TOTALLY UNIMODULAR: groups
partition the samples, so the class-side row supports form a laminar family and
Ghouila-Houri applies; every vertex is integral for free, and re-solving the
real 2,943 x 8 instance through the repo's own `solve_lp_assignment` reproduces
the stored predictions **2943/2943**. `cc-F1 = 2TP/(K+n)` is exact wherever its
precondition holds, verified to the printed digit. `effective_budget`'s
`min(global, local-sum)` matches what actually binds (global 352/433 inert
against local sums 333/411). And the LEAKAGE audit is clean: the only
label-derived input to any decision is `K` itself; checkpoint selection reads
counts against budgets and never a label-derived metric, `model.eval()` holds
through every transductive pass in all four trained arms, and warm-up touches no
test data at all.

**⛔ TASK CLOSED WITHOUT RUNNING: THE BUDGET-PERMUTED CONTROL.** It was queued as
the decisive experiment to close the warm-up confound. It is pre-empted:
permuting budgets across groups destroys strictly LESS information than
`tralo_coin`, which replaces the entire constraint gradient with a random
same-norm vector and lands at the RNG floor (2(z29)). Its outcome is forced.
Do not spend a campaign on it.

---

## 2(z44). THE REJECTION AUDIT -- WHICH CLOSURES REST ON EVIDENCE WE NOW KNOW IS BAD (2026-09-06)

**Section 2 was read end to end against the defects found on 2026-09-06. Many
closures still stand. Some do not, and the pattern is not random.**

A rejection is only as good as the regime it was measured in. Six contaminants
invalidate one:

| | contaminant | why it voids a closure |
|---|---|---|
| 1 | **DEAD-DATA** | measured on `dermmnist` / `octmnist` / `tissuemnist`. All removed; derm leaks 38.7% of its test set; the other two have `synth_group = index % 3`, i.i.d. by construction, so a per-group count constraint is EMPTY there |
| 2 | **NON-TASK-CAP** | measured at L20 / L30 / L50, where 24 of 24 cells fail at least one of {evicts >= 10, errors inside K, p@K < 0.99}. A null there is the absence of a question |
| 3 | **UNPRICED-NULL** | a tie judged against an RNG floor from ONE `_null`/`_reseed` pair at 4 seeds, under `MIN_FLOOR_OBS = 8`. "No difference" and "not enough measurement" are opposite conclusions from the same table |
| 4 | **INVISIBLE-ARM** | the conclusion came from `deployed_h2h` / `tralo_wins`, which ranked only `(tralo, alm, fioretto, hounie)` until 2026-09-06. Any other arm was structurally absent from every head-to-head |
| 5 | **DEAD-ARM** | `fioretto`/`hounie` at 28.00 vs 29.00 attempted steps in `dom1`/`dom1b`/`equaldose1` |
| 6 | **ALGEBRA** | a proof or an identity, not a measurement. These STAND regardless |

🔑 **THE DOMINANT CONTAMINANT IS NON-TASK-CAP, AND IT IS SYSTEMATIC.** Nearly
every "aggression hurts" closure was measured in the tight regime, and 2(z10)
later established WHY anything aggressive must lose there: the clipper's tight
selected set is 99.6% correct, so every swap trades a true positive away. That
is not a fact about the method under test. It is a fact about the cap.

### The closures that DO NOT stand as measured

| tag | idea | contaminant | what it would take |
|---|---|---|---|
| 2(b) | more constraint steps / dose axis | DEAD-DATA + NON-TASK-CAP | the whole dose sweep ran in the regime 2(z10) proves aggression must lose; at loose caps the arms already order by how hard they reach (`alm` +129 > `tralo` +95 > `tralo_uniform` +22) |
| 2(a)/2(a2) | penalty shape | DEAD-DATA + NON-TASK | ⇒ **ACTED ON, see 2(z45)** |
| 2(q)/2(v) | top-K / ranking surrogates | DEAD-DATA + NON-TASK | closed on a `frozen_head_probe` resolution of 35 items read at **L30_G50**, where p@K is 0.999 and the cut is uncontested BY CONSTRUCTION |
| 2(c) | `budget_margin`, `rankpair` | DEAD-DATA + NON-TASK | killed for "ccP does not move" in cells 2(z15) later showed have prec@K = 1.0000, i.e. ccP was PINNED |
| 2(r)/2(u)/2(w) | `tralo_uniform` | NON-TASK + UNPRICED + INVISIBLE-ARM | `uniform1` is 9 of 9 cells outside the window and quarantined |
| 2(z29) | the coin: direction carries nothing | UNPRICED-NULL | median 2.0 items against a floor of 2.0 from ONE pair at 4 seeds, one backbone. ⇒ being re-priced by `price1` |
| 2(w2) | `fmow` has no prize | borrowed calibration | priced off **iwildcam's** p@K, which the tool itself says does not transfer |

⛔ **AND ONE CORRECTION IN THE OTHER DIRECTION.** `protocol.yml`'s
`unproven_arms` said `tralo_coin` had "0 completed runs and no recorded
rationale". It has **24**, in `vitcoin1`, `coin1` and `coin2`. Nobody had read
them because the scorer could not display the arm. Corrected 2026-09-06.

### What still stands, and why it is worth saying

Every ALGEBRA closure survives untouched: `class_balanced` and `logit_adjust`
inert on a balanced train prior; `focal_alpha` and `fioretto_step_size`
cancelled by normalisation; `ortho_project` delivering 0.0% of its promised
CE-neutrality; the panel being allocator-blind by construction; top-K
invariance to a per-class prior shift; `<fam>_reseed` twins being
byte-identical at lambda 0. A proof does not care which dataset it was written
on.

---

## 2(z45). THE PENALTY SHAPE STARVES ITS WORST VIOLATOR, MEASURED ON iwildcam (2026-09-06)

The shipped `rational_bounded` penalty is BOUNDED in the excess, so
`d(pen)/dE` is NON-MONOTONE: near `1/s` at the boundary, peaking ~53-58% over,
decaying toward zero for anything deeper. With ONE term that divides out under
the single normalisation. With SEVERAL it sets their RELATIVE weights.

`scripts/penalty_starvation`, 232 epochs over 8 `dom1` runs, from
`training_log.csv` alone -- no GPU, no model, no re-run:

| | |
|---|---|
| live constraint scopes per epoch | **11** |
| deepest scope violated by | **29.8x** its budget |
| median scope violated by | 0.19x |
| **spread** | **147x** |

| shape | pull(deepest) / pull(median) |
|---|---|
| `rational_bounded` (shipped) | **0.075x** (rho=0.5) -> **0.014x** (rho=100) |
| `linear` | 92x |
| `squared` | 3926x |

**TraLO pulls its worst-violated constraint 13-71x LESS hard than one that is
19% over.**

⚠️ **WHY THIS WAS NOT KNOWN, and it is 2(z44)'s pattern exactly.** The algebra
is 2(a2) and was always correct. It was demonstrated on **dermmnist**, whose
LOCAL scope was EMPTY -- `lp_fallback_used` False with 0 candidates on all 52
runs. The one dataset where the effect was shown is the one where the many-term
case barely existed. iwildcam's spread is 147x against dermmnist's ~30x.

🔑 **CANDIDATE MECHANISM FOR THE `alm` GAP.** An augmented Lagrangian grows its
pull with violation depth without bound; this shape shrinks it. `alm` leads
TraLO in 12 of 17 testable cells. That is a STRUCTURAL difference, not tuning.

`tralo_linear` / `tralo_squared` are staged as `shape1` (RegNetY400MF,
L70-70_G95 + L80-80_G95, both cells in-window, with three lambda=0 RNG streams
so the cells can be PRICED).

---

## 2(z46). `constraint_step_rule: sgd` IS UNDER-DOSED ~89x, AND THE 0.013 COSINE DOES NOT REPRODUCE (2026-09-06)

`scripts/step_dose`, real MobileNetV2 config, real data, one shared Adam state
and one constraint gradient, optimizer restored between rules. Constraint-ALIGNED
weight displacement, `||dw|| * cos(dw, descent direction)`:

| CE steps of Adam state | `shared` \|\|dw\|\| | `shared` cos | `sgd` aligned / `shared` aligned |
|---|---|---|---|
| 60 | 0.0444 | 0.187 | 0.0121 (**83x under**) |
| 126 (a full epoch) | 0.0346 | **0.258** | 0.0112 (**89x under**) |

**`sgd` has a perfect direction and a tiny dose.** So a null from `tralo_sgd`
must be reported as the DOSE GAP, never as "delivering the direction does not
help". Pre-registered in `protocol.yml` before `price1` launched.

🛑 **AND A DISAGREEMENT TO RESOLVE, NOT TO SMOOTH OVER.**
`src/training/constraint_step.py` states that sharing CE's Adam leaves
`cos(parameter update, constraint gradient)` at **0.009-0.017**, i.e. the
constraint step is ~98% a 127th CE step. That figure is the entire motivation
for `tralo_sgd`. Measured here on MobileNetV2 it is **0.187 at 60 CE steps and
0.258 at 126** -- 15-20x higher, and moving in the WRONG direction as Adam's
state matures.

Three readings, and they are not equivalent:
* different DEFINITION -- the 0.013 may be cos of the update against the
  gradient including the CE step, not the constraint step in isolation;
* different STATE -- 0.013 may be measured deep in the constraint phase, on a
  warm-up-trained model 29 epochs in, not on 126 fresh CE steps;
* different BACKBONE -- this is MobileNetV2; the original may be ViTB16.

Until one of those is confirmed, **do not quote "92.6% of the update is stale CE
momentum" as settled**. The dose conclusion is unaffected -- `sgd` is
under-dosed at either cosine -- but the MECHANISM claim behind the whole
delivery program rests on the smaller number, and the smaller number did not
reproduce here.


---

## 2(z47). THE PRIZE IS 13-21 ITEMS AT TASK CAPS, NOT 0-1 (2026-09-06)

**"There is nothing to win" was measured on the TIGHT cells and generalised past
its evidence.** `scripts/headroom`, run on two backbones on 2026-09-06 -- the
entire gap from `clip` to a PERFECT RANKING, per capped class and summed to the
cell:

| campaign | backbone | cap | class 2 | class 7 | **cell** |
|---|---|---|---|---|---|
| `dom1` | MNv2 / MNv3 | L80_G95 | 7.8 | 5.0 | **12.8** |
| `dom1` | | L90_G95 | 11.9 | 8.1 | **20.0** |
| `dom1` | | L95_G80 | 6.5 | 7.2 | **13.7** |
| `vitdual2` | ViTB16 | L80-80_G95 | 7.3 | 5.7 | **13.0** |
| `vitdual2` | ViTB16 | L90-90_G95 | 12.0 | 8.7 | **20.7** |

Against `headroom` reading **0.0-1.0 items** on the iwildcam TIGHT cells. The
cap choice moves the available prize by roughly **15x**, and most of this
project's campaigns were run where there was nothing to win.

🔑 **WHAT THIS CHANGES, AND WHAT IT DOES NOT.**
* It does NOT make any past tight-cap null wrong. It makes those nulls
  UNINFORMATIVE about the method, which is 2(z44)'s NON-TASK-CAP contaminant
  restated in items.
* It does NOT mean TraLO wins. Observed `tralo` - `clip` deltas are +2 to +10
  items -- 10-50% of the prize -- against RNG floors of 3-10 items. The effect
  and the noise are the same size and both sit well under the prize. The honest
  reading is UNDER-POWERED, which is a different conclusion from EMPTY.
* It DOES mean the ceiling arguments that closed several directions
  ("the clipper's selected set is already 99.6% correct, so nothing can be
  won") are statements about the tight regime specifically.

⛔ **READ `binds n/N` BEFORE QUOTING ANY OF THESE CELLS.** `vitdual2` L90-90
class 2 binds in **1 of 3 seeds**; `dom1` L90_G95 binds in 5 of 8. The penalty
is `relu(hard - K)`, so a seed already under budget receives an identically ZERO
constraint gradient and IS its own null. Averaging over it dilutes the cell with
seeds that pose no question. `vitdual2` is the only campaign carrying all four
duals, and one of its two cells is half non-binding.

⚠️ **AND THE `frozen_head_probe` RE-PRICING OF THE RANKING SURROGATES IS
INCONCLUSIVE, NOT A NULL.** Run on `vitdual2`/ViTB16/L90-90_G95 with 8 seeds,
its own NULL CHECK FAILED: refitting a linear head on the frozen features moved
ccF1 by **+8.84 items** against a 2.7-item tolerance, i.e. the harness moves the
endpoint further than any treatment could. `topk` -0.11, `pauc` +0.00, `ptopk`
-0.51 are therefore unreadable on this input. Separately worth keeping: that
+8.84 says a BETTER HEAD ON THE SAME FROZEN FEATURES is worth ~8.8 items, which
is comparable to the whole prize.


---

## 2(z48). THE STARVATION IS REAL IN THE GRADIENT AND ABSENT IN THE OUTCOME (2026-09-06)

**2(z45) measured that the shipped penalty pulls its deepest-violated scope
13-71x less hard than one 19% over. `scripts/deep_scope` asked whether that has
an OUTCOME. It does not, and the pre-registered prediction is REFUTED.**

The test: for every (cell, seed) in `dom1` -- where `tralo` vs `alm` is at EQUAL
dose and explicitly unaffected by the PARTIAL quarantine -- take each group's
RAW argmax count per capped class from `final_predictions_raw.csv`, subtract the
group's budget, and bucket the scopes by how deeply the **lambda=0 null**
violates them. Bucketing on the null and never on the arm under test is what
keeps it from being endogenous: an arm that closed a scope would otherwise move
it out of the deep bucket by construction.

178 violated scope-instances over 24 cell(seed)s. Excess REMOVED vs the null, in
raw items:

| depth (E/K) | scopes | null E | K=0 | `alm` | `tralo` | `clip` REF | `lp` REF |
|---|---|---|---|---|---|---|---|
| shallow 0.00-0.19 | 59 | 11.3 | 0% | +0.5 | -0.9 | -0.5 | -0.5 |
| middle 0.19-1.00 | 59 | 25.5 | 22% | **+13.6** | **+8.9** | +2.3 | +2.3 |
| **DEEP 1.00-80.0** | 60 | 19.1 | **83%** | **+10.2** | **+9.8** | -2.5 | -2.5 |

🔑 **`clip` and `lp` ARE THE BAR AND THEY COST NOTHING.** Both are post-hoc
methodologies -- protocol rule 1 runs them at warm-up 30 / constraint 0, so they
take ZERO constraint steps and their raw predictions are plain CE. They read
-0.5 / +2.3 / -2.5, so the items table has essentially no artefact floor. Net of
them:

* **DEEP: `alm` +12.7 vs `tralo` +12.3 -- TIED.**
* **MIDDLE: `alm` +11.3 vs `tralo` +6.6 -- a 4.7-item gap.**

**The entire `alm` advantage in constraint satisfaction sits at MIDDLE depth.
TraLO closes its deepest scopes as well as `alm` does.** The prediction written
into 2(z45) and `protocol.yml` -- that `alm` leads because the bounded shape
leaves TraLO weak where violations are deepest -- is wrong as an outcome claim.

⚠️ **AND THE DEEP BUCKET IS 83% K=0 SCOPES**, the "predict none of this species
at this camera" ceilings. There `s = max(K,1) = 1`, so `E/K` is just the raw
count and "80x over" means 80 items. Read that row as the zero-ceiling regime,
not as a continuation of the depth axis.

⛔ **THE AIMING STATISTIC IS ALMOST ALL ARTEFACT, AND THE TOOL NOW SAYS SO.**
Correlating a scope's share of the penalty slope against the fraction of its
excess removed reads `tralo` +0.447 -- which looks like the shape aiming. **The
zero-constraint `clip` and `lp` read +0.400 on the same sets.** A shallow scope
has a tiny excess, so a few items of ordinary run-to-run movement remove a large
FRACTION of it; slope share is itself largest at shallow depth; so the two
correlate positively for any arm, including one that never took a constraint
step. `deep_scope` prints the post-hoc arms as REFERENCE rows and labels every
other arm as clearing the bar or not. Net of it: `tralo` +0.047, `tralo_uniform`
**-0.100**, `alm` **-0.400**. Only `alm`'s large negative is a real signal, and
it is the expected one -- an unbounded augmented Lagrangian should weight scopes
opposite to a bounded penalty.

🟢 **ONE ENCOURAGING RESULT, STATED WITH ITS CAVEAT.** Across runs, total raw
excess removed correlates with deployed capped-class TP moved: **rho +0.504 over
6 cells, 360 runs, 4 of 6 cells positive**. So the proxy the constraint
optimises is not orthogonal to the metric that is scored -- which is the
premise the whole program rests on and had never been checked. ⚠️ But 4 of 6 is
sign p=0.34, and the correlation POOLS ARMS within a cell, so it may be reading
"trained arms both close more and score better" rather than a causal link.
Suggestive, not established.

### What this changes for `shape1`, written BEFORE its results are read

`shape1` (RegNetY400MF, L70-70_G95 + L80-80_G95) is running `tralo_linear` and
`tralo_squared`. The prediction is REWRITTEN here, on the record, because the
old one is refuted and rewriting it afterwards would be worthless:

* **`squared`** (slope `2e/s`, weight growing without bound in depth) aims at the
  DEEP bucket. That bucket is already closed to `alm`'s level. **It is now
  predicted to be uninformative or harmful**, and it is retained as the CONTROL
  that discriminates the two stories rather than as a candidate.
* **`linear`** (slope `1/s`, so weight `∝ 1/K`) up-weights middle and deep
  scopes relative to shallow ones under `rational_bounded` at `initial_rho`.
  The middle bucket is where the measured gap actually is, so **`linear` keeps a
  live prediction** and is the arm to read.
* If `linear` moves the middle bucket toward `alm` and `squared` does not, the
  weighting story survives in its corrected form. If NEITHER moves it, the
  penalty shape is closed as a direction and the `alm` gap is somewhere else.

⛔ **AND NEITHER OUTCOME IS A WIN BY ITSELF.** Closing the middle bucket is
constraint satisfaction, not quality. The allocator emits exactly K per scope
regardless, so a shape that fixes the counts and moves no deployed TP must be
reported as exactly that. `flips` and proximity to feasibility are still not
metrics.


---

## 2(z49). THE LATCH NEVER FIRES, AND THE REAL GAP IS THE MULTIPLIER'S DYNAMIC RANGE (2026-09-06)

**Two results from `scripts/latch_probe` over 72 completed `dom1` runs, at zero
GPU cost. The first refutes a hypothesis raised the same day; the second is the
sharpest structural difference between TraLO and the three rival duals found so
far.**

### 1. ⛔ THE SATISFACTION LATCH IS DEAD ON iwildcam -- 0 of 72 runs

`tralo/train.py` gates BOTH adaptive channels on one latch: `ratchet_gate =
satisfaction_epoch is None`, and `satisfaction_epoch` is set on the first epoch
where `global_satisfied and local_satisfied` and is NEVER cleared. On paper that
permanently freezes the lambda ratchet and the rho ramp for every scope.

**It never happens.** `tralo`, `tralo_null` and `tralo_uniform` all latch in
**0 of 24 runs each**. Satisfaction is a global AND over every scope and SEVEN of
iwildcam's fourteen per-group ceilings are `K = 0`, so the conjunction is never
true. The ratchet gate is open for all 29 epochs and rho always completes its
0.5 -> 100 ramp.

Consequences, and they cut in several directions:
* The latch is **not** the mechanism behind the `alm` gap. Closed for free.
* FRAMEWORK's `no_freeze` ablation (+0.13 pp) is **moot on this dataset** rather
  than confirmed -- there is nothing to un-freeze. It remains uninterpretable
  for the reason already given (measured where the local scope was empty), but
  it is no longer even the right question here.
* `satisfaction_epoch` is **not persisted**: `runner.py` puts it in
  `best_metrics` and only a subset reaches `config.json`. It was reconstructed
  from `training_log.csv`'s `Global_Satisfied` / `Local_Satisfied` columns,
  which are the exact two booleans the latch ANDs.

### 2. 🔑 TraLO's MULTIPLIER IS A FREQUENCY COUNTER, AND ITS RANGE IS CAPPED BY THE EPOCH COUNT

The shipped ratchet adds a CONSTANT per violated epoch, so after T epochs

    lam_c = lambda_local + lambda_step * (epochs scope c was violated)

which counts **how often** a scope was violated, never **how much**. Its
achievable range is therefore bounded by the number of epochs:

| constants | max lam / min lam over 29 epochs |
|---|---|
| shipped (`lambda_local` 0.01, `lambda_step` 0.05) | **24.3x** |
| the MANUSCRIPT's (lam_0 0.05, ratchet step 0.002) | **2.1x** |

Measured on `dom1`: TraLO's lambda spans **13.3x** across the scopes of a run.
The raw violations it is responding to span **634x**.

| | `tralo` | `tralo_uniform` |
|---|---|---|
| Spearman(lam, cumulative violation) | +0.905 | +0.844 |
| lambda range across scopes | 13.3x | 13.3x |
| violation-magnitude range | **634x** | 254x |

🛑 **QUOTE THE RANGE, NOT THE RHO, AND SAY WHY.** A rank correlation is
invariant to any monotone rescaling, so +0.905 says TraLO orders the scopes
about right while saying nothing about how sharply it weights them. Under
`constraint_grad_mode: normalize` the summed gradient takes ONE norm over
`model.parameters()` (`constraint_step.py:263`) AFTER every scope has been
accumulated into `.grad`, so the overall size of the multiplier divides out and
**only the ratios across scopes steer**. TraLO is spreading its fixed-norm step
nearly evenly over scopes that ALM concentrates on. The rho hides exactly that.

### 2b. REPLICATED ACROSS SIX CAMPAIGNS AND FOUR BACKBONES (2026-09-06)

Both results above were measured on `dom1` first and then re-run on every live
campaign carrying `tralo`. 109 runs:

| campaign | backbone | runs | latched | rho | lambda range | violation range | ratio |
|---|---|---|---|---|---|---|---|
| `dom1` | MNv2/MNv3 | 24 | **0** | +0.905 | 13.3x | 634x | 48x |
| `dom1b` | MNv2/MNv3 | 12 | **0** | +0.900 | **24.3x** | 1934x | 80x |
| `equaldose1` | MNv2/MNv3 | 24 | **0** | +0.896 | 13.3x | 598x | 45x |
| `vitdual2` | **ViTB16** | 5 | **0** | +0.855 | **24.3x** | 716x | 29x |
| `taskwin2` | MNv3 | 5 | **0** | +0.930 | 13.3x | 282x | 21x |
| `uniform1` | MNv2/MNv3 | 24 | **0** | +0.899 | 13.3x | 1590x | 120x |

⛔ **THE LATCH FIRES IN 0 OF 109 RUNS**, on every backbone including ViTB16.
The closure in section 1 is not a `dom1` artefact.

🔑 **AND THE lambda RANGE IS QUANTISED, WHICH IS THE CLEANEST FORM OF THE
CLAIM.** It takes only two values across all six campaigns, and both are exact:

    24.3x = (0.01 + 29*0.05) / (0.01 + 1*0.05)     some scope violated in ALL
                                                    29 epochs, another in 1
    13.3x = (0.01 + 29*0.05) / (0.01 + 2*0.05)     ... another in 2

So the range is set ENTIRELY by the smallest violation COUNT, a small integer,
and **24.3x is the structural ceiling** -- `dom1b` and `vitdual2` are already
saturated at it. No violation profile, however extreme, can make the shipped
ratchet express more than 24.3x, while the violations themselves span 282x to
1934x. That is the defect in one line: **the multiplier's expressiveness is
bounded by the epoch count, and the problem's dynamic range is not.**

⚠️ `uniform1` and `taskwin2` are quarantined / partial for SCORING (their cells
sit outside the task window). That does not affect their use here: this is a
measurement of `tralo`'s OWN multiplier trajectory against its OWN violations,
not a contrast between arms, so no cell needs to pose a question for the
multiplier range to be readable. Do NOT promote any of these rows to a
head-to-head claim.

### 2c. A FALSIFIABLE PREDICTION FOR `tralo_dualprop`, REGISTERED BEFORE IT REPORTS

`latch_probe`'s "violation range" column is not just a contrast -- it IS the
lambda range the proportional ratchet will produce, because `proportional` adds
`lambda_step * excess` per violated epoch and therefore ends at
`lambda_0 + step * (cumulative excess)`, which is exactly the quantity that
column measures. So the table above already predicts the arm's behaviour from
logs that exist:

| | shipped `constant` | predicted `proportional` |
|---|---|---|
| lambda range across scopes | 13.3x - 24.3x, CEILING 24.3x | **282x - 1934x** |
| set by | the smallest violation COUNT | the cumulative violation MAGNITUDE |

**CHECK THIS ON THE FIRST COMPLETED `dualprop1` RUN.** If `tralo_dualprop`'s
lambda range comes back near 24.3x, the key is not reaching the ratchet and the
arm is a sixth inert flag -- md5 against `tralo` will NOT settle that (2(x2):
different predictions prove nothing), but the lambda range will, because it is
a direct readout of the mechanism.

⚠️ AND A LARGER lambda IS NOT A LARGER DOSE HERE. Under
`constraint_grad_mode: normalize` the summed gradient is rescaled to exactly
`constraint_grad_clip` by one norm over `model.parameters()`, so the absolute
size divides out and only the ratios steer. If the DOSE gate goes red on this
arm that is a finding about fp16 or about the clip, NOT about the mechanism.

### 3. WHY THIS WAS NOT KNOWN, AND IT IS 2(z44)'s PATTERN A FOURTH TIME

The manuscript defends the constant ratchet explicitly
(`main_edited_by_roei.tex:1490`): *"the count comes down no faster at
`lambda=53` than at TraLO's ratcheted `lambda <= 0.19` ... with the constraint
term dominant and the step norm-clipped, the update is independent of lambda."*

That argument is **correct for the global scale and silent on the per-scope
ratios**, which is the whole question once there is more than one live scope.
And its evidence, Fig. `fig_mechanism`, is **DermMNIST `L50/G50`,
RegNetY-400MF, seed 1** -- the dataset that ran `lp_fallback_used=False` with
**0 LP candidates on all 52 runs**, i.e. effectively a SINGLE-scope problem,
which is precisely the case where lambda genuinely does divide out. The claim
was verified where it cannot fail. iwildcam runs 11 live scopes per epoch.

### 4. `tralo_dualprop`, IMPLEMENTED AND GATED

`lambda_ratchet_mode: constant | proportional`, read at
`tralo/train.py`, defaulting to `constant` so every stored result is unchanged.
`proportional` integrates the RAW excess, matching ALM's residual
(`fioretto_alm/train.py:149`) rather than a K-normalised one, so the arm is that
difference and nothing else.

⚠️ **AND ONE CLAIM CORRECTED BY ITS OWN GATE.** "Proportional widens the
spread" is FALSE in general -- it REORDERS. On a fixture where frequency and
depth disagree (one scope 1 item over in 29 epochs, one 100 over in a single
epoch) the CONSTANT ratchet spreads them 29x and proportional only 3.45x. What
is always true is the degenerate case, and it is the defect under test: **at
equal violation frequency the constant ratchet is blind to depth entirely**,
however far apart the scopes are. That is what caps its range at 13.3x. The test
asserts the degenerate case and the reordering, not the false general claim.

Gate: `test_the_proportional_ratchet_is_LIVE_and_widens_the_multiplier_range`,
mutation-tested -- making `proportional` fall through to the constant increment
turns it red, and the restore was verified by EXECUTION, not by grep.

⛔ **PRE-REGISTERED**: the prediction is about the MIDDLE-depth scopes, where
2(z48) put the entire gap. If `dualprop` only moves the DEEP bucket it has
reproduced `tralo_squared`, which 2(z48) already demoted, and is not the
mechanism. It is NOT predicted to win outright -- `headroom` still bounds the
prize at 12.8-20.7 items per cell.


## 3. WHAT WE KNOW WORKS -- regime beats method, every time

### 3(0) 🛑 **STATUS BOARD, updated 2026-08-30 -- read this before section 3's older text**

Section 3 below was written against the warm-up-50 corpus and is HISTORY. This
board is the live state. It is updated every time a campaign lands; if it is
stale, that is a defect.

| claim | status | evidence | what would kill it |
|---|---|---|---|
| Constraint HELPS at loose caps | 🟢 holds | `loose1` AP +0.0253 5/1 vs a tying reseed | a loose-cap null that also moves |
| Constraint HARMS at tight caps | 🔴 holds | `iwc4` AP -0.0572 9/9; `vitu1` **-0.0933 0/3** | -- |
| TraLO > `clip` at loose caps | 🟢 holds | `dom1` 6/6 cells ccF1/AP/AUROC | dom1b reversing it |
| ~~TraLO > **hounie**~~ | ⛔ **UNANSWERABLE** | ~~`dom1` AP+AUROC 6/6, p=0.031, fails BH~~ -- **`hounie` is a DEAD ARM at 28.00 steps (2(z40))** | `vitdual2` |
| ~~TraLO > **fioretto**~~ | ⛔ **UNANSWERABLE** | ~~AP 3/6 cells, p=1.00 -- a coin flip~~ -- **`fioretto` is a DEAD ARM at 28.00 steps (2(z40))** | `vitdual2` |
| TraLO > **alm** (the ONLY surviving rival dual) | 🔴 **NOT shown** | 4/6 on every metric, p=0.69; as-deployed #1 in **0 of 15** cells, 2(z43) | -- |
| TraLO #1 of the duals, as deployed | 🔴 **REFUTED** | **2 of 15 cells namable, both `alm`, TraLO 0** once the dead arms drop (2(z43)) | -- |
| TraLO is a better ENFORCER | 🔴 **REFUTED** | pulls **+6.2 items** vs `alm` **+17.8** -- the weaker of the two surviving arms | -- |
| `tralo_uniform` fixes tight caps | 🟡 holds, tight only | `uniform1` AP -0.0754 -> +0.0030 | -- |
| macroF1 (the paper's headline) | 🔴 negative | `dom1` -0.0023, loses to `clip` | -- |
| Any result on ViTB16, the HEADLINE backbone | 🔴 **absent at loose caps** | `loosevit1` is 2 cells, p-floor 0.50 | running it properly |
| Second dataset | ⛔ none | `fmow` screened, needs ~21k images | -- |

### 3(0d) 🔑 **WHY EVERY dom1 CAP IS L80+ -- and the matched pair hiding in it**

Asked 2026-08-30: is the cap set so high because of dataset size, or because it
stops working when tightened? Neither. Measured against the count the lambda=0
model naturally emits:

| cap | class | n_true | K_global | K_local sum | **K binding** | natural count | must drop | K/n |
|---|---|---|---|---|---|---|---|---|
| L80_G95 | 2 | 370 | 352 | **296** | 296 | 358.1 | **62 (17.3%)** | 0.80 |
| L80_G95 | 7 | 456 | 433 | **364** | 364 | 468.9 | **105 (22.4%)** | 0.80 |
| L90_G95 | 2 | 370 | 352 | **333** | 333 | 358.1 | 25 (7.0%) | 0.90 |
| L90_G95 | 7 | 456 | 433 | **411** | 411 | 468.9 | 58 (12.3%) | 0.90 |
| L95_G80 | 2 | 370 | **296** | 352 | 296 | 358.1 | **62 (17.3%)** | 0.80 |
| L95_G80 | 7 | 456 | **365** | 433 | 365 | 468.9 | **104 (22.2%)** | 0.80 |

✅ **THE CAP IS NOT VACUOUS.** Even at "L95" the model must drop 17-22% of its
capped predictions. The tag is a percentage of the LOCAL budget, and the LOCAL
SUM is far below the global -- so `L80_G95` binds at K/n = **0.80**, not 0.95.
**Never read tightness off the tag; read `K binding` off the labels.**

🔑🔑 **AND `L80_G95` AND `L95_G80` ARE A MATCHED PAIR.** Both bind at K = 296 /
364-365, i.e. **the same total budget**, through **different scopes** -- local
in one, global in the other. That is the controlled contrast `scope_probe` was
built for, sitting inside `dom1` by construction.

⛔ **THE FINDING THAT USED TO BE HUNG ON IT IS GONE, 2026-09-04.** This read:
"So the finding that TraLO beats fioretto by **+0.0439 AP at L95_G80 and loses
by -0.0084 at L80_G95** is **not** a tightness effect: the budget is held fixed
and only the SCOPE moves." **`fioretto` is a dead arm, and it was the entire
contrast** -- the scope reading has no other arm behind it. The matched-pair
GEOMETRY is untouched and still worth having (same budget, different scope,
inside `dom1` by construction); what is missing is a live contrast to run
across it. `alm` could supply one and has not been read this way.

⚠️ **WHY NO TIGHT CAP IN dom1, stated honestly: it was selected out.** Tight
caps are measured to HARM (2(u): `iwc4` AP -0.0572 in 9/9; `vitu1` -0.0933), and
2(v) priced the tight-cap prize at **0.04-0.09x the paired noise**, i.e. below
detection at 4 seeds. So loose caps are both where the effect exists and where
it is measurable. That is defensible -- and it is still **selection on the
outcome**, because the regime was chosen after `loose1` reported it. The arms
and criterion in `launch_dom1.sh` were pre-registered; **the regime was not.**
Say so whenever the dom1 result is quoted.

### 3(0a) 🛑🛑 **dom1 IS NOT A HEADLINE RESULT -- IT IS A GENERALIZATION CHECK**

§1-pre fixed the headline backbone as **ViTB16**, a priori, on 2026-08-20,
precisely so that a win found on some other backbone could not be promoted after
the fact. **`dom1` contains no ViTB16 at all** -- it is MobileNetV2 +
MobileNetV3, i.e. one architecture family. `dom1b` adds RegNetY400MF.

So by this document's own binding rule, `dom1`'s positive result is a
**generalization check on a fixed headline, not the headline**. And on the
actual headline backbone the picture inverts:

| ViTB16, `vitu1`, tight caps, vs its OWN null | AP | macroF1 |
|---|---|---|
| **`tralo`** | **-0.0933, 0/3 cells** | -- |
| `clip` | -0.0045, 0/3 | -0.0089 |
| `focal_clip` (a POST-HOC baseline) | **+0.0219, 3/0** | **+0.0194** |

⇒ **On the headline backbone TraLO is 0.115 AP WORSE than a post-hoc focal
clipper.** The loose-cap ViTB16 evidence (`loosevit1`) is **2 cells**, whose
exact sign floor is p=0.50 -- nothing is callable there at any effect size.

🎯 **The single highest-value missing run is loose-cap ViTB16 at >= 6 cells.**
Until it exists, "TraLO wins" is a claim about MobileNet.

### 3(0b) 🔴🔴 **TraLO IS THE WEAKER ENFORCER -- OF THE TWO ARMS STILL COMPARABLE**

⛔ **THIS SECTION WAS HEADED "TraLO IS THE WEAKEST ENFORCER OF THE FOUR DUALS"
UNTIL 2026-09-04.** `hounie` and `fioretto` are `dom1` dead arms at 28.00
steps, so "of the four" cannot be said. **The finding SURVIVES on `alm` alone
and the direction is unchanged: `tralo` pulls +6.2 items against `alm`'s
+17.8, under a THIRD of the enforcement, while buying more quality
(dAP +0.0371 vs +0.0335, dccF1 +0.0141 vs +0.0136).** Only the superlative and
the denominator were wrong.

Measured on `dom1` from the PREDICTIONS (see 3(0c) -- the training logs cannot
be used for this). "pull" = items moved toward the cap vs the arm's own
lambda=0 twin, so 0 is "did nothing":

| arm | pull (items) | mean excess over K | dAP vs null | dccF1 vs null |
|---|---|---|---|---|
| ~~`hounie`~~ ⛔ **DEAD ARM** | ~~+23.4~~ | ~~+45.9~~ | ~~+0.0327~~ | ~~+0.0103~~ |
| ~~`fioretto`~~ ⛔ **DEAD ARM** | ~~+23.2~~ | ~~+46.2~~ | ~~+0.0268~~ | ~~+0.0101~~ |
| `alm` | +17.8 | +51.5 | +0.0335 | +0.0136 |
| **`tralo`** | **+6.2** | +63.1 | **+0.0371** | **+0.0141** |
| `tralo_uniform` | +4.9 | +64.4 | +0.0135 | +0.0095 |
| `tralo_reseed` (RNG floor) | -0.3 | +69.6 | +0.0104 | +0.0104 |

🔑 **TraLO buys the most quality for a THIRD of the enforcement.** Whatever
else it is, it is not a better constrained optimizer -- against the one rival
still comparable it is the one that does less. ⚠️ Do not fit a curve: the
correlation between pull and dAP was +0.61 at n=6 and meaningless then; with
two rows struck it is n=4 and not worth quoting at all. The FACT is the
`tralo`-vs-`alm` ordering. **The line read "a QUARTER of the enforcement"
until 2026-09-04, which was 6.2/23.4 against `hounie`, a dead arm; against
`alm` it is 6.2/17.8, a third.**

🛑 **AND NO ARM EVER SATISFIES THE CONSTRAINTS: `0 of 696` epochs, every trained
arm.** Mean excess stays +45.9 to +69.3 items above K. The post-hoc allocator
does 100% of the actual satisfying; the constraint phase only tilts the scores.

### 3(0c) ⛔ **THE TRAINING LOGS ARE NOT COMMENSURABLE ACROSS ARMS**

Found 2026-08-30, and it produced a wrong conclusion before it was caught. The
arms write **different log schemas**: `tralo`* 76 columns, `hounie` 16, `alm`
15, `fioretto` 14, post-hoc arms 34. `log_health` prints their count
trajectories in one table as if they were the same quantity. They are not.

⚠️ **`hounie` and `fioretto` are DEAD ARMS (2026-09-04, 2(z40)) and appear
below.** This section is UNAFFECTED and deliberately keeps them: it is a
statement about log SCHEMAS and about logs disagreeing with predictions, which
is a property of the writer, not of how many constraint steps an arm took. The
rule it sets -- **never read a count from `training_log.csv`, read it from
`final_predictions_raw.csv`** -- applies to every arm including the dead ones.
Do not quote any arm-vs-arm PERFORMANCE number out of this table.

Worse, for every **trained** arm the last logged hard count **disagrees with the
model's actual predictions**:

| arm | log's last `Hard_Class2` | actual raw prediction count | agree |
|---|---|---|---|
| `tralo_null` | 393 | 393 | **24/24** |
| `alm_null` | 393 | 393 | **24/24** |
| `tralo` | 428 | 419 | 0/24 |
| `alm` | **340** | **467** | 0/24 |
| `fioretto` | 343 | 402 | 0/24 |
| `hounie` | 332 | 413 | 0/24 |

The logs make `alm`/`fioretto`/`hounie` look like they drive the count to the
cap (340/343/332 against K=352) when their deployed counts are 402-467. Reading
that table alone gives **the exact opposite** of 3(0b).

⇒ **Never compare a count across arms from `training_log.csv`. Measure it from
`final_predictions_raw.csv`.** The nulls agreeing 24/24 is the control that
proves the disagreement is real and specific to arms that take constraint steps.


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
tests/             584 tests, ~200 s, no dataset required
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
