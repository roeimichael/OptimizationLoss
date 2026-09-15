# TraLO reset: execution state

Updated 2026-09-15 (Asia/Jerusalem). User approved recoverable large cleanup, fresh evidence,
validated logging/data/code, then a within-TraLO modification and monitored SSH
experiments. No historical acceptance tally is carried forward.

**Latest user direction (2026-09-14): take charge and drive it.** Manage the
classes and the data, get them to fit the requirements, then run on the GPU
servers; if results look decent, improve them by (1) hyperparameters, (2) extra
seeds for replication, (3) other modalities -- and validate that what comes back
is true. The acceptance bar, chosen by the user: **leading group on cc-F1 and
not dominated elsewhere**, judged on the full metric profile rather than a single
hard-capped condition.

**The dataset is `fmow2`.** iwildcam is RETIRED (2 of 8 candidate conditions;
see the retirement note in `configs/protocol.yml`) and the original `fmow`
oodslice is withdrawn (basename join collapsed distinct AOIs). Rebuilt fmow2
passes all 8 conditions of `scripts/candidate_gate.py`.

## THE COURSE -- read this first, every session

*This section is the contract. Everything below it is evidence. If a new result
changes the course, edit THIS section -- do not leave it stale and correct it
further down.*

### The one question

**Can a training-time constraint beat a post-hoc clipper on a capped-class
task?** The acceptance bar, set by the user: **leading group on cc-F1 AND not
dominated on the rest of the metric profile.** A win on accuracy or macroF1
while losing cc-F1 is a TRADE, not a win, and must be reported as one.

### What is settled -- do not re-open without new evidence

| # | Settled | Consequence |
|---|---|---|
| 1 | **Every cell saturates in 2-4 epochs.** 5/5 cells, 3 backbones, 2 datasets. Augmentation doubles it to 5, and no more. | The boundary is frozen for most of any long budget. `gate:saturation` is a HARD DROP. |
| 2 | **The four duals share ONE per-item gradient.** tralo/fioretto/alm/hounie all reduce to `sum_scopes w_scope * sum_i dp_i(c)/dz`, differing only in the scalar. Local scopes are disjoint. | Their only freedom is the per-group eviction COUNT, which the caps fix. **No better dual rule exists to find.** |
| 3 | **Eviction given the probabilities is already ~optimal.** A gradient push lands on 87% of the provably optimal set; closing the rest is worth 0.0002. | The prize is NOT in the loss shape. A margin-aware soft count is CLOSED. |
| 4 | **The exact allocator makes real results WORSE** (-0.01 to -0.04 acc, -0.02 to -0.08 cc-F1, 14/14 cells) because models run at 0.92 confidence against 0.60 accuracy. | The allocator "fix" is CANCELLED. Greedy's suboptimality is protective. |
| 5 | **The constraint DOES work -- it just does not convert.** Every dual beats both clippers on native obedience (excess 279-326 vs 381-413). | Obedience is not the missing piece. The metric does not reward it. |
| 6 | **On a frozen boundary the constraint makes the RANKING WORSE. 15 of 16 seed-deltas negative** (binomial p = 0.0005), gAP(tralo) - gAP(tralo_null) = -0.007 to -0.030 per cell. gAP is allocation-free, so this is the model, not the allocator. | The damage is UPSTREAM of the metric and about 2x the cc-F1 damage. The constraint is not reshaping the boundary badly -- it is **injecting noise into it**. |
| 7 | **4 seeds cannot detect this project's effect size.** Effects run ~0.01 with a seed sd of ~0.011, so n=4 carries roughly 15% power; detecting d = sd at 80% needs n ~ 13. | Every "not significant" verdict on n=4 is consistent with a real effect. `--seeds` is per-campaign, so **12 seeds for anything meant to SETTLE something.** Re-check the rejected ledger for directions closed at n=4. |
| 8 | **Training is BIT-DETERMINISTIC given (config, seed).** gx2 re-ran 30 of fm2_mn3's cells and matched every byte across trees and code versions. | Reproducibility is exact and publishable. But an overlapping arm adds ZERO information: **de-duplicate by prediction hash before counting n.** |

**Why it can only lose in this regime, stated as a mechanism.** A count says
HOW MANY, never WHICH. With CE alive the two signals compose: CE supplies the
"which", the constraint supplies the "how many". Once train CE hits ~0 the
"which" is gone, and `constraint_step.py:36` rescales the constraint gradient to
full size no matter how small the violation is. Full-size push, nothing
opposing it, no information about correctness -- so the ranking can only
degrade. **That is exactly what gAP measures, and exactly what it shows.**

⚠️ **This means the two complete campaigns do NOT measure TraLO.** They measure
what any count-based constraint does to a memorised model. The method has not
been given a boundary it could reshape.

**Therefore the ONLY channel left is the RANKING** -- the model's own margins,
measured by gAP. And the constraint can only change margins while the boundary
is still moving. **That is the whole remaining question.**

### The live experiment: `sweep1` / `sweep1_mn2` (launched 2026-09-15)

**Two campaigns, 384 runs each, one per GPU, ~3.6 GPU-hours each.** MobileNetV3
and MobileNetV2, fmow2, {L80_G95, L90_G95}, **12 seeds**, frozen at commit
`21fd13fb`, all 10 pre-flight gates GREEN.

This is a **within-campaign BUDGET SWEEP**, and it exists because the previous
evidence could not carry the claim. The account says the constraint only helps
while CE is still shaping the boundary, which predicts the effect scales with
the **live fraction** = live window / constraint epochs. That was only ever
varied BETWEEN campaigns, where it is confounded with backbone, code version and
cap set at once:

| campaign | backbone | live frac | d gAP (de-duplicated) |
|---|---|---|---|
| `fm2_mn3` | MobileNetV3 | 10.3% | -0.0174 |
| `fm2_mn2` | MobileNetV2 | 13.8% | -0.0274 |
| `live11` | MobileNetV3 | 30.0% | -0.0089 |
| `live6b` | MobileNetV3 | 60.0% | **+0.0079** |

On a fixed backbone that is monotone, 3 of 3. Spearman over all four is +0.80,
**p = 0.20**. It cannot reject anything, and the one campaign that breaks
monotonicity is also the only MobileNetV2.

The sweep makes the live fraction an **independent variable inside one
campaign**: budgets 5 / 8 / 12 / 20 / 30, live fraction 75% / 43% / 27% / 16% /
10%, with the backbone, the code version and the seeds held fixed. Every trained
arm keeps warm-up 1, so **all five budgets resume the SAME cached warm-up** --
the sweep varies the length of the constraint phase and nothing else. Each
budget carries its own null AND its own clipper, because the constraint phase
runs full CE every epoch (a 5-epoch null is a differently trained model) and
because if short budgets hurt every arm, the constraint's standing is relative.

**Pre-registered outcomes, fixed BEFORE the seeds land:**

| Result | Reading | What we do |
|---|---|---|
| d gAP rises monotonically with live fraction and crosses 0 | The account holds, now with power and without the confound. | Take the best budget to the cc-F1 bar against `aug_clip`, then the second dataset. |
| d gAP flat and negative across all five budgets | The ranking channel is SHUT and `live6b` was a 4-seed artifact. | Write the negative result. It is publishable. |
| d gAP rises but never crosses 0 | The constraint costs less on a live boundary but never pays. | Report as a bounded negative: the mechanism is real, the method still loses. |

⚠️ **The cc-F1 bar is a separate question from the gAP channel and must be
reported separately.** In `live11`, where the per-column nulls finally exist,
`aug_tralo_null` (0.6411) BEATS `aug_tralo` (0.6285): inside the augmented
column the constraint costs 0.013 cc-F1 even where it is neutral on gAP.

⛔ **`gx2`'s apparent TraLO win does not survive its own control.** gx2 ranks
`aug_tralo` (0.6565) first, 0.0012 above `aug_clip` -- far inside seed noise,
and gx2 has NO `aug_tralo_null`, so nothing in it can attribute that to the
constraint. This is the failure mode already in the ledger as "all four TraLO #1
calls were manufactured by a dead arm".

### Standing decision rules

- **`gate:saturation` first.** An arm that memorises in 3 epochs has not run the
  experiment; its numbers are not read at all.
- **Average over SEED only.** Never pool across cap levels, backbones or
  datasets. A cap level is not a seed.
- 🔑 **The epoch budget is DERIVED, not assumed.** `total_epochs = 30` was never
  measured -- the user confirmed 2026-09-14 that he picked it arbitrarily, and
  that **every hyperparameter in the config is movable.** The gate's criterion
  inverts into the rule: `total_epochs = 2 x (measured live window) + 1`.
  Measure the live window first, then set the budget. Do not defend 30.
- **A gate is not done until a mutation shows it FAIL**, and the restore is
  verified by EXECUTING -- stale bytecode has faked a pass before.
- **Never touch `src/`, `configs/`, `scripts/`, `main.py` on the server while a
  campaign is live** -- it splits `code_version`. Use `~/optloss-probe`.
- **Claims about what code reads come from AST or reading, never grep.**
- **Report a trade as a trade.** Retract in the same document, in place.

### Where we actually stand, 2026-09-15

🟢 **`live6b` is the first campaign to PASS `gate:saturation`, and in it the
constraint's ranking damage is ABOLISHED** (30-epoch 15/16 negative -> 6-epoch
6/8 positive, Fisher p = 0.0013). A benefit is NOT established -- 6/8 is p =
0.29.

🟡 **`live11` ran the pre-registered 2x2 with the per-column nulls in place, and
it came out in the predicted direction with no power to confirm it.** Constraint
effect on gAP, seed-paired, caps averaged within seed, n=4:

| column | d gAP | p |
|---|---|---|
| plain | -0.00784 +- 0.00938 | 0.19 |
| augment | **+0.00109** +- 0.00505 | 0.70 |
| focal | -0.00940 +- 0.02402 | 0.49 |
| **augment x constraint** | **+0.00892** +- 0.01142, 3/4 positive | **0.22** |
| focal x constraint | -0.00157 +- 0.01599, 1/4 positive | 0.86 |

Augment removes the damage, focal does not -- which is exactly the 2x2
amendment's discriminating prediction, and exactly what "augment raises the live
fraction, focal only enlarges the gradient" implies. But the effect and the seed
sd are the same size, so this is **settled fact 7** in action, not a result. It
is the direct reason the sweep runs at 12 seeds.

**Two 30-epoch cells complete, 112 runs, and both FAIL `gate:saturation`** -- so
they are the **frozen-boundary REFERENCE**, not a verdict on the method.

TraLO minus `tralo_null` on cc-F1, the like-for-like comparison (same warm-up,
same compute, the only difference is whether the constraint steps):

| cell | cap | d cc-F1 vs its own null |
|---|---|---|
| MobileNetV3 | L80_G95 | -0.0036 +- 0.0103 |
| MobileNetV3 | L90_G95 | **-0.0193 +- 0.0058** |
| MobileNetV2 | L80_G95 | -0.0092 +- 0.0153 |
| MobileNetV2 | L90_G95 | -0.0125 +- 0.0093 |

🛑 **4 of 4 negative**, two of them beyond their own seed sd. On a frozen
boundary the constraint phase COSTS cc-F1 relative to not stepping at all.

And the same comparison on **gAP**, which is computed from the probabilities
alone and is allocation-free -- so it is the MODEL, with the allocator removed
from the question entirely:

| cell | cap | d gAP vs its own null | per-seed |
|---|---|---|---|
| MobileNetV3 | L80_G95 | -0.0069 +- 0.0222 | 3 of 4 negative |
| MobileNetV3 | L90_G95 | **-0.0280 +- 0.0113** | 4 of 4 |
| MobileNetV2 | L80_G95 | **-0.0245 +- 0.0114** | 4 of 4 |
| MobileNetV2 | L90_G95 | -0.0303 +- 0.0185 | 4 of 4 |

**15 of 16 seed-deltas negative** (binomial p = 0.0005), and the ranking damage
is about **twice** the cc-F1 damage -- so it is the upstream cause, not a
side effect of allocation.

The per-cell verdicts are NOT consistent and must not be summarised as one:
mn3 reads LEADING GROUP at L80 and LOSS at L90; mn2 reads LOSS at L80 (trailing
`clip` by -0.0162 +- 0.0055) and LEADING GROUP at L90. **The accuracy-for-cc-F1
trade does not replicate either** -- real on mn3 (+0.013 acc), absent on mn2 at
L80, where macroF1 and accuracy are negative too. The only thing true in all
four cells is that TraLO never WINS cc-F1.

`fm2_vit` fails the gate hardest (1.6 live epochs) and is a failed experiment;
it completed 56/56 anyway and is archived as reference, not read as a verdict.

**`sweep1` / `sweep1_mn2` are the next thing that can change the answer**, and
they are the first campaigns built to have the power to do it.

## Current stage

### 🔬 PRE-REGISTERED 2026-09-14 (2) -- THE 2x2 GRADIENT-HEALTH TEST

**THE ACCOUNT THIS TESTS.** The chain that explains every null in the ledger:

1. The metric is scored AFTER post-hoc allocation. The allocator takes top-K per
   (group, class), is optimal given the probabilities, and the capped classes'
   top-K sets are pairwise disjoint (`lp` and `clip` score identically). So
   allocation adds nothing an arm can beat.
2. Therefore the only payoff channel is the **ORDER** of the probabilities.
3. The constraint touches the order only through `d(soft count)/d(theta) =
   sum_i d p_i(c)/d theta`, whose per-item weight is `p_i(1 - p_i)`.
4. On a memorised model that weight is **0.007-0.023 against a 0.25 maximum**,
   and **34% of it sits in the top 1% of items** (class 1, fm2_mn3). A few dozen
   borderline items choose the entire direction -- which `normalize` then
   rescales to full size.
5. So the constraint moves COUNTS a lot (which do not matter) and ORDER not at
   all (which does).

**THE TEST.** A 2x2: focal on/off crossed with the constraint on/off, at equal
compute. Focal is the intervention measured to raise `p(1-p)` to 0.015-0.045 and
cut the top-1% share to 5-16%.

| | post-hoc | trained |
|---|---|---|
| CE | `clip` | `tralo` |
| focal | `focal_clip` | **`focal_tralo`** |

**PREDICTION (the INTERACTION, not the main effect):**

    gAP(focal_tralo) - gAP(focal_clip)  >  gAP(tralo) - gAP(clip)

i.e. the constraint buys MORE when its gradient is alive. On fm2_mn3 the
right-hand side is currently **-0.0136** (5 of 6 cells negative).

⛔ **FALSIFIED IF** the two differences are equal within seed noise. That would
mean focal helps the MODEL and not the CONSTRAINT, the gradient-health account
is wrong, and it goes in the closed ledger rather than being rescued. A main
effect of focal alone does NOT confirm it -- `focal_clip` already has one.

#### AMENDMENT, same day, BEFORE any of it ran

The pre-registration above stands as written -- it is not being retro-edited --
but two of the premises it rests on were refuted later the same day, and the
amendment must be on the record before the seeds land.

**Premise 1 is false.** Allocation is NOT optimal given the probabilities: the
allocator is top-K by `p(c)` and the optimum is top-K by MARGIN, worth ~+0.01
accuracy. And the supporting evidence ("`lp` and `clip` score identically") was
already retracted -- it was a gAP comparison, and gAP is allocation-free.

**Premises 4-5 are mostly false.** `p(1-p)` concentration was said to let a few
dozen items choose the direction. Measured: a bisected gradient push lands on
**87% of the provably optimal eviction set**, and closing the last 13% is worth
**0.0002 accuracy**. Gradient-mass concentration barely changes WHICH items
flip. It changes how large a step is needed -- and `normalize` removes that.

**The corrected chain, which makes the SAME test sharper:**

1. Eviction given the probabilities is already ~optimal (0.0002 residual), and
   all four duals share one per-item direction whose only freedom is the
   per-group eviction count, which the caps fix.
2. So the ONLY payoff channel is a change in the PROBABILITIES -- the model's
   own margins, i.e. gAP.
3. The constraint can change those only while the boundary is still moving. CE
   reaches ~0 by epoch 3-5 of 30, so 24+ of 29 constraint epochs act on a frozen
   boundary.

🔬 **THE DISCRIMINATING PREDICTION.** The corrected chain says the intervention
has to UNFREEZE THE BOUNDARY, not enlarge the gradient. Focal does the second;
augmentation does the first. So the two interventions are now predicted to come
apart, and that is a much stronger test than running both and keeping the
winner:

    focal:        gAP(focal_tralo) - gAP(focal_clip)  ~=  gAP(tralo) - gAP(clip)
    augmentation: gAP(aug_tralo)   - gAP(aug_clip)     >  gAP(tralo) - gAP(clip)

**Focal is now the NEGATIVE CONTROL for augmentation.** If focal's interaction
comes out positive too, the "unfreeze the boundary" account is wrong and what is
really being measured is any regulariser at all. If NEITHER moves, the ranking
channel is shut and no training-time constraint can beat a clipper on this
corpus -- which is a publishable result, and the one the ledger currently
points at.

⚠️ Both arms must pass `gate:saturation` before their numbers are read at all.
An arm that still memorises in 3 epochs has not run the experiment.

⚠️ Pre-registered BEFORE `focal_tralo` has ever run. It is a schema line plus a
protocol arm; `make_ce_criterion` already honoured `warmup_loss` and is what
`tralo/train.py:80` and `dual_common.py:94` build their task criterion with.

⚠️ Run in a SEPARATE checkout. `src/` and `configs/` are inside
`source_inventory()`, and `fm2_vit` / `fm2_mn2` are still live on the frozen
release -- `validate_campaign` hashes the source root of the process that runs
it, so an isolated tree cannot disturb them.

### ⚠️ CORRECTION: THE ALLOCATOR IS A GREEDY HEURISTIC, NOT A PROVEN OPTIMUM

I have been resting the whole diagnosis on "post-hoc allocation is optimal given
the probabilities, so only the ORDER survives". Reading
`heuristic/train.py:apply_allocation_heuristic` rather than trusting the note:

- **Pass 1** sorts every (item, capped class) pair by probability and assigns
  greedily. Its own docstring says this equals top-K **only when there is a
  single capped class**. fmow2 has three and bcn two, so the classes compete.
- **Pass 2** gives uncapped classes only the items whose ARGMAX is that class.
- **Pass 3** sends the leftovers to their best still-feasible class.

Greedy on a joint assignment problem is **not** optimal in general -- which is
why an `lp` arm exists at all.

⛔ **And my evidence that greedy equals the LP was worthless.** I wrote that
`lp` and `clip` "score identically". They scored identically **on gAP**, which
is computed from the probabilities alone and is allocation-free by design --
two arms sharing a trained model must agree on it no matter what allocator
runs afterwards. That observation cannot distinguish allocators and I should
not have used it.

🔑 **This matters because it could overturn the diagnosis.** If the
greedy pass is leaving value on the table, then a model that obeys natively
hands the allocator less work, and TraLO's obedience advantage (bcn L70 excess
550 vs `clip`'s 852) would have somewhere to convert.

**The decisive test is free and queued**: `lp` and `clip` share the same trained
model, so any difference between them on cc-F1 is PURELY the allocator.

### 🔑 ANSWERED OFFLINE, AND THE ANSWER IS BIGGER THAN THE QUESTION

`scripts/alloc_gap.py` imports the shipped allocator and solves the identical
instance exactly (transportation LP -- items x classes with per-(group, class)
capacities, so the relaxation is integral and HiGHS returns the true optimum).
400 items / 8 classes / 4 groups / 5 seeds. `sharp` sets separation; **sharp=2.0
lands at 0.70 accuracy, which is our regime.**

| capped | sharp | obj gap | moved | **accuracy gap (LP - greedy)** |
|---|---|---|---|---|
| 1 | 4.0 | 0.007% | 0.5% | +0.0000 |
| 1 | 2.0 | 0.096% | 0.9% | +0.0030 |
| 1 | 1.0 | 0.299% | 2.5% | +0.0050 |
| 3 | 4.0 | 0.050% | 1.5% | +0.0000 |
| **3** | **2.0** | **0.588%** | **4.7%** | **+0.0095** |
| 3 | 1.0 | 1.204% | 8.7% | +0.0005 |
| 5 | 2.0 | 1.385% | 9.9% | **+0.0150** |
| 5 | 1.0 | 2.115% | 13.9% | +0.0040 |

**1. The allocator's own docstring hides a false premise.** It is right that a
single capped class makes pass 1 that class's top-K. But **top-K by p(c) is not
the maximiser.** An item not given `c` falls back to its best OTHER class, so a
slot is worth the MARGIN `p(c) - best_alt`, not `p(c)`. `margin_topk` in the
probe takes the top-K by that margin and **reproduces the LP to floating point
on every seed**, while the shipped rule is strictly worse on all five. So the
defect is present even in the case the code documents as exact, and it is not a
subtlety of the multi-class joint pass.

**2. The prize is not negligible at the separation our models actually have.**
+0.0095 (3 capped) to +0.0150 (5 capped) accuracy at sharp=2.0 -- **larger than
most arm-vs-arm gaps this project has ever measured.** It vanishes at sharp=4.0
because nothing is contested, and it DECOUPLES from the objective at sharp<=1.0,
where greedy loses more probability mass while the accuracy difference falls
back into the noise. The peak sits exactly where our models sit.

**What this does and does not change.** Every arm shares the allocator, so it
does not explain TraLO losing.

⛔ **And I immediately over-read it.** I wrote here that greedy reads `p(c)`
while the optimum reads the margin, so a TRAINED arm with margin-shaped
probabilities converts where a clipper cannot. **That is wrong, and the
refutation is sitting in the same file**: `margin_topk` needs nothing but the
probabilities and the caps. It IS post-hoc. Any clipper can run it. There is no
trained-arm channel here at all -- the only effect of fixing the allocator is
that **the clipper baseline gets ~+0.01 accuracy for free and the bar TraLO has
to clear goes UP.** That is the honest reading and it is the one to act on.

**Do NOT fix `apply_allocation_heuristic` now.** It is inside
`source_inventory()` and three campaigns are live; changing it splits
`code_version` exactly as the scorer deploy did on 2026-08-24. Queued as the top
post-campaign change -- and note it makes the CLIPPER BASELINE STRONGER, raising
the bar, which is the honest thing to do.

⚠️ **Synthetic softmaxes, well calibrated; real models are overconfident.** The
stored probability vectors settle it for free and are queued behind SSH.
Gated by `test_the_ALLOCATOR_is_NOT_optimal_even_with_a_SINGLE_capped_class`,
whose mutation control (rank `margin_topk` by `p(c)`) makes it FAIL -- and the
mutant reproduces the shipped allocator's objective to the digit, which is
independent confirmation that pass 1 is exactly top-K by `p(c)`.

### ⛔ REAL DATA REVERSES THE ALLOCATOR FINDING. THE FIX IS CANCELLED.

`scripts/alloc_real.py` on **fm2_mn3, all 7 arms x 2 caps x 4 seeds** (56 runs,
same stored probabilities, two allocators, so the only difference IS the
allocator):

| cap | arm | d objective | **d accuracy** | **d cc-F1** | moved |
|---|---|---|---|---|---|
| L90_G95 | fioretto | +5.369% | **-0.0403** | **-0.0770** | 7.4% |
| L90_G95 | alm | +4.417% | -0.0341 | -0.0632 | 6.3% |
| L90_G95 | tralo | +4.563% | -0.0279 | -0.0668 | 6.8% |
| L90_G95 | clip | +2.904% | -0.0214 | -0.0442 | 5.4% |
| L90_G95 | tralo_null | +2.164% | -0.0173 | -0.0371 | 4.2% |
| L80_G95 | focal_clip | +0.938% | -0.0108 | -0.0315 | 3.2% |

**The LP wins its own objective in all 14 cells and LOSES accuracy and cc-F1 in
all 14.** The synthetic said +0.0095 accuracy; real data says **-0.011 to
-0.040**, and -0.022 to -0.077 on cc-F1. Sign reversed, magnitude larger.

**Why.** The synthetic used calibrated softmaxes, where total assigned
probability is a good proxy for accuracy. Real models are overconfident, so
maximising `sum p` chases confident-and-wrong items. Greedy's "suboptimality" is
protective, and the 3-7% of items the LP moves are net-negative every time.

🛑 **CANCELLED: the queued change to `apply_allocation_heuristic`.** Making it
optimal would cost every arm 0.01-0.04 accuracy and 0.02-0.08 cc-F1. It is not a
defect to fix. The entry above stands as the record of a claim that did not
survive its own confirmation.

✅ **The gate earned its keep.** The first version asserted the LP must also win
on accuracy; it failed, and that is exactly the falsehood this table shows. The
probe reports the objective as the guaranteed quantity for that reason.

**Overconfidence, measured rather than assumed.** ECE (15 bins) on the same
runs: **mean confidence 0.92 against accuracy 0.60, ECE ~0.31** for every arm.
A 32-point confidence overstatement is more than enough for `sum p` to stop
proxying accuracy.

⛔ **And I got the second half of this wrong within the hour.** I wrote that
allocator disagreement measures calibration damage and that "the more a
constraint pushed, the worse the calibration". **ECE refutes it**: it is FLAT at
0.306-0.324 across alm, clip, fioretto, hounie, tralo and tralo_null, with only
`focal_clip` apart (0.228, confidence 0.831). There is no per-arm calibration
ordering to measure.

🔑 **What disagreement actually tracks is NATIVE OBEDIENCE, inversely.** Against
`obey_all` excess on the same 14 cells, L90_G95:

| arm | native excess | items moved |
|---|---|---|
| fioretto | 279 | 7.4% |
| hounie | 298 | 5.8% |
| tralo | 309 | 6.8% |
| alm | 326 | 6.3% |
| focal_clip | 381 | 4.6% |
| tralo_null | 409 | 4.2% |
| clip | 413 | 5.4% |

**Spearman rho = -0.79.** The better an arm already obeys, the MORE the two
allocators disagree about it. Mechanism not confirmed -- the plausible one is
that an obedient model leaves greedy filling the cap from lower `p(c)`, where
margins are tiny and there is more for the LP to rearrange. Stated as a
correlation, not a cause.

🔑 **The duals DO obey natively, and by a clear margin.** All four beat both
clippers at L90_G95 (excess 279-326 against 381-413), and `tralo` is the best
trained arm on compliant scopes (13.0/30 against `tralo_null`'s 10.0). The
constraint phase is working. **It just does not convert** -- which is the
finding the whole ledger keeps arriving at, now with the obedience side
measured on one scale for all seven arms.

### 📊 fm2_mn3 COMPLETE: THE 4-SEED VERDICT AGAINST ALL SIX RIVALS

56/56, zero failures. Seed-paired, sd = seed-paired sd of the difference.

**L90_G95 -- TraLO minus each arm:**

| arm | d cc-F1 | d macroF1 | d accuracy |
|---|---|---|---|
| tralo_null | **-0.0193 +- 0.0058** | +0.0032 +- 0.0137 | +0.0008 +- 0.0112 |
| alm | -0.0150 +- 0.0144 | +0.0059 +- 0.0126 | +0.0069 +- 0.0114 |
| fioretto | **-0.0142 +- 0.0048** | +0.0072 +- 0.0104 | +0.0086 +- 0.0140 |
| focal_clip | **-0.0134 +- 0.0052** | +0.0144 +- 0.0078 | +0.0129 +- 0.0077 |
| clip | -0.0112 +- 0.0147 | +0.0155 +- 0.0094 | +0.0119 +- 0.0128 |
| hounie | -0.0084 +- 0.0089 | +0.0045 +- 0.0096 | +0.0043 +- 0.0088 |

**VERDICT: LOSS.** TraLO trails **every one of the six** on cc-F1, and three of
those gaps clear their own seed sd by 2.6-3.3x -- including its own null. At
L80_G95 the same grid reads LEADING GROUP (-0.0036 +- 0.0103 vs `tralo_null`).

**The trade is real and it is consistent**: TraLO buys +0.013 accuracy and
+0.015 macroF1 against the clippers while paying -0.013 cc-F1. That is a
different operating point, not a win on the headline.

⚠️ **And this cell FAILS `gate:saturation` at 2.7 live epochs.** So the whole
table is the FROZEN-BOUNDARY REFERENCE, which is the role it should play: it is
the control arm of the augmentation experiment, not a verdict on the method.

### 🟢 FIRST GATE-PASSING CAMPAIGN: THE RANKING DAMAGE STOPS

`live6b` (total 6 = 1 warm-up + 5 constraint) is the **first campaign in this
project to PASS `gate:saturation`** -- required >= 2 of 5, measured **3.4**,
exit 0.

gAP (allocation-free) for `tralo` minus its own null, against the 30-epoch
reference where the same quantity was -0.007 to -0.030 with 15 of 16 seeds
negative:

| cap | n | mean | sd | per-seed |
|---|---|---|---|---|
| L80_G95 | 3 | -0.0016 | 0.0139 | +0.0023 +0.0100 -0.0171 |
| L90_G95 | 3 | **+0.0104** | 0.0117 | +0.0169 +0.0174 -0.0030 |

**Complete at n = 4, both caps:**

| cap | mean | sd | per-seed |
|---|---|---|---|
| L80_G95 | +0.0037 | 0.0155 | +0.0023 +0.0100 -0.0171 +0.0195 |
| L90_G95 | **+0.0121** | 0.0101 | +0.0169 +0.0174 -0.0030 +0.0170 |

**The sign flips, and the flip itself is significant.** Seed-level signs of
`gAP(tralo) - gAP(tralo_null)`:

| regime | signs | vs a coin |
|---|---|---|
| 30 epochs (fm2_mn3 + fm2_mn2) | **15 of 16 NEGATIVE** | p = 0.0005 |
| 6 epochs (live6b) | **6 of 8 POSITIVE** | p = 0.29 |

**Fisher exact between the two regimes: p = 0.0013.**

🔑 **Read that precisely, because the two halves say different things.**
- The 30-epoch damage was real: 15/16 negative does not happen by chance.
- **It is abolished at 6 epochs.** The regime change is significant at p =
  0.0013.
- **A BENEFIT IS NOT ESTABLISHED.** 6 of 8 positive is p = 0.29 -- a coin. The
  honest statement is that the constraint has gone from *reliably harmful* to
  *neutral or better*, and nothing stronger. It is not yet a win over the null,
  let alone over a clipper.

⚠️ One backbone, one dataset, 8 seed-deltas. Replication on a second backbone is
required before this is a result rather than a lead.

#### AND THE METRIC AGREES -- BUT THE BEST ARM IS STILL A CLIPPER

`profile_report` on `live6b`, L90_G95, n = 4:

| arm | accuracy | cc-F1 |
|---|---|---|
| **`aug_clip`** | **0.6632** | **0.7378** |
| `aug_tralo` | 0.6623 | 0.7338 |
| `clip` | 0.6469 | 0.7154 |
| `tralo` | 0.6484 | 0.7131 |
| `tralo_null` | 0.6467 | 0.7071 |

- 🟢 **`tralo` - `tralo_null` on cc-F1 = +0.0060 +- 0.0091.** POSITIVE, where the
  30-epoch cells gave **-0.0193 +- 0.0058**. The metric moves with gAP, which is
  the consistency the account needs. Still inside 1 sd.
- 🛑 **The best arm in the campaign is `aug_clip`, a CLIPPER.** TraLO trails it
  by -0.0247 +- 0.0160.

🔑 **Same shape as the allocator result: an improvement available POST-HOC
strengthens the baseline at least as much as the method.** Augmentation is a
rising tide, and it lifts the clipper. Any claim for TraLO has to be made
against `aug_clip`, not against `clip`.

⚠️ **`aug_tralo` vs `aug_clip` (-0.0040) is NOT the constraint's effect** -- they
differ in schedule as well. The only clean test is `aug_tralo` vs
`aug_tralo_null`, which `live6b` does not contain. **`live11` does** (72 runs,
all three columns controlled, RUNNING).

**Augmentation raises gAP on its own**: `aug_tralo - tralo` = +0.0221 +- 0.0112
(L80, 4 of 4 positive) and +0.0121 +- 0.0077 (L90, 3 of 3). Seven of seven
positive -- but that is a main effect on the MODEL, not evidence about the
constraint, and the pre-registration says exactly that a main effect confirms
nothing.

### ⛔ DESIGN GAP FOUND BY THAT SCORING: THE INTERVENTION COLUMNS HAD NO NULL

`gen_campaign` force-adds `tralo_null` whenever a trained arm is present, which
reads as "the campaign has a control". It is the PLAIN null. Inside the
augmented column the only comparisons that existed were:

- `aug_tralo` vs `aug_clip` -- different SCHEDULE (1+5 against 6+0)
- `aug_tralo` vs `tralo` -- different AUGMENTATION

**Neither isolates the constraint**, which is the entire quantity under test.
Added `aug_tralo_null` and `focal_tralo_null`, each sharing its column's warm-up
identity and schedule and differing only in whether the constraint steps.

Gated by `test_every_INTERVENTION_column_has_its_own_zero_constraint_control`.
The invariant is about WARM-UP IDENTITY, not arm names: with lambda at zero
every dual reduces to plain training, so `tralo_null` legitimately controls
`alm`/`fioretto`/`hounie`. What must never exist is a trained arm whose warm-up
no zero-constraint arm reproduces. Mutation removing `aug_tralo_null` FAILS it.

⚠️ `live6b`'s augmented column therefore cannot answer the pre-registered
question. `live11` will be regenerated with the new controls.

### ⚠️ `live11` MISSES THE GATE BY 0.3 EPOCHS -- AND THAT IS ITSELF THE RESULT

Live window per column, `live11` (constraint_epochs = 10, gate needs >= 5):

| column | live | verdict |
|---|---|---|
| `aug_tralo` / `aug_tralo_null` | **4.7** of 10 | fails by 0.3 |
| `focal_tralo` / `focal_tralo_null` | 3.0 of 10 | fails |
| `tralo` / `tralo_null` | 3.0 of 10 | fails |

I set total = 11 from the rule `2 x live + 1` using **5** live epochs measured on
a single `gx2` run. The real figure across 6 runs is **4.7**, so the budget
needed was 10, not 11. **The largest gate-passing budget for the augmented
recipe on this backbone is total ~10; `live6b` at 6 passes comfortably.**

🔑 **But the partial interaction lines up with the live windows, which is the
mechanistic check the correlation needed.** L80_G95, n = 3:

| column | live window | constraint effect on gAP |
|---|---|---|
| augment | 4.7 | **+0.0015** |
| plain | 3.0 | -0.0083 |
| focal | 3.0 | -0.0080 |
| **augment x constraint** | -- | **+0.0098 +- 0.0166** (registered interaction) |
| **focal x constraint** | -- | **+0.0003 +- 0.0284** (negative control) |

The two columns with a 3.0 live window both show damage; the one with 4.7 does
not. The registered interaction is positive and the negative control is ~zero,
**which is exactly the discrimination the amendment predicted** -- focal
enlarges the gradient and changes nothing, augmentation unfreezes the boundary
and changes the sign.

🛑 **Do NOT read any single column of `live11` as "the constraint works".** All
three fail the absolute bar, so by the standing rule no arm's number is a
result. What `live11` supports is a DOSE-RESPONSE on liveness -- the constraint
does better where the boundary is more alive -- at n = 3 with sd > mean.

➡️ **The squeeze is the problem, and lr is the way out.** Budgets of 9-11 sit
right on the bar because augmentation buys only 4.7 live epochs. If a lower lr
widens the window again the budget stops being a knife-edge. **Run the lr probes
before committing the decisive campaign's budget.**

### ⚠️ THE LIVE-REGIME CAMPAIGNS HAD NO RIVAL DUALS

`live6b` and `live11` carry `tralo`/`clip`/`focal`/`aug` and their nulls -- and
**no `alm`, `fioretto` or `hounie`.** They can answer the MECHANISM question
(does the constraint stop damaging the ranking) but not the ACCEPTANCE BAR,
which is *leading group on cc-F1* against every rival. A result from them could
only ever say "TraLO stopped hurting itself", never "TraLO is competitive".

Caught before the replication launched. **`live11_mn2` restaged with all 12
arms** -- the full seven-arm comparison PLUS the augmented column and its null
-- 96 runs, MobileNetV2, total 11, gates GREEN, frozen.

`live11` (MobileNetV3, 72 runs) is NOT wasted: it is the mechanism experiment
with proper per-column controls, which is what it was built for. The bar gets
tested on mn2 first; if the mechanism holds there too, mn3 gets a full
12-arm campaign as well.

### 📈 THE BUDGET AXIS, ONE INSTRUMENT, THREE CAMPAIGNS

`scripts/interaction.py` on the PLAIN column -- `gAP(tralo) - gAP(tralo_null)`,
same warm-up, same schedule, the only difference is whether the constraint
steps:

| campaign | budget | tree | L80_G95 | L90_G95 |
|---|---|---|---|---|
| `fm2_mn3` | 30 | optloss-lean | -0.0069 +- 0.0222 | **-0.0280 +- 0.0113** |
| `gx2` | 30 | optloss-probe | -0.0049 +- 0.0267 | **-0.0233 +- 0.0079** |
| `live6b` | **6** | optloss-live6 | **+0.0037 +- 0.0155** | **+0.0121 +- 0.0101** |

⛔ **RETRACTED 2026-09-15: `gx2` DID NOT replicate `fm2_mn3` -- it RE-RAN it.**
All 30 runs the two campaigns share are **byte-identical** in
`final_predictions_raw.csv`: distinct inodes, distinct `code_version`
(`1072a229317c` vs `4c099fc089f5`), separate executions hours apart, identical
output. Training is bit-deterministic given `(config, seed)`, and the code
change between the two stamps did not touch the training path. The apparent
difference above (-0.0233 vs -0.0280) is **entirely** gx2 missing seed 4 at
L90_G95: on the three shared seeds the numbers agree exactly.

Two consequences, one good and one costly:

- ✅ **The pipeline reproduces bit-exactly across trees and code versions.** That
  is worth stating in the paper and it makes a re-run a valid integrity check.
- 🛑 **An overlapping arm adds ZERO information.** The 30-epoch damage rests on
  **one** campaign, not two. Any future analysis must de-duplicate by prediction
  hash before counting n -- `~/dose_response.py` does, and drops 29 of 75 runs.

`gx2` still earns its place: `aug_tralo`, `aug_clip` and `focal_tralo` are new
arms that `fm2_mn3` never ran. Only its overlap with `fm2_mn3` is redundant.

⚠️ Still one backbone and one dataset throughout. `live11_mn2` (MobileNetV2,
72 runs, all columns controlled) is staged and frozen for the replication.

**And the instrument refused six numbers I would otherwise have been tempted to
read**: `gx2` and `live6b` both lack `aug_tralo_null` and `focal_tralo_null`, so
their intervention columns print REFUSED rather than a plausible-looking delta
against the wrong control. That is the gate doing its job on live data.

### 🔬 THE SECOND LEVER: LEARNING RATE (staged, not yet run)

With the budget free, `lr` is the other knob that sets how fast the boundary
freezes, and it had been treated as fixed at 1e-4. `core` holds only four:
`batch_size 64`, `dropout 0.3`, `lr 1e-4`, `pretrained True`.

A lower lr should stretch the live window roughly in proportion, which -- unlike
a short budget -- buys MORE constraint epochs rather than fewer. Augmentation
(2x) and lr could compose: 5 live epochs at 1e-4 might become 20+ at 2e-5, which
would make a long constraint phase legitimate for the first time.

**Staged and frozen: `lr5e-5`, `lr2e-5`, `lr1e-5`** -- 10 runs each, 12 epochs,
seed 1, arms `aug_tralo`/`aug_clip`/`clip` (+ mandatory). ~15 min each. The
probe measures ONE thing: the live window as a function of lr.

⚠️ **`lr_constraint` is set to match `core.lr` in every variant.** The LR TRAP of
2026-08-15 was exactly an unequal lr fabricating a result, and `gen_campaign`
refuses if they differ -- but the variant generator has to set BOTH, and it does.

⚠️ This is not weight decay or label smoothing. It is the learning rate, shared
by every arm, which is why it does not tilt the comparison.

### 🔑 MEASURED: AUGMENTATION DOUBLES THE LIVE WINDOW, AND 30 EPOCHS STILL FAILS

First completed `gx2` runs, epochs before train accuracy reaches 0.95:

| arm | live window |
|---|---|
| `clip` | 2 |
| `focal_clip` | 2 |
| `aug_clip` | 3 |
| `focal_tralo` | 3 |
| **`aug_tralo`** | **5** |

Augmentation roughly **doubles** the live window -- a real effect, and the
first intervention that has moved this number at all. **And it is nowhere near
enough at 30 epochs**, where the gate needs 14. `gx2` will fail
`gate:saturation` on every arm, so by our own hard rule its numbers are not to
be read.

🔑 **So the budget is the wrong way round.** We have been fixing
`total_epochs = 30` and discovering the boundary dies at epoch 3-5. The gate's
criterion -- live window >= half the constraint epochs -- inverts directly into
a design rule:

    total_epochs  =  2 x (measured live window)  +  1

With augmentation's 5 live epochs that is **11**: one warm-up plus ten
constraint epochs on a boundary that is live for half of them. Below that the
constraint gets too few steps to do anything; above it, every extra epoch is
spent kicking a frozen boundary and, by settled fact 6, actively damaging gAP.

**Staged and frozen, gates green, waiting on a GPU:**
- `live6b` -- total 6 (1 + 5). Live window covers 5 of 5 augmented, 2.7 of 5
  unaugmented, so it passes either way. **RUNNING.** The floor case: does the
  constraint stop damaging gAP when the boundary is alive?
- `live11` -- total 11 (1 + 10). The budget the rule picks. **STAGED**,
  `rig_status` correctly refuses to launch it at 3 GPUs (house limit 2).

⚠️ **`rig_status` caught me over the house limit**: it refuses at 3 in use,
limit 2. `fm2_vit` (failed the gate at 1.6 live epochs) and `gx2` (fails at 5 of
29) are both occupying GPUs on regimes we have already ruled out, while the two
designs that PASS the gate queue behind them.

### 🔑 TIGHT COMPARISON: WHAT ACTUALLY DIFFERS BETWEEN TraLO AND THE RIVALS

**Dose is closed.** `dose_landed` on both complete campaigns: `alm`, `fioretto`,
`hounie` and `tralo` each land **232 / 232 steps, 100%**. The historic dose gap
is gone, so it explains nothing. Combined with the shared per-item gradient,
the four arms differ in **exactly one thing: how a fixed total displacement is
split across scopes.**

**And TraLO splits it differently from all three rivals, in two ways.** Read
from `tralo/train.py:292-311` against `fioretto_ldf/train.py:104-124`:

| | TraLO | fioretto / alm / hounie |
|---|---|---|
| response to violation | `lam += lambda_step` -- a CONSTANT, identical for every violated scope | proportional to `excess`, so the worst scope gets the most |
| what triggers it | the **HARD** count crossing the cap (a step function) | the **SOFT** count, continuously |

1. **Depth blindness.** Under `normalize` only the RATIOS between scopes steer.
   TraLO gives every violated scope an equal share, so a scope over by 1 item
   pulls as hard as one over by 100 -- and because the total is normalised, that
   share is TAKEN FROM the scopes that need it. With ~33 scopes this is a large
   misallocation. The rivals concentrate where the violation is.
2. **A step-function gate.** TraLO's lambda only moves when the HARD count
   crosses. A scope sitting just under its cap gets nothing, then pops over and
   takes a constant kick. The rivals' response is continuous.

⛔ **Difference 1 has been tested and is a null** -- `tralo_dualprop`
(`lambda_ratchet_mode: proportional`), 12 cells, F1 -0.0038, every cell inside
its seed sd. **Do not re-propose it as-is.**

🛑 **BUT READ THAT NULL AGAIN.** All 12 of those cells ran the 30-epoch recipe,
which saturates in 3-5 epochs in **5 of 5 cells wherever it has been measured**.
By settled fact 6, the constraint in that regime is injecting NOISE into the
ranking. So what `tralo_dualprop` actually established is that **redistributing
noise across scopes does not help** -- which is exactly what one would predict,
and says nothing about the live-boundary regime.

⚠️ **This applies to the WHOLE rejected ledger.** Nearly every direction closed
in this document was closed on a frozen boundary. Those nulls are sound as
statements about that regime and are NOT evidence about a live one. **Do not
re-open them speculatively -- but do not treat the ledger as final either.** The
correct order is: get a live boundary first (`gx2`, `live6b`), confirm the
constraint stops damaging gAP, and only THEN re-test the closed mechanisms,
cheapest first. Re-opening before that would just re-measure noise.

**Difference 2 (hard step-function gate vs continuous soft response) does not
appear to have been isolated** -- `tralo_dualprop` changed the depth response,
not the trigger. Flagged as a candidate, NOT proposed: it belongs behind the
live-boundary result like everything else.

### 🔑 WHY IT CAN ONLY LOSE: THE CONSTRAINT IS **UNIDENTIFIED**

Read from `tralo/train.py:137-160` and `:232-263`. Each constraint epoch runs
TWO backward passes: CE over the **train** loader, then the count penalty over
the **test** set. So the test set -- the thing being reshaped -- carries a
count term and **nothing that knows about correctness. Ever.**

**And a count is permutation-invariant.** The penalty is a function of
`sum_i p_i(c)` within a scope. Permuting the `p_i(c)` values among the items of
that scope leaves the count, and therefore the loss, exactly unchanged. So the
objective's level sets contain every ranking with the same count: **the
constraint cannot prefer a good ranking to a bad one.** It is not being
optimised badly -- it is *unidentified*, and gradient descent resolves the tie
by whatever is cheapest in parameter space, which has nothing to do with
correctness.

- **CE alive** -> CE breaks the tie, toward rankings consistent with labels.
  Count says HOW MANY, CE says WHICH. They compose.
- **CE dead** (epoch 3-5 of 30) -> nothing breaks the tie, and
  `constraint_step.py:36` rescales the step to FULL SIZE however small the
  violation. Full-size push, no tie-breaker, no opposition.

**That is exactly what gAP measures and exactly what it shows** (settled fact 6,
15/16 seeds negative). Not a bad reshaping we could derive and correct -- noise.

⛔ **A weaker model or harder data does NOT fix this.** Any competent backbone
memorises 17k images in ~3 epochs; harder data lowers the ceiling but the train
set is still memorised, which is what kills CE. A weaker model has the same
structure with a worse boundary. **The lever is not capacity or difficulty.**

### 🔬 THE TWO FAMILIES OF FIX, AND WHICH ONE IS A PAPER

**A. Keep the existing tie-breaker alive.** A training recipe, not a claim.
  - `aug_*` -- augmentation makes the train set effectively infinite so CE never
    zeroes. **RUNNING as `gx2`.**
  - **`total_epochs ~6`, ZERO code change, 5x less compute** -- run the whole
    constraint phase inside the live window instead of outside it. The cheapest
    untried test in the project.

**B. Give the TEST set its own tie-breaker.** This is the method claim: a
count-constrained transductive learner is unidentified on the unlabeled set, and
the fix is an unsupervised task term there. **Nothing of this kind exists in the
codebase** -- entropy appears only as a logged metric (`metrics.py:96`), never
as a loss.
  - **Entropy minimisation on test.** Prefers confident predictions, so among
    count-equivalent solutions it evicts the LEAST CONFIDENT -- which is the
    margin rule, and `alloc_gap.py:margin_topk` proved that optimal. The
    theoretical fit is exact.
    ⚠️ Entropy minimisation normally collapses to one class; **here the count
    constraint itself prevents that on the capped classes.** The two terms fix
    each other's failure mode, which is the nicest thing about this direction.
  - **Consistency under augmentation on test.** Stronger, more machinery.
  - **Proximal/KL anchor to the warm-up predictions.** Minimal-change projection
    onto the feasible set. Note what this is: it makes TraLO CONVERGE to the
    clipper, so it is a safety FLOOR (stops the damage), not a win -- worth
    having, not worth claiming.

**C. Remove the noise amplifier.** `constraint_grad_mode: normalize` rescales to
a fixed norm even when the violation is tiny. Scaling with violation depth
instead is a config-level change and directly targets the "full-size push"
half. See `project_constraint_blind_to_violation_depth_2026-08-22`.

**Every one of these needs a matched post-hoc control** (`ent_clip` beside
`ent_tralo`, as `aug_clip` sits beside `aug_tralo`), or the intervention's own
main effect gets read as the constraint's.

🛑 **Do not implement while `gx2` is live** -- `src/` is in `source_inventory()`.
Build on the branch, gate it, deploy after gx2 lands.

### 🔑 THE FOUR "RIVAL" DUALS SHARE ONE PER-ITEM GRADIENT. READ, NOT SIMULATED.

Every dual in the comparison builds its constraint term as a weighted sum of the
SAME per-item quantity:

| arm | constraint term | source |
|---|---|---|
| `fioretto_ldf` | `sum_scope lam * sum_i p_i(c)` | `fioretto_ldf/train.py:145-160` |
| `fioretto_alm` | `sum_scope (lam + mu*aug) * sum_i p_i(c)` | `fioretto_alm/train.py:182-196` |
| `hounie_rcl` | `sum_scope lam * sum_i p_i(c)` | `hounie_rcl/train.py:173-178` |
| `tralo` | `sum_scope penalty(soft_scope, K)`, `soft_scope = sum_i p_i(c)` | `tralo/train.py:238-263` |

TraLO looks different and is not: `penalty` is a function of the SCOPE's summed
soft count, so `d penalty / d p_i(c)` is one scalar shared by every item in the
scope. So for all four,

    dL/dz = sum_scopes w_scope * sum_{i in scope} d p_i(c) / dz

and the arms differ **only in how they compute the scalar `w_scope`.** Within a
scope, the per-item direction is byte-for-byte the same function. `lam >= 0` and
`penalty' >= 0`, so the weights never even change sign.

**And the scopes are disjoint.** A local scope `(g, c)` sums only over group
`g`, and groups partition the test set. So the per-scope weights do not reweight
items against each other inside a group -- they set **how hard each GROUP is
pushed**, i.e. how many items get evicted from each group. Only the single
global scope overlaps.

🛑 **Chain this with the eviction-order result above and the family collapses.**
Any push monotone in `p(c)` evicts a group's lowest-margin items first, and the
order is set by the warm-up model's margins, not by the dual rule. So the whole
family's only real freedom is **the per-group eviction COUNT** -- and the caps
already specify that. Four methods, one direction, one number each, and the
number is given to them.

**This is why they tie**, and it is a stronger statement than "the effect is
below the noise floor": there is no per-item decision left for a better dual
rule to make. What genuinely remains free is narrower than it looks -- the
trajectory (how fast each group is driven to its target), the stopping rule
(satisfaction/ratchet, which on this corpus has never fired), and interaction
with the CE term. Those are schedule, not mechanism.

**How close to the ceiling is the family already?** `margin_topk` is proven
equal to the LP, so its eviction set is the provable optimum given the
probabilities. A plain gradient push, bisected to evict the same 20%, lands on
**87% of exactly those items** at sharp=2.0 (92% at 4.0, 88% at 1.0) -- and the
missing 13% is worth the <=6% damage difference measured above, i.e. 0.0002
accuracy. So the dual family is already at ~94% of everything reachable without
changing the probabilities. **The residual is 0.0002 and the prize is not
there.**

➡️ **The only channel that can change WHICH item is evicted is the ranking
itself** -- the model's own margins. That needs a boundary still moving when the
constraint phase starts, which is exactly what `gate:saturation` now measures
and what the corpus fails 5 cells out of 5. **The saturation problem is not
hygiene beside the real question; it is the whole question.**

### ❌ CLOSED: A MARGIN-AWARE SOFT COUNT IS NOT THE MISSING PIECE

The loss penalises `soft = sum_i p_i(c)`; the deployed quantity is
`hard = sum_i 1[argmax_i == c]`. The two disagree about *which* items matter, so
this looked like the core-loss defect. Measured, 4000 items / 8 classes /
5 seeds, at sharp=2.0 (0.70 accuracy, our regime):

- **68% of the soft-count gradient mass, by full per-item gradient norm, lands
  on items whose argmax is NOT class `c`.** Those items are not in the class;
  reducing their `p(c)` cannot lower the hard count by one item. Only 8.8% of
  the mass sits within 0.05 of a decision boundary.
- `soft` is nonetheless an excellent ESTIMATOR: 503.3 against a hard count of
  502. So the earlier reassuring "soft ~ hard" measurement was checking the
  value, not the derivative, and could not have caught this.

🛑 **And the 68% does not convert.** Replacing the surrogate with
`sum_i sigmoid(margin_i / tau)` -- whose gradient is concentrated on the
boundary by construction -- and bisecting the step size so BOTH evict the same
20% of the capped class:

| sharp | soft, accuracy cost | margin, accuracy cost | change |
|---|---|---|---|
| 4.0 | +0.0226 | +0.0226 | 0% |
| **2.0** | **+0.0039** | **+0.0037** | **-6%** |
| 1.0 | +0.0004 | +0.0003 | -14% |

**Why it cannot convert.** `constraint_step.py` rescales to a fixed norm and the
dual keeps stepping until the cap is met, so the budget is set by the EVICTION
TARGET, not by the step size. Any push that is monotone in `p(c)` flips the
lowest-margin items FIRST. The weighting changes how hard each item is pushed;
it barely changes the ORDER in which they cross. Gradient-mass concentration is
therefore close to irrelevant to which items move -- which also means the
`p(1-p)` collapse recorded above limits how large a step is needed, not what the
step does, and it was over-weighted in the diagnosis.

⚠️ **Caveat, stated rather than buried.** This moves logits per item freely;
real training moves WEIGHTS, and per-item displacements are coupled through the
network, so the absolute damage figures do not transfer. The comparison does --
the coupling handicaps both surrogates identically. Do not re-propose a
margin-aware count without new evidence that beats this control.

### 🟢 WE MAY BE JUDGING IT ON THE WRONG HEADLINE

`profile_report` on `fm2_mn3`, TraLO minus each clipper, seed-paired with the
sd of the difference (2-3 seeds, MobileNetV3 -- directional, not final):

| contrast | cap | d cc-F1 | d macro-F1 | d accuracy |
|---|---|---|---|---|
| vs `clip` | L80 | -0.0059 +- 0.0183 | **+0.0081** +- 0.0163 | +0.0050 +- 0.0119 |
| vs `clip` | L90 | **-0.0237** +- 0.0009 | **+0.0109** +- 0.0127 | +0.0074 +- 0.0162 |
| vs `focal_clip` | L80 | +0.0049 +- 0.0103 | **+0.0141** +- 0.0084 | **+0.0116** +- 0.0053 |
| vs `focal_clip` | L90 | -0.0115 +- 0.0027 | **+0.0199** +- 0.0072 | **+0.0169** +- 0.0086 |

🔑 **macro-F1 is positive in all four contrasts, and against
`focal_clip` it clears its own seed sd at both caps (1.7 and 2.8 sd), as does
accuracy (2.2 and 2.0 sd).** Collateral F1 -- the UNCAPPED classes -- is
+0.0164 against clip at L80 (0.6051 vs 0.5887).

So the constraint has a consistent, replicated effect and the sign depends on
which classes you look at: **it trades capped-class quality for uncapped-class
quality.** Pushing probability mass off the capped classes frees it for the
rest, and the rest are 5 of the 8.

⚠️ **That is exactly what main.tex claims** -- "beats post-hoc clipping on
macro-F1 in nearly every regime" -- and it is NOT the bar this project has been
judging against. cc-F1 has been the headline, and on cc-F1 TraLO trails.

⚠️ Under the user's stated bar (leading group on cc-F1, not dominated
elsewhere) this is a **TRADE**, not a win: at L90 the cc-F1 deficit against
`clip` is -0.0237 against a paired sd of 0.0009. It does not pass. But it does
mean the method is not inert, and that the question "where are we failing" has a
narrower answer than "everywhere": **we are failing on the capped classes
specifically, while winning on the other five.**

🔁 Confirm at 4 seeds on `fm2_mn3` the moment it lands, and check
whether `fm2_mn2` reproduces the sign.

### 📒 LEDGER -- WORKED / FAILED / PROMISING (running, 2026-09-14)

Kept current so no direction is tried twice. Add to it, never re-litigate it.

**✅ ESTABLISHED TRUE**

- **The constraint really does reduce violations.** Native excess items, raw
  argmax before the allocator: bcn L70 `tralo` **550** vs its own null **953**
  and plain `clip` 852; fmow2 L80 `tralo` **372** vs null 469, clip 473. This is
  not an inert arm.
- **The optimizer reset is worth real ranking quality.** `tralo_null` (zero
  constraint steps) minus `clip` on gAP: +0.0064/+0.0199 MobileNetV3,
  +0.0137/+0.0234 ViTB16, 4 of 4 positive at 4 seeds.
- **fmow2 passes all 8 `candidate_gate` conditions**; bcn passes 7 of 8
  (C8 balance 0.04) and is blocked separately on an integrity hold.

**⛔ CLOSED -- DO NOT PROPOSE AGAIN**

- **`tralo_dualprop`** (integrate the raw residual instead of counting violated
  epochs): built 2026-09-06, run, **null** -- F1 (Macro) -0.0038 negative in 9
  of 12 cells, Accuracy -0.0013 in 8 of 12, every cell inside its paired sd.
- **A two-sided / shrinking multiplier**: same evidence, plus main.tex's own
  280x lambda-scale probe and the fact that `normalize` discards the common
  scale.
- **A per-cell freeze**: the ratchet is ALREADY per-cell
  (`if hard_c > limit_c`); the global AND is a kill switch that never fires, so
  removing it changes nothing.
- **`tralo_squared`**: obeys WORSE than `tralo` (613 vs 550 at bcn L70). A
  peaked penalty spends the fixed-norm step on scopes that cannot be fixed.
- **"More data" as the saturation fix**: refuted. fmow2 has 17,670 training
  images to bcn's 8,270 and saturates FASTER (3.0 vs 5.0 live epochs).
- **Harder classes / weaker networks as the saturation fix**: the whole
  {3 backbones x 2 datasets} grid spans 2.0-5.0 live epochs. Worth ~2; we need 11.
- **iwildcam** (2 of 8 conditions) and the original **fmow** oodslice (basename
  join collapsed AOIs).

**🟡 NULL, BUT NOT A DEFECT**

- **Focal for the CLIPPER on the metric**: differences 0.002-0.004 against seed
  sd 0.005-0.014 on fmow2. It is not a metric win for `clip` -- see PROMISING.
- **Obedience does not convert to the metric.** The allocator takes top-K per
  (group, class) and is optimal given the probabilities, and the capped classes'
  top-K sets are pairwise disjoint (`lp` and `clip` score identically), so
  native compliance earned in training is worth ~0 at scoring time. **Only the
  ORDER survives.** This is why every shape/dose/multiplier variant moves
  obedience without moving the metric.

**🟢 PROMISING -- OPEN**

1. **The saturating penalty costs obedience.** `tralo_linear` reaches excess
   **472** where `tralo` reaches **550** at bcn's tightest cap. The shape the
   package is named after is the worst of the three tested at the cap that
   binds hardest. (It does NOT convert to ranking -- see NULL above -- but it
   locates a real defect in the core loss.)
2. 🔑 **Focal keeps the constraint's gradient ALIVE.** The constraint
   reaches the weights only through `d(soft count)/d(theta)`, whose per-item
   weight is `p(1-p)`. Measured on fm2_mn3, against a 0.25 maximum:

   | arm | mean p(1-p) | share carried by top 1% of items |
   |---|---|---|
   | `clip` / `tralo` / `tralo_null` / `alm` | 0.007 - 0.023 | **34%** (class 1) |
   | `focal_clip` | **0.015 - 0.045** | **5-16%** |

   On the saturated model a few dozen borderline items choose the entire
   constraint direction, which `normalize` then rescales to full size -- that is
   the "kicks the boundary at random" failure, quantified. Under focal the
   gradient is 2-2.5x larger and far less concentrated. New arm **`focal_tralo`**
   (schema line only; `make_ce_criterion` already honours `warmup_loss`).
3. **Augmentation** as the saturation fix -- the only intervention that attacks
   memorisation at its root. Default-off `augment` flag through the single
   `make_dataloader` seam.
4. **Run the constraint phase inside the live window** (`total_epochs 6`). Zero
   code change, passes `gate:saturation` because the gate is a ratio, 5x cheaper.

### 🛑 gate:saturation -- THE CONSTRAINT SPENDS ITS EPOCHS ON A FROZEN BOUNDARY

Once train CE collapses the task gradient is ~0, and `constraint_step.py:36`
rescales the summed gradient to EXACTLY `clip` whenever `raw_norm < clip` -- so
the constraint step stays full-size however small the violation is. Full-size
step, nothing opposing it: **the constraint is not reshaping a boundary, it is
kicking a frozen one.**

**Live window = epochs before train accuracy reaches 0.95. Required: half the
29 constraint epochs, i.e. 14.**

| backbone / dataset | live epochs | train acc after 1 constraint epoch |
|---|---|---|
| MobileNetV3 / bcn | 5.0 | 0.720 |
| ViTB16 / bcn | 4.2 | 0.746 |
| MobileNetV2 / fmow2 | 4.0 | 0.783 |
| MobileNetV3 / fmow2 | 3.0 | 0.844 |
| ViTB16 / fmow2 | **2.0** | 0.906 |

🔑 **The whole {3 backbones x 2 datasets} grid spans 2.0 to 5.0 epochs
against a requirement of 14.** The two ordering effects are legible -- a bigger
model saturates faster (ViT worst), a harder dataset saturates slower (bcn best)
-- and both are far too small to matter. **Changing dataset or backbone cannot
fix this**, which is why the dataset search kept failing to find one that
"works".

⛔ **And "more data" is refuted, not merely doubted:** fmow2 has 17,670
training images to bcn's 8,270 and saturates FASTER (3.0 vs 5.0). Size is not
the lever. Nor is class separability the whole story -- memorisation is
instance-level, so a harder class set buys the ~2 epochs bcn shows, not 11.

**What actually changes the regime**, given the above:

1. **Augmentation.** The only intervention that attacks memorisation at its
   root: each epoch shows a different view, so there is nothing fixed to
   memorise. Currently there is NONE -- `make_dataloader` wraps a
   `TensorDataset` over preprocessed uint8 arrays. Its absence is the anomaly,
   not its addition: the model it produces is 99.9% train against 63.4% test.
2. **Run the constraint phase inside the live window** (`total_epochs 6`). The
   gate is a RATIO, so this passes it, needs no code change, and costs 5x less
   compute. Framed as a design rule -- the constraint acts while the task loss
   is still live -- it is principled; framed as "we ran 30 epochs and 5
   mattered" it is not.
3. Weight decay / label smoothing: real but secondary, and the user has
   reservations about tuning toward an outcome.
4. ⛔ Weaker networks: rejected -- reads as handicapping the model.

**The gate.** `scripts/saturation_gate.py`, wired into `run_campaign`'s
**firstrun** stage as a REQUIRED gate, keyed on (backbone, dataset). It fails
5 of 5 existing cells.

### 🔧 WHY bcn AND fmow2 DISAGREE: bcn FAILS THE BALANCE CONDITION

`scripts/candidate_gate.py` on both live datasets, capped classes [0, 2]:

| condition | bcn | fmow2 |
|---|---|---|
| C1 classes >= 8 | 8 | 10 |
| C4 density >= .50 | **0.89** | 0.75 |
| C5 dead items <= 10% | 0% | 0% |
| C6 zero ceilings <= 25% | 0/16 | 3/20 |
| C7 binding ceilings | 11/16 | 6/20 |
| **C8 balance >= .25** | ⛔ **0.04** | ✅ 0.59 |
| verdict | **7 of 8** | **PASS ALL** |

bcn class supports: 32.7 / 23.5 / 22.3 / 8.8 / 6.5 / 3.7 / 1.3 / 1.3 percent --
five tail classes holding 21% of the data between them, and the rarest class at
**1/26th** of the commonest. fmow2 spans 9.3 to 15.9 percent.

🔑 **This is the most likely reason every intervention beats plain
`clip` on bcn and none does on fmow2.** `focal_clip` (+0.0136 / +0.0197) and the
optimizer reset (+0.0132 / +0.0185) are both IMBALANCE remedies, and they buy
about the same amount, and they do not stack. The constraint is not an imbalance
remedy and buys nothing on either dataset. On a dataset this skewed the plain
clipper is a weak baseline, so **no bcn claim may be reported without
`focal_clip` beside it.**

⚠️ **Standing decision, taken 2026-09-14 under the user's mandate to drive
this.** bcn STAYS as the second dataset -- 7 of 8 against iwildcam's 2 of 8, best
density in the corpus, no dead groups, no zero ceilings -- and the C8 failure is
recorded as a property to report rather than a defect to hide. A dermatology
slice legitimately has rare conditions; that is the "hospital story" the dataset
brief asked for. But bcn's image arrays are NOT in the lean tree (only
`train_meta.csv` / `test_meta.csv`), so a bcn campaign needs them restored from
the archive first.

### 🟢🔴 THE RANKING CHANNEL: THE RECIPE BEATS `clip`, AND `focal_clip` MATCHES IT

"gAP" is per-group average precision on the raw pre-allocator probabilities.
Post-hoc allocation is optimal given the probabilities and that optimality is
distribution-free, so **ordering is the only channel by which a trained arm can
beat the clipper.** `scripts/rank_paired.py` reports the seed-paired delta per
cell WITH its sd across seeds, and collapses caps that never reached the model.

⚠️ **Two analysis traps this section walked into first, both now gated.**
(1) A mean with no floor: I published "bcn, 6 of 6 cells positive, +0.0143 vs
`clip`" off `rank_probe`, and every one of those cells is inside its own seed
sd. (2) A cap is not a cell for an arm that takes no constraint step: md5 shows
`clip` and `tralo_null` write **byte-identical** predictions across L70/L80/L90
while `tralo` writes three different ones. The six cells were **two**.

**`tralo_null` minus `clip` -- this is the Adam-reset contrast.** `tralo_null`
takes ZERO constraint steps (`has_constraint` is false when all lambdas are 0),
so it is exactly `clip` plus the 1+29 split and the fresh Adam built after
warm-up. 4 seeds, caps collapsed:

| backbone | class 0 | class 2 |
|---|---|---|
| MobileNetV3 | +0.0064 (1.45 sd) | **+0.0199 (1.52)** |
| ViTB16 | +0.0137 (1.34) | **+0.0234 (0.82)** |

🟢 4 of 4 positive, +0.006 to +0.023. **The Adam reset is worth real
ranking quality**, which is what main.tex already credits.

⛔ **But `focal_clip` gets the same thing for free.** `tralo_null` minus
`focal_clip`: MobileNetV3 +0.0041 / -0.0050, ViTB16 +0.0023 / -0.0047 -- 2 of 4
positive, all far inside sd. So the recipe's advantage exists only against the
**plain** clipper, and the plain clipper is not the strongest baseline in the
protocol. This is the same shape as the 2026-08-17 retraction
(`project_loses_ccf1_to_focalclip_multiclass`): a win over `clip` that
`focal_clip` erases.

🔑 **And the constraint adds nothing on top.** `tralo` minus
`tralo_null` on bcn class 2 is +0.0080 / +0.0034 / +0.0026 on MobileNetV3 (all
inside sd) and 2 of 3 negative on ViTB16.

⛔ **On fmow2 the constraint is actively NEGATIVE.** `fm2_mn3`, `tralo` minus
`clip`: 5 of 6 cells negative, and at L90 the effect clears its own sd -- class 1
-0.0082 (1.59), class 2 -0.0142 (4.65), class 7 **-0.0410 (7.03)**. Against
`tralo_null` also 5 of 6 negative, cell-mean -0.0101.

**Net, stated plainly: there is no cell anywhere in the corpus where the
CONSTRAINT buys ranking.** What buys ranking is the optimizer reset, and
`focal_clip` matches that without any constraint machinery.

⚠️ fmow2 is 2-3 seeds and one backbone. `fm2_mn2` and `fm2_vit` at 4 seeds
decide whether the fmow2 reversal is the dataset or the backbone; at 1-2 seeds
they disagree with each other and are not read.

### ⛔ THE PAPER'S SECOND PHASE HAS NEVER RUN -- 2,563 runs, 0 freezes

`docs/paper/main.tex` describes TraLO as two-phase: lambda ratchets while the
count is violated, then **"is frozen the moment the count is first satisfied"**,
and rho likewise stops, so **"the tail of training descends one objective
instead of chasing a multiplier that is still moving."**

**That branch has never executed.** A recursive scan of every `optloss-*` tree on
dsisco02 -- `Satisfaction Epoch` in `evaluation_metrics.csv`, which
`tralo/train.py:433` -> `runner.py:154` -> `logging.py:213` carries from
`satisfaction_epoch` -- found **2,563 TraLO-family runs across 5 datasets and 22
arm variants, and 0 that ever reached satisfaction**:

| dataset | runs | reached |
|---|---|---|
| iwildcam | 1,568 | 0 |
| bcn | 663 | 0 |
| fmow | 272 | 0 |
| dermmnist | 46 | 0 |
| fmow2 | 14 | 0 |

Three independent signals agree: `Local_Satisfied` is 0 in every logged epoch;
`Lambda_Local` climbs monotonically 0.0400 -> 1.0416 over 30 epochs with a dead-
constant ~+0.035/epoch increment; `L_Local` grows 0.16 -> 1,481 while `L_CE`
falls to 0.011.

**Why.** `tralo/train.py:285` sets `is_satisfied = global_satisfied and
local_satisfied`, where `local_satisfied` is a conjunction over every
(group, class) cell. On fmow2 that is 3 global + 30 local = 33 cells that must
all be at-or-under budget **in the same epoch**. Measured: only **8-13 of the 30
local cells** comply in any epoch, and that does not improve over 30 epochs
(12 -> 8 at L80, 14 -> 13 at L90). Satisfaction is ~20 cells away, not one.

**Two consequences, present in every run ever made:**

1. **The multiplier can only grow.** `set_lambda_per_class(c, old + lambda_step,
   ...)` at lines 296 and 308 are the ONLY mutation sites after init, and the
   increment is a CONSTANT on a BOOLEAN violation. So `lambda_c = lambda_0 +
   step * (epochs violated)` is a violation-FREQUENCY counter whose dynamic
   range is capped by the epoch count.
2. **rho never freezes either.** `rho_frozen` is set only inside the
   satisfaction branch, so `increment_rho(rho_step)` runs all 29 epochs and rho
   lands on `rho_target = 100.0` in every run, from `initial_rho = 0.5`.
   (The paper says rho ramps "from 5 toward 100"; the config says 0.5.)

⚠️ **What this does NOT mean, and I had it wrong for an hour.** The
monotone lambda ramp is not an escalating force. `constraint_step.py:36`
renormalizes the summed gradient to exactly `clip` whenever `raw_norm < clip`,
so **the common scale of all multipliers divides out and only the per-scope
RATIOS steer**. The measured pre-clip `Grad_Norm` growth of 2.9 -> 27,000 is
discarded in full. Consequently "pressure on an already-compliant cell never
relaxes" is FALSE: a compliant cell stops accruing lambda while violating cells
keep accruing, so its RELATIVE weight decays and the system releases it.
Undershoot is therefore NOT structural on this argument. main.tex makes the same
point and reports a matched probe with lambda differing ~280x training almost
identically.

⛔ **THE IMPLIED FIX IS ALREADY IN THE LEDGER. DO NOT PROPOSE IT AGAIN.**
The obvious repair -- integrate the raw residual like `alm` does, instead of
counting violated epochs -- was built on 2026-09-06 as
`lambda_ratchet_mode: proportional`, arm **`tralo_dualprop`**, and run. Scored
here against `tralo` on seed-paired cells:

| cells | metric | mean delta | negative |
|---|---|---|---|
| 12 (bcn, fmow, iwildcam x MNv2/MNv3/ViTB16, 2-4 seeds) | F1 (Macro) | **-0.0038** | 9 of 12 |
| same | Accuracy | **-0.0013** | 8 of 12 |

Every cell is inside its own paired seed sd. **`tralo_dualprop` is a null, and
if anything slightly negative.** The direction is closed.

⚠️ This finding is also not new -- `scripts/latch_probe` established the
dead latch on 2026-09-06 over 72 `dom1` runs (0 of 24 for each of `tralo`,
`tralo_null`, `tralo_uniform`). What is new here is the SCOPE: 2,563 runs, all
five datasets, all 22 variants, read from the persisted `Satisfaction Epoch`
rather than reconstructed. The two-phase description in main.tex has never once
described a run that exists.

🔑 **What is still open.** Not the multiplier. The latch's own gate is:
satisfaction is a global AND over all 33 scopes when the ratchet it gates is
already per-cell (`if hard_c > limit_c` at 294 and 306). Removing the
conjunction changes nothing about which cells ratchet -- it is a kill switch
that never fires -- so a per-cell freeze is NOT the fix either. The live
questions are the ones the pre-registration below tests, and the CE budget:
`L_CE` is under 0.05 from epoch 6 of 30 with train accuracy 98.4%, so **24 of
the 29 constraint epochs run against a task loss that is 5% of its epoch-2
value**, on BOTH `tralo` and `tralo_null` (their CE trajectories match within
seed sd at every epoch). Test accuracy is 63.4% against 99.9% train, so fmow2 is
genuinely hard -- the saturation is memorization, not an easy task.

### 🔬 PRE-REGISTERED 2026-09-14, BEFORE THE SEEDS LANDED

**CLAIM: TraLO's ranking damage is caused by UNDERSHOOT, not by the constraint.**

TraLO decides a scope is violating from a SINGLE epoch's count. That count has a
measured epoch-to-epoch sd of **27-113 items** and oscillates just as much with
the constraint switched off -- `tralo_null` takes zero constraint steps and its
class-2 count still swings 468 -> 869. So the ratchet fires on noise:
`Local_Satisfied` was 0 in EVERY epoch of every run inspected, multipliers
climbed monotonically for all 29, and class 7 ended at **111-122 predictions
against a permitted 304 on BOTH backbones** -- a class whose global excess in
the null was +8 (MNv3, 0.17 sd) and -83 (MNv2, already compliant).

Evidence at pre-registration: 12 paired `tralo` vs `tralo_null` contrasts over
2 backbones x 2 caps x 3 classes.

| group | mean d gAP | n |
|---|---|---|
| ended UNDER budget | **-0.0273** | 5 |
| ended AT/OVER budget | **+0.0001** | 7 |

`pearson(excess/sd, d gAP) = +0.524`. The two largest GAINS in the table
(+0.0482, +0.0247) are both cases where TraLO corrected and stopped.

**PREDICTION.** At 4 seeds across `fm2_mn3`, `fm2_mn2` and `fm2_vit`, the split
holds: contrasts ending under budget stay negative, contrasts ending at/over
budget stay >= 0 within seed noise.

⛔ **FALSIFIED IF** the two groups are equally negative at 4 seeds, or if
`pearson` falls to ~0. Then the damage is not overshoot and this account is
wrong -- record it in the rejected ledger rather than rescuing it.

⚠️ At pre-registration n=12 and the contrasts are NOT independent: L80 and L90
share a warm-up and some share seeds. r=+0.524 at n=12 is suggestive, not
conclusive. What makes it worth testing is the mechanism, not the p-value:
undershoot forces the allocator to backfill to K from lower-ranked items, which
is mechanically guaranteed to cost quality.

🔑 **THE FIX THIS IMPLIES IS NOT IN THE REJECTED LEDGER.** Everything closed
there is the gradient EXPRESSION -- penalty shape, count function, cut window,
margin, scope re-weighting. This is the MEASUREMENT that triggers it: require a
scope's excess to clear its own measured epoch noise before ratcheting, or
average the count over recent epochs before declaring a violation. No gradient
changes and no extra compute. Test as a `tralo_hyst` arm against unmodified
`tralo` at equal compute, pre-registered above.

⚠️ **AMENDED 2026-09-14, same day, BEFORE 4 seeds.** Two checks since
pre-registration weaken parts of this, recorded here rather than quietly dropped:

- **The local scope is near-inert.** Against `tralo_null`, end-of-run local
  compliance is indistinguishable (8 vs 8, 13 vs 10, 9 vs 12, 13 vs 13 of 30
  cells); total local excess falls only ~10-15% (407 vs 471, 338 vs 403, 381 vs
  351, 275 vs 306). 29 epochs of ratcheting buy almost nothing locally.
- **"Class 7 driven to 111-122 on BOTH backbones" does NOT replicate.** At the
  current seed counts, end-of-run class-7 global counts against a permitted 304
  are 171.5 (MNv3 L90, n=2), 218/246 (MNv2, n=1), and **385 (ViTB16 L80, n=1,
  vs a null at 159 -- the opposite sign)**. The earlier figure is withdrawn.
- **Classes 1 and 2 never reach budget at all**: `tralo` ends 371-398 against
  347, and 519-703 against 519. The direction is right (it beats the null by up
  to 191 items on class 2) and the magnitude is insufficient.

The pre-registered prediction stands as written and will be judged at 4 seeds.
The corpus-wide freeze finding above does not depend on it.

⚠️ With `L80`/`L90` local against `G95` global, `sum(local K) < global K` for
every capped class, so the GLOBAL cap is inert and these campaigns are a pure
LOCAL-cap experiment. Do not read them as evidence about the global scope.

**Core and tool thinning committed and independently reviewed. Fresh identity,
common deployment, metrics and logging gates remain.**
Current source checkpoint `de760d40`: seven public arms; 52 historical scripts and the
task-window config machinery retired. All 17 formerly populated server result
trees remain in the external history archive, not the active results roots.

**RUN STATE, CHECKED 2026-09-14 19:42 Asia/Jerusalem.** Three campaigns live on
dsisco02, tree pinned at `1072a229`, at the 3-GPU ceiling and zero failures:

| campaign | backbone | GPU | done / planned | launcher PID |
|---|---|---|---|---|
| `fm2_mn3` | MobileNetV3 | 1 | 27 / 56 | 3290265 |
| `fm2_mn2` | MobileNetV2 | 0 | 15 / 56 | 3320254 |
| `fm2_vit` | ViTB16 | 2 | 7 / 56 | 3306622 |

`sat_fmow2` completed 16/16 earlier as the clipper saturation baseline. GPU 3 is
free and deliberately unused -- the standing ceiling is three GPUs total, not
three per host. This is a snapshot, not a reservation; recheck BOTH hosts before
any dispatch. **The source tree is FROZEN while these run** -- `src/`, `configs/`,
`scripts/` and `main.py` are all inside `source_inventory()`, so a deploy of any
of them splits the campaign identity.

## Ordered work

- [x] Preserve the pre-reset dirty diff and exact instruction files under `.codex/`.
- [x] Move the four large operational narratives into the local history archive;
  replace them with a short current protocol and state file.
- [x] Preserve historical artifacts outside Git tracking with verified backups;
  see `docs/GIT_TRACKING.md`. Archival copies are not fresh-clone dependencies.
- [x] Archive old local and remote result trees with manifests; keep data arrays,
  checkpoints, predictions, parent-extension links, and recovery paths intact.
- [x] Complete independent review of the committed operational-tool/test thinning;
  retained mathematical, AMP, caps, data, recovery and real-CLI checks stay in Git.
- [x] Reduce the public comparison to `tralo`, `tralo_null`, `clip`, `focal_clip`,
  `fioretto`, `hounie`, `alm`; remove retired variant runtime branches.
- [x] Generate the reference with `constraint_fp32: true` and
  `constraint_grad_mode: normalize`; trained 1+29/posthoc 30+0 task epochs.
- [ ] Remove the remaining unused weighted-CE option and protocol metadata;
  remove orphan package dependencies without changing the installed environment.
- [ ] Require a fresh campaign identity and explicit source/config/data/quota
  inventory for reporting; test rejection of archived, mixed and unmarked runs.
  Use a new `OPTLOSS_MODEL_CACHE` namespace so fresh runs cannot reuse historical
  warm-up checkpoints; permit sharing only inside the newly frozen release.
- [x] Replace the stale `keepworking` skill and forward-test the new reference.
- [x] Fix and regression-test AMP step/event accounting. The separately repaired
  optional uniform estimator was subsequently retired with the noncore variants.
- [ ] `scripts/alloc_real.py` -- run the greedy-vs-LP allocator comparison on
  STORED probabilities the moment SSH returns. Synthetic said +0.0095 accuracy
  at our separation; real models are overconfident and overconfidence is what
  makes greedy look good, so this either survives or it does not. Report the
  OBJECTIVE gap as the guaranteed quantity and accuracy/cc-F1 as proxies that
  can move either way -- the first version of its gate asserted the LP must win
  on accuracy, which is false, and the gate caught it.
- [ ] Use the same greedy deployment allocator and the same saved probabilities
  for every arm. Clippers currently allocate with 256-item inference, while eval
  saves a separate 512-item pass; trained arms also use a different allocator.
  Correct this explicitly as a deployment-protocol change, not a TraLO loss gain.
- [ ] Complete cc-F1-first, fixed-class metric reporting with paired native-unit
  uncertainty. Missing declared classes must count as zero, not disappear.
- [ ] Integrate shared structured logs: rival CSV initialization currently erases
  warm-up history; warm-up/rival task-step application and rival displacement/
  local-scope state are missing. Preserve model state/RNG while fixing producers
  and make the first-run gates consume the records. Missing evidence is unknown.
- [ ] Enforce exclusive canonical campaign ownership and safe crash recovery;
  fix queue failure propagation and per-runner orphan detection. The read-only
  audit found duplicate-root admission and a stale-running recovery mismatch.
  Initial two-GPU execution uses two manifest-disjoint complete campaign roots,
  one dispatcher per root, one queue per card; no multi-GPU scheduler rewrite.
- [ ] Audit current datasets and development cut saturation without selecting on
  a TraLO win. Resolve untouched holdout availability with the user.
- [ ] Commit the validated reference release and verify SHA-256 parity on the
  target host. Leave exact generate/freeze/verify/launch/inspect/report commands
  usable from any terminal or Claude Code, without Codex-only dispatch logic.
- [ ] Launch first-run pilots on two GPUs after gates pass, attach monitoring,
  inspect logs, then expand only if healthy (maximum three GPUs).

Implementation order: reviewed core/tool thinning; fresh identity/common deployment
and reporting; logging integration; dispatch/recovery repair; whole-change
verification; target-host/data
validation; then monitored reference experiments. New loss changes are deferred.
The reference loss, dual update ordering and training behavior stay unchanged
during structural cleanup. The named shared-allocator correction is separate.
Unknown or removed config keys must fail clearly, not be silently ignored.
Archive historical tests/probes through the existing recovery process; retain
compact tests for gradients, caps, metrics, data splits, logging and recovery.

The first GPU experiment tests pipeline/log validity and dataset headroom, not
superiority. Use an audited development split, one backbone and one host before
expanding. Inspect per-group allocation-cut errors, soft/hard residuals, dual
trajectories, actual applied updates and parameter displacement alongside cc-F1.
After those checks, compare the seven core methods at two distinct cap levels
with at least four seeds, paired native-metric uncertainty, equal task-epoch
budgets and recorded extra constraint compute. A later candidate requires a
separate reviewed change; no redesign is needed to run this reference comparison.

## Validation and release state

Core checkpoint `8e684211`: 77 deterministic CPU model/probability/deployment/RNG
arrays match the pre-thinning reference exactly. Independent review cleared its
dataset-scope and fixture-label fixes. Tool checkpoint `2f33fce7`: **335 passed,
1 skipped, no warnings** in 99.95 seconds. Its five review findings were fixed in
`de760d40`: **273 passed, 1 skipped, no warnings** in affected integration, then
all five independently cleared. This was not another full-suite run; the
real-log skip is not a pass. Receipts/review ledger are in the ignored
`.superpowers/sdd/lean-cleanup-plan/`; source recovery is in Git and the verified
external archive. These are software checks, not GPU or superiority evidence.

**fMoW: REPAIRED AND IN USE AS `fmow2` (2026-09-14).** The defect below was real
and is fixed. `prep_fmow` joined metadata to images on `os.path.basename`, but
the archive is laid out `split/class/class_seq/aoi/file` and the filename encodes
only `<class>_<class_seq>_<idx>` -- the AOI is not in it. Measured on
`val-metadata.tar.gz`: 63,422 records, 53,041 unique basenames, **7,429 basenames
under more than one AOI**, so 16.4% of records were silently dropped or
mis-joined. That is the mechanism behind the 436/146 rows: the `false_detection`
metadata row was dropped by `DROP`, but its IMAGE was still popped by a
surviving row sharing the basename. Both sides now key on `class_seq/aoi/file`
and `load()` refuses a non-unique key. The old arrays are preserved; the rebuild
is a separately versioned slice.

`fmow2` re-audited on the rebuilt arrays: 17,670 train / 3,442 test with row
counts consistent across images, labels and meta; **139 train countries vs 10
test countries, zero overlap**; **zero cross-split exact image duplicates**. It
passes **8 of 8** conditions in `scripts.candidate_gate` (density 0.82, 6% dead
items, 6/26 zero ceilings, class balance 0.57). Capped classes are drawn from
this slice's own labels: **1 crop_field, 2 place_of_worship, 7
ground_transportation_station** -- present in 10/10, 9/10 and 8/10 groups. Not
the 3 and 5 the old config declared; class 3 lives in 6 of 10 groups with 4 zero
ceilings.

Measured hardness on `fmow2`/MobileNetV3: train CE saturates by epoch 6 (99.7%
train accuracy), but **test accuracy is 0.634-0.648 over 4 seeds** and a cell
carries **187 errors inside K** against iwildcam's 11.7-21.2 item prize.
⚠️ On ~11 of 30 ceilings p@K >= 0.99 -- the model is confidently wrong, and the
penalty's `p(1-p)` gradient is near zero exactly there. That is a calibration
limit, not a data limit, and it is the open question on this slice.

**BCN is not launch-ready:** two exact duplicate pairs cross train/test with
conflicting class labels and different official lesion IDs. Public source JPEG
and annotation checks now confirm the conflict is upstream, not introduced by
our resize/export for these pairs. No images or labels were changed. A versioned
curation policy and renewed whole-split audit remain necessary; see
`docs/audits/2026-09-14-reset.md`. `fmow2` passes the exact cross-split image
check with zero duplicates and its crop/label failure is repaired above, so BCN
is the only runnable slice still blocked on integrity. Near-duplicates and
unused-holdout status remain open. BCN is otherwise the best-structured slice
available (candidate_gate 7/8, failing only class balance at 0.04), so repairing
it is worth doing rather than abandoning.

**iwildcam is RETIRED (2026-09-14).** It passes **2 of 8** conditions: 2 of its 8
classes can carry a local cap, half the per-group ceilings are K=0 before
training starts, and **72% of test items sit in groups holding NEITHER capped
class** -- those groups cannot produce one allocation decision. Its per-group
label shift is the best in the corpus (TV 0.737) and that is the SAME fact as
its density of 0.27: the shift IS the sparsity. Removed from `protocol.yml`,
from `data_loader.IMAGERY_DATASETS` and from every test fixture; data and the 41
completed runs are archived, not deleted.
⛔ **Every earlier TraLO number was measured through that**, so treat pre-fmow2
results as describing iwildcam rather than the method.
🔑 The two tools that could have caught it disagree by construction --
`dataset_screen` rewards shift, `tier_viability` rewards density -- and nothing
combined them, so whichever was run said "it passes". `scripts.candidate_gate`
now screens all eight conditions at once and its exit code is the verdict.

The server validation checkout `/home/dsi/michaer8/optloss-reset-validation-20260914`
is an OLDER source snapshot; its earlier CPU test pass does not validate the lean
source. No new cleanup commits have been pushed or synced. The old app display
+117,078/-33,372 was the committed `origin/main...62581d90` comparison, not
uncommitted dirt. **Re-sync and verify actual bytes before any campaign.**
iwildcam/fMoW arrays are linked there; BCN is deliberately not linked. Canonical
arrays are under `/home/dsi/michaer8/optloss-audit/data`.
dsisco01 uses older GPUs/fp16; dsisco02 Blackwell/bf16. Storage is shared NFS.
Server static-analysis dependency is isolated at
`/home/dsi/michaer8/optloss-reset-validation-deps-20260914` (`pyflakes==3.4.0`);
the shared training environment was not upgraded.

## Open user question

Are there untouched evaluation groups/splits on the three current datasets?
Until answered, do not describe rerunning the inspected splits as fresh
confirmatory evidence. Code cleanup and data-integrity checks can proceed.

The user's separate loss-research task owns `docs/research/RESEARCH_LEDGER.md`.
Its shortlist is a proposal, not an approved algorithm change or launch protocol.

## Preservation

No old scientific result is promoted or erased by the reset. Folder titles have
no evidential meaning. Archive records must say original path, destination,
inventory/hash verification, and restore procedure. Keep this state concise;
put completed audit receipts in `docs/audits/`, not a growing resume narrative.
