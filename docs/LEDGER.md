# LEDGER -- what is proved, what is measured, what is closed

**What this file is for.** So no direction is tried twice and no result is
over-read. Add to it; do not re-litigate it. If a new result contradicts an
entry, edit that entry in place and say what changed it.

**What it is not.** Not a list of wins. Several entries below are corrections to
claims this project previously made with confidence.

Every new entry needs: the hypothesis, the code/data/config identity, the actual
contrast, the metric, the uncertainty, the scope, and one disposition of
**supported**, **unfavorable in tested setting**, **inconclusive**, **invalid
comparison**, or **not tested**.

---

## PART 1 -- How to treat a result

These are the traps this project has actually fallen into. Each one produced a
wrong published-to-ourselves conclusion at least once.

### Arms and controls

- 🛑 **THE SAME CACHE TRAP TWICE: AN IDENTITY KEY MUST COVER EVERY INPUT THAT
  CHANGES THE WARM-UP, INCLUDING INPUTS THAT DO NOT LIVE IN `hp`.** 2026-09-15,
  caught at 2/40 runs in. `rank_clip` and `aug_rank_clip` hashed to ONE
  `base_model_id` across L80 and L90. The budgeted ranking loss cuts at the K-th
  order statistic with K derived from the CAP (`warmup.py` reads
  `config["constraint"][0]`), so the two caps train genuinely different models --
  but the cap lives in `config["constraint"]`, not in `hp`, and
  `compute_base_model_id` only hashes identity keys found in `hp`. L80 runs
  first, trains, caches; L90 then silently loads it.

  **Every L90 rank cell would have been a model trained for an 0.8 cut and then
  scored against an 0.9 allocation -- precisely the train/deploy mismatch this
  loss exists to remove.**

  🛑 **THIS IS THE DANGEROUS KIND.** The unpack defect recorded above crashed
  loudly and destroyed 48 runs; this one would have run clean, logged clean, and
  produced a full campaign of plausible, half-wrong numbers. It would also have
  corrupted the pre-registered reading: `rank_paired` marks an arm `(cap-inert)`
  when its probabilities do not move with the cap, and MISSION had recorded that
  as meaning "the loss never reached the model". Cap-inertness was GUARANTEED by
  the cache and said nothing about the loss. **A pre-registered inference rule is
  only as sound as the mechanism it assumes; when the mechanism changes, the rule
  must be retracted in place, not annotated.**

  `tests/test_rank_loss.py` already pinned that `rank_clip` cannot reuse `clip`'s
  warm-up via `rank_weight`, and passed throughout. Same trap, second instance,
  through an input the digest could not see.

  **How it surfaced: by COUNTING distinct warm-up identities in the generated
  campaign** (20 trained / 20 cached, then per-arm: each rank arm had 1 warm-up
  across 2 caps), while pricing an unrelated follow-up. Not by reading code, and
  no test or gate was looking for it.

  Fixed at `323edf44`: `gen_campaign` stamps `rank_cap_fraction` for rank arms
  only, so no existing digest moves; `warmup.py` REFUSES a stamp that disagrees
  with the cap it trains, so the two sources cannot drift;
  `tests/test_rank_cache_identity.py` pins both directions. **Verified in the
  produced predictions, not just the configs:** `aug_rank_clip` L80 vs L90 now
  differ on MobileNetV3 (`8c7a9316fa59` vs `e56ec773408d`) and RegNetY400MF
  (`fa08e8cc461c` vs `e227e7a25b52`), where the bug would have made them equal.

- 🛑 **A UNIT-TESTED MECHANISM CAN STILL BE 100% DEAD IN THE PIPELINE: 48 OF 120
  RUNS DIED ON A LINE ONE LEVEL BELOW THE THING UNDER TEST.** 2026-09-15.
  `tests/test_rank_loss.py` pins the budgeted ranking loss with 14 tests --
  competition at the cut, cutoff sensitivity, per-group independence, warm-up
  cache identity -- and every one passed while **every single `rank_clip` and
  `aug_rank_clip` run in all three Stage 1 campaigns failed in its first logged
  epoch**:

      ValueError: too many values to unpack (expected 2)

  The cause: the ranking arms pass `groups` to `make_dataloader`, so their
  loader yields 3-tuples; `warmup.py:208` hands that SAME loader to
  `compute_train_accuracy`, which read `for X, y in loader`. The loss was
  correct. The line that broke was in a different module, on a path no test ever
  executed with groups set.

  **Cost: 2h21m on three cards, and three campaigns that produced control arms
  only.** The trigger is `epoch < 3`, so failure was immediate and total -- it
  could have been caught by a single end-to-end warm-up on 32 random tensors,
  which is what `tests/test_warmup_executes_with_groups.py` now does.

  **The generalisable rule: a test of a COMPONENT is not a test of the PATH.**
  When a feature changes the SHAPE of something that flows through the pipeline
  -- an extra tensor in a batch, an extra column, an extra tuple field -- unit
  tests of the consumer prove nothing about the other consumers of the same
  object. Enumerate everything the changed object reaches (`make_dataloader`'s
  loader reached two consumers; only one was updated) and execute the real path.
  This is the same "verify by EXECUTING" lesson already recorded for stale
  bytecode and for restores, in a third costume.

  **Second-order lesson, same event.** The monitor reported the finished
  campaigns as `STALLED`. A failed run writes `config.json` but never writes
  `final_predictions_raw.csv`, so `done < total` stays true forever and a
  campaign that had FINISHED with 16 failures looked like a dead dispatcher --
  the wrong verdict, pointing at the wrong fix (relaunch the dispatcher rather
  than fix the bug). **A progress counter that only counts SUCCESSES cannot
  distinguish "still working" from "finished badly".** `scripts/rank_status.sh`
  now reads completion from the dispatcher's own `ALL DONE` line, counts
  failures, and prints distinct error signatures so a second bug cannot hide
  behind a known one.

- **A "null" arm can inherit lambda = 0 from a block and nobody notices.**
  `tralo_reseed` was built from `[constraint_phase, tralo_null, tralo_reseed]`
  and therefore inherited `lambda_step: 0.0`. For months `|tralo - tralo_reseed|`
  was used as an RNG-only floor while it was actually a treated-vs-untreated
  contrast. **Read the block composition, never the arm name.**
- **A named "#1" can be a dead arm.** Four separate TraLO first-place calls were
  produced by arms that were not running the treatment.
- **Every trained arm needs its own zero-constraint control** sharing the warm-up
  identity AND the schedule. `tralo_null` controls the plain column only.
- **`tralo_null` is cap-independent** (lambda = 0 means the cap never enters its
  training), so the L80 and L90 nulls are the SAME run. Share it; never count it
  twice.

### Counting n

- **Training is bit-deterministic given `(config, seed)`.** `gx2` re-ran 30 of
  `fm2_mn3`'s cells across different trees and code versions and reproduced every
  byte. An overlapping arm adds ZERO information. **De-duplicate by prediction
  hash before counting n.**
- **A cap level is not a seed. A copied warm-up is not a seed. A re-run is not a
  seed.** A shared warm-up implies Cov >= 0 between arms, so the independence
  formula over-estimates the sd.
- **Four seeds is a pilot.** Effects here run ~0.01 against a seed sd of ~0.011,
  so n = 4 carries roughly 15% power. Every "not significant" verdict taken at
  n = 4 is consistent with a real effect.

### Statistics

- **"1 of 158 rows clears 2 sd" IS the chance expectation, not a finding.** With
  a 4-seed mean against a per-seed sd, that bar is a Welch t >= 4 at df 3-6;
  chance alone yields ~1.1 rows at df 6 and ~4.4 at df 3. The honest statement
  was **0 of 158 resolve beyond chance**.
- **Label the sidedness of every sign test.** The project's headline p-values
  were one-sided and unlabelled; two-sided they are 0.125 / 0.0625 / 0.0078.
- **A ratio of medians of absolute differences cannot separate location from
  scale.** With sd ~5.9 items, a genuine +2-item dominance moves median |X| from
  3.98 to ~4.25. The correct reading is "unresolved at this resolution", never
  "refuted".
- **Report mean, seed sd, every seed delta, n, and a two-sided 95% t interval in
  native metric units.** A reseed spread is a diagnostic, not a confidence
  interval and not proof of equivalence.
- **Do not choose the inferential method after seeing which one declares a win.**

### Prizes

- **A nominal prize is not a reachable prize.** Counting wasted allocation slots
  and excluded true positives gives the headroom a PERFECT re-ranking would win.
  It is an upper bound, not a target: a slot held by an item the model scores at
  p(c) >= 0.99 and gets wrong cannot be evicted by any realistic training-time
  nudge. **Measured 2026-09-15: 39-46% of wasted slots at budget 30 are exactly
  that.** Always discount a headroom figure by the certain-wrong share before
  treating it as a target.

### Metrics

- **`items = d(F1) * (K+n) / 2` is exact PER CLASS only.** cc-F1 is macro over
  classes with different `(K+n)`, so the inversion lives on a two-quantum lattice
  and mis-states a single-class move by 1.116x / 0.906x.
- **Headroom figures do not transfer between datasets.** The "1.9-9.9 items"
  figure came from dermMNIST, which was removed for leakage.
- **Flips, raw counts and proximity to a cap are not classification quality.**
  Improving constraint satisfaction does not imply improving cc-F1; that link
  must be priced and tested, never assumed.
- **cc-F1 is the primary endpoint** and correlated metrics are not independent
  replications of it.

### Instruments

- **Gradient SIGNS cannot demonstrate that items compete.** Building the
  budgeted ranking loss, my own mechanism test asserted only that positives are
  pushed up and negatives down -- and it PASSED under a mutation that detached
  the cut, which removes the coupling entirely. Both a competitive loss and two
  independent one-sided pushes produce those signs. A claim of the form "these
  items interact" must be tested against an explicit non-interacting reference,
  not against the sign pattern it predicts.

- **An epoch chosen on the per-epoch curve is an ORACLE, not a method.**
  `scripts/epoch_curve.py` scores stored snapshots against the TEST set, so
  "stop at the best epoch" selects on the evaluation data and reports an
  optimistic upper bound. Quote it only as a BOUND -- it says what any stopping
  rule could win at most, and a small headroom over the final epoch closes the
  direction cheaply. A deployable stopping rule needs a held-out split that is
  not the test set, which is still an open question (see MISSION).
- **"It only observes" is not a property a gate can check.** The first version
  of the per-epoch trace scored inside the training loop, which put the test
  labels in `TrainInputs`. It was observational in fact and rejected anyway, by
  `test_no_methodology_reads_the_test_LABELS_except_to_count_them` and
  `test_train_inputs_do_not_expose_held_out_labels`. The gates were right:
  intent is not structure, and the only thing between an observation and a leak
  would have been that nobody edits the file. Storing probabilities and scoring
  offline makes the property structural. **Prefer the design that cannot leak
  over the design that promises not to.**

- **`scripts/deployed_h2h.py` is the maintained reporter.** Hand-rolled cc-F1 has
  produced three scorer bugs.
- **Training accuracy alone cannot diagnose test-cut saturation.**
  `gate:saturation` reads train accuracy and is a SCREEN, not a diagnosis. Pair
  it with the regime check at the real allocation cut.
- **A gate is not done until a mutation makes it FAIL**, and the restore is
  verified by EXECUTING it. Stale bytecode has faked a pass.
- **Six instruments hardcoded the 30-epoch protocol** and would have mis-read any
  campaign that did not use it. Assume a seventh exists.
- **A VIABILITY verdict is not a SCORE, and naming makes it one.** `~/triage.py`
  labelled cells KEEP / KILL while only ever testing whether a cell *can produce
  a valid measurement* (live fraction, model strength). Reported as-is on
  2026-09-15 it read as "TraLO is winning here", which was false in every cell.
  The column is now `viability (NOT a score)` with values VIABLE / UNUSABLE.
  **Who wins comes only from `scripts/deployed_h2h.py`.** Any screening column
  that could be mistaken for an outcome must be named so it cannot be.

---

## PART 2.1b -- THE GATE THAT WAS BLIND TO THE CAMPAIGN IT GATED (2026-09-16)

**`scripts/saturation_gate.py` skipped all 360 ranking runs and said nothing.**

The gate exists to catch exactly one condition: the boundary has frozen, so
whatever is pushing on it cannot reshape anything. It keyed off
`hyperparams.constraint_epochs` and contained `if con <= 0: continue`.

The ranking campaigns carry their objective in the **warm-up**, not in a
constraint phase, so every one of them sets `constraint_epochs = 0`. The gate
therefore skipped 100% of rank1/rank2/rank3 and printed **`no training_log.csv
matched`** -- which reads like a bad glob, not a verdict. It exited 0 through a
pipe. The campaign series was reported as gates-GREEN on a check that never ran.

Judged against the phase where the ranking loss was actually active (30 warm-up
epochs), all three backbones FAIL:

| cell | live window | live fraction | verdict |
|---|---|---|---|
| MobileNetV2/fmow2 | 3.0 of 30 | 10% | SATURATED |
| MobileNetV3/fmow2 | 2.5 of 30 | 8% | SATURATED |
| RegNetY400MF/fmow2 | 2.7 of 30 | 9% | SATURATED |

The gate requires >= 50%. **rank1, rank2 and rank3 -- 360 runs -- were run in the
dead regime**, the one this project established in August 2026 as where nothing
can happen ([[feedback-warmup1-is-the-only-regime]]: the regime is worth ~8 pp,
the method choice ~0.1 pp).

**This is the upstream cause of PARTS 2.2, 2.3 and 2.4.** The gradient being
uncertainty-weighted, the cap being destroyed by batch-level `round()`, and the
inversion set being identically empty on train are all downstream of a model that
has memorised the training set by epoch 6 of 30. **No reshaping of the loss could
have fixed any of it.**

**FIXED** (this commit), with four controls: the gate now falls back to
`warmup_epochs` when `constraint_epochs = 0` **and** `rank_weight > 0`; it
distinguishes "the glob matched nothing" from "every run was skipped, nothing was
checked"; and it names the runs it gated on the warm-up. Controls: (1) rank3 now
exits 1 SATURATED; (2) clipper-only arms correctly report no gateable phase --
a post-hoc arm has no in-training objective to gate; (3) the
`--constraint-epochs` override path is byte-identical to before; (4) rank2
reproduces independently.

**Augmentation is NOT the lever.** Measured per arm on the stored logs: augmented
arms reach a 3.00-epoch live window against 2.17 for plain `clip` -- 10% vs 7% of
a 30-epoch budget. It buys 0.8 epochs where 12 are needed.

**The generalisable trap.** A gate keyed on the name of a phase rather than on
*where the objective is active* will silently exempt any campaign that moves the
objective. Gate on the active phase, and never let "nothing matched" and "nothing
was checked" print the same message.

## PART 2.2 -- WHY THE RANKING LOSS FAILED: the gradient is UNCERTAINTY-weighted, not CUT-anchored

**The result first.** `rank3_*`, 120 runs, 3 backbones x 2 caps x 4 seeds, all
gates GREEN, 84 distinct models, zero cap collisions, measured 2026-09-16.

| contrast | cells positive | cell-mean gAP |
|---|---|---|
| `rank_clip` - `clip` | 4 of 18 | **-0.0090** |
| `aug_rank_clip` - `aug_clip` | 7 of 18 | **-0.0039** |

On the primary endpoint the ranking arm trails its control in **11 of 12
(backbone x cap x augmentation) cells**, mean cc-F1 delta about **-0.005**.
Against the measured null envelope (`focal_clip` -0.0048, `aug_clip` +0.0017)
`rank_clip` is MORE negative than either known null. **The budgeted ranking loss
does not supply the "which". It costs a little.**

**WHY, measured rather than argued** (`scripts/why_rank_failed.py`, 27 cells,
9794 items, within-cell standardised, gradients taken through the real loss on
real score distributions):

| | correlation with log per-item gradient |
|---|---|
| distance from the allocator's cut | -0.244 |
| softmax Jacobian p(1-p) | **+0.624** |
| **partial**, distance given Jacobian | **-0.145** |
| **partial**, Jacobian given distance | **+0.604** |

🔑 **Holding uncertainty fixed, distance from the cut explains almost nothing.
Holding distance fixed, uncertainty explains most of it.** The hinge is anchored
at the K-th order statistic in PROBABILITY space, but the gradient that reaches
the weights passes through the softmax Jacobian, and `p(1-p)` is a function of
the probability VALUE, not of its RANK. So the term the model actually receives
is an uncertainty weighting -- "push on the items you are unsure about" -- which
is close to entropy regularisation and is NOT the allocator's "which".

🛑 **THIS IS PART 2.1 IN A NEW COSTUME, AND THAT IS THE LESSON.** PART 2.1 proves
a count penalty fails because it reads the MULTISET of probabilities while the
allocator reads the RANKS. The budgeted ranking loss was designed to read ranks.
It does, in its forward pass. But **backward**, through the softmax, what lands
on the parameters is again a function of values. Anchoring a loss at an order
statistic is not sufficient; the GRADIENT has to stay rank-dependent too, and
here the Jacobian launders it back into a value-dependent quantity.

Two further mechanical defects, both exact rather than suggestive:

- **An irreducible floor.** `softplus(margin + t - s)` on probabilities bounded
  in [0,1] means the argument lies in [margin-1, margin+1], so softplus never
  reaches zero: best case 0.327 per term, worst 1.350. At most 76% of the loss
  is movable; the rest is a constant the optimiser carries. This is why rank
  arms log a total loss near 1.0 while controls log 0.009.
- **Built-in unsatisfiability.** `K = round(n_pos * cap_fraction)` with
  cap_fraction 0.9 forces **10.0% of true positives (123 of 1231) below the cut
  by construction**. They are penalised at every step and cannot be fixed -- and
  because the cut is deliberately NOT detached, lifting any positive raises `t`
  for all the others. The term grinds against itself.

**What this does NOT license.** The dose deficit recorded in PART 3 is real and
unfixed (fires on 8 of 139 train groups; trains a 2.3rd-of-12 order statistic to
serve a 41st-of-363 decision), so this is not a clean refutation of the ranking
CHANNEL. But it IS a specific, measured refutation of THIS loss, and the
diagnosis points at the surrogate rather than the dose: fixing the dose would
deliver more of a gradient that is still uncertainty-weighted.

**Saturation is NOT the explanation** -- 84% of the capacity sits at a decidable
cut (PART 3), so "nothing could have worked here" is not available.

## PART 2.3 -- THE CAP NEVER ENTERED THE RANKING LOSS: `round()` collapses on a batch slice

**Measured 2026-09-16 on `data/fmow2/oodslice/train_meta.csv`, the real train
split, by simulating the shuffled loader. This is mechanical, not statistical.**

`budgeted_rank_loss` takes its cut INSIDE a batch: `k = round(n_pos * cap_fraction)`
where `n_pos` is the positives of that class in that group's slice **of the
current batch of 64** (`src/training/rank_loss.py`, the `torch.topk(scores, k)`
line). fmow2 train has 17,670 items in 139 countries, so a group's slice is
small. Simulating 828 batches over 3 epochs:

| quantity | value |
|---|---|
| batches with a usable (group, class) cell | 96.5% -- **the term did fire** |
| usable cells per batch | 2.56 |
| mean group slice size in a usable cell | 12.4 items |
| cells holding exactly **one** positive | **49.1%** |
| cells where `K < n_pos` (the budget BINDS) at cap 0.90 | **16.7%** |
| same, at cap 0.95 | **0.1%** |
| same, at cap 0.80 | 30.7% |

`round(n_pos * 0.9) = n_pos` for every `n_pos <= 4`, and 83.3% of cells are that
small. **So in 83% of cells the budget was not binding and the loss reduced to a
plain "rank positives above negatives" pairwise term** -- precisely the
non-cutoff-sensitive surrogate that `rank_loss.py`'s own docstring identifies as
the wrong tool ("a plain pairwise or AP-style surrogate is NOT cutoff-sensitive").

**The decisive contrast.** How often do the two cap levels the campaign compared
produce an IDENTICAL cut?

| where the cut is taken | L90 and L95 give the same cut |
|---|---|
| per BATCH slice (what ran) | **83.0%** of cells |
| per FULL GROUP (what deploys) | **33.3%** of cells |

At group level the budget binds in 76.4% of cells at cap 0.90 and the two levels
differ by a mean of 2.14 items. **The cap information exists; the batch-level
`round()` destroys it before it reaches the gradient.**

**This supersedes nothing in PART 2.2 -- it is a second, independent defect.**
2.2 says the gradient that arrives is uncertainty-weighted rather than
cut-anchored. 2.3 says that for 83% of cells there was no cut to anchor to in the
first place. Either alone is sufficient to explain rank3's null on the cap axis.

**Not fixable by reparameterisation.** Carrying the cap as a rate
`q = K_full/n_full` and taking a quantile of the slice was tested and is WORSE
(L90 and L95 identical in 87.7% of cells): a 12-item slice admits ~12 distinct
cut positions, and the two caps differ by well under one item. The slice size is
the binding constraint, so the fix must enlarge the effective group -- a
full-group score buffer, or group-blocked batching -- not re-parameterise it.

## PART 2.4 -- THE ROOT CAUSE: the constraint is computed where its violation is EXACTLY ZERO

**Measured 2026-09-16 by loading the actual rank3 warm-up checkpoint
(`MobileNetV3_fmow2_5aa9cfb1e37a.pt`) and scoring the real train and test splits.
This is not a simulation and not an inference from a loss curve.**

Every constraint term this project has ever run -- the count penalty, all four
duals, `budgeted_rank_loss`, and the proposed inversion loss -- is computed on
TRAIN data. At the deployed cut (cap 0.90, classes 1/2/7):

| | TRAIN (17,670 items) | TEST (3,442 items) |
|---|---|---|
| accuracy | **0.9999** | 0.6322 |
| usable cells | 174 | 27 |
| **F**, false positives inside the budget | **0** | 246 |
| **V**, inversion set size `F*(n_pos-K+F)` | **0** | 5,507 |
| cells where the term is SILENT (V=0) | **174 of 174 (100%)** | 1 of 27 (3.7%) |
| F as a share of the budget | **0.0000** | 0.2220 |

**Every budget slot on the training set is filled by a true positive. There is no
violation to penalise, no inversion to swap, and nothing for a dual to
integrate.** The count family's violation is identically 0; the pairwise family's
gradient support is the empty set. Both are measuring a quantity that is zero
wherever they are allowed to look.

**This is not permanent, it is a decay.** The rank3 logs give train accuracy
0.803 / 0.877 / 0.942 at epochs 1-3, 0.984 at 6, 0.996 at 12, 0.999 at 30 (both
MobileNetV3 and MobileNetV2). Support scales with `(1 - accuracy)`, so the term
carried real signal for roughly **epochs 1-5 of 30** and was effectively silent
for the remaining 80% of the budget. The 719 train positives sitting outside the
budget are the ones the cap forces out by construction (PART 2.2); they generate
no inversions because no false positive is inside to swap with them.

**This SUBSUMES 2.2 and 2.3.** The gradient being uncertainty-weighted rather
than cut-anchored, and the cap being destroyed by batch-level `round()`, are both
real -- but they describe the shape of a gradient that is, for 80% of training,
multiplied by an empty support. **Fixing the loss FUNCTION cannot fix this. The
defect is which DATA the term is computed on.**

**What it reframes.** Train 0.9999 against test 0.6322 is a 37-point
generalisation gap, and the constraint term is being asked to repair an ordering
that is already perfect on every item it can see. This is consistent with every
null in PART 4 and with the theory: post-processing is optimal when the score is
Bayes on the observed data, and on TRAIN this score is effectively perfect.

**The direction this opens (see PART 5).** Compute the constraint term on a
held-out fold of the TRAIN GROUPS that cross-entropy does not fit. The model is
not memorised there, so `V > 0` and the term is alive for all 30 epochs. This
uses train labels only and touches no test label, so it is FRAMEWORK-legal;
`scripts/val_split.py` already exists. **This is a data-routing change, not a
loss-function change, and it is the first candidate in this project that is
motivated by a measured zero rather than by a theory of the surrogate.**

## PART 2 -- What is PROVED

From the adversarial-review theorem package (2026-09-02, three independent
reviewers, re-verified against source). This is the part of the old theory
document that survived. **These are proofs about the mechanism, not results.**

1. **No value-level selection.** The loss is invariant under permuting test
   items -- CE never reads them, and every `S = sum phi(p_ic)` is symmetric. So
   `L` is a function of the MULTISET while the allocator is a function of the
   RANKS. **Any procedure reading only values of `L` cannot prefer a correct
   ordering over the worst ordering with the same multiset.**
2. **The budget enters only as a scalar gain.** `grad_theta P = a(t) * V(theta)`
   with `a(t) = lambda(t) * psi'(S;K) >= 0` and `V(theta)` independent of K. The
   budget's information is reduced by the mechanism to a few non-negative
   scalars.
3. **Corollary: the four duals are one family.** TraLO, fioretto, alm and hounie
   differ only in the gain schedule and span the same cone. Their deployed
   differences SHOULD sit at the noise floor -- so measuring them tied is a
   confirmed prediction, not a disappointment.
4. **Binary and decoupled implies provable invariance.** The reordering channel
   exists only through softmax coupling (C >= 3) and weight sharing.
5. **Conditional harm lemma.** If current scores already exhaust the available
   label information, any label-blind reordering has non-negative expected
   deficit. Applied to this project's own numbers it predicted -28.3 against
   **-30.4 measured**.

**THE LITERATURE EXPLAINS THE NEGATIVE RESULT, AND NAMES THE ONE CRACK**
(verified via Semantic Scholar, 2026-09-15). For selection-rate constraints of
exactly our form -- "at most K(g,c) items predicted as c in group g" -- the
Bayes-optimal constrained classifier IS a group-wise thresholding rule on the
posterior. Our greedy top-K allocator is the plug-in version of that optimum,
which is why it keeps winning:

- **Zeng, Cheng & Dobriban (2024)**, "Bayes-Optimal Fair Classification with
  Linear Disparity Constraints via Pre-, In-, and Post-processing": via a
  Neyman-Pearson connection, the optimum is explicit group-wise thresholding
  with closed-form thresholds. Our caps meet the structural assumption.
- **Xian, Yin & Zhao (ICML)**, "Fair and Optimal Classification via
  Post-Processing": in the general multi-group multi-class case, post-processing
  a score attains the optimum **whenever that score is Bayes-optimal**.
- **Fukuchi (ICML 2025)**, "Meta Optimality for Demographic Parity Constrained
  Regression via Post-Processing": fair minimax optimality is achievable by
  post-processing; the explicit advice is to improve the underlying regression.
- **Alabdulmohsin (2020)** and **Zhang et al. (2026)** agree: in- and
  post-processing converge to the same Pareto frontier.

🔑 **THE CRACK: our score is NOT Bayes-optimal.** **Woodworth, Gunasekar,
Ohannessian & Srebro, COLT 2017**, "Learning Non-Discriminatory Predictors":
post-processing a FIXED, NON-Bayes predictor can be strictly suboptimal, and
in-processing is justified precisely through hypothesis-class restriction.
**So the theory permits a training-time win only by improving the SCORE** --
never by enforcing the count. PART 2.1 shows a count penalty cannot carry
score-improving information; `tralo_stab` showed even an informative weight
cannot. Both halves of our result now have citations.

**Dead by citation:** further Lagrangian/ALM variants (Chamon & Ribeiro
NeurIPS 2020 / IEEE TIT 2021 bound the duality gap and feasibility, never
accuracy over a feasible post-hoc rule; AL-CoLe ICASSP 2025 is one more
instance of our PART 2.3 family). Learning-from-label-proportions does NOT
transfer: LLP's counts are observed LABELS (new supervision), ours are caps on
PREDICTIONS (no supervision) -- the cleanest statement of why the penalty is
information-free.

**What is NOT proved, and was wrongly claimed:** the strong impossibility
conjecture is FALSE. A constructed two-cluster geometry with a shared linear head
gives a strict allocation improvement from one aggregate-count step, and TENT
(Wang et al., ICLR 2021) is a published counterexample. **Renounce impossibility;
claim the five results above.**

**Also corrected:** "both allocators are functions of the ranking" is false for
the LP, which maximises a linear functional of cardinal values. And the
transductivity argument does not hold -- the post-hoc allocator sees the budgets
too, so there is no information surplus. What differs at equal compute is the set
of REACHABLE RANKINGS, an optimisation-geometry object, not an information one.

---

## PART 3 -- What is MEASURED

### Settled, do not re-open without new evidence

- 🔑 **GATE 3 ON `rank3_*`: THE CAP BINDS EVERYWHERE, THE HEADROOM IS REAL, AND
  36% OF THE ALLOCATOR'S CUTS FALL IN A SATURATED REGION.** Measured 2026-09-16
  on 135 (backbone, cap, group, class) cells with `scripts/headroom.py`,
  aggregated by `scripts/gate3_summary.py`. This is the TEST-SIDE regime check
  RULESET section 2 requires and that train accuracy structurally cannot give.

  | backbone | cap | binds | outside_tp | selected errors | cut > 0.99 | median cut |
  |---|---|---|---|---|---|---|
  | MobileNetV2 | L80 | 27/27 | 458 | 211/984 | 8/27 | 0.915 |
  | MobileNetV3 | L80 | 27/27 | 459 | 212/984 | 11/27 | 0.981 |
  | MobileNetV3 | L90 | 27/27 | 382 | 259/1108 | 10/27 | 0.866 |
  | RegNetY400MF | L80 | 27/27 | 478 | 231/984 | 11/27 | 0.921 |
  | RegNetY400MF | L90 | 27/27 | 411 | 288/1108 | 8/27 | 0.864 |

  **The regime is informative, and the task is real.** The cap binds in 135 of
  135 cells (`emitted == K`, no slack), 380-480 true positives sit OUTSIDE the
  cut as correctable headroom, and 21-26% of everything the allocator selects is
  wrong. There is plenty to win.

  **But the cut-probability distribution splits the cells in two:**

  | where the cut falls | cells | share |
  |---|---|---|
  | contested, < 0.2 | 29 | 21.5% |
  | middling, 0.2-0.8 | 23 | 17.0% |
  | confident, 0.8-0.99 | 35 | 25.9% |
  | **SATURATED, > 0.99** | **48** | **35.6%** |

  In roughly a third of cells the allocator cuts through probabilities above
  0.99, where candidates are numerically indistinguishable. **No re-ranking loss
  can act there**, however live the training loop looks. In ~38% the cut is
  contested and there is something to learn.

  ⚠️ **This REFINES, and partly corrects, the train-side saturation screen.**
  Train accuracy reaches 0.95 by epoch 4-7 and 0.99 by epoch 7-13 for every arm,
  which invites the conclusion that the boundary is uniformly frozen. The
  test-side cut says otherwise: the boundary is frozen in about a third of the
  cells and contested in about a third. "The model memorised the train set" and
  "the deployed cut is undecidable" are DIFFERENT claims, and only the second
  bounds what a ranking loss can achieve. Do not quote the first as evidence of
  the second -- that is precisely the substitution RULESET section 2 warns
  against.

  ⚠️ **CORRECTED IN PLACE 2026-09-16, same day: the 35.6% is a share of CELLS,
  not of the prize, and quoting it as the ceiling OVERSTATES it by half.** The
  saturated cells are the SMALL ones -- median K 11 against 33 for the rest --
  so weighted by capacity (`scripts/where_saturated.py`):

  | where the cut is | cells | med K | % of slots | outside_tp | selected errors |
  |---|---|---|---|---|---|
  | contested < 0.2 | 29 | 51 | **34.9%** | 661 | 342 |
  | middling 0.2-0.8 | 23 | 46 | 26.7% | 628 | 362 |
  | confident 0.8-0.99 | 35 | 18 | 22.1% | 477 | 245 |
  | SATURATED > 0.99 | 48 | 11 | **16.2%** | 422 | 252 |

  **The real ceiling is ~16% of slots, ~19% of the correctable headroom, ~21% of
  the errors inside the selection. About 84% of the capacity sits at a cut a
  model could in principle re-rank**, and the single largest band is the
  CONTESTED one at 34.9% of slots.

  🛑 **This closes an escape hatch before it can be used.** If the Stage 1 gate
  comes back null, "the cuts were saturated so nothing could have worked" is NOT
  available as the explanation -- five sixths of the prize sits at a decidable
  cut. A null would have to be explained by the dose deficit already measured
  (8 of 139 groups, a 2.3rd-of-12 order statistic), or by the idea itself.
  Recording this NOW, before the numbers land, is the point.

- 📏 **THE gAP NOISE ENVELOPE, MEASURED ON TWO KNOWN-NULL CONTRASTS -- and an
  isolated `|mean|/sd > 2` cell APPEARS IN BOTH.** 2026-09-16, on `rank1_*`'s 72
  surviving control runs (3 backbones x 3 constrained classes x 4 seeds, scored
  with `scripts/rank_paired.py`). Both contrasts are arms whose only difference
  is a training-time lever, so they say what this instrument does when there is
  little or nothing to find:

  | contrast | cells positive | cell-mean | largest single cell |
  |---|---|---|---|
  | `focal_clip` - `clip` | 2 of 9 | **-0.0048** | -0.0261 at `|mean|/sd` 2.33 |
  | `aug_clip` - `clip` | 4 of 9 | **+0.0017** | +0.0316 at `|mean|/sd` 2.41 |

  **Two things follow, and the second is the load-bearing one.**

  1. Neither focal loss nor augmentation improves the SCORE that the allocator
     reads. Focal is directionally negative (7 of 9 cells); augmentation is a
     null. Augmentation was added to fight the saturation the ledger records --
     it may still do that, but it does not buy ranking quality.

  2. 🛑 **An isolated cell at `|mean|/sd` ~ 2.4 is what a NULL looks like here.**
     Both contrasts produced one, with opposite signs, at n=4 seeds. So when the
     Stage 1 ranking gate is read, a single strong-looking cell is NOT evidence
     -- only the COUNT of cells and their agreement across backbones is. This is
     exactly the shape that manufactured earlier retracted headlines in this
     project, and it is now measured rather than asserted.

  Cell-means of +-0.005 with per-cell sd of 0.01-0.04 are therefore the
  background, and the pre-registered `>= 7 of 9` screen in MISSION should be
  read against that, not against zero.

- 🟡 **THE STAGE 1 RANKING LOSS IS LIVE BUT NARROW: IT FIRES ON 8 OF 139 TRAIN
  GROUPS, AND TRAINS A 2nd-OF-12 ORDER STATISTIC TO SERVE A 41st-OF-363
  DECISION.** Measured 2026-09-15 on fmow2, before any `rank_*` run completed,
  by simulating the warm-up sampler against `train_meta.csv` (`rank_dose.py`,
  `rank_scale.py`). `make_dataloader` shuffles uniformly and is group-blind, so
  a batch of 64 spread over 139 country groups holds ~12 items of the largest
  group and fewer of the rest. With `rank_min_group = 8`:

  | quantity | value |
  |---|---|
  | usable (group, class) terms per batch | mean 2.50 of 417 possible |
  | batches with NO ranking gradient at all | 3.4% |
  | groups that ever contribute | 8 of 139 (USA 55%, FRA 22%, ITA 6% of batches) |
  | trained cut | 2.3rd order statistic of 12 items, k=1 in 49% of terms |
  | deployed cut | 41st order statistic of 363 items |
  | cut DEPTH, train vs test | 0.208 vs 0.148 (ratio 1.41 means, 1.52 medians) |

  **What this does and does not license.** The loss is NOT inert -- that was the
  first thing checked, because five flags in this project have died in exactly
  that shape. It is aimed at roughly the right quantile (within ~1.4x), so it is
  the intended mechanism. But it is estimated from 29x fewer items than the
  decision it serves, the `max(1, .)` floor pins k=1 in half the terms and so
  biases the trained cut SHALLOW, and 131 of 139 groups never enter the
  gradient. **Therefore: a positive gAP effect from `rank_clip` is trustworthy.
  A NULL is NOT informative about the ranking channel** -- it cannot separate
  "the channel does not help" from "this estimator is too noisy and too narrow
  to deliver it". This is the `hounie_rcl` 1%-dose trap and the `mc29` 100x-dose
  trap in a third costume, caught this time BEFORE the runs landed rather than
  after.

  The fix, if a null comes back, is a sampler change and not a loss change:
  group-batched sampling (draw each batch from one or a few groups) would put
  ~64 items of a single group in front of the cut and raise k from 2 to ~10,
  restoring both the width and the sample size. That is a relaunch, hence a
  compute-budget decision, hence a question for the user rather than a unilateral
  change.


- 🔴 **THE ENVIRONMENT IS EXONERATED. IN A BOUNDARY THAT NEVER FREEZES, WITH THE
  CAP BINDING AND THE PRIZE REACHABLE, TraLO STILL LOSES TO ITS OWN NULL.**
  `small60` (SmallCNN 100k params, fmow2, budget 60, 32/32 complete), measured
  2026-09-15. This is the regime every previous excuse asked for:

  | precondition | `small60` |
  |---|---|
  | boundary still moving | **live fraction 102%** -- train acc never reaches 0.95 in 59 constraint epochs |
  | cap actually binds | `emitted == K` in EVERY group x class row (FRAMEWORK gate 3) |
  | prize is reachable | `certain%` = **0** -- no wasted slot is held by a confident-and-wrong item |
  | cuts are contested | cut probability 0.12-0.90, not a wall of 0.99 |
  | headroom exists | `outside_tp` 2-69 correctable true positives per cell |

  Result, seed-paired cc-F1, TraLO minus its phase-matched zero-constraint null:
  **L80 -0.00957 (4/4 seeds negative), L90 -0.00925 (3/4 negative).** It also
  trails `clip` in both caps (-0.0189, -0.0169). The constraint is not merely
  failing to help in the ideal regime; it is costing something.

- 🔴 **"THE MODEL IS TOO GOOD" IS REFUTED TWICE.** `scratch60` (MobileNetV3,
  `pretrained=false`, 32/32) drops cc-F1 to 0.475-0.495 and TraLO still loses:
  L90 vs `clip` **-0.0258, CI [-0.0431, -0.0085], excludes zero**, and it loses
  to its own null (-0.0073). Caveat recorded honestly: `scratch60`'s live
  fraction is only 33%, so it FAILS `gate:saturation` and is the weaker of the
  two controls. `small60` above is the one that carries the argument, because it
  is the only campaign in the whole corpus that passes the live-boundary test.

- 🔴 **THE PER-ITEM INFORMATION CHANNEL IS CLOSED. `tralo_stab` FAILS ITS
  PRE-REGISTERED BAR.** `stab8` (72/72, budget 8, MobileNetV3), 2026-09-15.
  The weighting is real, not another inert flag: `gate:weight_bites` PASSES with
  median weight cv **0.83 / 0.88**, uniform controls at exactly 0.0, and 8/8
  twin pairs byte-different. Pre-registered bar was "clear plain TraLO's
  +0.001..+0.004 tie with a CI excluding zero". Measured, seed-paired,
  `tralo_stab` minus `tralo_null` (derived exactly from the reporter's paired
  deltas against `tralo`):

  | cap | seed deltas | mean |
  |---|---|---|
  | L80 | -0.0060, -0.0078, +0.0157, -0.0032 | **-0.0003** |
  | L90 | -0.0001, -0.0113, -0.0154, +0.0010 | **-0.0064** |

  Both negative. The bar is not cleared and it is not close. **A weight carrying
  correctness information the score lacks (knn_agree AUC 0.87 vs 0.68) still
  does not make the constraint useful** -- which is the sharpest available test
  of the harm lemma's one escape route, and it closes it.

- 🔴 **THERE IS NO EPOCH AT WHICH THE CONSTRAINT HELPS -- NOT EVEN WHILE CE IS
  STILL LIVE.** `trace30` (48/48, budget 30), per-epoch curve from stored
  probability snapshots, `tralo` minus `tralo_null`, per-epoch 95% Student-t
  intervals over 4 seeds.

  | cap | epochs whose CI excludes 0 | chance alone | mean over epochs | typical 95% half-width |
  |---|---|---|---|---|
  | L80 | 2 of 29 | ~1.5 | -0.00005 | 0.0162 |
  | L90 | 4 of 29 | ~1.5 | -0.00185 | 0.0184 |

  **L80 is indistinguishable from noise** (2 of 29 is chance; the mean over
  epochs is -0.00005). **L90 is a weak NEGATIVE drift, not symmetric noise:**
  all four excluding epochs are negative (-0.0059, -0.0083, -0.0148, -0.0195)
  and the largest is the final epoch.

  Two things this kills. First, the saturation story: on L90 the earliest epoch
  reaching significance is **epoch 4, train accuracy 0.92, still live -- and it
  is NEGATIVE** (-0.00592 +- 0.00460). There is no early phase in which the
  constraint is helping. Second, the stopping rule: the ORACLE best epoch is 28
  in one cap and 14 in the other, inconsistent and both deep in saturation, and
  its headroom (+0.016 / +0.030) sits at or below the per-epoch 95% half-width
  (0.016 / 0.018). The headroom IS the noise envelope, so there is nothing for a
  stopping rule to find.

  ⚠️ Corrected 2026-09-15, same day: an earlier version of this entry said
  "noise at every epoch". That is right for L80 and wrong for L90, where the
  drift is weakly negative. The correction does not change what it closes.

- **AT A LIVE BOUNDARY, TraLO STILL DOES NOT BEAT THE CLIPPERS, AND DOES NOT
  BEAT ITS OWN NULL.** `scripts/deployed_h2h.py` on the two COMPLETE budget-8
  screens (`scr_MobileNetV3`, `scr_RegNetY400MF`, 56/56 each, 4 seeds, fmow2,
  L80_G95 and L90_G95), measured 2026-09-15. This is the head-to-head the
  frozen-boundary story predicted we would win, run in the regime it asked for,
  and it is lost.

  cc-F1, seed-paired, TraLO minus the comparator (negative = TraLO loses):

  | cell | vs `tralo_null` | vs `clip` | vs `focal_clip` | vs `aug_clip` |
  |---|---|---|---|---|
  | MN3 L80 | +0.0007 | +0.0092 | +0.0113 | **-0.0124** |
  | MN3 L90 | +0.0030 | +0.0073 | +0.0110 * | **-0.0099** |
  | RegNet L80 | -0.0026 | -0.0088 | -0.0105 | **-0.0298** * |
  | RegNet L90 | +0.0042 | +0.0015 | -0.0030 | **-0.0197** * |

  `*` = 95% CI excludes zero, NOT multiplicity-adjusted over 24 comparisons.

  Three things this settles:

  1. **`aug_clip` is the best arm in all four cells.** Post-hoc clipping plus
     augmentation. The win is augmentation, and it is available to every method,
     TraLO included -- it is not a TraLO result.
  2. **The constraint buys nothing over its own phase-matched control.** TraLO
     minus `tralo_null` is +0.0007, +0.0030, -0.0026, +0.0042: every CI spans
     zero, and the sign is not even consistent. The whole constraint phase is
     worth less than the seed noise it runs in.
  3. **The only CI favouring TraLO is MN3 L90 vs `focal_clip`**, +0.0110 with CI
     [0.0002, 0.0218] -- 1 of 24 unadjusted comparisons, i.e. what multiplicity
     produces on its own. It is not a win.

  Three CIs exclude zero AGAINST TraLO, all against augmented arms on RegNet.
  **The live boundary was the last untested precondition, and supplying it did
  not change the verdict.** Whatever is wrong with TraLO is not the budget.

  Reading limit: the reporter emits seed-paired deltas only against `tralo`, so
  `aug_tralo` minus `aug_tralo_null` is available here as a difference of means
  (+0.0023, +0.0029, -0.0008, +0.0055) and NOT seed-paired. It must not be
  quoted as a paired effect.

| # | Finding | Consequence |
|---|---|---|
| 1 | **Every cell saturates in 2-4 epochs.** 5/5 cells, 3 backbones, 2 datasets. Augmentation doubles it to ~5, and no more. | The boundary is frozen for most of any long budget. |
| 2 | **The four duals share one per-item gradient**, read from source, not simulated. Dose is equal: 232/232 steps, 100%, all arms. | The historic dose gap explains nothing. No better dual rule exists to find. Matches PART 2.3. |
| 3 | **Eviction given the probabilities is already ~optimal.** A gradient push lands on 87% of the provably optimal set; closing the rest is worth 0.0002. | A margin-aware soft count is CLOSED. The prize is not in the loss shape. |
| 4 | **The exact allocator makes real results WORSE** (-0.01 to -0.04 accuracy, -0.02 to -0.08 cc-F1, 14/14 cells) because models run at 0.92 confidence against 0.60 accuracy. | The allocator "fix" is CANCELLED. Greedy's suboptimality is protective. |
| 5 | **The constraint obeys but does not convert.** Every dual beats both clippers on native obedience (excess 279-326 vs 381-413). | Obedience is not the missing piece; the metric does not reward it. |
| 6 | **On a frozen boundary the constraint makes the RANKING worse.** 15 of 16 seed-deltas negative, gAP(tralo) - gAP(tralo_null) = -0.007 to -0.030. gAP is allocation-free, so this is the model, not the allocator. | The damage is upstream of the metric and ~2x the cc-F1 damage. Predicted by PART 2.5. |
| 7 | **The cap IS binding and there IS headroom.** At the real allocation cut, `filled == slots` in every cell; ~1,000 of ~4,400 slots hold wrong items while ~1,500 true positives sit outside. | This is a real constrained problem with a real prize, ~20-25% of the deployed budget. |
| 8 | **The paper's second phase has never run.** 2,563 runs, 0 freezes: lambda and rho never freeze, so the described tail of training does not exist. The obvious repair, `tralo_dualprop`, was already built and is a null. | The implementation does not match the method description. |

### Measured and directional, not settled

- **Shorter budgets stop the damage.** gAP effect by live fraction, de-duplicated:
  10.3% -> -0.0174, 13.8% -> -0.0274, 30.0% -> -0.0089, 60.0% -> **+0.0079**.
  Monotone on a fixed backbone, 3 of 3, but Spearman p = 0.20 over four
  campaigns and the one break is also the only MobileNetV2.
- **The same flip appears on the DEPLOYED selection**, independently of gAP: at
  the real allocation cut the constraint removes wasted slots at a 6-epoch budget
  (-33, -21) and adds them at 30 epochs (+29, +88).
- **The augment x constraint interaction is positive and now REPLICATES across
  backbones, still underpowered.** First seen on one backbone (+0.00892 +-
  0.01142, 3/4 seeds, p = 0.22; focal flat at -0.00157, the predicted
  dissociation -- augmentation raises the live fraction, focal only enlarges the
  gradient). The budget-8 screens (2026-09-15) reproduce the sign in **5 of 6**
  computable cells across THREE backbones: +0.00744, +0.01036, +0.00835,
  +0.00423, +0.00941, with ViTB16 the lone negative (-0.00328) and ViTB16 is the
  cell that FAILS the saturation gate. This is the only direction in which the
  constraint is not harmful. It is NOT settled: at ~4 seeds the power against
  effects this size is ~15% (PART 1, Statistics), so ~13 seeds are needed.
- **A LIVE boundary does not rescue the plain column -- the freeze was not the
  whole story.** Four budget-8 screens on fmow2 (`scr_MobileNetV2`,
  `scr_MobileNetV3`, `scr_RegNetY400MF`, `scr_ViTB16`), measured 2026-09-15 at
  live fractions of 72 / 55 / 69 / 37 %, i.e. the boundary is demonstrably alive
  for most of the constraint phase in three of the four. The plain constraint
  effect on gAP is nonetheless **NEGATIVE in 7 of 8 cells**:

  | backbone | cap | constraint effect on gAP | n |
  |---|---|---|---|
  | MobileNetV3 | L80 | -0.00530 +- 0.00865 | 3 |
  | MobileNetV3 | L90 | +0.00064 +- 0.02027 | 3 |
  | MobileNetV2 | L80 | -0.01634 +- 0.01168 | 3 |
  | MobileNetV2 | L90 | -0.00177 +- 0.00265 | 2 |
  | RegNetY400MF | L80 | -0.00270 +- 0.00360 | 3 |
  | RegNetY400MF | L90 | -0.00413 +- 0.02061 | 3 |
  | ViTB16 (gate FAIL) | L80 | -0.03540 | 1 |
  | ViTB16 (gate FAIL) | L90 | -0.02736 | 1 |

  Every sd is 2-3x its mean, so no single cell is significant; the sign
  consistency over three backbones and two caps is the signal. **What this costs
  us:** the live-boundary account said the 30-epoch damage happened because the
  boundary had frozen by epoch 3-5 and the constraint was pushing a dead
  surface. Short budgets were the repair. At 55-72% live the damage persists, so
  a frozen boundary is at most a PART of the mechanism. It does not overturn the
  budget-30-vs-6 flip above (different campaigns, different live fractions); it
  says the flip is not explained by liveness alone. gAP is allocation-free, so
  these numbers are the MODEL moving, not the allocator -- the boundary IS being
  reshaped, and in the plain column it is being reshaped for the worse.
- **Contested cuts rise as the budget falls.** The share of allocation cuts whose
  marginal probability sits in [0.05, 0.95] goes 36-43% at budget 30 to 44-50% at
  budgets 6-11. Half of all cuts sit where the model has collapsed and no
  re-ranking can move them.
- **Between a third and a half of the prize is unreachable, and the share falls
  with the budget.** At the real cut, the proportion of wasted slots held by
  items the model scores p(c) >= 0.99 AND gets wrong:

  | campaign | budget | L80_G95 | L90_G95 |
  |---|---|---|---|
  | `fm2_mn3` | 30 | 46% of 208 | 39% of 262 |
  | `live6b` | 6 | 33% of 212 | 28% of 265 |

  So LEDGER fact 7's 20-25% nominal prize is really ~12-15% reachable at budget
  30 and ~15-18% at budget 6. Shorter training leaves the model less certain and
  more of the headroom in play -- a THIRD independent measurement pointing the
  same way as the gAP effect and the deployed-selection sign flip, and it is a
  unit-free proportion rather than a metric delta. **4-seed pilot; no interval
  quoted.** `~/reachable.py` on dsisco02.

  ⚠️ The same probe's `gap1` (probability distance from the weakest selected
  error to the strongest excluded true positive) is 0.013-0.014 at budget 30 and
  0.024-0.050 at budget 6. **Do not read that as the short budget being harder.**
  Raw probability gaps are not comparable across regimes when calibration itself
  differs -- an overconfident model compresses every gap toward zero while its
  ordering is more entrenched, not less. The certain-wrong SHARE is the
  scale-free quantity; the gap is not.

- 🔑 **THE INFORMATION THE CONSTRAINT LACKS EXISTS, AND IT IS NOT IN `p`.**
  Read from source: the soft count is an UNWEIGHTED sum of probabilities
  (`chunk_eff = chunk_proba`; `chunk_global = chunk_eff.sum(dim=0)`,
  `tralo/train.py:240`), so `dL/dp_i(c) = psi'(S_c)` is identical for every item
  in the scope and the only per-item differentiation is the softmax Jacobian
  `p_i(1-p_i)`. Everything the constraint knows about item i is `p_i(c)` -- the
  quantity the allocator already ranks by. **That one line is the bottleneck
  PART 2.1 proves cannot re-order.**

  Measured 2026-09-15 on stored embeddings, fm2_mn3 / clip, predicting WRONG
  among items placed in a capped class at p(c) >= 0.99 (2,595 and 2,671 such
  items, 15% wrong):

  | signal | L80 AUC | L90 AUC | from `p`? |
  |---|---|---|---|
  | `knn_agree`, 20-NN argmax agreement in embedding space | **0.869 +- 0.028** | **0.874 +- 0.033** | no |
  | `knn_pc`, neighbours' mean p(c) | 0.790 | 0.790 | no |
  | `knn_dist` inverted, local density | 0.705 | 0.705 | no |
  | `centroid_cos` | 0.656 | 0.648 | no |
  | `margin` (control) | 0.678 | 0.678 | **yes** |

  Neighbourhood agreement identifies the confidently-wrong far better than
  anything derivable from `p`. ⚠️ The margin control did NOT land at 0.5 as
  predicted -- p-derived quantities carry more than expected among the confident
  set -- so the claim is the GAP (0.87 vs 0.68), not that `p` is uninformative.
  Unit is the seed; 12 cells = 3 capped classes x 4 seeds per cap.
  `~/separability.py` on dsisco02.

  **This is the escape route the harm lemma leaves open.** PART 2.5 assumes
  current scores exhaust the available label information. They do not.

- **cc-F1 still goes the wrong way.** In `live11`, where the per-column nulls
  exist, `aug_tralo_null` (0.6411) beats `aug_tralo` (0.6285). The augmentation
  does the work, not the constraint.
- **The apparent TraLO lead in `gx2` does not survive its own control**:
  `aug_tralo` leads `aug_clip` by 0.0012, far inside seed noise, and that
  campaign has no `aug_tralo_null` to attribute it to the constraint at all.

### Datasets

- **`fmow2` is the dataset.** It passes all 8 conditions of
  `scripts/candidate_gate.py`.
- **iwildcam is RETIRED** -- 2 of 8 conditions; its per-group label shift IS its
  sparsity.
- **The original `fmow/oodslice` is WITHDRAWN** -- a basename join collapsed
  distinct AOIs, mis-joining 16.4% of records.
- **dermMNIST is REMOVED for leakage** -- 38.7% of test lesions appear in train.
- **`bcn` fails the balance condition** -- five tail classes holding 21% of the
  data between them.

---

## PART 4 -- Closed and rejected

- **Early stopping / per-epoch boundary selection -- CLOSED 2026-09-15.** The
  per-epoch curve shows noise at every epoch and an oracle that picks a
  different late epoch in each cap. No stopping rule can find those without the
  test labels, so there is nothing for one to exploit. The instrument
  (`epoch_trace` + `scripts/epoch_curve.py`) is kept: it is cheap, it rides
  along on every tralo arm, and it is how this was closed in one campaign.
- **`tralo_stab`, the weighted soft count -- CLOSED 2026-09-15.** Built, gated,
  proven live (weight cv 0.83-0.88, twins differ 8/8), and it fails its
  pre-registered bar in both caps. The escape route the harm lemma left open is
  now measured and shut.
- **"The recipe saturates, so the constraint never had a chance" -- CLOSED
  2026-09-15** by `small60`, which never saturates and still loses. Retained as
  a true description of the OTHER campaigns; retired as an EXPLANATION for the
  damage.

Historical hypotheses and their disposition after the 2026-09-14 evidence reset.
Historical results can identify risks and tests; they cannot establish the new
campaign's success or failure.

| Hypothesis or intervention | What the record supports | Disposition |
|---|---|---|
| Aggregate counts cannot reorder examples | Contradicted by the shared-parameter gradient | **False general claim.** Reordering may help or harm. |
| Changing penalty shape, weights, units or dose must improve quality | Several settings improved a proxy without improving deployed quality; early tests include invalid regimes | No default repeat. Requires a distinct mechanism and a valid regime. |
| More constraint satisfaction means better classification | Interventions changed satisfaction without improving selection | **Wrong implication.** Measure cc-F1 and collateral separately. |
| Ranking/pairwise methods are universally rejected | `rankpair` was called rejected without a run receipt | **Missing evidence is not a negative experiment.** Still needs justified headroom. |
| Selective head / joint objective | Losses and collapse in tested configurations, older datasets | Historical unfavorable evidence, not a universal closure. Not the next default. |
| Class-balanced and logit-adjusted CE on a uniform train prior | Algebraic equivalence to CE under the stated definitions | **Conditional identity holds.** Verify the actual prior and gradients, not folder names. |
| A constant scalar before gradient normalization changes the update | Positive scaling cancels under exact normalization | **Conditional algebraic identity.** Scope weights and optimizer state need separate analysis. |
| Snapshot averaging shows a TraLO-specific gain | Zero-constraint twins shared the gains | Possible general training improvement, not constraint novelty. Give averaging to every control. |
| Below a reseed floor means no effect | The floor estimates run-to-run spread, not uncertainty of the paired mean | **Inconclusive.** Extra seeds improve mean precision. |
| cc-F1 is invalid because it shares counts with TP | At fixed emitted counts F1 is a weighted function of TP | cc-F1 remains the primary endpoint. |
| A harder dataset or ALM's success guarantees a TraLO win | Neither implication follows | Inspect learnable errors at the actual cuts. |
| Graph diffusion / transductive geometry | Helps the NULLS most | **Null.** Baseline improvement, not a constraint effect. |
| `class_balanced`, `logit_adjust`, `graph_probe --dump` | Inert flags: no behavioural difference | **Inert.** Verify a flag changes behaviour before running an arm on it. |
| The exact (LP) allocator | Loses accuracy and cc-F1 in 14/14 real cells | **Cancelled.** |
| Margin-aware soft count | A push already lands on 87% of the optimal set | **Closed.** |
| Weight decay, label smoothing | -- | **Rejected by the user as cheating.** Not to be used. |

---

## PART 5 -- Open, and not yet tested

- 🟢 **A GROUP-DISJOINT VALIDATION SPLIT IS CONSTRUCTIBLE, AND IT IS THE ONLY
  THING STANDING BETWEEN `epoch_curve` AND A REPORTABLE STOPPING RULE.**
  Measured 2026-09-15 (`scripts/val_split.py`). `data/fmow2/oodslice/` ships
  train and test only -- there is no validation split anywhere on this dataset,
  so every curve this project has ever looked at was read against the test set.
  Train and test are group-disjoint by construction (139 countries vs 10, zero
  overlap), which means a row-shuffled val split would be the WRONG instrument:
  it would leave the same countries on both sides and measure an easier problem
  than deployment poses.

  35 train countries carry >= 80 items, enough to carve a group-disjoint split
  that imitates the test profile -- ARG, BRA, CHE, CHL, DEU, KEN, KOR, PER, SVN,
  SYR gives 2920 items in 10 groups against test's 3442 in 10, constrained-class
  shares matching to a total mismatch of 0.021, leaving 83% of train.

  **Why this is worth the compute.** `epoch_curve.py` already carries the
  warning that its best epoch is an oracle. If the constraint's contribution
  really does flip sign at CE saturation, a val-selected stopping epoch is a
  legitimate, deployable method -- and it is the first candidate in this project
  that would produce a POSITIVE reportable result rather than another null. It
  costs a retrain of every arm on 83% of the data, so it is a compute-budget
  decision and needs to be asked, not taken.

  🛑 The val split must be carved from TRAIN countries only, and its labels must
  never reach a gradient -- it selects an epoch, nothing else. FRAMEWORK forbids
  individual evaluation labels entering gradients, checkpoint selection or
  hyperparameter search, and a val split does not become exempt by being called
  validation: it is legitimate for CHOOSING among already-trained checkpoints
  only because it is disjoint from the test set, not because it is unlabelled.


- **`tralo_stab` -- the weighted soft count -- is BUILT, GATED, and NOT RUN.**
  `constraint_weight: knn_disagree` replaces `S_c = sum_i p_i(c)` with
  `S_c = sum_i w_i p_i(c)`, w = label-free neighbourhood disagreement, mean 1
  per local group, applied to the counting pass and the gradient pass alike.
  Hard counts stay unweighted. `uniform` is exactly ones and is bit-identical to
  the reference arm, dtype included. **What is verified:** the unweighted
  per-item gradient really is constant across items (so PART 2.1 still describes
  the code), the weighted one is exactly `w_i * psi'(S_c)`, the weight lands on
  the item whose neighbours disagree, and `gate:weight_bites`
  (`scripts/weight_bites.py`) rejects an inert arm, a perturbed control, a mixed
  declaration, and an arm that reproduces its twin byte-for-byte.
  **What is NOT verified:** it has never executed inside a real training loop.
  No arm is declared in `configs/protocol.yml` and nothing has been launched.
  The first campaign must pass `gate:weight_bites` with
  `--pair tralo_stab:tralo` before any number from it is read.

- **The budget-permuted twin.** Identical code and schedule, budgets permuted
  across groups within a class. By the scalar-gain lemma (PART 2.2) this changes
  only the gain trajectory and leaves the field direction untouched, so it is the
  closest matched control obtainable. **If the effect survives permutation the
  budgets are not doing the work and both the transductive claim and the
  constraint claim fail. If it dies, the constraint claim survives its strongest
  available test.** Cheap. Never run.
- 🔑 **`tralo_stab` -- a stability-weighted soft count. The one falsifiable TraLO
  modification currently on the table**, stated in the form FRAMEWORK requires.

  **Mechanism.** Replace the unweighted soft count with
  `S_c = sum_i w_i * p_i(c)`, where `w_i` is the item's label-free neighbourhood
  DISagreement (`1 - knn_agree`, computed on the model's own test embeddings each
  constraint epoch, renormalised to mean 1 per scope so the dose is unchanged).

  **Derivative.** `dL/dp_i(c) = psi'(S_c) * w_i`, against `psi'(S_c)` today. The
  per-item weight stops being a function of `p_i` alone, which is exactly the
  condition PART 2.1 identifies as necessary to re-order. Eviction pressure
  concentrates on items whose neighbourhood disagrees with them -- measured AUC
  0.87 for being wrong.

  **Expected log signature.** Same dose (attempted == applied == constraint
  epochs); `w` mean 1.0 and sd > 0 per scope; the evicted set diverging from the
  unweighted arm's by more than the RNG floor. If `w` sd is ~0 the arm is inert
  and the run is void.

  **Matched control.** `tralo_stab_null` at lambda = 0. Note the weight can act
  ONLY through the constraint, so a null is unaffected by construction -- unlike
  graph diffusion, which modified predictions and therefore helped the nulls most
  (PART 4). Any gain here is attributable.

  **Failure criterion.** No improvement in gAP over `tralo` at matched budget and
  seeds, or an improvement that the null also shows. Either kills it.

  ⚠️ **Not yet run, and identifying wrong items is not the same as fixing them.**
  PART 1 requires that link to be priced, not assumed.

- **Learning rate as the second lever.** `lr` sets how fast the boundary freezes
  and has been fixed at 1e-4 throughout. Any change must set
  `constraint_phase.lr_constraint` to match, or unequal lr fabricates a result --
  it did once already.
- **Candidate loss modifications**, each with a stated mechanism and
  falsification criterion, none run: task-protected constraint displacement;
  temporal signed-residual controller; constrained posterior targets.

---

*Superseded documents are in `docs/archive/`, recoverable in full from git
history. The pre-reset theory document is `docs/archive/THEORY.md`; PART 2 above
is the part of it that survived review.*
