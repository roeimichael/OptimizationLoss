# LEDGER -- what is proved, what is measured, what is closed

**What this file is for.** So no direction is tried twice and no result is
over-read. Add to it; do not re-litigate it. If a new result contradicts an
entry, edit that entry in place and say what changed it. **It is not a list of
wins** -- several entries are corrections to claims this project made with
confidence.

Every new entry needs: the hypothesis, the code/data/config identity, the actual
contrast, the metric, the uncertainty, the scope, and one disposition of
**supported**, **unfavorable in tested setting**, **inconclusive**, **invalid
comparison**, or **not tested**.

Five parts: **1** how a result can mislead, **2** mechanism, **3** measurements,
**4** closed, **5** open. Mechanism results are labelled **M1-M5** so they are
never confused with the numbered sections of PART 2.

---

## PART 1 -- How a result can mislead

Each trap below produced a wrong conclusion at least once.

### Arms and controls

- 🛑 **AN IDENTITY KEY MUST COVER EVERY INPUT THAT CHANGES THE WARM-UP, INCLUDING
  INPUTS OUTSIDE `hp`.** 2026-09-15, caught at 2/40 runs. `rank_clip` and
  `aug_rank_clip` hashed to ONE `base_model_id` across L80 and L90 because the
  cap lives in `config["constraint"]` while `compute_base_model_id` hashes only
  keys found in `hp`: L80 trained and cached, L90 silently loaded it, so every
  L90 rank cell would have been a model trained for an 0.8 cut and scored against
  an 0.9 allocation -- the exact train/deploy mismatch the loss exists to remove.
  **The dangerous kind: it runs clean and logs clean.** It also corrupted the
  pre-registered reading, since `rank_paired`'s `(cap-inert)` flag was registered
  as "the loss never reached the model" while the cache guaranteed cap-inertness
  regardless. **A pre-registered inference rule is only as sound as the mechanism
  it assumes; when the mechanism changes, retract the rule in place.** Fixed at
  `323edf44` -- `gen_campaign` stamps `rank_cap_fraction` for rank arms only so no
  existing digest moves, `warmup.py` REFUSES a stamp disagreeing with the cap it
  trains, and `tests/test_rank_cache_identity.py` pins both directions. Verified
  in the produced PREDICTIONS, not the configs:
  `aug_rank_clip` L80 vs L90 now differ on MobileNetV3 (`8c7a9316fa59` vs
  `e56ec773408d`) and RegNetY400MF (`fa08e8cc461c` vs `e227e7a25b52`). Found by
  COUNTING distinct warm-up identities in a generated campaign; no test or gate
  was looking for it.
- 🛑 **A UNIT-TESTED MECHANISM CAN BE 100% DEAD IN THE PIPELINE: 48 of 120 runs
  died one level below the thing under test.** 2026-09-15. 14 passing tests pin
  the ranking loss while every `rank_clip` / `aug_rank_clip` run in all three
  Stage 1 campaigns failed in its first epoch with `ValueError: too many values
  to unpack (expected 2)` -- ranking arms pass `groups` to `make_dataloader`, so
  the loader yields 3-tuples, and `warmup.py:208` handed that loader to
  `compute_train_accuracy`, which read `for X, y in loader`. Cost 2h21m on three
  cards. **A test of a COMPONENT is not a test of the PATH**: when a change alters
  the SHAPE of an object flowing through the pipeline, enumerate every consumer
  and execute the real path (`tests/test_warmup_executes_with_groups.py` does).
- 🛑 **A progress counter that only counts SUCCESSES cannot distinguish "still
  working" from "finished badly".** The same event was reported `STALLED`: a
  failed run writes `config.json` but never `final_predictions_raw.csv`, so
  `done < total` stays true forever. That pointed at the wrong fix (relaunch the
  dispatcher, not fix the bug). `scripts/rank_status.sh` now reads completion
  from the dispatcher's `ALL DONE` line, counts failures, and prints distinct
  error signatures so a second bug cannot hide behind a known one.
- **A "null" arm can inherit lambda = 0 from a block and nobody notices.**
  `tralo_reseed` was built from `[constraint_phase, tralo_null, tralo_reseed]`
  and inherited `lambda_step: 0.0`; for months `|tralo - tralo_reseed|` was used
  as an RNG-only floor while it was a treated-vs-untreated contrast. **Read the
  block composition, never the arm name.**
- **A named "#1" can be a dead arm.** Four separate TraLO first-place calls were
  produced by arms that were not running the treatment.
- **Every trained arm needs its own zero-constraint control** sharing the warm-up
  identity AND the schedule. `tralo_null` controls the plain column only;
  `aug_tralo` needs `aug_tralo_null`.
- **`tralo_null` is cap-independent** (lambda = 0 keeps the cap out of training),
  so the L80 and L90 nulls are the SAME run. Share it; never count it twice.

### Counting n

- **Training is bit-deterministic given `(config, seed)`.** `gx2` re-ran 30 of
  `fm2_mn3`'s cells across different trees and code versions and reproduced every
  byte. An overlapping arm adds ZERO information. **De-duplicate by prediction
  hash before counting n.**
- **A cap level is not a seed. A copied warm-up is not a seed. A re-run is not a
  seed.** A shared warm-up implies Cov >= 0 between arms, so the independence
  formula over-estimates the sd.
- **Four seeds is a pilot.** Effects here run ~0.01 against a seed sd of ~0.011,
  so n = 4 carries roughly 15% power. Every "not significant" verdict at n = 4 is
  consistent with a real effect.

### Statistics

- **"1 of 158 rows clears 2 sd" IS the chance expectation, not a finding.** That
  bar is a Welch t >= 4 at df 3-6; chance alone yields ~1.1 rows at df 6 and ~4.4
  at df 3. The honest statement was **0 of 158 resolve beyond chance**.
- **Label the sidedness of every sign test.** The project's headline p-values were
  one-sided and unlabelled; two-sided they are 0.125 / 0.0625 / 0.0078.
- **A ratio of medians of absolute differences cannot separate location from
  scale.** With sd ~5.9 items a genuine +2-item dominance moves median |X| from
  3.98 to ~4.25: "unresolved at this resolution", never "refuted".
- **Report mean, seed sd, every seed delta, n, and a two-sided 95% t interval in
  native metric units.** A reseed spread is a diagnostic, not a confidence
  interval and not proof of equivalence.
- **Do not choose the inferential method after seeing which one declares a win.**
- 🛑 **At n=2 a t-statistic has ONE degree of freedom.** An interim sweep emitted
  t=+29.3, t=+24.3, t=+23.0 -- two seeds landing near each other, not effects.
  Only the SIGN across cells is readable at that n.

### Prizes and metrics

- **A nominal prize is not a reachable prize.** Counting wasted slots and excluded
  true positives gives what a PERFECT re-ranking would win. A slot held by an item
  scored p(c) >= 0.99 and wrong cannot be evicted by any realistic nudge, and
  **39-46% of wasted slots at budget 30 are exactly that** (2026-09-15). Always
  discount headroom by the certain-wrong share.
- **`items = d(F1) * (K+n) / 2` is exact PER CLASS only.** cc-F1 is macro over
  classes with different `(K+n)`, so the inversion lives on a two-quantum lattice
  and mis-states a single-class move by 1.116x / 0.906x.
- **Headroom figures do not transfer between datasets.** The "1.9-9.9 items"
  figure came from dermMNIST, which was removed for leakage.
- **Flips, raw counts and proximity to a cap are not classification quality.**
  Improving constraint satisfaction does not imply improving cc-F1; price and test
  that link, never assume it.
- **cc-F1 is the primary endpoint** and correlated metrics are not independent
  replications of it.
- **A result quoted without its backbone is not a result.** The backbone x loss
  interaction (+0.03 for focal on ViTB16, PART 3) is larger than any method effect
  this project has chased.

### Instruments and gates

- **Gradient SIGNS cannot demonstrate that items compete.** The ranking loss's own
  mechanism test asserted only that positives are pushed up and negatives down --
  and it PASSED under a mutation that detached the cut, removing the coupling
  entirely. A claim of the form "these items interact" must be tested against an
  explicit non-interacting reference.
- **An epoch chosen on the per-epoch curve is an ORACLE, not a method.**
  `scripts/epoch_curve.py` scores stored snapshots against the TEST set. Quote it
  only as a BOUND on what any stopping rule could win.
- **"It only observes" is not a property a gate can check.** The first per-epoch
  trace scored inside the training loop, putting test labels in `TrainInputs`. It
  was observational in fact and rejected anyway, by
  `test_no_methodology_reads_the_test_LABELS_except_to_count_them` and
  `test_train_inputs_do_not_expose_held_out_labels`. **Prefer the design that
  cannot leak over the design that promises not to.**
- **`scripts/deployed_h2h.py` is the maintained reporter.** Hand-rolled cc-F1 has
  produced three scorer bugs.
- **Training accuracy alone cannot diagnose test-cut saturation.**
  `gate:saturation` reads train accuracy and is a SCREEN. Pair it with the regime
  check at the real allocation cut.
- 🛑 **A gate keyed on the NAME of a phase silently exempts any campaign that moves
  the objective.** `scripts/saturation_gate.py` contained `if con <= 0: continue`
  and so skipped **all 360 rank1/rank2/rank3 runs**, printing `no training_log.csv
  matched` -- indistinguishable from a bad glob -- and exiting 0 through a pipe.
  The series was reported gates-GREEN on a check that never ran. **Gate on where
  the objective is ACTIVE, and never let "nothing matched" and "nothing was
  checked" print the same message.** Fixed with four controls: falls back to
  `warmup_epochs` when `constraint_epochs = 0` and `rank_weight > 0`; rank3 now
  exits 1 SATURATED; clipper-only arms correctly report no gateable phase; the
  `--constraint-epochs` override path is byte-identical; rank2 reproduces.
- 🛑 **A DEFAULT CAN FABRICATE AN OUTPUT.** `log_progress_to_csv` defaulted
  `global_satisfied=True, local_satisfied=True`, and `src/pipeline/warmup.py:223`
  called it with neither, so every warm-up epoch wrote `Global_Satisfied=1,
  Local_Satisfied=1` on a row whose `L_Global`, `L_Local`, `Lambda_Global` and
  `Grad_Norm` are all 0. ⛔ **RETRACTION: the claim that `focal_tralo` reached
  joint global+local satisfaction (1 epoch of 30 at L80) is WITHDRAWN** -- that
  row is the warm-up. No arm reaches joint satisfaction and no rho freeze fires.
  Worse than a constant offset, because the warm-up is CACHED: `focal_tralo` at
  L80 has 30 rows from Epoch 1 while the same arm at L90 has 29 from Epoch 2, so
  the contamination varies with cache hits. Fixed 2026-09-16 (both default to
  `None`; `_sat_cell()` writes an empty cell, read back as NaN by the existing
  `pd.to_numeric(errors="coerce")` idiom); mutation-verified with `__pycache__`
  cleared each way; 445 tests pass. 🛑 **Historical logs still carry the bad rows:
  restrict any satisfied-epoch count to rows at or after `warmup_epochs + 1`.**
  This is the SIXTH defect of the "a flag reports something it never measured"
  family, after `class_balanced`, `logit_adjust`, `hounie_alpha`,
  `graph_probe --dump` and `disable_lambda_t` -- the others were inert INPUTS;
  this one is a fabricated OUTPUT, which nothing downstream can detect.
- **A gate is not done until a mutation makes it FAIL**, and the restore is
  verified by EXECUTING it. Stale bytecode has faked a pass.
- **Six instruments hardcoded the 30-epoch protocol** and would mis-read any
  campaign that did not use it. Assume a seventh exists.
- **A VIABILITY verdict is not a SCORE, and naming makes it one.** `~/triage.py`
  labelled cells KEEP / KILL while only testing whether a cell can produce a valid
  measurement; reported as-is on 2026-09-15 it read as "TraLO is winning here",
  false in every cell. The column is now `viability (NOT a score)` with values
  VIABLE / UNUSABLE. **Who wins comes only from `scripts/deployed_h2h.py`.**
- **Check the campaign inventory before calling anything "not run".** Four stale
  "not run" claims were found in one day (`tralo_stab`, `small60`/`scratch60`,
  `fm2_mn2`, `fm2_vit`) -- all complete on disk and unscored. An inventory across
  all worktrees now exists.

### Datasets and joins

- 🛑 **`scripts.candidate_gate` READS LABELS AND GROUP IDS ONLY. It never opens an
  image, so it cannot see a broken image-to-label join** -- its own docstring says
  INTEGRITY "needs the source archive". A slice whose pixels are attached to the
  wrong rows passes 8 of 8. **A gate pass is necessary, never sufficient.**
- 🛑 **PERFECT UNIQUENESS OF A JOIN KEY IS EVIDENCE THE JOIN WAS LOSSY.**
  Measured 2026-09-16:

  | slice | split | rows | distinct basenames | ambiguous rows |
  |---|---|---|---|---|
  | `fmow` | train | 15386 | 15386 | **0 (0.0%)** |
  | `fmow` | test | 4168 | 4168 | **0 (0.0%)** |
  | `fmow2` | train | 17670 | 16409 | 1261 (7.1%) |
  | `fmow2` | test | 3442 | 3145 | 297 (8.6%) |

  The fMoW archive is laid out `split/class/class_seq/aoi/file` and the filename
  does not encode the AOI, so 7-9% of basenames genuinely collide. `fmow2`, keyed
  on `class_seq/aoi/file`, shows them; `fmow` shows zero, which is only possible
  because its basename join dropped or mis-assigned every colliding record. **The
  cleaner-looking table is the broken one**, and row counts, duplicate-filename
  checks and cross-split image hashes all pass it GREEN by construction. Only
  re-deriving the join from the source archive can detect it.

### Freezes and re-validation

- 🛑 **A COMPLETED CAMPAIGN STOPS RE-VALIDATING ONCE THE WORKING TREE MOVES PAST
  ITS FREEZE, AND THE MESSAGE BLAMES THE DATA.** Observed 2026-09-19: all four
  `polc*` campaigns reported `REFUSED: source bytes differ from frozen release`.
  They were complete and already scored; `polc2` was frozen at `b2051c88` and the
  server tree is now `8085c356`. I had just repointed a data symlink, so the
  obvious reading was that I had corrupted the data. **I reverted the symlink and
  the refusal PERSISTED** -- which is the only thing that separates the two
  causes. The refusal is over `source_inventory()` (102 entries: `src/`,
  `configs/`, `scripts/`, `main.py`), not over the dataset.
  **Rule: this refusal on a finished campaign is expected code drift and means
  nothing about the results already scored. Never "fix" it by re-freezing a
  completed campaign against newer source -- that would silently re-stamp runs
  with a code version that did not produce them.** Revert-and-retest is what
  distinguishes drift from damage; do it before believing the message.

---

## PART 2 -- Mechanism

### 2.1 The five proved results (M1-M5)

From the adversarial-review theorem package (2026-09-02, three independent
reviewers, re-verified against source). **Proofs about the mechanism, not
results.**

- **M1 -- No value-level selection.** The loss is invariant under permuting test
  items: CE never reads them and every `S = sum phi(p_ic)` is symmetric. `L` is a
  function of the MULTISET while the allocator is a function of the RANKS. **Any
  procedure reading only values of `L` cannot prefer a correct ordering over the
  worst ordering with the same multiset.**
- **M2 -- The budget enters only as a scalar gain.** `grad_theta P = a(t) *
  V(theta)` with `a(t) = lambda(t) * psi'(S;K) >= 0` and `V(theta)` independent of
  K. The budget's information is reduced to a few non-negative scalars.
- **M3 -- Corollary: the four duals are ONE family.** TraLO, fioretto, alm and
  hounie differ only in the gain schedule and span the same cone. Their deployed
  differences SHOULD sit at the noise floor, so measuring them tied is a confirmed
  prediction, not a disappointment.
- **M4 -- Binary and decoupled implies provable invariance.** The reordering
  channel exists only through softmax coupling (C >= 3) and weight sharing.
- **M5 -- Conditional harm lemma.** If current scores already exhaust the
  available label information, any label-blind reordering has non-negative
  expected deficit. Applied to this project's own numbers it predicted -28.3
  against **-30.4 measured**.

**What is NOT proved, and was wrongly claimed.** The strong impossibility
conjecture is FALSE: a constructed two-cluster geometry with a shared linear head
gives a strict allocation improvement from one aggregate-count step, and TENT
(Wang et al., ICLR 2021) is a published counterexample. **Renounce
impossibility; claim M1-M5.** Also corrected: "both allocators are functions of
the ranking" is false for the LP, which maximises a linear functional of cardinal
values; and the transductivity argument does not hold, because the post-hoc
allocator sees the budgets too. What differs at equal compute is the set of
REACHABLE RANKINGS -- an optimisation-geometry object, not an information one.

### 2.2 The literature explains the negative result and names the one crack

Verified via Semantic Scholar, 2026-09-15. For selection-rate constraints of
exactly our form -- "at most K(g,c) items predicted as c in group g" -- the
Bayes-optimal constrained classifier IS a group-wise thresholding rule on the
posterior, and our greedy top-K allocator is its plug-in version.

- **Zeng, Cheng & Dobriban (2024)**: via a Neyman-Pearson connection the optimum
  is explicit group-wise thresholding with closed-form thresholds; our caps meet
  the structural assumption.
- **Xian, Yin & Zhao (ICML)**: multi-group multi-class post-processing attains the
  optimum **whenever the score is Bayes-optimal**.
- **Fukuchi (ICML 2025)**: fair minimax optimality is achievable by
  post-processing; the advice is to improve the underlying regression.
- **Alabdulmohsin (2020)**, **Zhang et al. (2026)**: in- and post-processing
  converge to the same Pareto frontier.

🔑 **THE CRACK: our score is NOT Bayes-optimal.** **Woodworth, Gunasekar,
Ohannessian & Srebro (COLT 2017)**: post-processing a FIXED, NON-Bayes predictor
can be strictly suboptimal, and in-processing is justified precisely through
hypothesis-class restriction. **The theory permits a training-time win only by
improving the SCORE**, never by enforcing the count. M1 shows a count penalty
cannot carry score-improving information; `tralo_stab` (PART 4) shows even an
informative weight cannot. Both halves of the result now have citations.

**Dead by citation:** further Lagrangian/ALM variants (Chamon & Ribeiro NeurIPS
2020 / IEEE TIT 2021 bound the duality gap and feasibility, never accuracy over a
feasible post-hoc rule; AL-CoLe ICASSP 2025 is one more instance of the M3
family). Learning-from-label-proportions does NOT transfer: LLP's counts are
observed LABELS (new supervision), ours are caps on PREDICTIONS (no supervision)
-- the cleanest statement of why the penalty is information-free.

### 2.3 ⛔ RETRACTED 2026-09-20 -- THE TERM IS NOT COMPUTED ON TRAIN DATA

**The premise below is false, and the conclusion drawn from it is withdrawn.**
The claim was that every constraint term runs on TRAIN data, where violation is
identically zero, making the term silent for ~80% of training. Two independent
checks refute it:

1. **By reading the source** (not grep -- the files are read end to end):
   `src/experiments/runner.py:109` binds `TrainInputs.X_test = data.X_test`, the
   real deployment pool, and `group_ids = groups_test`.
   `src/methodologies/tralo/train.py` computes its soft counts and runs its
   entire backward pass over `model(X_test[...])`. All four duals do the same
   (`fioretto_alm`, `fioretto_ldf`, `hounie_rcl` each bind `X_test_dev =
   inputs.X_test`). **Only `budgeted_rank_loss` uses train data -- and no arm
   using it has ever run.**
2. **By the logs**, across every Option C run: **0 of 288 constraint epochs in
   60 `tralo` runs was ever satisfied -- 0.0%.** A representative L25 run:

   | epoch | train acc | L_Global | L_Local | grad norm | Hard_Class1 vs limit 91 |
   |---|---|---|---|---|---|
   | 2 | 0.842 | 0.04 | 0.27 | 2.7 | 410 |
   | 4 | 0.961 | 10.3 | 70.0 | 1080 | 333 |
   | 7 | 0.985 | 56.1 | 404.7 | 6399 | 402 |

   The term grows ~1500x, the gradient norm reaches 6399, and the class is
   over-budget **4x in every epoch**. It is not silent; it is maximally alive.

🔑 **This makes the negative result STRONGER, not weaker.** The constraint is
computed exactly where the transductive setting says it should be, is violated
throughout training, and delivers a large gradient -- and TraLO still does not
beat a post-hoc clipper (PART 3, PART 4). **"The term was never alive" is now
closed as an explanation.** M1 stands as the operative mechanism: the loss is a
function of the probability MULTISET while the allocator is a function of their
RANKS, so a live, large, well-aimed term still cannot prefer a correct ordering.

⚠️ **The direction this section opened in PART 5 -- route the constraint to a
held-out fold of the train groups -- is therefore CANCELLED.** It would move the
term from a pool where it is 4x violated onto a fold where it would be violated
less. That is strictly worse, and it was approved by the user on the strength of
this section before the section was checked.

**What the original measurement DID establish**, and which stands: on train the
term would be silent (V=0 in 174/174 cells), on test it is not (V=5,507, 26 of
27 cells live). Its own TEST column always contained the refutation of the
premise sentence. The error was asserting the code used the TRAIN column.

Original measurement, 2026-09-16, from the rank3 warm-up checkpoint
(`MobileNetV3_fmow2_5aa9cfb1e37a.pt`) at the deployed cut (cap 0.90, classes 1/2/7):

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
gradient support is the empty set.

The 719 train positives sitting outside the budget are the ones the cap forces
out by construction (2.4); they generate no inversions because no false positive
is inside to swap with them.

**It is a decay, not a constant.** rank3 logs give train accuracy 0.803 / 0.877 /
0.942 at epochs 1-3, 0.984 at 6, 0.996 at 12, 0.999 at 30 (MobileNetV3 and
MobileNetV2 alike). Support scales with `(1 - accuracy)`, so the term carried real
signal for roughly **epochs 1-5 of 30** and was silent for the remaining 80%.

⛔ **WITHDRAWN with the premise:** this paragraph claimed to subsume 2.4 and 2.5
because the gradient was multiplied by an empty support for 80% of training. The
support is not empty -- it is the test pool, violated in 288 of 288 epochs. **2.4
and 2.5 are NOT subsumed and stand on their own.**
Train 0.9999 against test 0.6322 is a 37-point generalisation gap, and the term is
asked to repair an ordering that is already perfect on every item it can see,
which is consistent with 2.2 and with every null in PART 4. ⛔ The direction it opened in PART 5 is cancelled, above.

### 2.4 The ranking gradient is UNCERTAINTY-weighted, not CUT-anchored

`scripts/why_rank_failed.py`, 2026-09-16, 27 cells, 9794 items, within-cell
standardised, gradients taken through the real loss on real score distributions:

| | correlation with log per-item gradient |
|---|---|
| distance from the allocator's cut | -0.244 |
| softmax Jacobian p(1-p) | **+0.624** |
| **partial**, distance given Jacobian | **-0.145** |
| **partial**, Jacobian given distance | **+0.604** |

🔑 **Holding uncertainty fixed, distance from the cut explains almost nothing.**
The hinge is anchored at the K-th order statistic in PROBABILITY space, but the
gradient reaching the weights passes through the softmax Jacobian, and `p(1-p)` is
a function of the probability VALUE, not its RANK. What the model receives is an
uncertainty weighting -- close to entropy regularisation, not the allocator's
"which".

🛑 **THIS IS M1 IN A NEW COSTUME, AND THAT IS THE LESSON.** The loss reads ranks
in its FORWARD pass; backward, the Jacobian launders it back into a
value-dependent quantity. **Anchoring a loss at an order statistic is not
sufficient; the GRADIENT has to stay rank-dependent too.**

Two further exact defects:

- **An irreducible floor.** `softplus(margin + t - s)` on probabilities in [0,1]
  gives an argument in [margin-1, margin+1], so softplus never reaches zero: best
  case 0.327 per term, worst 1.350. At most 76% of the loss is movable -- why rank
  arms log a total loss near 1.0 while controls log 0.009.
- **Built-in unsatisfiability.** `K = round(n_pos * cap_fraction)` at 0.9 forces
  **10.0% of true positives (123 of 1231) below the cut by construction**. They
  are penalised every step and cannot be fixed, and because the cut is
  deliberately not detached, lifting any positive raises `t` for the rest.

### 2.5 The cap never entered the ranking loss: `round()` collapses on a batch slice

Measured 2026-09-16 on the real train split by simulating the shuffled loader over
828 batches / 3 epochs. Mechanical, not statistical. `budgeted_rank_loss` takes
`k = round(n_pos * cap_fraction)` INSIDE a batch of 64
(`src/training/rank_loss.py`), and fmow2 train spreads 17,670 items over 139
countries:

| quantity | value |
|---|---|
| batches with a usable (group, class) cell | 96.5% -- the term did fire |
| usable cells per batch | 2.56 |
| mean group slice size in a usable cell | 12.4 items |
| cells holding exactly **one** positive | **49.1%** |
| `K < n_pos` (budget BINDS) at cap 0.90 | **16.7%** |
| same at cap 0.95 | **0.1%** |
| same at cap 0.80 | 30.7% |

`round(n_pos * 0.9) = n_pos` for every `n_pos <= 4`, and 83.3% of cells are that
small, **so in 83% of cells the loss reduced to a plain pairwise term** -- exactly
the non-cutoff-sensitive surrogate `rank_loss.py`'s own docstring rejects. The
decisive contrast: L90 and L95 give the **same cut in 83.0% of cells per BATCH
slice against 33.3% per FULL GROUP** (at group level the budget binds in 76.4% of
cells at cap 0.90 and the two levels differ by a mean of 2.14 items). **The cap
information exists; batch-level `round()` destroys it before the gradient.**
Not fixable by reparameterisation -- carrying the cap as a rate `q = K_full/n_full`
and taking a slice quantile is WORSE (identical in 87.7% of cells). The slice size
binds, so a fix must enlarge the effective group (full-group score buffer, or
group-blocked batching).

### 2.6 The dead regime: where the objective was actually active

Judged against the phase where the ranking loss was active (30 warm-up epochs),
all three backbones FAIL `gate:saturation`, which requires >= 50% live:

| cell | live window | live fraction | verdict |
|---|---|---|---|
| MobileNetV2/fmow2 | 3.0 of 30 | 10% | SATURATED |
| MobileNetV3/fmow2 | 2.5 of 30 | 8% | SATURATED |
| RegNetY400MF/fmow2 | 2.7 of 30 | 9% | SATURATED |

**rank1, rank2 and rank3 -- 360 runs -- ran in the dead regime**, the one
established in August 2026 as worth ~8 pp against ~0.1 pp for the method choice.
This is upstream of 2.3-2.5. **Augmentation is NOT the lever:** augmented arms
reach a 3.00-epoch live window against 2.17 for plain `clip` (10% vs 7% of a
30-epoch budget) -- 0.8 epochs where 12 are needed.

⚠️ **But liveness is not the barrier either.** See PART 3: at ~100% live
(`small60`) the constraint is still negative. A short budget and a live window
explain why the constraint is INERT; they do not explain, and do not repair, the
fact that it is HARMFUL when active.

### 2.7 The global cap is REDUNDANT on fmow2: every fmow2 result is a LOCAL-cap result

Verified 2026-09-16 from code and live training logs. **Both scopes are genuinely
implemented**: `compute_local_constraints` gives K per (group, class),
`compute_global_constraints` gives K per class over the whole pool; both enter the
objective (`chunk_loss + lg + ll`, `tralo/train.py:290`) with separate multipliers
and both are logged. **Both are violated during training** -- over 116 real
constraint epochs of `vit_a`'s `tralo` arm: both violated in 115, local-only in 1,
global-only in 0, satisfied in 0 (global 64% over cap, median slack -33, n=348;
local 64% over cap, median slack -4, n=3480).

🛑 **But the global cap cannot bind AFTER allocation.** The allocator enforces the
per-group ceilings, so the most a deployed model can emit for a class is the SUM
of its local ceilings -- below the global ceiling at **all 15 (cap level x capped
class) combinations** (sum of local K vs global K):

| cap | class 1 | class 2 | class 7 |
|---|---|---|---|
| `L40_G95` | 145 vs 347 | 218 vs 519 | 128 vs 304 |
| `L55_G95` | 202 vs 347 | 300 vs 519 | 176 vs 304 |
| `L70_G95` | 256 vs 347 | 381 vs 519 | 224 vs 304 |
| `L80_G95` | 291 vs 347 | 437 vs 519 | 256 vs 304 |
| `L90_G95` | 328 vs 347 | 491 vs 519 | 289 vs 304 |

Headroom 15 to 301 items. **The global term optimises a constraint the allocator
already guarantees** -- not inert, but unable to change the deployed outcome
except through side effects on the score, and by M1 the LEAST item-selective part
of the objective (one sum over the entire pool, so its gradient is identical for
every item in the dataset). **Every fmow2 number is a local-cap result and must be
described as such; dropping the global term loses nothing at deployment.**
⚠️ **Redundant after allocation is NOT the same as inactive during training** --
both statements above are true and are about different stages.

🔑 **CAUSE: the redundancy is OUR cap pairs, not the method.** Danit's LP
(Shifman et al. 2025) constrains the same two scopes and sets *the two percentages
equal*, so with one percentage p, `sum_lambda Phi_lambda(c) = p*N_c = Psi(c)` up
to rounding and both constraints bind exactly. `configs/gen_campaign.py` still
defaults to `L30_G30 L50_G50`, which has that property; **every campaign this
project has run overrode it with `--caps L80_G95 L90_G95`**, opening the headroom
above by construction. An equal-percentage pair restores the global term.

### 2.8 The information the constraint lacks EXISTS, and it is not in `p`

Read from source: the soft count is an UNWEIGHTED sum of probabilities
(`chunk_eff = chunk_proba`; `chunk_global = chunk_eff.sum(dim=0)`,
`tralo/train.py:240`), so `dL/dp_i(c) = psi'(S_c)` is identical for every item in
the scope and the only per-item differentiation is the softmax Jacobian
`p_i(1-p_i)`. **Everything the constraint knows about item i is `p_i(c)` -- the
quantity the allocator already ranks by. That one line is the bottleneck M1 proves
cannot re-order.**

Measured 2026-09-15 on stored embeddings, `fm2_mn3` / `clip`, predicting WRONG
among items placed in a capped class at p(c) >= 0.99 (2,595 and 2,671 such items,
15% wrong; unit is the seed, 12 cells = 3 capped classes x 4 seeds per cap):

| signal | L80 AUC | L90 AUC | from `p`? |
|---|---|---|---|
| `knn_agree`, 20-NN argmax agreement in embedding space | **0.869 +- 0.028** | **0.874 +- 0.033** | no |
| `knn_pc`, neighbours' mean p(c) | 0.790 | 0.790 | no |
| `knn_dist` inverted, local density | 0.705 | 0.705 | no |
| `centroid_cos` | 0.656 | 0.648 | no |
| `margin` (control) | 0.678 | 0.678 | **yes** |

⚠️ The margin control did NOT land at 0.5 as predicted, so the claim is the GAP
(0.87 vs 0.68), not that `p` is uninformative. `~/separability.py` on dsisco02. **This was the escape route M5
leaves open** -- M5 assumes current scores exhaust the available label information
and they do not. ⛔ **The route is now measured and shut**: `tralo_stab` used
exactly this weight and is a null (PART 4). Identifying wrong items is not fixing
them, and that link is now priced at zero.

---

### 2.9 The local cap was a PREVALENCE, never a POLICY -- fixed 2026-09-16

Read from source, then fixed and gated. Shifman et al. Eq. (2) takes
`Phi[lambda][i]` as a GIVEN integer bound; the deleted
`danits_lp/constraints_builder.py` (recovered from `.codex/git-clean-checkout`,
removed at `cb516cb3`) derived it as `round(feature_pct * count of class i IN
GROUP lambda)`, and `compute_local_constraints` did the same. **A ceiling
proportional to a group's own positives is proportional to the answer**: a group
holding twice the positives is handed twice the budget, so no cap in this
project could ever express a policy that favours a small group. That is the
entire point of a local feature, and it was inexpressible.

`compute_local_constraints` now takes `group_budget_shares`: the class's pooled
budget apportioned by externally given shares, by LARGEST REMAINDER so the parts
reconstruct the total exactly (independent rounding does not -- three groups at
1/3 of 100 round to 33 each and lose an item no log shows). **The default path is
unchanged and asserted byte-identical**, so every completed run stays comparable.

Validated on the paper's own worked example (`scripts/hospital_smoke.py`, 1000
patients, 3 tiers sized 334/333/333, shares 10/30/60 deliberately
anti-correlated with demand). At `L50_G50`:

| class | Psi | policy split | prevalence split |
|---|---|---|---|
| normal_bed | 100 | 10 / 30 / 60 | 60 / 25 / 15 |
| special_bed | 200 | 20 / 60 / 120 | 100 / 60 / 40 |

Same totals, near-reversed order. The allocator saturates all six policy
ceilings exactly with no violations. PDF at
`docs/validation/hospital_constraints.pdf` (gitignored; regenerate with
`hospital_smoke.py --json` then `hospital_report.py`).

⚠️ **Two consequences that are NOT bugs but change what the loss means.**

1. **With `L == G` the global term becomes a strictly redundant CONSTRAINT**, not
   merely redundant after allocation (2.7): `sum_lambda Phi = Psi` exactly, so
   local feasibility implies global feasibility. Measured: locals exactly at
   ceiling gives both terms 0.000000; +1 item over in the smallest-budget group
   gives local 0.0959 and global 0.0100. The global term fires on the same
   violation while adding no constraint, and by M1 its gradient is identical for
   every item in the pool.
2. **The K-normalised penalty now weights groups by BUDGET, not by size.**
   `d(penalty)/d(count)` at one item over: 0.0924 at K=10, 0.0323 at K=30,
   0.0164 at K=60 -- pressure scales like 1/K. Under the prevalence derivation K
   tracked group size, so this was a per-capita normalisation; under a policy it
   is not, and the smallest-budget tier exerts 5.6x the gradient of the largest
   regardless of how many candidates it holds. **Left as-is deliberately**: the
   relative form is what every completed run and every rival dual used, and
   changing it would break comparability. Recorded as an ablation, not a defect.

🛑 **The ranking term is REFUSED alongside a policy** (`check_rank_budget_agreement`,
`warmup.py:126`). `budgeted_rank_loss` cuts each TRAIN (group, class) at that
group's own prevalence; under a policy the allocator cuts at share x pooled
budget, differing by up to 6x per tier, and train/test groups are disjoint so the
shares cannot be carried across by group identity. Currently latent --
`rank_weight` is 0 in every live config -- so the guard stops it arriving the
first time someone turns it on.

19 tests, 7 of them negative controls; four mutations (round() for largest
remainder, dropped sum check, dropped group check, disabled rank guard) each fail
the suite.

⛔ `danits_lp` is ABSENT from the live tree (deleted at `d4aea81c` / `cb516cb3`,
recoverable from git). **The LP rival cannot currently be run at all**, so no
comparison against it is available until it is restored.

## PART 3 -- What is measured

### The settled table -- do not re-open without new evidence

| # | Finding | Consequence |
|---|---|---|
| 1 | **Every cell saturates fast.** Train acc >= 0.95 in 2-4 epochs, 5/5 cells, 3 backbones, 2 datasets; augmentation doubles it to ~5 and no more. Per backbone (2026-09-16): ViTB16 at epoch **1**, MobileNetV3 at 3, MobileNetV2 and RegNetY400MF at 4; >= 0.99 by epoch 6-13. | The boundary is frozen for most of any long budget, and a single budget does NOT hold the live fraction fixed across backbones. |
| 2 | **The four duals share one per-item gradient**, read from source. Dose is equal: 232/232 steps, 100%, all arms. | The historic dose gap explains nothing. No better dual rule exists to find. Confirms M3. |
| 3 | **Eviction given the probabilities is already ~optimal.** A gradient push lands on 87% of the provably optimal set; closing the rest is worth 0.0002. | A margin-aware soft count is CLOSED. The prize is not in the loss shape. |
| 4 | **The exact (LP) allocator makes real results WORSE** (-0.01 to -0.04 accuracy, -0.02 to -0.08 cc-F1, 14/14 cells) because models run at 0.92 confidence against 0.60 accuracy. | The allocator "fix" is CANCELLED. Greedy's suboptimality is protective. |
| 5 | **The constraint obeys but does not convert.** Every dual beats both clippers on native obedience (excess 279-326 vs 381-413). | Obedience is not the missing piece; the metric does not reward it. |
| 6 | **On a frozen boundary the constraint makes the RANKING worse.** 15 of 16 seed-deltas negative, gAP(tralo) - gAP(tralo_null) = -0.007 to -0.030. gAP is allocation-free. | The damage is upstream of the metric and ~2x the cc-F1 damage. Predicted by M5. |
| 7 | **The cap IS binding and there IS headroom.** At the real cut `filled == slots` in every cell; ~1,000 of ~4,400 slots hold wrong items while ~1,500 true positives sit outside. | A real constrained problem with a nominal prize of ~20-25% of the budget -- **~12-15% reachable** after the certain-wrong discount (PART 1). |
| 8 | **The paper's second phase has never run.** 2,563 runs, 0 freezes: lambda and rho never freeze. The obvious repair, `tralo_dualprop`, was already built and is a null. | The implementation does not match the method description. |
| 9 | **At the right dose, the constraint direction knows HOW MANY but not WHO** (2026-09-26, branch `claude/rebuild-validation-20260925`, knee end-to-end ResNet18, grade-3 cap 76, n=24, preregistered). The published separate-Adam step overshoots ~10x: one step moves the grade-3 soft count 82.6 -> 8.2 against a cap of 76. A step along the same direction, bisected to the smallest radius that meets the HARD cap, lands the count exactly; a sham of the same radius and per-tensor norms in a random direction leaves the count unchanged. Yet on capped_first grade-3 F1: target-null +0.96 [-0.96, +2.88], **target-sham +0.14 [-1.21, +1.48]** (Holm ns, bounded), target-adam +2.98 [+1.24, +4.71] (p=0.002). Slot turnover vs the null: target 0.223, sham 0.230 (reseed floor 0.301). | The published arm's damage is the dose. Fixing it (and D1-D4, D7) leaves no measurable information about which patients fill the slots: any such gain is below +1.5 F1 points (~1.3 of 76 slots) at 95%. Why (post hoc, dev labels offline): the targeted step evicts 83% the same items as the post-hoc cut on the same probabilities, at slightly LOWER precision (57.3% vs 58.6% not-grade-3); the published dose evicts 4x too many at 36%. The step IS the clipper, through the weights. Record: `experiments/claude_targeted_step_protocol_20260925_result.md` on that branch. **Replicated at cap 50** (seeds 1901-1924, preregistered, n=24): target-null +0.96 [-0.87, +2.79], target-sham -0.75 [-2.45, +0.96], step/post-hoc eviction overlap 87%. Two studies, 48 seeds, same verdict. |
| 10 | **Why the count cannot move the capped boundary the right way: it only demotes, it peaks at the ARGMAX boundary, and it carries nothing beyond p3** (2026-09-26, stored snapshots of the two #9 studies, n=24 per cap, dev labels offline; `analysis/lab_synthesis/BOUNDARY_DIRECTION.md` on `claude/rebuild-validation-20260925`). Items whose grade-3 log-odds rose under the count gradient: 0 of 99,120 per cap. The push peaks at rank ~89 (p3 = 0.50, the argmax boundary), not at the cap; 24% (cap 76) / 35% (cap 50) of it lands on true grade-3 items OUTSIDE the slots. AUC for wrong occupants, push minus p3: -0.010 [-0.016, -0.006] / -0.001. Cosine with the oracle swap direction -0.157 / -0.067 (random sd 0.017). Idealised free-logit flow to the cap admits 0 items and keeps 99% of the top-cap set. The real network step adds a little who-information through shared weights (+0.62 [+0.37, +0.86] correct slots vs sham per step at cap 50; +0.11, ns, at cap 76), and the next CE epoch removes it (-0.72 [-1.32, -0.12]). Reachable prize: 16.5 wrong of 76 / 6.4 of 50; swaps within 10 ranks of the cut recover at most 3.8 / 2.5 slots. **Depth probe** (seeds 2601-2624, cap 50, n=23 after one preregistered exclusion; `experiments/claude_step_probe_20260926_result.md` on `claude/bandcons-20260926`): the exact-cap step gives tralo-sham +0.57 [-0.04, +1.17] (Holm ns); pushing past the cap costs slots, -2.39 at 40% of cap and -6.09 [-7.73, -4.45] at 20%, slope -8.1 [-10.3, -5.9] slots per unit depth; the evicted share that is not grade 3 falls 47% -> 32%. No depth carries more who-information than the exact cap. | capped_first is the plug-in Bayes rule under a cardinality constraint; the KL projection onto a count is a tilt that never changes top-K. **Stop tuning dose, schedule, controller or targeting inside the count family** -- at best it ties the clipper. A training term can win only with a promote side, aimed at the cap, fed by who-information from OUTSIDE the count (training labels at the cut = CUTPAIR; augmentation stability = BANDCONS). Synthetic lab: a label-free count helps only when the dense uncertain mass at the cut is mostly wrong, hurts by the same mechanism when it is mostly right, and nothing label-free separates the two. |

### The head-to-head record

- 🔴 **THE CONTROL KILLS EVERY MobileNet TraLO WIN: `tralo_null` DELIVERS THE SAME
  MARGIN.** Measured 2026-09-16 over all **1,364** fmow2 runs carrying both
  predictions and metrics, de-duplicated by prediction hash, averaged over SEED
  only, (campaign, backbone, cap) as the unit, nothing pooled:

  | cell | contrast | L80 | L90 |
  |---|---|---|---|
  | stab8 MNv3 (live) | `tralo` - `focal_clip` cc_f1 | +0.0113 (t+2.7) | +0.0110 (t+3.2) |
  | stab8 MNv3 (live) | **`tralo_null` - `focal_clip`** | **+0.0106 (t+5.2)** | **+0.0080 (t+2.0)** |
  | stab8 MNv3 (live) | `tralo` - `tralo_null` | +0.0007 | +0.0030 |
  | fm2_mn3 MNv3 (30e) | `tralo` - `focal_clip` macroF1 | +0.0136 (t+3.9) | +0.0144 (t+3.7) |
  | fm2_mn3 MNv3 (30e) | **`tralo_null` - `focal_clip`** | **+0.0134 (t+4.0)** | **+0.0112 (t+3.8)** |
  | fm2_mn3 MNv3 (30e) | `tralo` - `tralo_null` | +0.0003 | +0.0032 |
  | live11 MNv3 (live) | `tralo` - `clip` cc_f1 | +0.0164 (t+4.6) | +0.0102 (t+1.2) |
  | live11 MNv3 (live) | **`tralo_null` - `clip`** | **+0.0226 (t+5.5)** | **+0.0224 (t+15.0)** |
  | live11 MNv3 (live) | `tralo` - `tralo_null` | **-0.0061** | **-0.0122** |

  Focal does not rescue it: `focal_tralo` - `focal_tralo_null` is -0.0079 /
  -0.0005, and `focal_tralo_null` - `focal_clip` (+0.0092 / +0.0069) again beats
  `focal_tralo` - `focal_clip` (+0.0012 / +0.0063). 🛑 **Any TraLO-vs-clipper
  number quoted without the matching null-vs-clipper number is uninterpretable** --
  8 of the 12 "both caps, |t|>=2" hits in the first scoring pass were recipe
  effects the control removes. Note the algebra:
  `(tralo - clip) - (tralo_null - clip) == tralo - tralo_null`.

- 🟢 **ViTB16 IS THE ONLY CELL WHERE THE ATTRIBUTABLE CONTRAST SURVIVES ITS OWN
  CONTROL -- A CANDIDATE, NOT A RESULT.** `fm2_vit` (ViTB16 x fmow2, 56 runs,
  budget 30 = 1+29, lr 1e-4, ~10% live, 7 arms), scored 2026-09-16 after sitting
  complete and unscored. Paired, n=4, `tralo` minus the named arm:

  | contrast | cc-F1 L80 | F1 Macro L80 | cc-F1 L90 | F1 Macro L90 |
  |---|---|---|---|---|
  | - `tralo_null` | +0.0117 (t+0.6) | +0.0230 (t 2.47) | +0.0141 (t+1.7) | +0.0149 (t+1.1) |
  | - `clip` | +0.0060 | +0.0114 | +0.0104 (t 2.45) | +0.0049 |
  | - **`alm`** | **+0.0160** | **+0.0284 (t 2.62)** | **+0.0179 (t 1.86)** | +0.0000 |
  | - `fioretto` | +0.0011 | +0.0082 | +0.0177 | +0.0168 |
  | - `hounie` | +0.0041 | +0.0056 | +0.0180 | +0.0175 (t 2.60) |
  | **`tralo_null` - `clip`** | **-0.0057** | **-0.0116** | **-0.0037** | **-0.0100** |

  Every sign is positive against every dual rival, the plain clipper and its own
  null -- the most pro-TraLO evidence in the corpus -- and it is the INVERSE of
  the MobileNet pattern, because here the null LOSES to `clip` while `tralo` beats
  it, so the margin cannot be the recipe. 🛑 **Why it is not a result:** not one CI
  excludes zero at n=4; only 1 of 4 (cap x metric) combinations reaches |t| >= 2;
  the effect is carried by 3 of 4 seeds with **seed 3 reversing in every metric at
  both caps** (cc_f1 L80: +0.0439, +0.0476, -0.0348, -0.0100). `tralo_null` sits in
  the normal 0.59-0.71 range, so this is variance, not a collapsed control -- the
  ~15% power regime. **And `focal_clip` still beats `tralo` on ViT** (-0.0221 /
  -0.0202 cc_f1), so it is a live-null result, not a rival-beating one.

- 🔴 **`focal_clip` BEATS TraLO DECISIVELY ON THE HEADLINE BACKBONE**, both caps,
  CIs excluding zero (`fm2_vit`, n=4):

  | contrast | F1 (Macro) | Precision (Macro) |
  |---|---|---|
  | `tralo` - `focal_clip` L80 | **-0.0223** [-0.0380, -0.0067] | **-0.0261** [-0.0480, -0.0042] |
  | `tralo` - `focal_clip` L90 | **-0.0309** [-0.0561, -0.0058] | **-0.0315** [-0.0532, -0.0099] |

  This reproduces the long-standing record (PART 4: the cc-F1 win dies against
  focal+clip). **`focal_clip` is the arm to beat, not `clip` and not ALM.**

- 🔑 **FOCAL'S 2-POINT WIN IS ViT-ONLY, AND IT IS AN ANTI-SATURATION EFFECT.**
  `focal_clip` - `clip`, paired, same seeds, same allocator, only the training loss
  differs:

  | campaign | backbone | cc-F1 L80 | cc-F1 L90 | F1 Macro L80 | F1 Macro L90 |
  |---|---|---|---|---|---|
  | `fm2_vit` | ViTB16 | **+0.0280** (t 2.36) | **+0.0306** (t 2.10) | **+0.0337** (t 2.29) | **+0.0358** (t 2.37) |
  | `fm2_mn3` | MobileNetV3 | -0.0037 | +0.0022 | -0.0015 | +0.0010 |
  | `fm2_mn2` | MobileNetV2 | -0.0095 | -0.0074 | -0.0023 | -0.0014 |
  | `live6b` (budget 6) | MobileNetV3 | -0.0092 | **-0.0079** [-0.0153,-0.0005] | +0.0050 | +0.0058 |

  Across the whole corpus the same contrast on MobileNetV3, MobileNetV2 and
  RegNetY400MF stays between -0.017 and +0.006 and never reaches |t| >= 2 on both
  caps. **Mechanism:** focal is a saturation counter-measure, and the live window
  (epochs before train acc 0.95) is ViTB16 1.0 -> 1.5 with focal, MobileNetV3
  2.0 -> 2.0, MobileNetV2 2.8 -> 2.8. ViT memorises in ONE epoch and focal is the
  only arm that extends its window. 🛑 **`focal_tralo` on ViTB16 has never run** --
  `fm2_vit` has no `focal_tralo`, and `live6b` has it only on MobileNetV3 where
  focal does nothing (`focal_tralo` - `focal_clip` = -0.0074 / -0.0009).

- **`fm2_mn2` -- no win.** `tralo` - `clip` is -0.0162 cc-F1 [-0.0250, -0.0075] at
  L80 (CI excludes zero, TraLO loses); `tralo` - `tralo_null` is -0.0125 (t -2.69)
  at L90. Small positives against `alm` (+0.0076) and `fioretto` (+0.0057 F1 Macro,
  t 2.10) at L90 only. **The backbone is a real moderator: ViTB16 directionally
  ahead of everything, MobileNetV2 behind.**

- **AT A LIVE BOUNDARY, TraLO STILL DOES NOT BEAT THE CLIPPERS OR ITS OWN NULL.**
  `deployed_h2h` on the two complete budget-8 screens (`scr_MobileNetV3`,
  `scr_RegNetY400MF`, 56/56 each, 4 seeds, fmow2), 2026-09-15. cc-F1, seed-paired,
  TraLO minus comparator:

  | cell | vs `tralo_null` | vs `clip` | vs `focal_clip` | vs `aug_clip` |
  |---|---|---|---|---|
  | MN3 L80 | +0.0007 | +0.0092 | +0.0113 | **-0.0124** |
  | MN3 L90 | +0.0030 | +0.0073 | +0.0110 * | **-0.0099** |
  | RegNet L80 | -0.0026 | -0.0088 | -0.0105 | **-0.0298** * |
  | RegNet L90 | +0.0042 | +0.0015 | -0.0030 | **-0.0197** * |

  `*` = 95% CI excludes zero, NOT adjusted over 24 comparisons. **`aug_clip` is
  the best arm in all four cells** and augmentation is available to every method.
  The constraint buys nothing over its phase-matched control (+0.0007, +0.0030,
  -0.0026, +0.0042; every CI spans zero, sign not even consistent). The only CI
  favouring TraLO is MN3 L90 vs `focal_clip` (+0.0110, [0.0002, 0.0218]) -- 1 of 24
  unadjusted, i.e. what multiplicity produces on its own. **The live boundary was
  the last untested precondition and supplying it did not change the verdict.**
  Reading limit: the reporter emits seed-paired deltas only against `tralo`, so
  `aug_tralo` - `aug_tralo_null` is a difference of MEANS here (+0.0023, +0.0029,
  -0.0008, +0.0055) and must not be quoted as paired.

- 🔴 **THE ENVIRONMENT IS EXONERATED, AND LIVENESS IS NOT THE BARRIER.** `small60`
  (SmallCNN 100k params, fmow2, budget 60, 32/32) is the only campaign in the
  corpus that passes the live-boundary test, 2026-09-15:

  | precondition | `small60` |
  |---|---|
  | boundary still moving | **live fraction 102%** -- train acc never reaches 0.95 in 59 constraint epochs (0.491 by epoch 7) |
  | cap actually binds | `emitted == K` in EVERY group x class row |
  | prize is reachable | `certain%` = **0** |
  | cuts are contested | cut probability 0.12-0.90 |
  | headroom exists | `outside_tp` 2-69 per cell |

  Seed-paired, `tralo` - `tralo_null`: **L80 -0.00957 (t -2.23; F1 Macro -0.0108
  t -2.10; Accuracy -0.0109 t -2.55), 4/4 seeds negative; L90 -0.00925 (F1 Macro
  -0.0096, Accuracy -0.0091), 3/4 negative.** Negative in 6 of 6 primary cells.
  It also trails `clip` in both caps (-0.0189, -0.0169). **At ~100% live the
  constraint arrives on time to a permanently-moving boundary and damages it.**
  ⚠️ `SmallCNN` is diagnostic-only under FRAMEWORK 1 and can never carry a paper
  WIN; a negative result from it is a legitimate mechanism refutation.

- 🔴 **"THE MODEL IS TOO GOOD" IS REFUTED TWICE.** `scratch60` (MobileNetV3,
  `pretrained: false`, 32/32) drops cc-F1 to 0.475-0.495 and TraLO still loses:
  L90 `tralo` - `clip` **-0.0258 [-0.0431, -0.0085]** (Precision Macro -0.0180
  [-0.0308, -0.0051]; F1 Macro -0.0180; Accuracy -0.0164), and it loses to its own
  null (-0.0073). Two CIs exclude zero on the NEGATIVE side. Caveat recorded
  honestly: `scratch60`'s live fraction is 33% (19.6 / 59), so it FAILS
  `gate:saturation` and is the weaker of the two controls -- `small60` carries the
  argument.

- 🔴 **THERE IS NO EPOCH AT WHICH THE CONSTRAINT HELPS -- NOT EVEN WHILE CE IS
  LIVE.** `trace30` (48/48, budget 30), per-epoch curve from stored probability
  snapshots, `tralo` - `tralo_null`, per-epoch 95% Student-t intervals over 4
  seeds:

  | cap | epochs whose CI excludes 0 | chance alone | mean over epochs | typical 95% half-width |
  |---|---|---|---|---|
  | L80 | 2 of 29 | ~1.5 | -0.00005 | 0.0162 |
  | L90 | 4 of 29 | ~1.5 | -0.00185 | 0.0184 |

  **L80 is indistinguishable from noise; L90 is a weak NEGATIVE drift, not
  symmetric noise** -- all four excluding epochs are negative (-0.0059, -0.0083,
  -0.0148, -0.0195) and the largest is the final epoch. ⚠️ Corrected 2026-09-15,
  same day: an earlier version said "noise at every epoch", right for L80 and
  wrong for L90. Two consequences: the earliest L90 epoch reaching significance is
  **epoch 4, train accuracy 0.92, still live -- and it is NEGATIVE** (-0.00592 +-
  0.00460), so there is no early phase in which the constraint helps; and the
  ORACLE best epoch is 28 in one cap and 14 in the other, with headroom (+0.016 /
  +0.030) at or below the per-epoch 95% half-width (0.016 / 0.018). **The headroom
  IS the noise envelope.**

- 🟡 **INTERIM, NOT A VERDICT: at 1/3 completion the budget sweep shows NO signal,
  an order of magnitude below seed noise.** Read 2026-09-16 19:50 at 302 of 900
  runs, n=1-2 seeds per cell; the campaign continues and this must not be quoted as
  its result. Only the SIGN across the 36 (backbone x cap x budget) cells is
  readable at this n, for the attributable contrast `tralo` - `tralo_null`:

  | slice | cc_f1 | F1 (Macro) |
  |---|---|---|
  | ALL CELLS | 20/36 (56%), mean **+0.0009**, p=0.62 | 23/36 (64%), mean **+0.0015**, p=0.13 |
  | LIVE budgets b5-b8 | 13/24 (54%), +0.0008, p=0.84 | 16/24 (67%), +0.0025, p=0.15 |
  | SATURATED b12/30ep | 7/12 (58%), +0.0012, p=0.77 | 7/12 (58%), -0.0006, p=0.77 |
  | MobileNetV2 | 6/12 (50%), -0.0017 | 8/12 (67%), +0.0008 |
  | MobileNetV3 | 8/12 (67%), +0.0031 | 7/12 (58%), +0.0026 |
  | RegNetY400MF | 6/12 (50%), +0.0015 | 8/12 (67%), +0.0010 |

  **cc_f1, the primary endpoint, is a coin flip.** At the planned n=6 the smallest
  detectable paired difference is ~2 x 0.011/sqrt(6) ~ **0.009**, and every slice
  above is between +0.0008 and +0.0031, so the effect would have to be 3-10x larger
  than anything visible for the sweep to return a positive primary result.

### Regime, headroom and the reachable prize

- 🔑 **GATE 3: THE CAP BINDS EVERYWHERE, THE HEADROOM IS REAL, AND ~16% OF THE
  ALLOCATOR'S CAPACITY SITS IN A SATURATED REGION.** Measured 2026-09-16 on 135
  (backbone, cap, group, class) cells with `scripts/headroom.py`, aggregated by
  `scripts/gate3_summary.py`. This is the TEST-SIDE regime check that train
  accuracy structurally cannot give.

  | backbone | cap | binds | outside_tp | selected errors | cut > 0.99 | median cut |
  |---|---|---|---|---|---|---|
  | MobileNetV2 | L80 | 27/27 | 458 | 211/984 | 8/27 | 0.915 |
  | MobileNetV3 | L80 | 27/27 | 459 | 212/984 | 11/27 | 0.981 |
  | MobileNetV3 | L90 | 27/27 | 382 | 259/1108 | 10/27 | 0.866 |
  | RegNetY400MF | L80 | 27/27 | 478 | 231/984 | 11/27 | 0.921 |
  | RegNetY400MF | L90 | 27/27 | 411 | 288/1108 | 8/27 | 0.864 |

  The cap binds in 135 of 135 cells (`emitted == K`, no slack), 380-480 true
  positives sit outside the cut, and 21-26% of everything selected is wrong.

  ⚠️ **CORRECTED IN PLACE 2026-09-16: the cut-distribution split must be weighted
  by CAPACITY, not counted by cell.** 35.6% of CELLS cut above p=0.99, but those
  are the small ones (median K 11 against 33 for the rest), and quoting the cell
  share as the ceiling overstates it by half (`scripts/where_saturated.py`):

  | where the cut is | cells | med K | % of slots | outside_tp | selected errors |
  |---|---|---|---|---|---|
  | contested < 0.2 | 29 | 51 | **34.9%** | 661 | 342 |
  | middling 0.2-0.8 | 23 | 46 | 26.7% | 628 | 362 |
  | confident 0.8-0.99 | 35 | 18 | 22.1% | 477 | 245 |
  | SATURATED > 0.99 | 48 | 11 | **16.2%** | 422 | 252 |

  **The real ceiling is ~16% of slots, ~19% of the correctable headroom, ~21% of
  the selected errors. About 84% of capacity sits at a cut a model could in
  principle re-rank**, and the largest single band is CONTESTED. 🛑 **This closes
  an escape hatch: "the cuts were saturated so nothing could have worked" is NOT
  available** as an explanation for a ranking null. ⚠️ It also refines the
  train-side screen: "the model memorised the train set" and "the deployed cut is
  undecidable" are DIFFERENT claims and only the second bounds what a ranking loss
  can achieve.

- **Contested cuts rise as the budget falls.** The share of cuts whose marginal
  probability sits in [0.05, 0.95] goes from 36-43% at budget 30 to 44-50% at
  budgets 6-11.
- **Between a third and a half of the prize is unreachable, and the share falls
  with the budget.** Proportion of wasted slots held by items scored p(c) >= 0.99
  AND wrong: `fm2_mn3` (budget 30) 46% of 208 at L80, 39% of 262 at L90;
  `live6b` (budget 6) 33% of 212 and 28% of 265. So the 20-25% nominal prize is
  ~12-15% reachable at budget 30 and ~15-18% at budget 6 -- shorter training leaves
  the model less certain and more headroom in play. **4-seed pilot; no interval.**
  `~/reachable.py` on dsisco02.
  ⚠️ The same probe's `gap1` (probability distance from the weakest selected error
  to the strongest excluded true positive) is 0.013-0.014 at budget 30 and
  0.024-0.050 at budget 6. **Do not read that as the short budget being harder** --
  raw probability gaps are not comparable across regimes when calibration differs.
  The certain-wrong SHARE is the scale-free quantity.

- 🟡 **THE STAGE 1 RANKING LOSS WAS LIVE BUT NARROW: it fires on 8 of 139 TRAIN
  GROUPS and trains a 2nd-of-12 order statistic to serve a 41st-of-363 decision.**
  Measured 2026-09-15 on fmow2 before any `rank_*` run completed, by simulating the
  warm-up sampler (`rank_dose.py`, `rank_scale.py`). `make_dataloader` shuffles
  uniformly and is group-blind, so a batch of 64 over 139 countries holds ~12 items
  of the largest group. With `rank_min_group = 8`: 2.50 usable (group, class) terms
  per batch of 417 possible; 3.4% of batches carry no ranking gradient at all;
  contributing groups are USA 55%, FRA 22%, ITA 6% of batches, so **131 of 139
  never enter the gradient**; the `max(1, .)` floor pins k=1 in 49% of terms and
  biases the trained cut SHALLOW;
  cut depth train vs test 0.208 vs 0.148 (ratio 1.41 means, 1.52 medians).
  **Therefore a POSITIVE gAP effect would have been trustworthy; a NULL cannot
  separate "the channel does not help" from "this estimator is too noisy and too
  narrow".** The fix would be a sampler change, not a loss change: group-batched
  sampling would put ~64 items of one group in front of the cut and raise k from 2
  to ~10. That is a relaunch, hence a compute-budget question for the user.

- **THE RANKING LOSS RESULT.** `rank3_*`, 120 runs, 3 backbones x 2 caps x 4 seeds,
  all gates GREEN, 84 distinct models, zero cap collisions, 2026-09-16:
  `rank_clip` - `clip` is **4 of 18 cells positive at cell-mean gAP -0.0090**;
  `aug_rank_clip` - `aug_clip` is **7 of 18 at -0.0039**. On cc-F1 the ranking arm
  trails its control in **11 of 12 cells**, mean delta about **-0.005**. Against
  the measured null envelope `rank_clip` is MORE negative than either known null.
  **The budgeted ranking loss does not supply the "which"; it costs a little.**
  Mechanism in PART 2.4 and 2.5; the dose caveat above limits it to a refutation of
  THIS loss, not of the ranking CHANNEL.

- 📏 **THE gAP NOISE ENVELOPE, MEASURED ON TWO KNOWN-NULL CONTRASTS.** 2026-09-16
  on `rank1_*`'s 72 surviving control runs (3 backbones x 3 constrained classes x 4
  seeds, `scripts/rank_paired.py`):

  | contrast | cells positive | cell-mean | largest single cell |
  |---|---|---|---|
  | `focal_clip` - `clip` | 2 of 9 | **-0.0048** | -0.0261 at `|mean|/sd` 2.33 |
  | `aug_clip` - `clip` | 4 of 9 | **+0.0017** | +0.0316 at `|mean|/sd` 2.41 |

  Neither focal loss nor augmentation improves the SCORE the allocator reads
  (focal directionally negative, 7 of 9 cells; augmentation a null). 🛑 **And an
  isolated cell at `|mean|/sd` ~ 2.4 is what a NULL looks like here** -- both
  contrasts produced one, with opposite signs, at n=4. Cell-means of +-0.005 with
  per-cell sd 0.01-0.04 are the background; read any screen against that, not
  against zero.

### Corpus identity: how many independent cells actually exist

- 🛑 **TEN fmow2 CAMPAIGNS ARE BYTE-IDENTICAL RE-RUNS OF EACH OTHER.** Verified by
  prediction hash over every shared (model, cap, arm, seed): `fm2_mn3` == `gx2` ==
  `trace30` == `sat_fmow2`, `scr_MobileNetV3` == `stab8` (56/56), `rank1_*` ==
  `rank3_*` for all three backbones (24/24 each). **Counting campaigns, or pooling
  them, inflates n.** ⚠️ **CORRECTED 2026-09-16, same day:** this entry first said
  "1,364 runs carry only 822 distinct prediction hashes", inviting the reading that
  ~40% of the corpus is redundant seeds. **That conflates two things and the
  headline number is withdrawn.** Over all 1,414 fmow2 prediction files (858
  distinct hashes, 556 redundant):

  | cause | files | is it n-inflation? |
  |---|---|---|
  | cross-campaign re-run at the SAME cap | 46 | **yes** |
  | cross-CAP within one campaign | 231 | **no -- expected, and a proof; see below** |
  | both campaign and cap differ | 273 | mixture |
  | same hash across DIFFERENT arms | 6 | benign |

- 🟢 **ARM-IDENTITY AUDIT BY PREDICTION HASH: THE LAMBDA TOGGLE IS NOT INERT AND
  THE CLIPPERS REALLY ARE POST-HOC.** Five separate inert flags have been found in
  this project, so "the arm does what its name says" is not assumable. The test:
  if an arm's TRAINING does not depend on the cap, two caps at the same seed must
  give a byte-identical raw prediction file while metrics still differ. **All 231
  cross-cap identical pairs are nulls or clippers** -- `tralo_null` (39),
  `focal_clip` (37), `clip` (36), `aug_clip` (21), `aug_tralo_null` (17),
  `clip_b5/b6/b7/b8/b12` (8 each), `tralo_null_b5/b6/b7/b8/b12` (7 each),
  `focal_tralo_null` (6). **Not one constraint-active arm appears**; `tralo` and
  `aug_tralo` appear only as cross-campaign re-runs (25 and 16). So: `tralo_null`
  is a genuine null, `tralo` is genuinely cap-dependent (a null result for it is
  about the method, not an inert flag), and the clippers are genuinely post-hoc.
  The 6 same-hash-different-arm pairs are `clip` == `clip_b8` and `aug_clip` ==
  `aug_rank_clip` variants resolving to the same 8-epoch budget.

- 🛑 **ONLY TWO CAP LEVELS HAVE EVER RUN ON fmow2: `L80_G95` and `L90_G95`, both
  LOOSE.** Across all 1,364 runs there is no third pair, so "holds across
  constraint pairs" means "holds across two adjacent loose caps". Cap tags are
  parsed free-form by `cap_pair()` in `configs/gen_campaign.py`, so more levels need
  **no protocol.yml edit and no worktree** -- the gap is compute, not code.
  The dose axis, verified 2026-09-16 with the pipeline's own
  `compute_local_constraints` on `test_meta.csv` (not a hand-rolled approximation --
  `_round_to_K` rounds where an earlier check floored):

  | cap | binding ceilings (evict >= 10) | items evicted | share of test |
  |---|---|---|---|
  | `L40_G95` | 18 / 30 | 740 | **21.5%** |
  | `L55_G95` | 15 / 30 | 553 | 16.1% |
  | `L70_G95` | 12 / 30 | 370 | 10.7% |
  | `L80_G95` | 9 / 30 | 247 | 7.2% |
  | `L90_G95` | 5 / 30 | 123 | **3.6%** |

  Zero K=0 ceilings where the class is present. **A 6x dose range, and the two caps
  this project has always used are the two WEAKEST rungs.**

### Directional, not settled

- ✅ **CONFIRMED 2026-09-26 on fresh seeds (preregistered, set 1 = BANDCONS 2401-2424, cap 50): ENS - BASE clipper +1.62 [+0.53, +2.72] slots, Holm 0.011; tralo_null +0.96 [+0.02, +1.89], Holm 0.045; every other arm +1.6 to +3.0.** The post-hoc bar is now the clipper on the snapshot ensemble; compare TraLO ENS vs ENS. Original exploratory entry: **The final-epoch p3 is a noisy score at the cut; a snapshot ensemble beats the cut by 2-3.4 slots** (2026-09-26, post hoc, knee, the two #9 studies, n=24 per cap). capped_first on the mean of the epoch 6-10 dev probability snapshots, minus capped_first on the final ones: tralo_null +3.17 [+1.92, +4.46] slots (cap 76), +2.46 [+1.46, +3.58] (cap 50); clipper +3.42 [+2.54, +4.29], +2.04 [+1.08, +3.04]. A label-free gate found epoch history predicts wrong occupants beyond p3 (residual AUC 0.66-0.74); the plain ensemble is the stronger form of it. This exceeds every training-time effect measured, and it helps the clipper as much as TraLO: it is a stronger POST-HOC bar, and TraLO must be compared at equal ensembling. Chosen after seeing the data -> preregistered confirmation on fresh seeds (BANDCONS 2401-2424, CUTPAIR 2701-2724): `experiments/claude_snapshot_ensemble_prereg_20260926.md` on `claude/bandcons-20260926`.
- **Shorter budgets stop the damage.** gAP effect by live fraction,
  de-duplicated: 10.3% -> -0.0174, 13.8% -> -0.0274, 30.0% -> -0.0089,
  60.0% -> **+0.0079**. Monotone on a fixed backbone, 3 of 3, but Spearman p = 0.20
  over four campaigns and the one break is also the only MobileNetV2. **The same
  flip appears on the DEPLOYED selection**, independently of gAP: the constraint
  removes wasted slots at a 6-epoch budget (-33, -21) and adds them at 30 (+29,
  +88). ⚠️ **Superseded as an EXPLANATION** by `small60` (liveness is not the
  barrier) and by the interim budget sweep (the budget axis is not the missing
  ingredient); retained as a true description of the measured campaigns.
- **A LIVE boundary does not rescue the plain column.** Four budget-8 screens on
  fmow2 at live fractions of 72 / 55 / 69 / 37 %, 2026-09-15; the plain constraint
  effect on gAP is **NEGATIVE in 7 of 8 cells**:

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

  Every sd is 2-3x its mean, so no single cell is significant; the sign consistency
  over three backbones and two caps is the signal. gAP is allocation-free, so this
  is the MODEL moving -- the boundary IS being reshaped, and for the worse.
- **The augment x constraint interaction is positive and REPLICATES across
  backbones, still underpowered.** First seen on one backbone (+0.00892 +- 0.01142,
  3/4 seeds, p = 0.22; focal flat at -0.00157, the predicted dissociation --
  augmentation raises the live fraction, focal only enlarges the gradient). The
  budget-8 screens reproduce the sign in **5 of 6** computable cells across three
  backbones: +0.00744, +0.01036, +0.00835, +0.00423, +0.00941, with ViTB16 the lone
  negative (-0.00328) and ViTB16 the cell that FAILS the saturation gate. **The only
  direction in which the constraint is not harmful.** ~13 seeds are needed.
- **`stab8`: the constraint IS non-null in the live regime, and augmentation
  matches it for free.** 72 runs, warm-up 1 + 7 constraint epochs (~43% live),
  MobileNetV3 x fmow2 x 2 caps x 4 seeds, full panel scored 2026-09-16. The one
  real effect is in the only attributable contrast, `aug_tralo` against its own
  zero-constraint null: Precision (Macro) **+0.0118 [+0.0061, +0.0174], t +6.62**
  at L80 and +0.0169 [-0.0113, +0.0451], t +1.91 at L90; Recall (Macro) +0.0088
  (t +1.93) and +0.0118 (t +1.24). **This is NOT the eviction trade** -- a
  selection-rate constraint that merely evicts must raise precision and LOWER
  recall, and both move up in both caps. `aug_tralo_stab` reproduces the sign at
  L90 (Precision Macro +0.0134, t +2.99). **And it is worth nothing against the
  matched rival:** `aug_tralo` - `aug_clip` Precision (Macro) +0.0014 (t +0.10) at
  L80, +0.0077 (t +0.79) at L90, cc-F1 negative in both; the large wins against
  `clip` and `focal_clip` (cc-F1 +0.018 to +0.020, t 5.6-7.0) are the AUGMENTATION.
  The full panel is 658 paired tests and produced **17 CIs excluding zero on the
  positive side against ~33 expected by chance** -- fewer hits than noise.
  ⚠️ Contradicted by `small60` at higher liveness, so more likely a multiplicity
  artifact than first stated. **Pre-registered: it counts only if `aug_tralo` -
  `aug_tralo_null` on Precision (Macro) is positive in the `bud_*` confirmation
  set too.**
- **IS A SHORT BUDGET "UNDERTRAINED"? Only without augmentation.** MobileNetV3 x
  fmow2, same allocator, 4 seeds; `live6b` trains post-hoc arms for 6 CE epochs,
  `rank3` for 30 (both `constraint_epochs = 0`, verified from config):

  | arm | budget | cc-F1 L80 | cc-F1 L90 | F1 Macro L80 | F1 Macro L90 |
  |---|---|---|---|---|---|
  | `clip` | 6 | 0.6927 | 0.7154 | 0.6310 | 0.6422 |
  | `clip` | 30 | **0.7101** | **0.7301** | **0.6443** | **0.6551** |
  | `aug_clip` | 6 | **0.7172** | **0.7378** | **0.6510** | **0.6625** |
  | `aug_clip` | 30 | 0.7107 | 0.7332 | 0.6438 | 0.6558 |

  Without augmentation the short budget IS undertraining (`clip` loses -0.0174 /
  -0.0147 cc-F1 by stopping at 6). With augmentation it is NOT (`aug_clip` at 6
  beats itself at 30 by +0.0065 / +0.0046 and is the best arm in the table on every
  metric). **So "stop while still live" is ordinary early stopping and it is
  correct, provided the model is regularised enough to still be learning.** Budget
  and augmentation are one lever, not two. ⚠️ **And the constraint loses in every
  cell:** `tralo` at budget 6 is 0.6862 / 0.7131 (below `clip`), `aug_tralo` is
  0.7113 / 0.7338 (below `aug_clip`). **The best configuration measured anywhere in
  this corpus is `aug_clip` at budget 6 -- a post-hoc clipper with flip-and-crop and
  early stopping, carrying no constraint at all.** Cross-campaign, not paired
  (different code versions), so sizes are indicative; direction consistent across 2
  caps and 4 metrics. 🛑 **The running `bud_*` sweep has NO augmented arms**, so the
  strongest known configuration is invisible to it; an augmented budget sweep is
  required before any budget recommendation.
- **cc-F1 still goes the wrong way where the per-column nulls exist.** In `live11`,
  `aug_tralo_null` (0.6411) beats `aug_tralo` (0.6285). The apparent TraLO lead in
  `gx2` does not survive either: `aug_tralo` leads `aug_clip` by 0.0012, far inside
  seed noise, and that campaign has no `aug_tralo_null` at all.

### Datasets

- **`fmow2` is the dataset.** 17,670 train / 3,442 test, row counts consistent
  across images, labels and meta. **139 train countries vs 10 test countries, zero
  overlap; zero cross-split exact image duplicates.** Passes **8 of 8** conditions
  in `scripts.candidate_gate` (density 0.82, 6% dead items, 6/26 zero ceilings,
  class balance 0.57). Capped classes come from its own labels: **1 crop_field,
  2 place_of_worship, 7 ground_transportation_station**, present in 10/10, 9/10 and
  8/10 groups. **Measured hardness (MobileNetV3):** train CE saturates by epoch 6 at
  99.7% train accuracy while test accuracy is **0.634-0.648 over 4 seeds**, and a
  cell carries **187 errors inside K**; on ~11 of 30 ceilings p@K >= 0.99, exactly
  where the penalty's `p(1-p)` gradient is near zero. **That is a calibration limit,
  not a data limit.**
- **The original `fmow/oodslice` is WITHDRAWN.** `prep_fmow` joined metadata to
  images on `os.path.basename`, but the archive is laid out
  `split/class/class_seq/aoi/file` and the filename does not encode the AOI; on
  `val-metadata.tar.gz` 7,429 of 53,041 basenames appear under more than one AOI, so
  **16.4% of records were silently dropped or mis-joined**. Both sides now key on
  `class_seq/aoi/file` and `load()` refuses a non-unique key. Old arrays preserved;
  the rebuild is a separately versioned slice.
- **iwildcam is RETIRED.** 2 of 8 conditions: 2 of 8 classes can carry a local cap,
  half the per-group ceilings are K=0 before training starts, and 72% of test items
  sit in groups holding NEITHER capped class. Its per-group label shift is the best
  in the corpus (TV 0.737) and that is the SAME fact as its density of 0.27 -- the
  shift IS the sparsity. **Every pre-fmow2 TraLO number was measured through it, so
  treat those results as describing iwildcam rather than the method.** The two tools
  that could have caught this disagree by construction (`dataset_screen` rewards
  shift, `tier_viability` rewards density); `scripts.candidate_gate` screens all
  eight at once.
- **`bcn` is BLOCKED on integrity.** Two exact duplicate pairs cross train/test with
  conflicting class labels and different official lesion IDs; public source JPEG and
  annotation checks confirm the conflict is upstream, not introduced by our export.
  No images or labels were changed. Otherwise the best-structured slice available
  (candidate_gate 7/8, failing only class balance at 0.04 -- five tail classes
  holding 21% of the data), so repairing it is worth doing. A versioned curation
  policy and a renewed whole-split audit are required.
- **dermMNIST is REMOVED for leakage** -- 38.7% of test lesions appear in train.

### Infrastructure

- 🔑 **dsisco01 IS 3.1x SLOWER THAN dsisco02**, 2026-09-16, from the only paired
  comparison the corpus contains (`bud_mn3`/`bud_mn3b` and `bud_mn2`/`bud_mn2b` are
  the SAME grid frozen per host). An earlier "~3x faster" claim from a single
  campaign's rate was WITHDRAWN; this replaces it.

  | grid | host | runs | median gap between completions |
  |---|---|---|---|
  | `bud_mn3` | dsisco01 | 114 | 3.60 min |
  | `bud_mn3b` | dsisco02 | 134 | **1.17 min** |
  | `bud_mn2` | dsisco01 | 100 | 4.17 min |
  | `bud_mn2b` | dsisco02 | 113 | **1.32 min** |

  **MobileNetV3 3.08x, MobileNetV2 3.16x** -- two backbones agreeing to within 3%.
  Median gap, not span/(n-1), which absorbs startup and stalls. 🛑 **There is NO
  ViTB16 timing on dsisco01 at all** -- every ViTB16 campaign ever run
  (`fm2_vit`, `pilot_vit`, `scr_ViTB16`, `vit_a`, `vit_b`) ran on dsisco02, so any
  dsisco01 ViT estimate is the dsisco02 rate scaled by 3.1x (ViT is 10.4 min/run on
  dsisco02, hence ~32 min/run on dsisco01), not a measurement.

- 🔑 **THE WHOLE OPTION C CORPUS IN ONE TALLY -- 60 contrasts, 1 tail for
  TraLO and 4 against (re-scored from raw predictions 2026-09-20).** Both
  campaigns, both backbones, every cap x rival x primary endpoint. MobileNetV3
  252 runs / 12 seeds per cell, ViTB16 112 runs / 8, both BALANCED, both
  dsisco02/bf16, every prediction hash distinct (no re-run counted as a seed).

  | backbone | contrasts favouring TraLO | \|t\| >= 2 for | \|t\| >= 2 against |
  |---|---|---|---|
  | MobileNetV3 | 24 of 36 | 1 (L75 cc_f1 vs `fioretto`, +0.0058) | 0 |
  | ViTB16 | 4 of 24 | 0 | 4 (largest -0.0181, t -3.3) |

  ⚠️ **The 24-of-36 is NOT corroboration and must never be quoted as a sign
  test.** All 36 contrasts share the same `tralo` runs and the same seeds, so
  they are one heavily-correlated family, not 36 draws. The contrast that
  actually separates the method from the recipe -- `tralo` - `tralo_null` --
  peaks at **+0.0034, t +1.6 (p ~ 0.14)** on MobileNetV3 and is negative on
  ViTB16. **There is no pocket of positive evidence left in the corpus.**

  ⚠️ Stated symmetrically: n=12 detects ~0.009 and n=8 ~0.011 while these
  effects are ~0.003, so MobileNetV3's null is **underpowered, not evidence of
  absence**. The ViTB16 negatives do clear that bar, so the power caveat
  excuses the flat cells and not the negative ones.

---

## PART 4 -- Closed and rejected

- ⛔ **BANDCONS -- augmentation consistency on unlabeled development items at the cut -- DAMAGES the score, and its placement carries nothing. 2026-09-26, knee, cap 50, seeds 2401-2424, n=24, preregistered (run under declared deviation D-A1).** Holm over B1-B4 on capped_first grade-3 F1: bandcons - bandcons_rand -0.53 [-2.59, +1.52], bandcons - bandcons_unc -0.69 [-3.44, +2.05], **bandcons - tralo_null -3.47 [-5.38, -1.56] (Holm 0.004)**, **bandcons - aug_clip -2.88 [-5.03, -0.74] (Holm 0.032)**. All three variants lose ~3 F1 to the null; the term drives the natural grade-3 count between 0 and 792 across epochs (null 51-148), and 41.5% of its slot swaps go the correct way (clipper 45.9%). Augmentation stability is CLOSED as a training-time who-source on this cell; the cap-76 replication block is not run. Record: `experiments/claude_bandcons_cap50_result_20260926.md` on `claude/bandcons-20260926`.

- ⛔ **ViTB16 DOES NOT RESCUE OPTION C -- and it REVERSES the old corpus's only
  positive backbone. 2026-09-17, 112 usable runs, n=8.**

  Verdict by the pre-registered mapping (written before launch): **outcome 3,
  AMBIGUOUS**, because `cc_f1` splits sign across caps. ⚠️ **But the ambiguity is
  between "no effect" and "harmful", never between "no effect" and "helps": every
  cell that moves beyond noise moves NEGATIVE.**

  `tralo - tralo_null`, n=8 per cell:

  | cap | cc_f1 | F1 (Macro) |
  |---|---|---|
  | `L25_G25` | +0.0007 (t +0.2) | **-0.0165 (t -2.4)** |
  | `L75_G75` | **-0.0139 (t -2.0)** | -0.0187 (t -1.2) |

  Mechanism, precision of the filled ceilings (all arms saturate all ceilings, so
  only WHICH items differ):

  | arm | L25 | L75 |
  |---|---|---|
  | `tralo` | 0.5908 | **0.4610 (worst of seven)** |
  | `tralo_null` | 0.5932 | 0.4786 |
  | `focal_clip` | **0.5993** | **0.4838** |
  | `clip` | 0.5920 | 0.4809 |

  Attributable `tralo - tralo_null`: **-0.0024 at L25, -0.0176 at L75.** TraLO
  evicts **805 true positives at L75 against 784-798 for every rival** -- it is
  the only arm that throws away more than the budget forces. **On ViTB16 the
  constraint does not merely fail to help; it damages the ranking**, and the
  endpoint and the mechanism agree.

  🔑 **This REVERSES `project_the_vit_task_cells_are_underpowered` and the
  old corpus's ViT-only positive.** That result was measured under the
  PREVALENCE cap (2.9), where the ceiling restated the per-group label histogram.
  With the cap correctly specified, ViTB16 flips from the one hopeful backbone to
  the clearest negative one.

  ⚠️ **SCOPE AND A REAL DEFECT IN THIS RUN. 56 of 168 runs were LOST to
  `OSError 28: No space left on device`** -- `/home` hit 100% mid-campaign
  because the cleanup was blocked at the permission layer. The loss is uniform
  (every cell has exactly 8 seeds; `polcv2_a` kept seeds 1-4, `polcv2_b` 7-10),
  so the surviving set is balanced and matched-pairs, but **n=8 detects ~0.011,
  not the ~0.009 the pre-registration assumed. This should be re-run at full n
  once there is disk.**


- ⛔ **OPTION C IS REFUTED ON (MobileNetV3, fmow2) -- 252 runs, 12 seeds, 2026-09-17.**
  The pre-registered mapping (MISSION, written at 130/252 before any score was
  read) called outcome 3 when the pooled mean sits inside the seed sd. It does,
  on both primary endpoints.

  `tralo - tralo_null`, the only attributable contrast, n=12 per cap:

  | cap | cc_f1 | F1 (Macro) |
  |---|---|---|
  | `L25_G25` | -0.0004 (t -0.2) | -0.0047 (t -1.6) |
  | `L50_G50` | +0.0034 (t +1.6) | -0.0022 (t -0.6) |
  | `L75_G75` | +0.0023 (t +0.9) | +0.0003 (t +0.1) |

  Pooled mean +0.0018 (cc_f1) and -0.0022 (F1 Macro) against a seed sd of ~0.011.
  Detectable at n=12 is ~0.009. **The predicted dose-response is inverted**: the
  effect was to be LARGEST at the tightest cap, and L25 is the only negative cell
  on cc_f1 and the most negative on F1 Macro.

  🔑 **THE MECHANISM, AND IT IS THE REAL RESULT.** Every arm saturates all
  30 per-group ceilings (verified), so all seven spend an identical budget in
  identical cells and can differ ONLY in which items they place. Precision of the
  filled ceilings, n=12:

  | arm | L25 | L50 | L75 |
  |---|---|---|---|
  | `tralo` | 0.5890 | **0.5301** | 0.4812 |
  | `tralo_null` | 0.5888 | 0.5257 | 0.4801 |
  | `clip` | **0.5931** | 0.5245 | **0.4834** |
  | `alm` | 0.5909 | 0.5268 | 0.4792 |
  | `fioretto` | 0.5907 | 0.5285 | 0.4756 |
  | `hounie` | 0.5879 | 0.5276 | 0.4812 |
  | `focal_clip` | 0.5904 | 0.5255 | 0.4801 |

  **All seven methods land within 0.005 of each other at every cap -- a spread
  smaller than the seed noise.** True positives evicted are ~1050 / 906 / 787 and
  agree across arms to within 1.6 items: **the budget decides how many are
  evicted, and the method decides almost nothing about which.** Choosing a method
  moves roughly 5 items in 1000.

  ⚠️ **What this does and does not close.** The mis-specification was real
  (2.9) and is fixed; the global scope binds for the first time; the constraint
  is verifiably applied. TraLO still does not win. So **"TraLO looked flat only
  because the cap was mis-specified" is REFUTED.** Scope: one backbone, one
  dataset, one share rule (`proportional_to_group_size`). It does not yet close
  ViTB16, `equal` shares, or a second dataset.


- **`tralo_stab`, the weighted soft count -- CLOSED 2026-09-15/16.** Mechanism:
  replace `S_c = sum_i p_i(c)` with `S_c = sum_i w_i * p_i(c)`, so
  `dL/dp_i(c) = psi'(S_c) * w_i` instead of `psi'(S_c)`, with `w_i` the item's
  label-free neighbourhood DISagreement (`1 - knn_agree`, computed on the model's
  own test embeddings each constraint epoch, renormalised to mean 1 per scope so
  the dose is unchanged; hard counts stay unweighted). Shipped as
  `constraint_weight: knn_disagree`, k=20, floor 0.05, matched control
  `tralo_stab_null`. The weight can act ONLY through the constraint, so a null is
  unaffected by construction and any gain would have been attributable -- unlike
  graph diffusion, which modified predictions and helped the nulls most. Built,
  gated, proven live (`gate:weight_bites` PASSES, median weight cv **0.83 / 0.88**,
  uniform controls at exactly 0.0, 8/8 twin pairs byte-different), and it FAILS its
  pre-registered bar ("clear plain TraLO's +0.001..+0.004 tie with a CI excluding
  zero"). `stab8` (72 runs, budget 8, MobileNetV3, ~43% live), `tralo_stab` minus
  `tralo_null`, seed-paired:

  | cap | seed deltas | mean |
  |---|---|---|
  | L80 | -0.0060, -0.0078, +0.0157, -0.0032 | **-0.0003** |
  | L90 | -0.0001, -0.0113, -0.0154, +0.0010 | **-0.0064** |

  Also `tralo_stab` - `tralo` (L90) cc-F1 **-0.0095**; `aug_tralo_stab` -
  `aug_clip` cc-F1 -0.0053 (L80) and -0.0012 (L90), Precision (Macro) -0.0042
  (t -0.46) and +0.0042 (t +0.40). **A weight carrying correctness
  information the score lacks (knn_agree AUC 0.87 vs 0.68) still does not make the
  constraint useful** -- the sharpest available test of M5's one escape route, and
  it closes it. ⛔ An entry claiming this arm was "NOT RUN" was stale and is
  retracted; the campaign was complete on disk and unscored.
- **Capacity restriction -- REJECTED 2026-09-16.** Woodworth et al. justify
  in-processing only through hypothesis-class restriction, so a model that cannot
  memorise should be where the constraint finally wins. Both `scratch60` (33% live)
  and `small60` (102% live, never saturates) were complete on disk and unscored; the
  theory's own precondition was created and the constraint got WORSE (PART 3).
  **With it, liveness as the explanation of the damage is closed too.**
- **"The recipe saturates, so the constraint never had a chance" -- CLOSED
  2026-09-15** by `small60`. Retained as a true description of the OTHER campaigns;
  retired as an EXPLANATION for the damage.
- **Early stopping / per-epoch boundary selection -- CLOSED 2026-09-15** for the
  CONSTRAINT's contribution. `trace30` shows noise at every epoch on L80, a weak
  negative drift on L90, and an oracle that picks a different late epoch in each
  cap whose headroom is at or below the per-epoch noise half-width. The instrument
  (`epoch_trace` + `scripts/epoch_curve.py`) is kept: cheap, rides along on every
  tralo arm, and closed this in one campaign. ⚠️ **This bounds, but does not
  formally close, the val-split stopping rule in PART 5** -- that candidate must be
  read against this measured headroom before it is worth compute.
- ⛔ **THE ON-DISK SECOND-DATASET HUNT IS CLOSED: there is no second corpus on these
  servers.** Measured 2026-09-16, labels-only, no GPU. Of the four corpora with
  images on disk, `iwildcam` is retired (2/8), `bcn` is blocked and fails C8 (7/8),
  and the only 8/8 pass is `fmow`, the withdrawn basename-join slice -- whose clean
  bill of health is the defect's signature (PART 1). `fmow` is ALSO not independent
  of `fmow2`: compared on basename, the only shared key, `fmow2` test n `fmow` test
  = **2242 (53.8% of fmow test)**, `fmow2` TRAIN n `fmow` test = **1926 (46.2%)**,
  `fmow2` test n `fmow` train = 903. **An earlier reading in the same session that
  the two slices were item-disjoint was WRONG** -- it compared `fmow`'s bare
  basenames against `fmow2`'s full `class_seq/aoi/file` paths, which never match,
  and read incomparability as disjointness. A genuine second corpus requires a
  download and a fresh `prep_*` keyed on a provably unique join: a data-access and
  compute-budget decision to be asked, not taken.

### Historical hypotheses and their disposition

After the 2026-09-14 evidence reset. Historical results can identify risks and
tests; they cannot establish the new campaign's success or failure.

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

## PART 5 -- Live candidates, not yet tested

- ⛔ **RETRACTED AS A CANDIDATE -- OPTION C HAS NOW RUN AND IS CLOSED ON BOTH
  BACKBONES (2026-09-20).** MobileNetV3 refuted (252 runs, 12 seeds), ViTB16
  ambiguous-by-mapping and negative in direction (112 runs, 8 seeds). The
  verdicts are in PART 4. What remains true and reusable is the plumbing
  check below; it is kept because it is the evidence the cap is now specified
  correctly, not because the direction is open.

  ✅ **OPTION C IS CORRECTLY APPLIED END TO END -- verified 2026-09-17 on 52
  live runs.** Allocated predictions: **0 violations, 30.0/30 per-group ceilings
  saturated exactly**, in every arm at every cap level. Raw pre-allocation
  predictions: **52/52 violating**, 0.7/30 saturated. So the policy ceilings bind
  and the allocator delivers them; without it the model breaks every one.

  ⚠️ **Saturation at 30/30 is expected and does NOT mean the budget is slack.**
  The earlier "wasted capacity" figures counted ceilings above a group's TRUE
  positives; the allocator fills a ceiling from any item, and with 3442 items
  against a few hundred slots there is always something to place. **The cap is
  therefore a pure ranking question now: every method spends the same budget in
  the same cells, and only WHICH items differ.** That is the cleanest form this
  comparison has ever had.

  Mechanism during training (`L25_G25`, tralo, 6 live epochs): `L_Global`
  0.035 -> 56.1, `L_Local` 0.27 -> 404.7, `Grad_Norm` 2.7 -> 6399, satisfied
  flags 0 throughout, `Hard_Class1` 333-410 against a limit of 91. ALM likewise
  holds `total_excess` at 1769-2023. **Neither dual reaches feasibility in
  training at a strict cap** -- which the allocator makes moot at deployment, but
  it means the constraint term is a shaping pressure, never a satisfied one.

- 🔑 **THE BUDGET IS NOW 7, DERIVED, NOT INHERITED (2026-09-16).** The first
  Option C pilot ran at the protocol default of 30 and FAILED the saturation
  gate at 14 runs: MobileNetV3/fmow2 reaches train acc 0.962 by epoch 4, so the
  live window is 3 epochs and only **10%** of a 29-epoch constraint phase. The
  other 90% of the steps push a frozen boundary with CE ~0 opposing them.
  Killed early per the standing rule rather than run to completion; 18 completed
  runs preserved in `~/quarantine_2026-09-16/polc_*__29ep_saturated`.

  `total_epochs` moved 30 -> 7 (`2 x live + 1`) on the protocol DEFAULT, so every
  arm in the seven-arm comparison moves together and the dose stays equal. 30 was
  never measured; it was inherited.

  ⚠️ **Two tests were pinning 30 and went red**: `tests/gates/test_g4_grid.py`
  asserted `(total, warm) == (30, 1)`, and `test_lean_protocol` asserted
  `warmup + constraint == 30`. Both now check the PROPERTY -- a warm-up exists,
  the constraint phase is non-empty, every arm splits its own budget -- which is
  the same correction `scripts/check_parity.py` already carries in its header
  ("it was comparing to a number, not checking parity"). **A gate that pins the
  one knob the diagnostics tell you to move is a gate that blocks the fix.**

- ⛔ **RETRACTED: THE EQUAL-PERCENTAGE POLICY CAP HAS RUN.** Rule C
  (`proportional_to_group_size`) was selected by the user on 2026-09-17 and
  carried both Option C campaigns at L25/L50/L75. It answered both of the
  open design questions below, and the answer was negative on each: a cap
  decoupled from group size DOES change the deployed outcome, and the change
  does not favour TraLO. The measurement that motivated it still stands and
  is kept for the record.

  🟢 **THE EQUAL-PERCENTAGE POLICY CAP (`L50_G50` + `group_budget_shares`).**
  The code exists and is gated (2.9). It is the first
  configuration in which **both constraints bind** (2.7 says none so far did) and
  the first in which the local cap carries information the prevalence cannot
  reproduce. Two open design questions it would settle, which no existing cell
  can: whether a cap decoupled from group size changes the deployed outcome at
  all, and whether the 1/K gradient asymmetry (2.9) helps or hurts.

  🔑 **Measured 2026-09-16 on the real fmow2 test pool** (3442 items, 10
  countries, `L50_G50`). Forced-out / wasted items, summed over groups, where
  forced-out = `sum max(0, n - Phi)` and wasted = `sum max(0, Phi - n)`:

  | share rule | crop_field (Psi=182) | place_of_worship (Psi=273) | ground_transport (Psi=160) |
  |---|---|---|---|
  | A prevalence (current) | 185 / **0** | 273 / **0** | 161 / **0** |
  | B equal shares | 211 / 28 | 385 / 112 | 207 / 47 |
  | C share ~ group SIZE | 212 / 29 | 381 / 108 | 195 / 35 |

  ⚠️ **Under A the wasted capacity is exactly zero in all three classes, because
  `Phi_lambda(c) = round(pct * n_{lambda,c})` makes the ceiling vector an affine
  image of the per-group label histogram.** The cap is then a restatement of the
  per-group class counts, and a model given it has been handed the aggregate
  test-label distribution rather than an external budget. B and C decorrelate
  the ceiling from prevalence: 15-40% more items are forced out AND 28-112 slots
  are allocated where the group has nothing to fill them with. **That residual
  is the information the constraint would carry that the data does not.**

  B and C are also label-free at the group level -- equal shares need nothing,
  size shares need only `location` counts -- so they use strictly less label
  information than A, which needs the per-group class counts. Psi still needs
  the class totals under all three.

  **Blocked on a decision, not on code.** fmow2's local feature is a COUNTRY,
  not an entitlement tier, so no external policy supplies the shares the way a
  hospital's membership rules would. Picking the rule is picking the scientific
  question and needs the user.
- ✅ **DONE 2026-09-19: `danits_lp` IS RESTORED** to `reference/danits_lp/`
  (8 files, each recovered from its own last-present commit and verified by
  EXECUTION via `selfcheck.py`, not by reading). It is reference material:
  nothing under `reference/` is imported by `src/`. Wiring it in as a rival
  arm still needs OR-Tools on the server and is not done.

  ⛔ **RESTORE `danits_lp` BEFORE CLAIMING ANYTHING AGAINST THE LP.** Was absent
  from the live tree; `lp_solver.py`, `heuristic.py`, `cost_matrices.py`,
  `constraints_builder.py` and `train.py` all recoverable from `cb516cb3^`.
  Needs OR-Tools. Until then the manuscript's second post-hoc clipper has no
  implementation in this repo.

- ⛔ **CANCELLED 2026-09-20 -- ITS PREMISE WAS RETRACTED (2.3).** The constraint
  already runs on the test pool, where it is violated in 288 of 288 epochs.
  Routing it to a held-out train fold would make the term LESS violated, not
  more. Cancelled before any compute was spent.

  🟢 **COMPUTE THE CONSTRAINT ON A HELD-OUT FOLD OF THE TRAIN GROUPS.** The
  direction PART 2.3 opened, and the first candidate motivated by a measured zero
  rather than a theory of the surrogate. The term's violation is identically 0 on
  train because the model memorises it; on a fold cross-entropy does not fit,
  `V > 0` and the term is alive for all 30 epochs. Uses TRAIN labels only, touches
  no test label, so it is FRAMEWORK-legal. **A data-routing change, not a
  loss-function change.** `scripts/val_split.py` already exists.
- 🟢 **A GROUP-DISJOINT VALIDATION SPLIT IS CONSTRUCTIBLE** (`scripts/val_split.py`,
  2026-09-15), and it is the only thing standing between `epoch_curve` and a
  reportable stopping rule. `data/fmow2/oodslice/` ships **train and test only** --
  there is no validation split, so every epoch, checkpoint or hyperparameter ever
  chosen by looking at a curve was chosen against the TEST set. Train and test are
  group-disjoint by construction (139 countries vs 10, zero overlap; test is CAN,
  DZA, EGY, IND, IRQ, JPN, MEX, NLD, PHL, TUR), so a **row-shuffled val split would
  be the wrong instrument** -- same countries on both sides, an easier problem than
  deployment poses. 35 train countries carry >= 80 items, enough for a
  group-disjoint split imitating the test profile: ARG, BRA, CHE, CHL, DEU, KEN,
  KOR, PER, SVN, SYR gives **2920 items in 10 groups** against test's 3442 in 10,
  constrained-class shares matching to a total mismatch of **0.021**, leaving 83% of
  train. ⚠️ **Price it against PART 4 first:** `trace30` measured the oracle
  headroom at or below the per-epoch noise half-width, so a val-selected epoch is
  chasing a prize already bounded by noise. It costs a retrain of every arm, so it
  is a compute-budget decision to be asked, not taken. 🛑 The split must be carved
  from TRAIN countries only and its labels must never reach a gradient -- it selects
  among already-trained checkpoints and nothing else.
- 🟢 **THE BUDGET-PERMUTED TWIN -- BUILT AND GATED 2026-09-20, ready to launch.**
  `permute_group_budgets: <seed>` on a dataset config, plumbed through
  `data_loader`, `campaign` and `gen_campaign` exactly like `group_budget_shares`.
  Within each constrained class the multiset of ceilings is preserved exactly, so
  the per-class total -- and the tie to Psi that makes both constraints bind --
  is untouched; only the group-to-ceiling assignment moves.
  `tests/test_permuted_budgets.py`, 14 tests, and the gate is proved by MUTATION:
  an identity permutation fails 2 tests, a one-item change to a class total fails
  6. It WARNS instead of pretending when a class has one distinct ceiling across
  all groups, which is a genuine no-op rather than a permutation.

  Identical code and schedule, budgets permuted across
  groups within a class. By M2 this changes only the gain trajectory and leaves the
  field direction untouched, so it is the closest matched control obtainable. **If
  the effect survives permutation the budgets are not doing the work and both the
  transductive claim and the constraint claim fail; if it dies, the constraint claim
  survives its strongest available test.** Cheap. Never run.
- **Differentiable top-K through the allocator** (Petersen ICML 2022; Xie NeurIPS
  2020; Berthet NeurIPS 2020). Differentiates the SELECTION itself so the gradient
  stays rank-dependent through the backward pass -- the specific defect PART 2.4
  identifies. `rank_tralo` / `rank_tralo_null` are BUILT and gated but deliberately
  unrun. Build + 1 campaign, ~5.5h on 3 cards.
- **Group-batched sampling.** Draw each batch from one or a few groups, so ~64 items
  of a single group sit in front of the cut and k rises from 2 to ~10 -- fixes both
  the 8-of-139-groups dose deficit and the batch-`round()` cap collapse (PART 2.5).
  A sampler change plus a relaunch, ~5.5h. **Weakest of the live options**, because
  PART 2.4's diagnosis points at the surrogate: more gradient of a still
  uncertainty-weighted term.
- **An equal-percentage cap pair** (e.g. `L80_G80`), which by PART 2.7 is the only
  way the global term can bind at deployment. Needs no code change.
- **An augmented budget sweep.** The best configuration measured anywhere is
  `aug_clip` at budget 6, and the running `bud_*` sweep carries no augmented arms.
- **Learning rate as the second lever.** `lr` sets how fast the boundary freezes and
  has been fixed at 1e-4 throughout. The `lr1e-5` / `lr2e-5` / `lr5e-5` campaigns
  were GENERATED and never run (0/10 each), so "slow the boundary down" is
  unmeasured. Any change must set `constraint_phase.lr_constraint` to match, or
  unequal lr fabricates a result -- it did once already.
- **Candidate loss modifications**, each needing a stated mechanism and
  falsification criterion, none run: task-protected constraint displacement;
  temporal signed-residual controller; constrained posterior targets.

---

*Superseded documents are in `docs/archive/`, recoverable in full from git
history, and are QUARANTINED -- never source a claim from them. The pre-reset
theory document is `docs/archive/THEORY_pre-reset_2026-09-02.md`; PART 2 above is
the part of it that survived review.*
