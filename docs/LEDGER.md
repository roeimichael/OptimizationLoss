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
