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

- **`scripts/deployed_h2h.py` is the maintained reporter.** Hand-rolled cc-F1 has
  produced three scorer bugs.
- **Training accuracy alone cannot diagnose test-cut saturation.**
  `gate:saturation` reads train accuracy and is a SCREEN, not a diagnosis. Pair
  it with the regime check at the real allocation cut.
- **A gate is not done until a mutation makes it FAIL**, and the restore is
  verified by EXECUTING it. Stale bytecode has faked a pass.
- **Six instruments hardcoded the 30-epoch protocol** and would have mis-read any
  campaign that did not use it. Assume a seventh exists.

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
- **The augment x constraint interaction is positive but underpowered**:
  +0.00892 +- 0.01142, 3/4 seeds, p = 0.22. Focal is flat (-0.00157, 1/4), which
  is the predicted dissociation -- augmentation raises the live fraction, focal
  only enlarges the gradient.
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

- **The budget-permuted twin.** Identical code and schedule, budgets permuted
  across groups within a class. By the scalar-gain lemma (PART 2.2) this changes
  only the gain trajectory and leaves the field direction untouched, so it is the
  closest matched control obtainable. **If the effect survives permutation the
  budgets are not doing the work and both the transductive claim and the
  constraint claim fail. If it dies, the constraint claim survives its strongest
  available test.** Cheap. Never run.
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
