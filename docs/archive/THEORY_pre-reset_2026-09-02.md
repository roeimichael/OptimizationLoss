# TraLO: transductive prediction-count constraints — theory, method, and evidence

**Status: written 2026-09-02 as a review document.** It states the thesis, the
method, the experimental design and everything measured so far, in the form a
reviewer needs to attack it. It is deliberately written against the project's
own interest: every negative result is here, and Section 9 lists the places the
argument is weakest.

Operational law lives in `docs/FRAMEWORK.md`. This document is the *theory*, and
where the two disagree, FRAMEWORK wins on protocol and this wins on nothing —
report the conflict instead.

---

## 0. Corrections from adversarial review (2026-09-02)

Three independent reviewers attacked this document — loss mathematics,
experimental statistics, theoretical framing — each instructed to verify against
source rather than trust the prose. **Everything below was then re-verified
here before being accepted.** Findings I could not reproduce are not listed.

### 0.1 A defect in the RNG floor, and it is in the shipped scorer

`configs/protocol.yml` builds `tralo_reseed` from blocks
`[constraint_phase, tralo_null, tralo_reseed]`. It therefore **inherits
`tralo_null`'s `lambda_step: 0.0`** and adds only `rng_reseed: True`.
`tralo_reseed` **is a lambda = 0 arm.**

So `|tralo − tralo_reseed|` is a *treated-vs-untreated* contrast, not an
RNG-only one — and `scripts/deployed_h2h.rng_floor` was computing exactly that
while its docstring asserted the two "differ in the RNG stream and in nothing
else". The genuine RNG-only pair is `tralo_null` vs `tralo_reseed`.

Measured on `dom1`, 24 paired cell-seeds, captured TP items over classes 2+7:

| contrast | median | mean | what it is |
|---|---|---|---|
| `\|tralo − tralo_reseed\|` | **4.0** | 5.50 | what the floor *used* |
| `\|tralo_null − tralo_reseed\|` | **6.5** | 8.62 | the true RNG-only floor |
| `(tralo − tralo_null)` signed | **+7.5** | +9.21 | the treatment, 19/24 positive |

🔑 **The direction of the error is the opposite of the obvious one.** The
contaminated floor was too **low**, so the tool was refusing too *seldom*. The
correction makes `deployed_h2h` more conservative and makes §8.1's "the
head-to-head is inside the noise" a **stronger** statement. Fixed, with the
self-test fixtures rebuilt around the lambda = 0 pair.

⚠️ **`MIN_PRIZE = 3.0` in `configs/task_windows.yml` was derived from the same
contaminated quantity.** On this evidence the RNG-only floor is ~6.5 items, so
the prize bar is roughly half what it should be — and that bar gates the task
windows, which gate the strict-cell set, which gates the unit ledger, which
gates the headline. **Not yet re-derived. This is the highest-priority open
item and it propagates.**

### 0.2 §3.2's chain rule was wrong, and the correction strengthens the claim

The document stated `d penalty / d z_ic = penalty'(S) · p_ic(1 − p_ic)`. That is
the **single-capped-class** expression. The protocol caps **two** classes, and
softmax couples them. The true gradient of the shipped loss is

```
    dL/dz_ij  =  sum over capped c of  pen_c'(S_c) · p_ic · (delta_cj − p_ij)
```

Verified against fp64 autograd on the shipped penalty with two binding budgets:
the document's formula errs by **5.98e-03**; the full-Jacobian expression is
exact to **3.5e-18**. Consequences the old formula denied:

* the **cross term** `− pen_7'(S_7) p_i7 p_i2` reaches **1.131×** the diagonal
  term, and can flip the sign of the net push on the capped logit — measured:
  1 sign disagreement in 200 items;
* the penalty also pushes **uncapped** logits (max |grad| 5.98e-03 where the
  document implied exactly zero) — a second reordering channel;
* the per-item scalar is not one scalar: with local scopes it is
  `lambda_g pen'(S_c) + lambda_{(g(i),c)} pen'(S_{g(i),c})`, which varies **by
  group** (14 groups, 7 of them permanently active at K = 0).

Every omitted term is *also* item-dependent, so §3.2's conclusion — the penalty
reorders — survives and is reinforced. But nothing quantitative may be built on
`pen' · p(1−p)` as an equality.

### 0.3 The "harmless path" is harmless for one capped class only

Subtracting a constant from `z_c` preserves class `c`'s order exactly (verified:
order identical, all `p_ic` fell). But with **two** capped classes it reorders
the *sibling*: a −1.0 shift on class 2 changes class 7's top-K by **3 items at
K = 50, 11 at K = 200, 41 at K = 800** (n = 5000). So "the cap is satisfiable
with zero reordering" is a theorem for |Ccap| = 1 and false for the protocol
actually run. The same qualifier applies to §8.8's "only an update confined to
`b_c` provably cannot reorder".

### 0.4 §3.5 is a heuristic, not a theorem

"A new count can only matter if it changes the direction" ignores the **value**
channel: `src/methodologies/tralo/train.py` gates the entire backward on
`has_constraint = total_constraint > 0`. A count whose *value* can reach `≤ K`
switches whole epochs off that `sum p` — permanently violated at K = 0 groups —
would have pushed. The repo's own `cut_window_count` records the extreme case: a
count pinned at `K − 0.5`, reporting excess 0.2 against a true 636.4, which
"cannot push" **with no direction change at all**.

### 0.5 Statistical claims that did not survive

* **`items = d(F1)·(K+n)/2` is exact per class only.** The reported cc-F1 is
  macro-averaged over two classes with different `(K+n)` (666 and 820 at
  dom1/MNv2/L80_G95), so `d(macro)` lives on a **two-quantum lattice** and the
  single-scale inversion over-states a class-2-only move by **1.116×** and
  understates a class-7-only move by **0.906×**. §5.1 presented the algebra
  clean; `paper_rows.py` and FRAMEWORK 2(z26) already carry the caveat.
* **The "1.9–9.9 items" headroom figure is from dermmnist** — the removed,
  leaking dataset (`full_panel.py` says so at the definition site). On iwildcam
  `ceiling_screen` gives 0.21–0.58 items at the old tight caps and 13.5–24.9 at
  the live loose caps. §5.1's calibration paragraph is wrong-dataset.
* **"1 of 158 rows clears 2 sd" is the chance expectation, not a finding.** The
  bar `|d| ≥ 2·sd` with `d` a 4-seed mean and `sd` the per-seed scale is a Welch
  `t ≥ 4` with df 3–6; over 158 rows chance alone yields ≈1.1 rows at df = 6 and
  ≈4.4 at df = 3. The honest statement is **0 of 158 resolve beyond chance** —
  which is *more* pessimistic than the document claimed, not less. The "sd is a
  lower bound" gloss is also directionally unsupported: a shared warm-up implies
  Cov ≥ 0, making the independence formula an over-estimate.
* **The sign-test p-values are one-sided and were never labelled.** Two-sided
  they are 0.125 / 0.0625 / 0.0078.
* 🛑 **The shipped `paper_rows.MEASURED_UNITS` contains FOUR units** (A1, A2,
  B1, C1), and FRAMEWORK 2(z26) states the current-recipe result as **4/4,
  p = 0.0625**. This document's §7 claims **5/5, p = 0.031**. The two cannot both
  be right, and the discrepancy is **unresolved as of writing**. Until it is,
  treat the headline as 4/4, p = 0.0625.

### 0.6 §8.1 does not refute dominance — it fails to measure it

A ratio of medians of *absolute* differences cannot separate location from
scale. With `sd ≈ 5.9` items, a genuine uniform **+2-item** dominance moves
median|X| only from 3.98 to ≈4.25 — indistinguishable from 1.00×. The correct
statement is **"dominance is unresolved at this resolution"**, not "refuted".
Note this is *more* favourable to the thesis than what was written, and it also
resolves the standing contradiction between §7 ("leads all four rival duals")
and §8.1 ("dominance refuted") — both were over-claims in opposite directions.

### 0.7 §1.2's transductivity argument does not hold

The claim that sample-specific budgets are "the only reason a training-time
method could beat a test-time one" is wrong: the post-hoc allocator sees the
budgets too, so there is **no information surplus**. Auditing the information
sets, the trained arm knows nothing the post-hoc arm cannot. What actually
differs, at equal compute, is the **set of reachable rankings** — an
optimisation-geometry object, not an information one.

Worse for the current framing: given the *true* posterior, the optimal feasible
labelling is an assignment LP over the posterior and the budgets — **computable
entirely post hoc**. So no training-time method can beat post-hoc-on-the-truth,
and the thesis question reduces to whether ~14 bag-marginal numbers improve a
*finite-sample* ranking estimate already trained on thousands of labels. That is
an LLP question, and LLP theory says bag marginals are weak supervision.

### 0.8 §9.2 is not one theorem — but a real theorem package sits just below it

The strong conjecture is **false**: a constructed two-cluster feature geometry
with a shared linear head produces a strict, deterministic allocation
improvement from one aggregate-count step, and **TENT** (Wang et al., ICLR 2021)
is a published counterexample — test-batch entropy minimisation is a separable,
permutation-invariant, multiset-only objective that demonstrably helps.

What *is* provable, and covers TraLO, every φ-count variant, and all four rival
duals:

1. **No value-level selection.** `L` is invariant under permuting test items
   (CE never reads them; every `S = Σφ(p_ic)` is symmetric), so `L` is a function
   of the *multiset* while the `clip` allocation is a function of the *ranks*.
   Any procedure reading only values of `L` cannot prefer a correct ordering
   over the worst ordering with the same multiset.
2. **The budget enters only as a scalar gain.** `∇_θ P = a(t)·V(θ)` with
   `a(t) = λ(t)·ψ'(S;K) ≥ 0` and `V(θ)` **independent of K**. The budget's
   information is reduced by the mechanism to a few non-negative scalars.
3. **Corollary — the four duals are one family.** They differ only in the gain
   schedule `μ(t)`, spanning the same cone. Their deployed differences *should*
   sit at the noise floor. **§8.1 is therefore a confirmed prediction, not an
   embarrassment.**
4. **Binary + decoupled ⇒ provable invariance.** The order is preserved for all
   time; the reordering channel exists only through softmax coupling (C ≥ 3) and
   weight sharing.
5. **Conditional harm lemma.** If current scores exhaust the available label
   information (`P(y=c|·) = q(s)`, q nondecreasing), any label-blind reordering
   has non-negative expected deficit. Applied to §8.3's own numbers:
   `73 × (0.688 − 0.301) ≈ −28.3` predicted against **−30.4 measured**.

Also corrected: §2's "both allocators are functions of the ranking" is **false
for the LP**, which maximises a linear functional of cardinal values — a
within-class monotone recalibration can flip its optimum with all rankings
intact. §8.4's invariance argument is exact for `clip` only, and only up to the
multiclass renormalisation channel (`p'_ic = w_c p_ic / Σ_k w_k p_ik` is not
within-class monotone for C ≥ 3).

### 0.9 What the reviewers agree the thesis should become

All three converge: **the negative structural result is the stronger and more
defensible thesis**, provided it renounces *impossibility* and claims instead:
no selection pressure (proved), provable invariance in degenerate regimes
(proved), expected harm under measured monotone calibration (proved
conditionally, and it predicts §8.3 to ~7%), and measured nulls here.

Two blocking inconsistencies must be fixed first: **§7 vs §8.3** (5/5 positive
vs its own twin, against 16/16 negative vs the same twin — presumably different
corpora, never stated), and the **asymmetric corpus boundary** (§10 restricts the
corpus to one recipe; the positives obey it, the negative mechanism numbers in
§8.3/§8.5/§8.6 quote archived off-recipe campaigns).

### 0.10 The decisive experiment nobody has run

For §9.3's warm-up confound: a **budget-permuted twin** — identical code and
schedule, budgets permuted across groups within a class. By the scalar-gain
lemma this changes *only* the gain trajectory, leaving the field direction
untouched, so it is the closest matched control obtainable. If the effect
survives permutation, the budgets are not doing the work and both the
transductive claim and the constraint claim fail. If it dies, the constraint
claim survives its strongest available test. **This is cheap and it is the next
campaign.**

---

## 1. The problem

### 1.1 Setting

A classifier is trained on a labelled training set and deployed on a **known,
finite, unlabelled test set**. At deployment we are additionally given
**prediction-count budgets**: upper bounds on how many items may be assigned to
particular classes.

Formally. Let the test set be `X = {x_1..x_n}` with unknown labels
`y_i in {1..C}`. A partition into **groups** `g(i) in {1..G}` is known (in our
data, the camera trap that took the photo). We are given:

* **local budgets** `K_{g,c}`: at most `K_{g,c}` items in group `g` may be
  predicted class `c`;
* a **global budget** `K_c`: at most `K_c` items overall may be predicted `c`.

Only a subset `Ccap` of classes is capped. A prediction vector
`yhat in {1..C}^n` is **feasible** iff

```
    |{i : g(i)=g, yhat_i=c}| <= K_{g,c}     for all g, all c in Ccap
    |{i :           yhat_i=c}| <= K_c       for all c in Ccap
```

The objective is prediction quality on the capped classes subject to
feasibility.

### 1.2 Why this is transductive, and why that is the whole claim

The test set is *available at training time* (unlabelled), and the budgets are
properties *of that specific test set*. So the constraint is not a property of
the data distribution — it is a property of the particular finite sample we must
label. This is what makes the setting transductive rather than inductive, and it
is the only reason a training-time method could beat a test-time one: a
test-time method sees the budgets too, so if the budgets carry information, both
methods have it.

**The thesis claim is therefore narrow and should be stated narrowly:**

> Folding the count constraint into training produces a better feasible
> labelling than applying the same constraint post hoc to an unconstrained
> model, at equal compute.

### 1.3 Where the budgets come from

In the experiments, `K_{g,c} = round(frac * n_{g,c})` where `n_{g,c}` is the
**true** number of class-`c` items in group `g` and `frac` is a cap tag (e.g.
`L80` = 0.80). A cap tag is written `L<frac_c2>-<frac_c7>_G<frac_global>`.

⚠️ **This is an oracle budget.** It is derived from test labels. That is
defensible as an *upper bound on what budget information can buy* — it models a
deployment where counts are known from an external source (a census, a manifest,
a prior survey) — but it is not a realistic operational budget, and any claim
must say so. See Section 9.1.

---

## 2. The baseline: post-hoc allocation

Given probabilities `p_ic` from *any* model, feasibility can be restored after
the fact. Two allocators are used:

* **`clip` (greedy).** For each capped class, keep the top-`K` items by `p_ic`
  and demote the rest to their best uncapped class; then fill remaining budget
  bidirectionally. Implemented in `src/utils/posthoc_adjustment.py`.
* **`lp` (LP-LG).** A two-phase linear program over the same probabilities.

**Key structural fact.** Both allocators are functions of the *ranking* of
`p_ic` within each group, not of the calibrated values. With exactly `K`
predictions emitted for a class, which items are chosen depends only on the
order. Two models with identical rankings and wildly different calibration
produce **identical** allocations.

This is the fact the entire thesis has to defeat, and Section 8 is largely the
story of it not being defeated.

---

## 3. TraLO

### 3.1 The objective

```
    L_total = L_CE  +  lambda_g * L_global  +  lambda_l * L_local
```

`L_CE` is ordinary cross-entropy on the *training* set. The two constraint terms
are computed on the *test* set (no labels needed — only the model's own
probabilities and the budgets).

**Soft counts.** The hard count `|{i : argmax_c p_ic = c}|` is not
differentiable, so it is replaced by

```
    S_c = sum_{i in scope} p_ic
```

(`soft_count_mode: sum`, the shipped default; alternatives in §3.5).

**The penalty.** For a capped (class, scope) with soft count `S` and budget `K`,
let the excess `E = relu(S - K)`, the scale `s = max(K, 1)`, and `e = E/s`:

```
    penalty(S, K)  =  E / (E + s)  +  rho * e^2 / (1 + e^2)
```

Rational saturation plus a bounded quadratic. The term is bounded in
`[0, 1 + rho)` regardless of how far over budget the model is. `rho` ramps over
the constraint phase; `lambda` ratchets per class.

```
    L_constraint = sum over capped (class, scope) of  lambda * penalty(S, K)
```

**Why `s = max(K,1)`.** At `K = 0` both forms pin at their bound
(`E/(E+0) = 1`, `(E/0)^2/(1+(E/0)^2) = 1`), giving a nonzero constant with
exactly zero gradient — so a group with no true instances of the capped class
would contribute nothing. The scaling is the identity for all `K >= 1`, so it is
bit-identical on every run made before the fix.

⚠️ **On iwildcam, 7 of 14 per-group ceilings are `K = 0`.** Since `sum_i p_ic > 0`
for any softmax, `relu(S - 0) > 0` always: **the soft constraint is never
satisfiable in those groups**, and contributes a permanent non-vanishing
downward pressure on `p_ic`. Satisfaction and the ratchet gate are decided from
**hard** counts, which *can* be exactly zero, so this does not stall the run.
Whether the permanent pressure is a feature (the class really is absent) or a
pathology is **an open question** — see §9.4.

### 3.2 The gradient, and the structural objection

The penalty is a function of the aggregate `S_c = sum_i p_ic`. By the chain
rule, the per-item derivative is

```
    d penalty / d p_ic  =  penalty'(S) ,   the same scalar within one (class, scope) term
```

⚠️ **CORRECTED — see §0.2.** The full gradient of the shipped loss, over both
capped classes and every scope, is

```
    dL/dz_ij  =  sum over capped c of  pen_c'(S_c) * p_ic * (delta_cj - p_ij)
```

The single-capped-class collapse `pen'(S) * p_ic(1 - p_ic)` is what this
section originally stated; it omits a cross-class term reaching 1.131x the
diagonal, omits the push on uncapped logits, and hides the per-group variation
of the scalar. Use it for intuition about the `p(1-p)` profile, never as an
equality.

**So the penalty pushes different items by different amounts, peaking at
`p = 1/2`, and therefore reorders the class.** This is not a side effect — it is
the only way an aggregate-count objective can interact with a ranking-based
allocator at all.

The measured consequence is Section 8.3 and it is negative.

**The harmless path exists and the loss does not take it.** Subtract a constant
from the capped class's logit: every `p_ic` falls monotonically and the
within-class order is exactly preserved. Nothing in the objective values the
order, so nothing selects this path — and worse, it is *anti*-selected, since
steepest descent concentrates the demotion on mid-`p` items precisely because
that maximises count-change per unit step norm.

⚠️ **But "zero reordering" holds for ONE capped class only (§0.3).** With
classes 2 and 7 both capped, a bias shift on `z_2` reorders class 7 — measured
at 3 / 11 / 41 items moved at K = 50 / 200 / 800.

### 3.3 Schedule

30 optimizer epochs in every arm, split two ways:

| arm type | warm-up epochs | constraint epochs |
|---|---|---|
| trained (`tralo`, duals) | 1 | 29 |
| post-hoc (`clip`, `lp`, ...) | 30 | 0 |

Equal compute. **Warm-up 1 is not a hyperparameter choice, it is the only live
regime**: at warm-up 50 CE saturates, `p(1-p)` at the cut falls by ~60x, and
every method becomes identical. Warm-up 5 is a measured dead zone; do not
interpolate.

### 3.4 Delivery: the part that is not the loss

A gradient in `prm.grad` is not a step. Measured facts:

* **`constraint_grad_mode`.** `clip` scales the constraint gradient by
  `min(raw_norm, 1)/raw_norm`; `normalize` rescales it to *exactly*
  `constraint_grad_clip`. Different arms have raw norms differing by ~20x, so a
  single absolute clip is a dose that varies per arm. **The two modes coincide
  exactly whenever the raw norm >= 1.**
* **`constraint_fp32`.** Running the constraint forward/backward outside
  autocast, bypassing the GradScaler. Measured over every completed run:
  `true` lands **15284/15284** constraint steps across 532 runs; `false` lands
  86.9% over 189 runs. A non-finite constraint gradient makes the step get
  dropped **while the run still writes `status: completed`**.
* **Adam.** The constraint step passes through Adam's moments. `train.py` puts
  ~126 CE steps between constraint steps, so `b1^126 = 1.7e-6` and the
  accumulation factor at a constraint step is `(1-b1)/(1-b1^(c+1)) = 0.1000` —
  the single-step value, forever. A gradient-level intervention (`ortho_project`)
  delivers **0.0%** of its promised CE-neutrality in 16/16 conditions: 92.6% of
  the momentum is stale CE the projection never touches, and `sqrt(v)` breaks the
  orthogonality of the rest.

**The general rule: verify any gradient-level arm at the weight-delta level.**

### 3.5 Count-function variants (all measured)

`S_c = sum_i phi(p_ic)` for various `phi`. The head gradient is `sum_i g_i f_i`,
a `g`-weighted mean feature; under `normalize` the magnitude is discarded, so a
new count can only matter if it changes the **direction**. Measured on real
post-ReLU features, the family forms three clusters at ~0.99 cosine within and
0.58–0.87 between:

```
    {uniform, 1-p}   {sum, margin}   {p, linear, cut-window}
```

`tralo_margin` sits at cosine 0.989 from `tralo` and would mostly reproduce it.

⚠️ A Gaussian toy says 1.0000 for all six and is **wrong** — real features are
non-negative and anisotropic.

---

## 4. The rival methods

Nine methodologies, all claimed in the paper:

* **TraLO** — the above.
* **Duals**: `fioretto_ldf` (Lagrangian dual framework), `hounie_rcl`
  (resilient constrained learning), `fioretto_alm` (augmented Lagrangian).
* **Allocators**: `heuristic` (= `clip`, greedy), `danits_lp` (LP-LG).
* **Imbalanced recipes**: `focal`, `class_balanced`, `logit_adjust`, each
  LP-clipped.

⛔ **Two of the three imbalanced baselines are mathematically inert on
iwildcam.** The TRAIN set is exactly 2500/class — imbalance 1.0x (the 4.5x
figure everyone quotes is the *test* set). So `class_balanced`'s weights are
exactly 1.0 and weighted CE is plain CE **bitwise**; `logit_adjust` adds
`tau * log(prior)`, a constant vector, and `log_softmax` is shift-invariant.

🛑 **And they fail differently, which is the methodological point.** `cb_lp`'s
raw predictions are byte-identical to `clip`'s in 24/24. **`la_lp`'s DIFFER in
24/24** — the constant moves float rounding by ~1e-9 and 30 epochs compound it.
Measured gradient delta vs CE: `max|g_v - g_ce| = 9.3e-10`, eight orders inside
the noise.

> **md5 divergence is not evidence of a live mechanism.** Identical predictions
> prove inertness; different predictions prove nothing. To clear a *loss*
> variant, compare its gradient against CE on the real training prior.

`focal` survives: it reweights per **example** and never reads the prior.

---

## 5. Experimental design

**Atomic cell** = (dataset, backbone, cap, method) over 4 seeds. Seed is the
only axis ever collapsed. Never pool across cap levels, backbones or datasets.

**Dataset.** `iwildcam` only (8 species; classes 2 = impala, 7 = cattle capped;
groups = camera traps; test cameras held out entire). The three MedMNIST sets
are removed: dermmnist leaked 38.7% of its test set, and octmnist/tissuemnist
built groups as `index % 3`, so their groups are i.i.d. draws from one
distribution and the local scope is empty **by construction**.

> **Triage rule.** A dataset famous for *domain shift* is not automatically one
> with *per-group label shift*, and only the second is usable. `rxrx1` fails
> despite 1,139 classes and real batch effects: every siRNA appears in every
> experiment by design.

**Backbones.** `ViTB16` (headline, fixed a priori 2026-08-20), `MobileNetV3`,
`MobileNetV2`, `RegNetY400MF`.

**Required controls in every campaign:**

* `tralo_null` — same warm-up, allocator and seed, `lambda = 0`. Isolates the
  constraint, and doubles as a post-hoc clipper at equal compute.
* `tralo_reseed` — that null with the RNG stream perturbed and nothing else.
  **This is the noise floor.**
* Both clippers (`clip` and `focal_clip`) *inside the same campaign*.
* At least two cap levels.

### 5.1 The metric, and its algebra

**cc-F1** — F1 on the capped classes. With exactly `K` predictions emitted for a
class with `n` true instances:

```
    F1 = 2 TP / (K + n)
```

so `d(F1)` must be an integer multiple of `2/(K+n)` (**not** `1/(K+n)`: TP is an
integer). **Convert to items:**

```
    items = d(F1) * (K + n) / 2
```

**Why this matters.** The whole gap from `clip` to a *perfect* allocator is
**1.9–9.9 items**. A paired seed sd is worth ~2.7. So `d(F1) = 0.02` is not a
small effect — it can be the entire headroom — and a sub-item delta is a
re-allocation, not a difference.

**A hard ceiling nobody's method can cross.** Emitting only `K` predictions for
a class with `n` true instances caps cc-F1 at `2K/(K+n)`. The whole prize for
*any* method is `(1-p@K) * K` items. No loss, dual, allocator or optimizer
changes that bound.

### 5.2 The task window

A cap poses a *question* only where three conditions hold at once:

1. **BINDS** — it evicts >= 10 predictions the model would have made;
2. **PRIZE** — there are >= 3.0 errors inside the budget the **local** allocator
   actually emits (3.0 = the measured RNG floor);
3. **WIGGLE** — `p@K < 0.99`, i.e. the cut is not buried in saturated scores.

Outside the window a cell measures the *absence* of a question, and a null there
is not evidence about any method.

🛑 **The screen was wrong until 2026-09-02, in two directions at once.** It
counted the prize over a **global** top-K while every allocator here is
per-group with 7 of 14 ceilings at zero — 8.5 errors global vs 2.0 local on
MobileNetV3 class 2, a **4.25x overstatement**. And it read wiggle at that same
global cut, which sits far above any group's own cut (p@K 0.99998 globally vs
0.98258 locally). The windows *moved*; they did not merely shrink.

### 5.3 The independent unit, and the power ceiling

🛑 **The unit is (backbone, HOST), not the campaign.** Across 14 worktrees there
are exactly **two** `tralo_null` models per (backbone, seed) — one per host,
dsisco02/bf16 and dsisco01/fp16 — with *identical* `base_model_id`. Two campaigns
on the same (backbone, host) are **one model, byte-identically**: `dom1` and
`loose1` produce byte-identical `tralo` predictions in 4/4 seeds. Two cap levels
in one campaign share a warm-up. So eight apparent cells can be four units.

**A sign test floors at `0.5^n`.** Four unanimous units cannot go below
p = 0.0625 *at any effect size*. Five reach 0.031.

### 5.4 The four noises, which differ by up to 12x

Quote which one you mean, every time:

| noise | what it is | typical |
|---|---|---|
| `unpaired` | one arm across seeds | 0.8–13.5 items |
| `reseed` | RNG only — the floor under any paired contrast | ~3.0 items/class |
| `treated` | the contrast actually run | 7.6–29.1 items |
| `full_panel`'s "paired seed sd" | macro-averaged `d ccF1`, **different units** | — |

⚠️ **Pairing GROWS the noise on this design, 6–12x.** `tralo` and `tralo_null`
share one warm-up epoch then train 29 apart — they are two *models*, not two
readings of one.

### 5.5 The power/binding dilemma

Measured on iwc3, class 2, seeds needed per cell at 80% power:

```
    K/n = 0.2  ->  2607 seeds
    K/n = 0.3  ->   546
    K/n = 0.5  ->   546
    K/n = 0.9  ->     7-8      <- the protocol runs 4
```

🛑 **Say this every time the result is quoted: at K/n = 0.9 the cap barely binds.
Where the constraint BINDS nothing is measurable, and where something is
measurable the constraint hardly constrains.** Half the prize costs 4x the
seeds. This is closed by the *cap choice*, not by physics — which is precisely
why it is uncomfortable.

---

## 6. Instruments

Scoring is split deliberately, because the two answers differ:

* **`full_panel`** re-derives its *own* equal-budget allocation from raw
  probabilities. It is therefore **allocator-blind by construction**: two arms
  sharing a warm-up score `+0.0000` on every budget-equalized metric however
  differently they allocate. It answers "whose *ranking* is better".
* **`deployed_h2h`** reads `final_predictions.csv` — what would actually be
  deployed — in exact captured items. It answers "which arm wins".

**They disagree in rank order.** At dom1/MNv2/L80_G95 the panel puts `tralo`
+5.77 over `alm` +5.49 while both capture *exactly 2602 items* — an artefact of
cc-F1 being macro-averaged over two classes whose `(K+n)` differ.

Pre-GPU screens that closed directions without spending a campaign:
`dataset_screen`, `ceiling_screen`, `task_window`, `paired_noise`,
`frozen_head_probe`, `scope_probe`, `graph_probe`, `straddle_probe`,
`step_direction_probe`, `ortho_survival`, `bias_shift_probe`.

---

## 7. What has been measured — positive

* **`tralo` > `clip` in 5/5 independent units, sign p = 0.031** (as-deployed,
  exact captured items).
* **`tralo` > its own `lambda = 0` twin in 5/5, p = 0.031.**
* **The constraint helps at loose caps** (loose1, L80/L90, 696/696 dose) — the
  first attributable positive effect.
* **`tralo` leads all four rival duals in dom1** (384 runs, 16 arms).
* **The uniform count removes the damage** (uniform1, 252 runs, 1044/1044
  steps): the constraint becomes approximately free.

---

## 8. What has been measured — negative, and this is the larger list

### 8.1 The head-to-head is the RNG floor

Over 19 cells: a #1 arm can be named in **6** and must be refused in **13**, and
of the 6 it splits `alm` 2 / `tralo` 2 / `fioretto` 2. Paired per seed,
`|tralo − rival|` has median **4.0 items** and `|tralo − tralo_reseed|` — the
same arm with only the RNG perturbed — has median **4.0 items**. **Ratio 1.00x.**
10 of 19 cells change their #1 when one seed is dropped.

**Dominance over the rival duals is refuted.**

### 8.2 Paper-level resolution

**1 of 158 strict-task rows clears 2 sd** — and that sd is a *lower* bound (it
assumes the arms are independent; they are two models sharing a warm-up,
measured at 6–12x). Everything else is a **sign**, not a measurement.

### 8.3 The constraint evicts the *correct* items

Against its own `lambda = 0` twin, the shipped `sum` count moves ~73 items per
cell. Items pushed **out** of budget are true positives **68.8%** of the time;
items pulled **in** are true positives **30.1%** of the time. **Net −30.4
items/cell on ViTB16, −3.4 on MobileNetV3, 16/16 negative.**

The control settles it: `tralo_reseed` moves a comparable 63 items and nets
**+0.38**, with evicted and admitted precision equal to three decimals.

**It is not a boundary effect.** The cut sits at p = 0.536 but evicted items
average p = 0.788 and admitted ones p = 0.251. The damage spans the whole range.

### 8.4 Top-K is invariant to prior shifts

A **1000x prior correction moves fewer top-K items than one RNG reseed.** Since
the allocator thresholds a ranking, any monotone recalibration is invisible to
it.

### 8.5 The constraint moves the count less than a dropout reseed

Constraint: 75–95 items RMS on the capped count. Reseed alone: 83–95.

### 8.6 The representation channel is negative

`tralo` vs its own `lambda = 0` twin: **AP −0.0306** (iwc1), −0.094 (iwc2).

### 8.7 Closed directions (do not re-propose)

Penalty-shape variants; more constraint steps (**more steps are worse** — the
starvation was protecting us); a dedicated constraint optimizer; the joint
objective (holds the cap 98.8% of epochs vs 6% **but −0.067 AP**); the undershoot
hinge; finer granularity (LLP); `rank`; `beta`; `select` (worst arm measured,
−22 items); one-vs-rest (dead twice); `tralo_uniform` (0/4 task cells, and its
founding claim is *refuted* — see below); local-cap scope pinning (−0.86 items
against wrong-shape controls costing 5.3–5.5); graph diffusion; KL anchoring.

### 8.8 A refuted founding claim, kept as a cautionary tale

`tralo_uniform`'s docstring argued that a uniform step in log-odds is "a pure
bias shift, which cannot reorder". **The step is taken in PARAMETERS, not
logits:** `dz_i = -lr * g * n * (fbar . f_i + 1)`, which varies with
`fbar . f_i`. It reorders, and it does so with the backbone frozen — the leak is
in the linear head. The only update that provably cannot reorder is one confined
to the bias `b_c`, and *that* one is useless: a constant added to `z_c` leaves
the within-class order untouched, so the emitted top-K is bit-identical.

Pure algebra, no artefact needed. **This is the shape of error most worth
hunting in this document.**

---

## 9. Where the argument is weakest — please attack these

### 9.1 The budgets are oracle-derived

`K = round(frac * n_{g,c})` uses test labels. Framed as an upper bound on what
budget information can buy, that is defensible; framed as a deployment scenario,
it is not. **Is there a formulation where the budget is estimated, and does the
whole effect survive it?** Note that a *noisy* budget would interact with §8.4:
if top-K is invariant to prior shifts, is it also invariant to budget error?

### 9.2 The aggregate-count / ranking impedance mismatch

§3.2 shows the penalty is a function of the aggregate and the allocator is a
function of the ranking. §8.4 shows monotone recalibration is invisible to the
allocator. **Is there a theorem here?** Something of the form: *any objective
that depends on the probabilities only through `sum_i phi(p_ic)` can affect a
top-K allocation only via the reordering it incidentally induces, and that
reordering is not selected for.* If that is provable, the negative results stop
being a series of unlucky experiments and become a structural statement — which
is a *better* thesis than the current one, and an honest one.

### 9.3 Warm-up 1 as the only live regime

The effect exists at warm-up 1 and vanishes at warm-up 50. The stated mechanism
is CE saturation dropping `p(1-p)` at the cut by ~60x. But warm-up 1 also means
the model is *bad*, and "a constraint helps a bad model" is a much weaker claim
than "a constraint helps". **Is the effect a constraint effect or a
regularisation-of-an-undertrained-model effect?** The `lambda = 0` twin controls
for compute but *not* for "any additional signal at all".

### 9.4 K = 0 groups

7 of 14 per-group ceilings are zero, and the soft constraint is *never
satisfiable* there — permanent non-vanishing downward pressure on `p_ic`. Is
this doing the work? It is the one place the constraint is guaranteed to be
active in every seed and every epoch, and it is also the place where the
"constraint" is really just "this class is absent here", which a model could be
told far more cheaply.

### 9.5 The power/binding dilemma is not obviously escapable

§5.5. If the only measurable regime is the one where the cap barely binds, what
exactly is being measured?

### 9.6 The unit count caps the achievable p-value

§5.3. With 4 backbones x 2 hosts, there are **8** possible independent units,
and four are spent. Even perfect unanimity across all eight gives p = 0.0039 by
sign test. Is the sign test the right instrument, or is there a better-powered
one that respects the clustering?

### 9.7 The metric is macro-averaged over two classes with different (K+n)

§6. This already produced a rank-order disagreement between two scorers on the
same runs. Is cc-F1 the right target at all?

---

## 10. Reproduction

```bash
python -m pytest tests -q                    # 526 regression tests
python -m scripts.preflight --before-launch  # staged experiment gates
python -m scripts.run_campaign --root <root> --step <step>
```

The corpus boundary is one recipe: `iwildcam + constraint_fp32: True +
constraint_grad_mode: normalize`. Anything else is a *different method*, not a
variant — five distinct TraLO configurations existed across 277 completed runs
and only 106 were current.
