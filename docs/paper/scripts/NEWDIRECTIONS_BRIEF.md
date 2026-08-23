# TraLO: change of direction — shared brief

You are one of six research arms trying to find a genuine improvement to TraLO's
constraint loss. Read this whole file before proposing anything.

## Why this exists

TraLO's headline claim was that it beats baselines on constrained-class F1 at
tight caps. That claim is **dead**. A budget-equalized control — refill every
method's predictions to exactly the cap `K` from stored probabilities, then
re-score — collapsed a +0.044 lead to **−0.002**. The "win" was quota
utilization: TraLO spent 73 of its 75-prediction budget, Fioretto spent 62,
Hounie 51, and the post-hoc clippers spend exactly 75 by construction. Filling a
budget is a post-processing step that costs nothing, so it cannot be a
contribution.

Verified three ways before anyone acted on it: 72/72 runs reproduce their own
stored cc-F1 to 4dp; 24/24 clipper runs' realized count equals the reconstructed
`K`; and allocation-free average precision reproduces the same ordering.

**Your job is to find something that is not that.** A change that only moves
counts around has already failed. The metric that decides everything is average
precision on the constrained class, because it uses the scores directly, picks
no threshold and spends no budget, so no allocator can manufacture it.

## The four structural defects

Read `src/losses/transductive_loss.py` and `src/methodologies/tralo/train.py`.
The loss is

```
L = L_ce + Σ_c λ_c [ sat(E_c) + ρ · quad(E_c) ],
    E_c  = relu(S_c − K_c),      S_c = Σ_i p_{i,c}   (soft count over the pool)
    sat  = E/(E+K),              quad = (E/K)² / (1 + (E/K)²)
```

**1. The constraint gradient is rank one. This is the fatal one.**
The penalty depends on the weights *only* through the scalar `S_c`. So

```
∂L_pen/∂θ = (dL_pen/dS_c) · (∂S_c/∂θ)
```

— a scalar times one fixed direction. The constraint can push the network along
a single line in weight space and nothing else. Two consequences that the corpus
confirms: Adam's per-coordinate normalization absorbs the scalar entirely (a 297×
change in λ produced a 0.000e+00 change in the update direction), and every
ablation on penalty *scale* or *shape* came back null. If your proposal leaves
the penalty a function of the aggregate count alone, it will do nothing, and you
should expect the critique stage to say so.

**2. It is blind to *which* samples.** The loss cares that the count is right,
never that the *right K* samples were chosen. The clipper wins precisely by
choosing the top-K by score. Nothing in the objective rewards a better top-K.

**3. It is one-sided.** `relu(S_c − K)` charges overshoot and lets undershoot go
free, which is why an `undershoot_hinge` had to be bolted on as a separate term
instead of falling out of the formulation.

**4. Soft/hard mismatch.** The loss optimizes `Σ_i p_{i,c}`; satisfaction is
verified on `argmax`. These are different numbers and post-hoc adjustment exists
to paper over the gap.

Also known: per-group caps and the global cap are rounded independently and can
disagree (they sum to 76 against a global 75), so the local and global terms can
be mutually unsatisfiable. Fixing that touches shared code — see Blast radius.

## The regime you are optimizing for

**Short warm-up, compute-matched. Not warm-up 50.**

At warm-up 50 the CE-saturation gate (train accuracy ≥ 0.995) has already fired,
so during the constraint phase nothing is learning: every method can only
re-threshold a frozen score vector, and optimal re-thresholding *is* the post-hoc
clipper. That regime is unwinnable by construction and tells you nothing about a
loss function.

At short warm-up the representation is still plastic and the constraint term can
shape what is learned. That is the only place a real win can live.

**The comparison must be compute-matched.** The post-hoc arms do no
constraint-phase training at all — they train `warmup_epochs` and allocate. A
naive short-warm-up comparison pits a ~26-epoch trained model against a 1-epoch
clipper, which is why the corpus appears to show a huge TraLO win there. It is a
compute artifact. Every comparison you make is pinned to the same total
optimizer epochs: post-hoc arms get `warmup_epochs = B`, trained arms get
`warmup_epochs = 1, constraint_epochs = B − 1`.

## Scorecard

Run `python paper/scripts/score_arm.py --arm results/<yours> --name <yours>`.

| metric | meaning | role |
|---|---|---|
| `AP` | average precision, constrained class | **PRIMARY.** Allocation-free. An arm that does not move this has not won. |
| `ccF1eq` | constrained-class F1, all arms filled to exactly K | budget held fixed, so it measures allocation quality |
| `macroEq` | macro-F1 at equal budget | **GUARD.** Buying AP by wrecking other classes is not a win. |
| `count/K`, `sat` | realized count, native satisfaction | must not regress badly |

## Rules

- **Never change a field in the warm-up cache key.** `compute_base_model_id`
  hashes model, lr, dropout, batch size, warm-up epochs, pretrained,
  class-weighted CE, dataset, data dir, num classes, image size and **seed** —
  but *not* the methodology. Clone a reference config and change only the loss,
  and you start from bit-identical warm-up weights as the baseline you are
  compared against. Touch anything in that key and the warm-up silently
  retrains, breaking the pairing and reintroducing the cross-campaign drift
  (0.027 cc-F1) that already corrupted a published ablation row.
- **Novelty gate.** Free rein on the maths, but every proposal is checked
  against the literature before implementation. If your idea is already
  published, we are not going to reinvent it and claim it. Say so honestly and
  propose the delta.
- **Blast radius.** You may propose changes to shared code (post-hoc adjustment,
  constraint computation, verification), but you must state explicitly what it
  invalidates and which baselines would need recomputing. TraLO-internal changes
  cost 4 runs; shared changes cost a grid.
- **Disk.** `data/` (71G) and `model_cache/` are symlinks to the shared
  originals. Do not copy them. Do not train new warm-ups you do not need.
- **Smoke, don't sweep.** Four cells: {octmnist, dermmnist} × {MobileNetV3,
  RegNetY400MF}, `L30_G30`, seed 1, budget 30 epochs. Baselines already exist —
  never recompute them.
- **Your workspace** is `newdirections/arm_<name>/`, a git worktree on branch
  `nd/<name>`. Commit your changes there. Do not touch `~/OptimizationLoss`
  itself; it is the frozen reference checkout.
- Experiments run on the server with `conda activate optloss`, dispatched by
  `EXPERIMENT_DIR=<dir> python main.py <<< 0` (main.py prompts for a GPU on
  stdin). GPUs 0–3 are available; check `nvidia-smi` first and never share one.

## What a good proposal looks like

A specific change to the mathematics, with: the new objective written out; the
gradient, and an argument for why it is **not** rank-one; the mechanism by which
it should improve top-K quality rather than count accuracy; a falsifiable
prediction on the scorecard; and the cheapest experiment that could kill it.

"Tune λ / ρ / the schedule" is not a proposal. It has been swept and it is null.
