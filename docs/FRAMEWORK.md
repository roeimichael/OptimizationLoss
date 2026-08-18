# THE FRAMEWORK

**One file. Read it before proposing, running, or scoring anything.**
Everything else in `docs/` is history. If this file and any other document disagree, this file wins.

---

## 0. Where we actually are (2026-08-18)

We set out to build a **dual-loss** method that caps a class's prediction count while still
optimizing that class, beating both the dual baselines (Fioretto-LDF, Hounie-RCL) and the
post-hoc clippers.

**Current status: it does not beat a plain post-hoc clipper anywhere we have measured.**
Head-to-head against `clip` (plain CE + post-hoc), pooled over 48 seed-pairs:
macro-F1 -0.0017 (p=0.478, tie), accuracy -0.0027 (tie), and a **significant loss on
eight other metrics**. A baseline with no constraint training at all beats `focal_clip`
by more than our method does.

There is a structural reason, and it is the single most important thing in this file:

> **Post-hoc allocation thresholds the ranking at the budget. The score IS the ranking.**
> cc-F1 is precision@K rescaled; AP is the integral of precision@k. A gradient that is a
> function of the *aggregate count* cannot change *which* items are on top, so it cannot
> beat post-hoc, structurally. A count says how many, never which.

Every arm that varied the count penalty tied for exactly this reason. **A winner must act
per-item at the operating point, or it must operate in a regime where post-hoc is not optimal.**

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
bind. Measured on the real test sets (`python -m scripts.verify_caps`):

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

**Imbalanced-recipe hyperparameters are the PAPER's**: focal alpha=0.25 gamma=2,
class-balanced beta=0.9999, logit adjustment tau=1. (The `mcbar` campaigns ran focal at
alpha=1.0, which is **not** the paper's focal -- any focal number quoted from those runs is a
different arm from the one the manuscript describes.)

- **Historical dual runs are unusable** (300 epochs + the LR trap) -- re-run them in-campaign.

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
  (two-allocator confound).
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
  (oct L50 = 0.625, derm L30 = 0.809, tissue L30 = 0.795).
- ccP / ccR / ccF1 are one metric in three costumes on single-class problems.

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
| `src/losses/transductive_loss.py` | the per-constraint penalty math was **duplicated verbatim** between the global and local paths; `penalty_mode` carried four shapes (`rational`/`quadratic`/`both`/`linear`) of which three are rejected arms | one `_penalty()` and one `_sum()` shared by both scopes; one shape. **Verified numerically identical** to the old code across 532 randomized comparisons -- values, gradients, satisfaction flags, and the empty-constraint edge cases that must still return an autograd-connected zero |
| `main.py` (304 -> 164) | two dispatch paths; the threaded multi-GPU one ran several cards in **one process sharing one `model_cache`**, which races on the warm-up write | single GPU per process, matching how campaigns are actually launched (one process per card, own `EXPERIMENT_DIR`) |
| `configs/common.py` (94) | 94 lines for one live function | folded into `gen_campaign.py`; **`configs/` is now one file** |
| `src/training/__init__.py` (30) | re-exports nobody imported, including `ConstraintTrainer` which no longer exists | 4-line docstring |
| `LogitAdjustedLoss`, `_class_counts` | orphaned when their methodologies were deleted | gone |

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

✅ **An AST reachability pass now reports zero dead definitions** (the only hits are `forward`
methods, which PyTorch dispatches through `__call__`).

**Result: 23,180 lines of Python -> 4,680.** The pipeline imports and generates cleanly, the dispatcher
runs end to end, and a generated config now carries 11 hyperparameters, all of them live.

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
6. **Post-hoc local adjustment never re-violates the global cap** -- 199 runs, zero violations.
   Retired as a concern.
7. **The scorer is validated.** `equalize_multi` is arm-independent, budget-constant, feasible
   144/144, order-independent, not calibration-convertible, and agrees with an exact LP.
8. **Multi-class caps are genuinely supported** by all three methods -- no `[0]` truncation anywhere.

### Measured and open, no clean answer yet

- **Warm-up 1 delays CE saturation, it does not prevent it**: dermmnist saturates at epoch 15
  (8/8 runs), octmnist 25, tissuemnist 30. Across those three, TraLO does *best* on the *most*
  saturated dataset and *worst* on the least -- the opposite of the "room to move the boundary"
  prediction. **Confounded**: saturation timing tracks dataset difficulty. The clean test is
  within one dataset, using augmentation to hold CE unsaturated. Not yet run.

---

## 4. THE ONE OPEN QUESTION

Given section 0, only two kinds of thing can still win, and they are the only things worth building:

1. **A per-item objective at the operating point** -- something whose gradient depends on an
   individual item's position relative to the budget, not on the aggregate count.
   Three attempts are null so far; `budget_margin` on multi-class is the one untried variant.

2. **A regime where post-hoc is not optimal.** Post-hoc greedy is optimal only over its own
   candidate neighbourhood. It is weakest where the assignment is **coupled**: several capped
   classes plus local per-group caps. It is also uninformative where **train and test prevalence
   are identical**, which is true of every cap we have run -- the cap tells the model nothing it
   does not already know. **Distribution shift is the untested regime where the cap carries real
   information.** Novelty must be checked against label-shift / prior-correction work first.

**Anything that is not one of those two is a repeat of section 2. Do not run it.**

---

## 5. Repository layout

```
main.py            dispatcher (kill -INT to stop; interrupted runs reset to pending)
configs/           gen_campaign.py (THE generator) + common.py (hashing, cap tags)
data/              dermmnist, octmnist, tissuemnist -- nothing else
docs/FRAMEWORK.md  this file
docs/archive/      history, not instructions
docs/paper/        the TMLR manuscript (main.tex is the professor's -- never edit)
results/           experiment outputs
scripts/           full_panel.py + score_arm.py (THE scorer) + dataset prep
src/               the pipeline: losses, methodologies, models, pipeline, training, utils
evidence/          archived provenance + predictions from every run ever made
```

Nine methodologies: `tralo` - the duals `fioretto_ldf` / `hounie_rcl` / `fioretto_alm` -
the two allocators `heuristic` (greedy) / `danits_lp` (LP-LG) - and the imbalanced recipes
`focal` / `class_balanced` / `logit_adjust`, which are LP-clipped.

## 6. Evidence appendix

The full run-by-run record, with numbers, p-values and cell counts, is preserved at
`docs/archive/REJECTED_full_2026-08-18.md`. It is history, not instructions.
