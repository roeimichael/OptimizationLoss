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

⛔ **REFUTED: "the octmnist slice is balanced 25/25 while the paper describes
8%/25%".** There is no 8% claim about octmnist anywhere in the manuscript; 7.1%
is tissuemnist's GE and it is correct. The paper already states that octmnist's
test split is class-balanced by construction, that drusen is therefore not the
minority the screening motivation assumes, and that the octmnist result must not
be read as evidence about rare-class screening. It is the most carefully hedged
dataset paragraph in the document. `scripts/prep_octmnist.py` taking 3,000 per
class is deliberate and matches what is written.

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

**Imbalanced-recipe hyperparameters are the PAPER's**: focal alpha=0.25 gamma=2,
class-balanced beta=0.9999, logit adjustment tau=1. (The `mcbar` campaigns ran focal at
alpha=1.0, which is **not** the paper's focal -- any focal number quoted from those runs is a
different arm from the one the manuscript describes.)

- **Historical dual runs are unusable** (300 epochs + the LR trap) -- re-run them in-campaign.

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
| `alm` | 701.2 | 701 |
| `hounie` | 0.0225 | 1.12e-05 |

Fioretto is 2.0e6 times hounie. Both fioretto and ALM blow past the unit-norm
clip for any plausible `||dS/dtheta||`, so the clip renormalizes them to the
same norm-1 step -- with a single active constraint the two arms take a
**bit-identical update** and differ only in how they weight the local caps
against each other. Hounie never reaches norm 1, so its constraint phase is 29
epochs of CE plus a numerically negligible nudge.

Deciding this requires choosing a convention and re-deriving each paper's step
size in it. **Until then, do not claim these are three distinct dual
baselines.** `Grad_Norm` is now logged per epoch in all four trained arms
(replacing the dead `L_KL` column), so the first campaign under this protocol
answers empirically how often each arm's raw norm crosses 1.0. Read it before
deciding.

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

**Result: 23,180 lines of Python -> 4,680 on 2026-08-15, and ~7,000 today** (4,801 `src` +
1,789 `scripts` + 608 `tests` + 267 `configs` + 163 `main.py`). It went back UP, on purpose: the
six restored baselines, five new gate scripts, and 96 tests. **Do not quote a line count as a
quality measure** -- it moved 4,680 -> 7,020 while the repository got strictly more correct.

What is actually load-bearing is that every one of those lines is reachable and every knob is
read: `audit_config` (no orphan hyperparameters), `smoke_arms` (all 10 arms run end to end),
`verify_caps` (the caps bind on the real slices), `check_parity` (equal compute, shared knobs,
no cross-objective warm-up sharing), and `pytest tests` (96 tests, ~16 s, no dataset needed).

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

**The meta-pattern behind 1, 8, 9 and 12: a thing that silently does nothing looks exactly like
a thing that does nothing useful.** An inert flag, a cancelled gradient, a non-binding cap and a
dropped pair all produce the same observable -- a tie -- as a real negative result. That is why
every guard above is a *pre-launch assertion* rather than a post-hoc analysis: after the fact the
two are indistinguishable.

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
- **Never let `lr_constraint` differ from `lr`.**
- **Never quote a metric that post-hoc filling can produce for free** (`flips`, raw count over K,
  proximity to feasibility). When quality ties, the honest report is "this arm produced nothing."
- **Never build a self-referential per-item term.** `rank` used top-K vs rest with no labels: it
  can sharpen a cut but cannot reorder, and ordering is the whole score.

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
configs/protocol.yml   EVERY experimental constant -- epochs, seeds, lr, caps, arms, backbones
configs/gen_campaign.py  THE generator: reads protocol.yml, holds no constant of its own
data/              dermmnist, octmnist, tissuemnist -- nothing else
docs/FRAMEWORK.md  this file
docs/archive/      history, not instructions
docs/paper/        the TMLR manuscript (main.tex is the professor's -- never edit)
results/           experiment outputs
scripts/full_panel.py  THE scorer (+ score_arm.py = the equalizer it calls)
scripts/audit_config.py  every hyperparameter has a reader; base_model_id is complete
scripts/smoke_arms.py    all 10 arms run end to end on synthetic tensors, ~40 s
scripts/verify_caps.py   the caps bind, on the real dataset slices
scripts/check_parity.py  equal compute, shared knobs, warm-up cache sharing
scripts/prep_*.py        dataset preparation
src/               the pipeline: losses, methodologies, models, pipeline, training, utils
tests/             96 tests, ~16 s, no dataset required
evidence/          archived provenance + predictions from every run ever made
```

Nine methodologies: `tralo` - the duals `fioretto_ldf` / `hounie_rcl` / `fioretto_alm` -
the two allocators `heuristic` (greedy) / `danits_lp` (LP-LG) - and the imbalanced recipes
`focal` / `class_balanced` / `logit_adjust`, which are LP-clipped.

## 6. Evidence appendix

The full run-by-run record, with numbers, p-values and cell counts, is preserved at
`docs/archive/REJECTED_full_2026-08-18.md`. It is history, not instructions.
