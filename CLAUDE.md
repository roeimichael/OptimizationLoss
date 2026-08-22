# OptimizationLoss

Thesis project: train neural networks to satisfy **transductive prediction-count constraints**
via soft constraint optimization, and beat a post-hoc clipping baseline.

---

# STOP. READ `docs/FRAMEWORK.md` FIRST.

**It is the only operational document.** It holds the fixed experimental protocol, every idea
that has already failed and why, and the one open question. Everything else in `docs/` is history.

**Do not propose, run, or score anything before reading it.** If any other file disagrees with
it, `docs/FRAMEWORK.md` wins.

## The five rules that get broken most

1. **Warm-up 1 / constraint 29 for trained arms; warm-up 30 / constraint 0 for post-hoc arms.**
   30 optimizer epochs on both sides. **Never run warm-up 50** -- CE saturates and every method
   becomes identical. Never run warm-up 5 -- it is a dead zone; never interpolate between them.
2. **Score at equal compute, with BOTH clippers (`clip` and `focal_clip`) inside the campaign.**
   `clip` is the stronger quality bar. An arm-vs-arm delta is not a result until the bar is in
   the same campaign.
3. **md5 the raw predictions across arms before reading any metric.** Inert flags are this
   project's most frequent failure mode -- four occurrences and counting.
4. **Atomic cell = (dataset, backbone, cap, method) over 4 seeds. Count cells.** Never pool
   across cap levels, backbones or datasets. Always sweep at least two cap levels -- a
   single-cap claim has been retracted three times.
5. **`flips`, raw count over K, and "proximity to feasibility" are NOT metrics.** Post-hoc
   filling is free. When quality ties, the honest report is "this arm produced nothing."

## Do not run

Anything already in `docs/FRAMEWORK.md` section 2. In particular: penalty-shape variants,
more constraint steps, a dedicated constraint optimizer, the joint objective, the undershoot
hinge, finer constraint granularity. **All of them are measured, and all made things worse.**

## Where things are

```
main.py            dispatcher (kill -INT to stop; interrupted runs reset to pending)
configs/           gen_campaign.py = THE generator (asserts the protocol, refuses to
                   emit a single-cap campaign, always adds both clippers)
data/              iwildcam -- THE ONLY dataset. The original three are removed and
                   unrunnable, not merely discouraged; see `docs/FRAMEWORK.md` 2(n)
docs/FRAMEWORK.md  THE framework -- protocol, rejected ideas, code purge, open question
docs/archive/      history, not instructions
docs/paper/        the TMLR manuscript
results/           experiment outputs
scripts/           full_panel.py + score_arm.py = THE scorer; plus dataset prep
src/               the pipeline: losses, methodologies, models, pipeline, training, utils
evidence/          two tarballs: provenance for 14,524 runs, predictions for 128
                   (`mcbar` + `multiclass` only). Extract BOTH into one tree --
                   neither alone yields a scorable run. 0.9% is re-scorable.
```

Nine methodologies, all claimed in the paper: `tralo` - duals `fioretto_ldf` / `hounie_rcl` /
`fioretto_alm` - allocators `heuristic` (greedy clip) / `danits_lp` (LP-LG, Shifman) - and the
imbalanced recipes `focal` / `class_balanced` / `logit_adjust`, each LP-clipped.
**Before launching anything, run all three** -- each refuses a different way to waste a week:

```bash
python -m pytest tests -q                   # 246 regression tests, ~105s, no dataset needed
python -m scripts.audit_config              # no config key without a reader, no reader without a key
python -m scripts.smoke_arms                # every arm actually RUNS and respects its caps
python -m scripts.smoke_arms --matrix       # + {1,2} capped classes x {L30_G30, L50_G30},
                                            #   caps verified for the TRAINED arms too
python -m scripts.flag_live <armA> <armB>    # md5 across arms: is the new flag LIVE
                                            #   or a fifth inert one? (rule 3)
python -m scripts.verify_caps               # what integer budget each cap tag really produces
python -m scripts.check_parity <root>       # equal compute, same knobs, >=2 caps, sane warm-up sharing
python -m scripts.reachability <early-run>  # CAN the penalty even reach this cell's cut?
```

## Reading a result

```bash
python -m scripts.full_panel --campaign <root> --control clip   # THE scorer, seed-paired
python -m scripts.log_health <root>        # what the OPTIMISATION did, per run, from
                                            #   training_log.csv -- collapse, divergence,
                                            #   satisfaction, count trajectory vs K
python -m scripts.paired_seeds <scan-root>  # each arm minus its OWN lambda=0 twin, per seed
python -m scripts.score_scan <root>         # AUROC / prec@K / Jaccard, grouped by CELL
python -m scripts.headroom <root>           # items from `clip` to a PERFECT allocator,
                                            #   per cell -- the ceiling any arm is chasing
```

⚠️ `full_panel` now prints a **RESOLUTION** block per contrast: the within-cell
seed sd in items, and the seeds needed at 80% power beside the seeds present. **Read it
before the verdict.** A tie means "no effect" OR "not enough seeds", and those are
opposite conclusions from the same table -- on the live `dualbar2` one contrast reads
`observed +0.36 items, needs ~174 seeds per cell`. It refuses to print a figure at all
when no cell has two seeds, rather than deriving one from nothing.

## Pricing a direction BEFORE spending a GPU

All five run on CPU in minutes against artefacts that already exist, and every one carries
its own liveness control, so a null from them is a measurement rather than silence. Each
closed a direction this project would otherwise have spent a campaign on.

```bash
python -m scripts.frozen_head_probe --run-dir <run> --seeds 1 2 ... # refit ONLY a linear
                                            #   head on the frozen features under a
                                            #   different loss; verdicts in ITEMS, and
                                            #   `seeds_needed` prices any survivor in
                                            #   CAMPAIGN seeds (topk/ptopk: +1.2-1.3
                                            #   items but ~24-36 seeds/cell => unaffordable)
python -m scripts.dataset_screen <slice-dir> ...  # CAN a count constraint carry
                                            #   information here? Labels + metadata only,
                                            #   no images/model/GPU. Read the NET column:
                                            #   the DIFFERENTIAL per-group shift, after
                                            #   subtracting BOTH a sampling-noise null and
                                            #   the global shift. octmnist -7, tissuemnist
                                            #   -55 = DEAD (`synth_group` is `index % 3`);
                                            #   derm slice_1 +65 passes stage 1 and STILL
                                            #   nulls, so stage 1 is necessary only --
                                            #   stage 2 is `scope_probe --calibrate`
python -m scripts.scope_probe --campaign <root>   # `L20_G50` and `L50_G20` impose the
                                            #   SAME TOTAL, so the local-vs-global SCOPE
                                            #   question is answerable with the model held
                                            #   fixed. CLOSED the local-cap direction:
                                            #   pinning the split -0.86 items while
                                            #   wrong-shape controls cost 5.3-5.5.
                                            #   `--oracle-split` ALWAYS prints its
                                            #   transfer: the best split found with labels
                                            #   gains +4.18 and transfers at -0.89, so an
                                            #   oracle quoted alone is selection noise
python -m scripts.graph_probe --campaign <root>  # diffuse the scores over a kNN graph of
                                            #   the stored embeddings -- the one input the
                                            #   allocator provably lacks. NULL: +0.50
                                            #   items, 10/19, while its shuffled-graph and
                                            #   shuffled-feature controls lose 5.8-8.4
python -m scripts.straddle_probe --campaign <root>  # how much of the ORACLE headroom is
                                            #   REACHABLE by a step the size ours actually
                                            #   is? `headroom.py` assumes the ranking can
                                            #   be rewritten arbitrarily; 2(a3) measured
                                            #   that we deliver exactly `lr*clip`, so an
                                            #   item misranked by a wide margin is not
                                            #   reachable at any dose. delta is MEASURED
                                            #   from each arm's own `_null` twin, not
                                            #   assumed. `--self-test` gates it.
                                            #   `contested` is LABEL-free but NOT
                                            #   model-free -- no model, no ranking, no
                                            #   cut. `dataset_screen` is the pre-GPU one
```

`frozen_head_probe`, `graph_probe`, `scope_probe` and `straddle_probe` need
`test_embeddings.npz`, written by `src/pipeline/features.py` at the end of every run
finished after 2026-08-22. Runs predating it cannot be probed and **must not be
substituted for with synthetic data** -- the probes refuse rather than fall back.
`dataset_screen` is the exception: labels and metadata only, so it runs on a candidate
slice before a single image is ever loaded.

⚠️ **Read `straddle_probe`'s shuffled control in the right DIRECTION.** Shuffling the
scores does not send `reachable` to zero, it RAISES it -- a random top-K scatters
positives on both sides of the cut. It is a *reference* (it depends on n, K and prevalence
only, measured at 10.8 vs 11.6 items across two regimes whose error structures differ 5x),
and the SIGN of the deviation is the result: `reachable << ctrl` means the ranking already
took the easy swaps, `~= ctrl` means the statistic is reading the score distribution and
means nothing, and `>> ctrl` means positives are parked BELOW the cut -- the one case in
which a cut-local method has something real to win.

**Three rules that cost a night each to learn:**

1. **Carry the `_null` arm AND `tralo_reseed`** (`--arms all+null`). The null is the same
   warm-up, allocator and seed with lambda=0, so it isolates the constraint -- and it
   doubles as a post-hoc clipper at equal compute with the allocator held fixed. Without
   it **no count trajectory is attributable**: CE alone swings the capped counts
   242 -> 227 -> 324 -> 233. `tralo_reseed` is that null with the RNG stream perturbed and
   nothing else, and it is the **noise floor**: the constraint moves the capped count RMS
   75-95 items, a reseed moves it 83-95. `gen_campaign` REFUSES a campaign that holds a
   trained arm without it.
2. **Read `d capF1` beside `d macroF1`.** Paired over seeds their precision differs by an
   order of magnitude, and macro-F1 is carried by the UNCAPPED classes, which swing with
   the seed. `d capF1` is quantised -- with exactly K predictions emitted, `F1 = 2TP/(K+n)`
   -- so it must be an integer multiple of `1/(K+n)` or there is an arithmetic bug.
   **CONVERT IT TO ITEMS: `items = dF1 * (K+n)/2`.** `full_panel` prints the scale per
   cell. The whole gap from `clip` to a PERFECT allocator is **1.9-9.9 items**, and the
   paired seed sd is worth ~2.7 -- so 0.02 is not a small effect, it can be the entire
   headroom, and a sub-item delta is a re-allocation, not a difference.
3. **Check reachability before choosing a cap.** The penalty's per-item gradient scales
   with `p(1-p)`. At the K-th RANKED item that is 0.026 at `L30_G20` (0/4 seeds respond)
   vs 0.055 at `L50_G30` (4/4), and converging the model drops it 60x -- which is what
   "CE saturates" means and why warm-up 50 makes every method identical.
   ⚠️ **But rank K is NOT the decision boundary**, and the two get conflated. When the
   hard count is 300 against K=44, the boundary is at item 300 and rank 44 is buried
   inside the class. At the boundary `p(1-p)` is near its MAXIMUM, and `sum` already puts
   29.4% of its gradient there. Say which point you mean; `docs/FRAMEWORK.md` section 4
   has the measurement.

`smoke_arms` exists because the config gates are structurally blind to a runtime
crash: three arms once shipped with an undefined name in `train()`, burned all 29
constraint epochs, died, were reset to `pending`, and the campaign came back
looking merely unfinished -- with `audit_config` and `check_parity` both green.

**The global cap is redundant at `L30_G30` / `L50_G50` and inert at any `G > L`** -- local caps
are per-group ceilings, so their sum already bounds the count. To make the global scope bind,
sweep `G < L` (e.g. `L50_G30`). See `docs/FRAMEWORK.md` section 1.
Generate a campaign with:

```bash
python -m configs.gen_campaign --root results/<name>     --datasets iwildcam --models MobileNetV3     --caps L20_G50 L30_G50 --arms all+null
```

## Datasets

**`iwildcam` is the only one.** 8 species, classes 2 (impala) and 7 (cattle) capped,
`location` = camera trap, and the test cameras are held out ENTIRE. **No AIDER, no
EuroSAT, no others.**

`dermmnist`, `octmnist` and `tissuemnist` are REMOVED -- the rows below are the evidence
for why, not an offer to run them.

🟢 **`iwildcam` is the ONE that can carry a constraint**, and the other three are now
understood not to. Screen them with `scripts.dataset_screen`, which reports the
DIFFERENTIAL per-group novelty net of sampling noise and the global shift:

| dataset | NET items | z | unseen groups |
|---|---|---|---|
| **iwildcam/oodslice** | **+3131** | **97.4** | **7 (all test cameras)** |
| dermmnist/slice_1 | +65 | 2.9 | 0 |
| octmnist/slice_1 | -7 | -0.4 | 0 |
| tissuemnist | -56 | -1.9 | 0 |

⚠️ **octmnist and tissuemnist are structurally dead** -- `synth_group` is
`np.arange(len(y)) % 3`, so their groups are i.i.d. draws from one distribution
and the local scope is empty **by construction**. Two of the original three
could never have tested the thing being tested. `data/dermmnist/shift_1` looks
better at LOCAL=160 but 110 of that is the global shift replicated across
groups; it has never been used and should not be.

🛑 On `iwildcam`, **7 of 14 per-group ceilings are K=0** ("predict none of this
species at this camera"). A zero ceiling binds regardless of sum slack, so the
LOCAL scope constrains the output at every cap level -- unlike dermmnist, where
`lp_fallback_used` was False with 0 candidates on all 52 runs. `gen_campaign`
now reads the real budgets and says so; do NOT trust the sum-arithmetic line
alone. See `docs/FRAMEWORK.md` section 2(n).

## Backbones

`MobileNetV3` (headline), `MobileNetV2`, `RegNetY400MF`, `ViTB16`. **Nothing else** -- these are
exactly the four the paper claims. ShuffleNetV2 and the small CNNs were deleted; they appear in
no `.tex` file.

## Loss

```
L_total = L_ce + lambda_g * L_global + lambda_l * L_local
```

Rational saturation `E/(E+K)` plus bounded quadratic. Soft counts (differentiable) for the
gradient, hard counts (argmax) for verification; post-hoc adjustment closes the gap.
**KL is out of scope.** The `alpha_kl` key and the whole KL anchor are DELETED from the
pipeline -- there is no setting to get wrong. Same for the CE-saturation skip
(`enable_ce_skip`), the undershoot hinge, and the `bounded_only` penalty branch.

## Infrastructure

- **Never run experiments locally.** SSH `dsisco01` / `dsisco02`, `conda activate optloss`.
- 🛑 **NEVER touch `src/`, `configs/` or `main.py` on the SERVER while a campaign is
  running** -- not even a comment. `code_version` is a git hash, so any edit splits the
  campaign into two non-comparable halves and turns `check_parity`'s "every arm from one
  commit" red. Deploy after the last run, never during. `scripts/` is exempt and safe to
  update mid-flight: nothing under it is on `src.experiments.runner`'s import path, which
  is why the scorer and the offline probes can be iterated while runs land. **Check
  `git status --porcelain src/ configs/ main.py` on the server, not just `git status`** --
  a tree dirty only in `scripts/` is the normal working state and says nothing.
- **Max 2 GPUs.** Run `nvidia-smi` **with owner lookup** first; never share a GPU with another user.
- dsisco01 = Quadro RTX 6000 (FP16 + GradScaler). dsisco02 = RTX PRO 6000 Blackwell (BF16 AMP).
  Record which one a result came from.
- Any hyperparameter that changes what warm-up optimizes **must** be in `compute_base_model_id`,
  or the second arm silently loads the first one's cached model.

## Paper

`docs/paper/main.tex` is the professor's file -- **never edit it**. Edit `docs/paper/main_edited_by_roei.tex`.
Appendix tables stay in the appendix.

**Five manuscripts sit in `docs/paper/`. `main_edited_by_roei.tex` is the paper of
record** -- it is the one to edit and the one to read a claim out of.

| File | What it is | Reads |
|---|---|---|
| `main_edited_by_roei.tex` | ✅ **the paper of record**, additions in blue | `tables/` + `tables_rev/` |
| `main.tex` | the professor's file. **Never edit** | `tables/` |
| `main_rev.tex` | the revision `main_edited_by_roei` was branched from | `tables/` + `tables_rev/` |
| `main_clean.tex` | a de-marked-up snapshot | `tables/` + `tables_clean/` |
| `main_old.tex` | pre-TMLR history | `tables/` |

Only the first two are live. A fix applied to one of the other three has no
effect on anything anyone reads.

**EIGHT of the eleven tables in `docs/paper/tables/` regenerate from
`docs/paper/data/corpus/corpus_final.csv` byte-for-byte** via
`docs/paper/scripts/make_*.py` -- run them and `git diff docs/paper/tables/` must
be empty. 🛑 **`make_main_table.py` needs `--two-metrics`**; the bare
invocation writes a DIFFERENT table over the same `tab_ccf1.tex` (verified
2026-08-21: bare = 54 insertions / 63 deletions, `--two-metrics` = byte-identical).
It is the one generator whose default is not the shipped artefact, so run:

```bash
python docs/paper/scripts/make_main_table.py --two-metrics   # tab_ccf1.tex
```

⚠️ `tab_ablation_complete`, `tab_deploy` and `tab_oct_backbone` have
**no generator and never did**, so an empty diff says nothing about those three. See `docs/paper/data/PROVENANCE.md`, including what the corpus itself
can no longer be rebuilt from.
