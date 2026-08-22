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
data/              dermmnist, octmnist, tissuemnist -- nothing else
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
python -m pytest tests -q                   # 190 regression tests, ~35s, no dataset needed
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
```

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
python -m configs.gen_campaign --root results/<name>     --datasets dermmnist tissuemnist --models MobileNetV3     --caps L30_G30 L50_G50 --arms all
```

## Datasets (the only three)

`dermmnist` (7 classes, MEL=4 capped, `loc_group`) - `octmnist` - `tissuemnist` (8 classes,
class 4 capped, `synth_group`). **No AIDER, no EuroSAT, no others.**

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
