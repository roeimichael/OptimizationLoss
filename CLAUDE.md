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
python -m pytest tests -q                   # 96 regression tests, ~17s, no dataset needed
python -m scripts.audit_config              # no config key without a reader, no reader without a key
python -m scripts.smoke_arms                # every arm actually RUNS and respects its caps
python -m scripts.verify_caps               # what integer budget each cap tag really produces
python -m scripts.check_parity <root>       # equal compute, same knobs, >=2 caps, sane warm-up sharing
```

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
