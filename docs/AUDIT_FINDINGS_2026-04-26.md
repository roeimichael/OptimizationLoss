# Adversarial Audit — 2026-04-26

Four parallel reviewers (code-reviewer agents) did adversarial passes on the
codebase from a thesis-defense perspective. Aggregated findings below.
Severity grouped, file:line cited where verified, "VERIFY" tag means I
have not yet confirmed against current code.

Reviewer scopes:
- **A1**: our_approach training loop + loss (`trainer.py`, `transductive_loss.py`, `schedulers.py`, `run_experiment.py`)
- **A2**: baseline fairness (`run_fioretto.py`, `run_heuristic.py`, `posthoc_adjustment.py`, `danits_research/*`)
- **A3**: data pipeline + cache (`data_loader.py`, `model_cache.py`, `model_factory.py`, imagery models, `inference.py`)
- **A4**: code quality, dead code, duplication

Reviewer verdicts:
- A1: NEEDS_CHANGES
- A2: SUBTLE_BIAS_PRESENT
- A3: NEEDS_HARDENING
- A4: NEEDS_CLEANUP

---

## SHOWSTOPPER (defense will fail without explanation)

### S1. Constraint budget K leaks test labels [CONFIRMED]
`src/training/constraints.py:21,37` — K is computed as `int(np.round(count * percentage))` where `count = (test_df[target_col] == c).sum()`. The constraint count budget is therefore defined using test-set true labels.

This means: the loss "knows" how many class-c test samples exist when defining the cap. The "transductive prediction-count constraint" framing assumes K is exogenous (regulatory budget, business decision, prior knowledge). Today K = 0.5 * (true count of class c in test).

**Decision needed:**
- (a) Frame the setup honestly as "K is a fraction of ground-truth class proportion" — defensible evaluation methodology for studying optimization behavior, but loses the "real-world prediction-budget" narrative.
- (b) Make K independent of test labels: pre-fix integer values, or fraction of |X_test|/num_classes (uniform allocation), or fraction of expected count from a held-out validation set.

### S2. Checkpoint selection by F1-macro on test set [CONFIRMED]
`src/experiments/run_experiment.py:117` — `best = max(candidates, key=lambda x: x[3]['f1_macro'])` where f1_macro is computed against `y_test`. The bracket-best/previous/final selection picks the checkpoint that scores highest against test labels. Same selection is NOT applied to fioretto_ldf (which uses best-by-excess) — so the comparison metric is asymmetric.

**Decision needed:**
- (a) Split off a validation set from current train, select on val, evaluate on test (~10% of train as val).
- (b) Apply identical best-by-F1 post-hoc selection to fioretto_ldf for symmetry.
- (c) Drop bracket selection entirely; use final-epoch model.

### S3. UNLIMITED constant inconsistency [CONFIRMED]
- `1e10` is the sentinel in `transductive_loss.py:14`, `trainer.py:30`, `posthoc_adjustment.py:16`, `constraints.py:7`, `run_heuristic.py:27`, `run_fioretto.py:38`.
- `1e9` is the threshold in `metrics.py:111`, `logging.py:21-73`, `run_experiment.py:79,124`, `run_heuristic.py:142-145,268`, `run_fioretto.py:443,465`.

Effect: a constraint set to UNLIMITED (1e10) is correctly skipped by the loss (`>= 1e10`) BUT the metric layer uses `< 1e9` as its "active constraint" filter. Since 1e10 is NOT < 1e9, the metric treats UNLIMITED as ACTIVE → reports phantom violations on unconstrained classes. MEMORY.md claimed this was unified — it was not.

**Fix:** single `src/utils/constants.py` defining `UNLIMITED = 1e10`. All sites import.

---

## CRITICAL (algorithmic correctness)

### C1. Pass A vs Pass B forward in train() mode [CONFIRMED]
`trainer.py:297,316,348` — both no-grad bookkeeping pass and grad pass run with `self.model.train()` active. Two consequences:
- Dropout fires twice per epoch differently → soft counts in pass A differ from pass B → "subtract-then-add-back" identity in line 364-371 is broken.
- BatchNorm running stats updated from full test set every epoch → quiet form of test-data influence on the model.

**Fix:** `model.eval()` for the bookkeeping pass; restore `model.train()` afterward. Or freeze BN running stats during constraint epochs.

### C2. KL temperature applied asymmetrically [CONFIRMED]
`trainer.py:170` — `warmup_proba = F.softmax(warmup_logits.float() / kl_temperature, dim=1)` (warmup softened by T).
`transductive_loss.py:151-155` and `trainer.py:373-376` — current logits use `log_softmax(logits)` and `softmax(logits)` at T=1.

With kl_temperature > 1, p_warmup is flatter than p_current. KL(p_current || p_warmup) becomes "pull toward uniform", not "anchor to warmup distribution". Default in DEFAULT_HP is `kl_temperature: 1.0`, so today this is benign — but kl_temperature was swept >1 in some configs and the loss meant something different than the docstring claims.

**Fix:** apply T to BOTH sides (current and warmup) symmetrically. Or document this as a calibration knob, not an anchor.

### C3. po_lp silent fallback to argmax masks failure [CONFIRMED]
`posthoc_adjustment.py:227-229` — `lp_constrained_assignment` returns `argmax_preds, 0` with only a `log.warning` on solver failure. The `po_lp` branch in `run_heuristic.py:206` calls this directly. No `lp_solver_failed` flag in the saved metrics. A `po_lp` row in master_results.csv could silently be "raw warmup argmax that violates constraints", indistinguishable from a successful run.

`targeted_correction` does record `lp_fallback_used` for our_approach/fioretto_ldf — but `po_lp`'s branch does not get that flag.

**Fix:** add `lp_solver_failed` field to evaluation_metrics.csv for `po_lp`; refuse to mark experiment `completed` on solver failure.

### C4. Fioretto step_size default mismatch [CONFIRMED]
- `run_fioretto.py:127` uses `hp.get('fioretto_step_size', 0.01)`.
- `gen_multimethodology.py:72` defaults `fioretto_step_size: 0.005`.
- `gen_fioretto_experiments.py:66` sweeps `[0.001, 0.005, 0.01]`.

Configs from different generators get different defaults silently. Two "default Fioretto" runs in master CSV with no flag. Plus the runner default differs from the generator default — bug-prone.

**Fix:** runner asserts `fioretto_step_size` present in hp; no default. Generator defaults are the single source of truth.

### C5. TissueMNIST class names disagree across files [CONFIRMED — verify]
`data/tissuemnist/download_data.py:37-46` (per A3): `CDI/CDP/CT/DCT/GE/INT/PTC/PTS`.
`data/tissuemnist/create_slices.py:27-30` and CLAUDE.md: `CDI/CDS/CST/EPI/GE/PTC/STR/TUB`.

Class index 4 ("GE") agrees by accident; every other index is named differently between download and slice. Meta CSVs' `class_name` column is provenance garbage. **VERIFY** the actual download script content.

**Fix:** pick one canonical mapping; replace the other.

### C6. Cache key (`base_model_id`) missing critical state [CONFIRMED]
`generate_configs.py:93-109` hashes: model_name, lr, dropout, batch_size, warmup_epochs, pretrained, class_weighted_ce, dataset_mode, data_dir, seed.

**Missing keys that affect warmup output:**
- `num_classes` — TissueMNIST 8 vs CIFAR-100 100; cached models incompatible if num_classes changes silently.
- `image_size` — affects input resolution, model behavior under torchvision adaptive pooling.
- AMP dtype (bf16 vs fp16 + scaler) — different numerics produce different weights.
- torchvision version — pretrained weights can change with version.

**Fix:** add `num_classes`, `image_size`, AMP dtype string, torch+torchvision versions to the hash. Memory says caches were invalidated once for normalization — same risk now for these.

### C7. `safe_execute` swallows `load_state_dict` mismatch silently
`model_cache.py:52-56` — wraps `load_state_dict` in `safe_execute` which catches all exceptions. If cached state dict shape disagrees with the new model (e.g. n_classes changed from 7 to 8), it silently returns None and the trainer fresh-trains over the cached file.

**Fix:** call with `strict=True`; let it raise. Cache invalidation requires explicit deletion.

### C8. CIFAR-100 has no slice generator [CONFIRMED — verify]
`data/cifar100/` has only `download_data.py`; no `create_slices.py`. Only one slice exists; multi-slice statistical claims are unreproducible. **VERIFY** server has matching state.

**Fix:** add `create_slices.py` parallel to `data/tissuemnist/create_slices.py`.

---

## SERIOUS (subtle bias, fairness, robustness)

### B1. Fioretto best-checkpoint selection differs from our_approach
`run_fioretto.py:335-338` restores best-by-excess (lowest pre-post-hoc violation).
`run_experiment.py:117` selects best-by-F1 (post-hoc-corrected).

Two different objectives → asymmetric comparison → our_approach can win on F1 by construction (its own selector) and Fioretto can win on satisfaction (its own selector).

**Fix:** apply identical post-hoc-then-pick-by-F1 to all training-based methods.

### B2. Fioretto only iterates `constrained_classes`, our_approach iterates all
`run_fioretto.py:215-228, 263-296` — penalty applied only to listed `constrained_classes`. `transductive_loss.py:_global_loss` iterates `range(num_classes)` filtered by `K < UNLIMITED`. Latent asymmetry — today the lists agree, but a configuration with finite K on a non-constrained class produces invisible asymmetry.

**Fix:** mirror the trainer's iteration pattern in run_fioretto.

### B3. Fioretto dual ascent monotonic-only
`run_fioretto.py:309-315` — λ ← max(0, λ + α · violation), violation already clipped to ≥0. λ never decreases. Standard subgradient ascent on Lagrangian permits λ to decrease when slack > 0. Plausibly defensible (Fioretto's published Algorithm 1 also clips at 0) but should be cited.

**Fix:** document as design choice in thesis or implement two-sided update behind a flag.

### B4. `raw_constraint_satisfaction` semantically asymmetric
`metrics.py:104-157` — for our_approach/fioretto_ldf this is argmax of constraint-trained model (had training pressure to satisfy). For heuristic/po_lp/danits_lp this is argmax of raw warmup (no constraint pressure). The metric name implies a fair comparison; semantics are not.

**Fix:** rename to `pre_posthoc_satisfaction` and document the asymmetry in thesis figures.

### B5. `training_time` differently scoped across methods
- our_approach: warmup + constraint phase.
- fioretto_ldf: warmup + constraint phase.
- heuristic / po_lp / danits_lp: post-hoc allocation only (~seconds).

Direct comparison makes LP baselines look 100x faster than reality.

**Fix:** split into `warmup_time + constraint_train_time + posthoc_time` columns.

### B6. Greedy heuristic ascending + constrained-first
`run_heuristic.py:85-89,259` — sorts constrained classes ascending by limit, processes them before unconstrained classes. The tightest-budget class gets to drink top-K confidence picks before others. Not necessarily optimal — descending or argmax-confidence-first might yield higher accuracy. Means the "heuristic" baseline is not the strongest greedy possible.

**Fix:** add a second greedy variant (best-greedy) as an additional baseline OR justify the current ordering.

### B7. Per-class lambda mode never decrements
`trainer.py:477-507` — `per_class_ratchet` only increments. No path to zero or freeze for a satisfied class. λ stays large after temporary violation, pushes soft count below K → under-prediction.

**Fix:** add per-class freeze-when-satisfied, mirroring the legacy toggle but per-class.

### B8. Lambda toggle halving has no floor
`trainer.py:594-602` — `toggle_count >= 10` halves λ each subsequent toggle. No floor; λ → 0 indefinitely. After 30 toggles λ ≈ original/8000.

**Fix:** windowed toggle count (last K epochs) + floor at λ_step.

### B9. CE saturation skip + α_kl + zeroed lambdas → posterior collapse
`trainer.py:639-648` — when `train_acc ≥ 0.995`, `skip_ce = True`. Combined with toggle's λ=0 on satisfaction and `alpha_kl > 0`, only loss term left is asymmetric KL pulling toward uniform (see C2). Comment in code admits this and tells user to set `disable_ce_skip=True`. **Footgun guarded by a flag is not a safeguard.**

**Fix:** auto-disable CE skip whenever `alpha_kl > 0` or whenever lambdas are zeroed.

### B10. anchor_id NOT a superset of base_model_id
`gen_multimethodology.py:77-91` — anchor_id excludes `dropout`, `batch_size`, `class_weighted_ce`, `data_dir`. Two configs with same anchor_id can have different warmups; paired comparison assumption breaks.

**Fix:** anchor_id = hash(base_model_id_keys ∪ {scenario, ctag, lr_constraint, constraint_epochs}).

### B11. CIFAR-100 `coarse_label` group is deterministic function of `label`
A3 — local constraints become trivial: "the constrained class is always one specific superclass". Need an alternative group construction (random binary, demographic-like) to make local constraints meaningful.

### B12. K=0 silently skipped
`transductive_loss.py:86-87,127-128` — `if K <= 0: log.warning(...); continue`. A rounding artifact that produces K=0 makes the constraint vanish. Run records as "satisfied" because no constraint exists.

**Fix:** raise on K=0; refuse to start the experiment.

---

## CODE QUALITY (cleanup before defense)

### Q1. Empty `src/models/tabular/` directory contradicts CLAUDE.md
CLAUDE.md describes `BasicNN`, `FTTransformer`, `TabularResNet`. Directory is 0-byte `__init__.py` only. Easy committee gotcha.

### Q2. Three copies of CE warmup boilerplate
`trainer.py:_train_warmup`, `run_heuristic.py:train_fixed_warmup`, `run_fioretto.py:_train_warmup`. ~50 lines each. Drift already happened (saturation skip in trainer only, post-train acc log in heuristic only).

**Fix:** consolidate into `src/training/warmup.py:train_warmup(config, ...) → model`.

### Q3. 13 config generators in `danits_research/`
Most reference dead `disable_lambda_toggle` or DermMNIST defaults. Only `gen_200run_thesis.py` and `gen_cifar100_experiments.py` align with the current plan.

**Fix:** delete the 11 historical ones; document the surviving canonical generators (`src/config_generators/gen_multimethodology.py`, `fioretto_research/gen_fioretto_experiments.py`).

### Q4. `generate_configs.py` defaults to DermMNIST
`generate_configs.py:1` docstring + lines 10-15, 93, 116. Active dataset is TissueMNIST.

**Fix:** archive this generator; `gen_multimethodology.py` is the new canonical.

### Q5. `disable_lambda_toggle` flag still wired
`trainer.py:561,578` — the flag the user proved harmful (MEMORY 2026-04-14) is still respected. Multiple `danits_research/gen_*.py` files set it to True.

**Fix:** delete the flag, the flag-handling branch, and the configs that set it.

### Q6. `minimal_correction` is dead
`posthoc_adjustment.py:130-161` — only called from `danits_research/benchmark_smoke.py`. Production uses `targeted_correction`.

### Q7. Two bare `except:` in analysis code
`analyze_all_experiments.py:138,143` — swallows KeyboardInterrupt and SystemExit.

### Q8. `'binary'` and `'sex'` defaults in data_loader
`data_loader.py:66` (`dataset_mode='binary'` — would raise ValueError today) and `:40` (`group_column='sex'` — Adult dataset leftover).

### Q9. trainer.py is 734 lines monolith
4 lambda modes, AMP, CE skip, 3 brackets, diagnostics, KL caching, 2-pass constraint compute. Should split: LambdaScheduler, AmpHelper, slim trainer.

### Q10. Unused imports / vars
- `run_fioretto.py:170-172`: `group_indices` computed never used.
- `run_heuristic.py:215-217`: `build_priority_cost_matrix` imported never used.
- `dispatch_multi_gpu.py:12,26`: `import logging` + logger never used.
- `model_factory.py:24-25`: `kwargs.pop('input_dim'/'hidden_dims')` — vestigial tabular-era plumbing.

### Q11. Hardcoded paths
- `build_master_results.py:15`: `BASE = Path(r"C:\Users\roeym\Desktop\projects\OptimizationLoss\results_fetched")`. Will not run for anyone else.
- `scripts/compare_mar27.py`, `show_grad_evolution.py`, `lp_vs_heuristic_sweep.py`: hard-coded result paths.

### Q12. `chunked_forward` duplication
`src/utils/inference.py` plus inline copies in `trainer.py:313-332,345-378`, `posthoc_adjustment.py`, `metrics.py`. Five callers, three implementations.

---

## KILL LIST (user pre-approves to delete)

Status: PROPOSED — needs your sign-off per item.

- `src/models/tabular/` (empty, contradicts CLAUDE.md)
- `src/utils/posthoc_adjustment.py:130-161` (`minimal_correction`, dead)
- `src/config_generators/generate_configs.py:39-56,201-210` (DermMNIST round structures, `reset_all_to_pending`)
- `danits_research/gen_pilot_configs.py`, `gen_fixed_grid.py`, `gen_fixed_grid_tissuemnist.py`, `gen_diagnostic_grid.py`, `gen_improvement_test_grid.py`, `gen_proportional_test.py`, `gen_march27_replay.py`, `gen_smoke_configs.py`, `gen_40run_sweep.py`, `gen_multiclass_sweep.py`, `gen_baseline_seeds.py` (historical sweeps; 11 of 13 unused)
- `danits_research/benchmark.py`, `benchmark_multi.py`, `benchmark_smoke.py`, `_benchmark_core.py`, `_verify_smoke.py`, `analyze_smoke.py`, `smoke_test.py`, `report.py`, `lp_vs_greedy_diag.py`, `diagnose_unfair.py`, `diagnostics.py`, `preflight.py`, `fill_to_budget.py` (research detritus; production uses only `solve_lp_assignment`, `build_psi_phi_from_percentages`)
- `danits_research/cost_matrices.py:103-135` (DERMMNIST cost matrices, never used since pivot)
- `scripts/compare_mar27.py`, `show_grad_evolution.py`, `lp_vs_heuristic_sweep.py` (single-purpose throwaway scripts, hard-coded paths)
- `disable_lambda_toggle` config key everywhere
- `src/utils/data_loader.py:40,66` (`'sex'`, `'binary'` defaults — Adult/Churn era)
- `model_factory.py:24-25` (vestigial tabular kwargs.pop)

---

## TRIAGE — recommended order

**Block 1 (must address before any thesis run, ~1 day):**
- S1 K-from-y_test (decision + implementation)
- S2 checkpoint selection on test (decision + implementation)
- S3 UNLIMITED unification
- C2 KL temperature symmetry
- C3 po_lp silent fallback flag

**Block 2 (high-leverage correctness, ~1 day):**
- C1 Pass A/B model.eval()
- C4 Fioretto step_size default
- C6 cache key additions
- B10 anchor_id superset
- B12 K=0 raise

**Block 3 (fairness for paired comparison, ~1 day):**
- B1 Fioretto best-by-F1 selection
- B4 raw_satisfied rename + framing
- B5 training_time split into 3 columns
- B9 CE skip auto-disable

**Block 4 (cleanup before defense, ~1-2 days):**
- Q1-Q12 dead code, unification, tabular/ removal
- Kill list
- trainer.py split (optional, pure code quality)

**Out-of-scope for now:**
- A2's note that danits_research's heuristic.py is paper-faithful while run_heuristic.py's heuristic isn't paper-faithful (different naming convention) — clarify in thesis text only
- Fioretto's monotonic-only λ update (B3) — design choice, document
