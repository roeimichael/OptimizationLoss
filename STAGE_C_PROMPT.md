# Stage C — Modularize the OptimizationLoss training pipeline

You are continuing a multi-session cleanup of a thesis project on transductive prediction-count constraints. Stages A + B (LOC reduction, dead-code removal, methodological alignment with Fioretto LDF) are already complete on branch `cleanup-pipeline`. Your job is Stage C: extract the shared pipeline scaffolding from three runners into a clean modular structure, with each methodology getting its own folder.

**This prompt is self-contained. The previous session is closed; you do not have its working memory. Read this carefully and analyze the codebase before making any changes.**

---

## 0. Working environment

- Code lives on a remote SSH host: `dsisco02` (`~/OptimizationLoss/`). The local Windows checkout is on a different branch and is stale — work via `ssh dsisco02 ...`.
- Branch: `cleanup-pipeline` (HEAD: `40ffba7` "Cleanup Stage B: per-class lambda is the only mode"). Confirm with `ssh dsisco02 'cd ~/OptimizationLoss && git log --oneline -3'`.
- Conda env on the server: `optloss` (`source ~/anaconda3/etc/profile.d/conda.sh && conda activate optloss`). Python 3.10.
- GPU: dsisco02 has 4× RTX PRO 6000 Blackwell 96GB. Use **GPU 0** for smoke tests. Verify it's idle first with `nvidia-smi`.
- Always check GPU exclusivity before launching: `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i 0` should be empty.
- Each smoke test runs ~7 minutes in a tmux session named `smoke`. Kill any existing `smoke` session before starting a new one.

---

## 1. Project context (read this BEFORE you read code)

This is a master's thesis on **transductive prediction-count constraints**. Three methodologies are compared on the same TissueMNIST + CIFAR-100 benchmarks:

- **`our_approach`** — bounded penalty `E/(E+K) + ρ·(E/K)²/(1+(E/K)²)` + per-class λ ratchet + KL anchor to warmup distribution. The paper's contribution.
- **`fioretto_ldf`** — Fioretto et al. 2020 ECML-PKDD baseline. Linear penalty `λ·max(0, soft − K)` + per-constraint subgradient ascent. No KL.
- **`heuristic`** + **`danits_lp`** — warmup-only baselines. Heuristic uses greedy top-K allocation post-hoc; danits_lp uses an LP allocation. Both skip the constraint training phase.

(`po_lp` was dropped in Stage A — do not re-introduce it.)

**Methodological invariants you must NOT change:**

| Methodology | Penalty | λ update |
|---|---|---|
| our_approach | `E/(E+K) + ρ·(E/K)²/(1+(E/K)²)` (bounded) + KL anchor | per-class ratchet, freeze on first satisfaction |
| fioretto_ldf | `λ·max(0, soft − K)` (linear) | per-constraint subgradient ascent |
| heuristic / danits_lp | n/a (post-hoc allocation only) | n/a |

If your refactor changes any of those three core algorithms, you've made a mistake. Smoke metrics will likely flag it; if not, audit before committing.

---

## 2. Target structure (final state after Stage C)

```
src/
  pipeline/                   # SHARED across all methodologies
    __init__.py
    setup.py        ~80      # seed_all(seed); setup_runtime(device) -> (use_amp, amp_dtype, scaler); cudnn flags
    data.py         ~90      # load_data(config) -> tensors + groups + constraints + num_classes
    model_setup.py  ~70      # get_or_load_warmup_model(config, ...); make_optimizer; make_dataloader
    warmup.py       ~110     # run_warmup(model, X_train, y_train, config, device) -> trained model + cache hit
    eval.py         ~140     # evaluate_with_posthoc(model, X_test, y_test, group_ids, ...) -> metrics + preds
    io.py           ~80      # save_predictions, save_evaluation_metrics, save_results_to_config
    contracts.py    ~50      # @dataclass TrainInputs / TrainOutputs

  methodologies/              # METHODOLOGY-SPECIFIC: only the constraint-training loop body
    __init__.py               # registry: dict[str, train_fn]
    our_approach/
      __init__.py
      train.py      ~280     # bounded penalty + per-class ratchet + KL anchor + rho schedule
      hp_defaults.py ~25     # {'lambda_step', 'initial_rho', 'rho_target', 'alpha_kl', ...}
    fioretto_ldf/
      __init__.py
      train.py      ~220     # linear penalty + subgradient ascent + per-constraint dict
      hp_defaults.py ~10     # {'fioretto_step_size': 0.005}
    heuristic/
      __init__.py
      train.py      ~30      # no-op + greedy allocation post-hoc dispatch hint
      hp_defaults.py ~8
    danits_lp/
      __init__.py
      train.py      ~30      # no-op + LP allocation dispatch hint
      hp_defaults.py ~10     # {'danits_cost_preset': 'identity'}

  experiments/
    runner.py       ~120     # ONE entry point: load config → resolve methodology → shared setup
                             # → methodology train → shared eval → save
    run_experiment.py ~10    # thin shim → runner (back-compat with main.py / run_anchor.sh)
    run_heuristic.py  ~10    # thin shim → runner

  losses/                              UNCHANGED (transductive_loss.py)
  models/                              UNCHANGED
  training/
    metrics.py                         UNCHANGED
    logging.py                         UNCHANGED
    model_cache.py                     UNCHANGED
    constraints.py                     UNCHANGED
    trainer.py      DELETED  (body moved to methodologies/our_approach/train.py + pipeline/warmup.py)
  utils/                               UNCHANGED (constants, posthoc_adjustment, data_loader, inference,
                                       error_handler, filesystem_manager)

fioretto_research/run_fioretto.py      DELETED (logic moved to methodologies/fioretto_ldf/train.py)
fioretto_research/{gen_*.py}           UNCHANGED (config generators stay)
danits_research/                       UNCHANGED
main.py                                Update dispatch table to point all methodologies at the new runner
scripts/run_anchor.sh                  UNCHANGED (it invokes main.py)
```

---

## 3. The interface contract for `methodologies/<name>/train.py`

Every methodology module exposes ONE function:

```python
def train(inputs: TrainInputs) -> TrainOutputs:
    ...
```

Where the dataclasses (defined in `pipeline/contracts.py`) are roughly:

```python
@dataclass
class TrainInputs:
    model: nn.Module                         # warmed-up model (post-warmup), eval-ready
    X_train: torch.Tensor                    # for any methodology that wants train data
    y_train: torch.Tensor
    X_test: torch.Tensor                     # transductive: constraints applied here
    group_ids: torch.Tensor
    global_con: list[float]                  # length = num_classes; UNLIMITED sentinel for unconstrained
    local_con: dict[int, list[float]]        # group_id -> per-class limits
    constrained_classes: list[int]
    num_classes: int
    config: dict
    hyperparams: dict
    device: torch.device
    experiment_path: Path
    csv_log_path: Path                       # already opened/header-written by pipeline.warmup

@dataclass
class TrainOutputs:
    model: nn.Module                         # final-epoch weights, eval-ready
    summary: dict                            # satisfaction_epoch, soft_hard_gap, constraint_train_time, ...
    skip_targeted_correction: bool = False   # heuristic / danits_lp set True (they did their own posthoc)
    precomputed_predictions: np.ndarray | None = None  # heuristic / danits_lp put their allocation here
```

The shared eval (`pipeline/eval.py:evaluate_with_posthoc`) accepts these flags:
- if `skip_targeted_correction=True` and `precomputed_predictions` is given, use them directly (no `targeted_correction` invocation)
- else: run `targeted_correction(y_proba, group_ids, ...)` exactly as today

---

## 4. Migration order (8 steps, smoke between each)

Each step ends with: same baseline smoke config (`results/pending_runs/smoke/baseline/config.json` is already there from the previous session, evaluation_metrics.csv from baseline saved as well — confirm with `ssh dsisco02 'ls ~/OptimizationLoss/results/pending_runs/smoke/'`). Drop a copy of it into `smoke/v9_<step>` etc, run the experiment, compare metrics.

**Acceptance tolerance per step:** F1 within ±0.02 of v8 (`F1=0.3601`), `sat_epoch` within ±20, no tracebacks.

The previous session's smoke baseline (post-Stage-B) is:
- F1 = 0.3601
- Accuracy = 0.4813
- Flips = 9
- Excess = 6
- Sat epoch = 122

These are your reference numbers. Reproduce a v9 sanity-check smoke FIRST before any extraction (just to confirm you can drive the pipeline before refactoring).

### Step-by-step

1. **`pipeline/setup.py` + `pipeline/io.py`** — pure extraction, no logic change. Move seed/AMP/cudnn block + the `config['results']` writeback. Update `run_experiment.py`/`run_fioretto.py`/`run_heuristic.py` to import from there. Smoke our_approach.

2. **`pipeline/data.py`** — extract data load + tensor build + `constrained_classes` derivation. Update all three runners. Smoke our_approach + heuristic + fioretto.

3. **`pipeline/warmup.py`** — merge the THREE near-identical CE-warmup loops (`ConstraintTrainer.train_warmup`, `run_heuristic._train_warmup` aliased `train_fixed_warmup`, `run_fioretto._train_warmup`). They differ only in log strings + `compute_train_accuracy` call placement. After merging, all three runners delegate to `run_warmup(...)`. Smoke all three methodologies.

4. **`pipeline/eval.py`** — extract evaluation + posthoc + Track1 metrics + per-class limit log. Use `evaluate_with_posthoc(model, X_test, y_test, group_ids, ..., skip_correction=False, precomputed_preds=None)`. Smoke our_approach + fioretto.

5. **`methodologies/our_approach/train.py`** — lift `ConstraintTrainer.train_constraints` body verbatim. Convert instance attributes (`satisfaction_epoch`, `final_soft_hard_gap`) into the returned `TrainOutputs.summary` dict. Delete `src/training/trainer.py`. Update `run_experiment.py` to call `pipeline.warmup.run_warmup` then `methodologies.our_approach.train.train(inputs)`. Smoke our_approach.

6. **`methodologies/fioretto_ldf/train.py`** — lift `_train_fioretto_constraints` from `fioretto_research/run_fioretto.py`. Move the dual-checkpoint pick (final vs best_excess by F1) inside this train fn, since it's methodology-specific. Delete `fioretto_research/run_fioretto.py`. Smoke fioretto.

7. **`methodologies/heuristic/train.py` + `methodologies/danits_lp/train.py`** — wrap each existing post-hoc allocator in `train(inputs) -> TrainOutputs(skip_targeted_correction=True, precomputed_predictions=...)`. Drop the dispatch in `run_heuristic.py`. Smoke both.

8. **`experiments/runner.py`** — single dispatcher: read methodology from config, call shared `pipeline.data.load`, `pipeline.warmup.run_warmup` (cache hit/miss), `methodologies.<name>.train()`, `pipeline.eval.evaluate_with_posthoc()`, `pipeline.io.save_results()`. Replace `run_experiment.py` and `run_heuristic.py` with thin shims that import from runner. Update `main.py` dispatch tuple to point all four methodologies at the same module. Smoke all four sequentially.

---

## 5. Smoke harness — exact commands

```bash
# 1) Set up smoke config (copy from previous session's baseline)
ssh dsisco02 "cp -r ~/OptimizationLoss/results/pending_runs/smoke/baseline ~/OptimizationLoss/results/pending_runs/smoke/v9_step1"
ssh dsisco02 "python3 -c 'import json; p=\"/home/dsi/michaer8/OptimizationLoss/results/pending_runs/smoke/v9_step1/config.json\"; c=json.load(open(p)); c[\"status\"]=\"pending\"; c[\"exp_name\"]+=\"_v9_step1\"; [c.pop(k,None) for k in (\"final_train_acc\",\"final_test_acc\",\"final_f1\",\"results_comparison\",\"selected_checkpoint\",\"satisfaction_epoch\")]; json.dump(c,open(p,\"w\"),indent=2)'"
ssh dsisco02 "rm -f ~/OptimizationLoss/results/pending_runs/smoke/v9_step1/{evaluation_metrics,training_log,final_predictions,final_predictions_raw}.csv"

# 2) Verify GPU 0 idle
ssh dsisco02 "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i 0"

# 3) Launch in detached tmux
ssh dsisco02 "tmux kill-session -t smoke 2>/dev/null; tmux new-session -d -s smoke 'source ~/anaconda3/etc/profile.d/conda.sh && conda activate optloss && cd ~/OptimizationLoss && export CUDA_VISIBLE_DEVICES=0 && export CUDA_MODULE_LOADING=EAGER && time python -m src.experiments.runner results/pending_runs/smoke/v9_step1/config.json 2>&1 | tee smoke_v9_step1.log; echo EXIT=\$?'"

# 4) Wait + read result (~7 min)
# Use a single sleep+tail call run_in_background, do NOT poll repeatedly.
ssh dsisco02 "sleep 480 && tail -8 ~/OptimizationLoss/smoke_v9_step1.log; cat ~/OptimizationLoss/results/pending_runs/smoke/v9_step1/evaluation_metrics.csv | head -10"
```

**Important**: in step 8 the entry point becomes `src.experiments.runner` not `src.experiments.run_experiment`. Until step 8 lands, all smokes go through the existing `src.experiments.run_experiment` entry.

---

## 6. Fioretto + heuristic smokes

You'll need separate smoke configs for fioretto and heuristic. Generate them via:

```bash
ssh dsisco02 "cd ~/OptimizationLoss && python -m src.config_generators.gen_multimethodology --output_root results/pending_runs/smoke_fioretto --warmup_epochs 50 --constraint_epochs 100 --datasets tissuemnist --scenarios single_GE --constraint_pairs 0.5,0.5 --models MobileNetV3 --seeds 1 --methodologies fioretto_ldf"
```
(read `src/config_generators/gen_multimethodology.py:main()` for exact CLI flags)

Reference numbers for these smokes — establish them BEFORE step 1 by running each methodology once on the current `cleanup-pipeline` HEAD:
- **our_approach**: F1=0.3601 / Acc=0.4813 / flips=9 / sat_ep=122 (already known)
- **fioretto_ldf**: ??? — establish by running once
- **heuristic**: ??? — establish by running once
- **danits_lp**: ??? — establish by running once

---

## 7. Workflow rules

1. **Analyze first.** Before any extraction, read all the affected files (`src/training/trainer.py`, `src/experiments/run_experiment.py`, `src/experiments/run_heuristic.py`, `fioretto_research/run_fioretto.py`, `main.py`, `src/utils/data_loader.py`). Sketch the dependency graph mentally. Know what `targeted_correction` returns and how each runner consumes it.
2. **One step at a time.** Don't batch step 3 + step 4 into one commit. Smoke between each.
3. **Per-step commit message format:**
   ```
   Stage C step N: extract <module> from <source(s)>

   <description of what moved + before/after LOC>

   Smoke: F1=X.XXXX (baseline 0.3601, Δ +/- Y).

   Co-Authored-By: <model> <noreply@anthropic.com>
   ```
4. **If a smoke fails (F1 drift > 0.02 or traceback):** revert that step's changes, investigate, fix, re-smoke. Do NOT proceed to the next step on a failing baseline.
5. **No new abstractions beyond the target structure.** No "PipelineBase" classes. No "registry decorator" patterns. Plain functions, plain dicts. Match Fioretto's simplicity that the user asked for.
6. **The methodology core stays methodology-specific.** Do not try to share the constraint-training loop body across our_approach and fioretto_ldf — they are algorithmically different.
7. **Stale-cache awareness:** existing warmup caches (`~/OptimizationLoss/model_cache/MobileNetV3_tissuemnist_*.pt`) work because `compute_base_model_id` is unchanged. Don't touch the cache key composition without a clear reason.
8. **Never commit without smoking.** Never push without committing.
9. **Don't re-run agents.** The user's prior session already ran 3 architecture/cleanup agents. Their findings are baked into this prompt. Read code yourself; if you must search, use Grep.
10. **GPU discipline:** dsisco02 driver crashes on shared GPUs. Always check GPU 0 is empty before launching. If another user's process appears, use GPU 1 instead and note the reason.

---

## 8. Definition of done

- Branch `cleanup-pipeline` has 8 new commits on top of `40ffba7`.
- The target file tree in section 2 is realized: `pipeline/` exists, `methodologies/` exists with 4 subfolders, `runner.py` is the single dispatcher, `trainer.py` and `run_fioretto.py` are deleted.
- All 4 methodologies smoke-pass: F1 within tolerance, no tracebacks.
- `main.py`, `scripts/run_anchor.sh`, `scripts/dispatch_multi_gpu.py`, `scripts/validate_anchor.py` still work (drop a config in `results/pending_runs/<dataset>/...` and run `python main.py`, get a completed run).
- A final commit titled "Stage C complete: modular pipeline + per-methodology training" with a summary table: file tree before/after, LOC totals, smoke results for all 4 methodologies.

---

## 9. Things you DO NOT need to do (deferred / out of scope)

- Don't refactor `posthoc_adjustment.py` internals (Phase 1/2 vs Phase 3a/3b duplication is real but not Stage C scope).
- Don't change Fioretto's dual-checkpoint selection (final vs best_excess) — keep it inside `methodologies/fioretto_ldf/train.py`.
- Don't switch lambda init from 0.01 to 0.0 (deferred per user).
- Don't drop `class_weighted_ce` from cache key (would invalidate every cached warmup).
- Don't touch `gen_multimethodology.py` config schema beyond moving METHODOLOGY_HP into per-methodology hp_defaults.py modules.

---

## 10. First action

1. `ssh dsisco02 'cd ~/OptimizationLoss && git log --oneline -10 && wc -l src/training/trainer.py src/losses/transductive_loss.py src/experiments/run_experiment.py src/experiments/run_heuristic.py fioretto_research/run_fioretto.py'` — confirm you're on the right HEAD and the LOC counts match the previous session's checkpoint.
2. Read `src/experiments/run_experiment.py`, `src/experiments/run_heuristic.py`, `fioretto_research/run_fioretto.py`, `src/training/trainer.py`, `src/utils/data_loader.py`, `main.py` in full.
3. Run a v9_pre baseline smoke for each of the 4 methodologies on the current HEAD to lock in reference numbers.
4. Then start Step 1.

Good luck. Pay attention to the user's directives in section 1 (methodological invariants) — those are immutable. Everything else is fair game for the modularization.
