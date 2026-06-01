# OptimizationLoss

Thesis project: train neural networks to satisfy **transductive prediction-count constraints** using soft constraint optimization, and compare against a greedy heuristic baseline.

## Datasets (active)

Three datasets in `data_loader.py`'s `IMAGERY_DATASETS`. Source of truth for paper scope: `docs/PAPER_PLAN.md`.

### TissueMNIST (active)
- **Source**: MedMNIST kidney tissue
- **Classes**: 8; **Constrained class**: 4 (GE, ~7.1% test)
- **Group column**: `synth_group`
- **Status**: F1 headline — paired-significant TraLO win vs all 5 baselines on L20-L50 × MobileNetV3 (`winning_results/headline_f1.md`).

### DermMNIST (active)
- **Source**: MedMNIST HAM10000 skin lesions
- **Classes**: 7 -- AKIEC, BCC, BKL, DF, MEL (constrained), NV, VASC
- **Constrained class**: MEL (class 4); local cap via `loc_group`
- **Status**: F1 3W/2T/0L, flips 5W/0/0; richest fairness story (3 imbalanced anatomical groups).

### AIDER (active)
- **Source**: aerial disaster image recognition dataset
- **Classes**: 4; **Constrained class**: 0
- **Group column**: `synth_group` (binary)
- **Status**: flips 5W/0/0; F1 ties / small loss — saturated-warmup regime, itself a paper finding.

**Previously tried and dropped:** see `docs/REJECTED.md`. Do not re-introduce without reading that file.

## Pipeline

```
main.py                                  # Dispatch pending experiments via subprocess
  -> src/experiments/runner.py            # Single dispatcher: tralo | fioretto_ldf | hounie_rcl | heuristic | danits_lp
```

**Config generation**: `src/config_generators/generate_configs.py`
**Evaluation**: `src/evaluation/generate_all.py`, `src/evaluation/thesis_figures.py`

## Models

### Imagery (`src/models/imagery/`)
- `MobileNetV3` -- torchvision MobileNetV3-Large (~5.4M params) — **headline backbone**
- `MobileNetV2` -- torchvision MobileNetV2 (~3.5M params) — Blackwell-validated co-winner
- `RegNetY400MF` -- torchvision RegNetY-400MF (~4M params, group-conv+SE) — corroboration
- `ShuffleNetV2` -- torchvision ShuffleNet V2 — corroboration
- Input: `(B, 3, H, W)` image tensors, ImageNet-normalized by data_loader

Registry: `src/models/model_factory.py` -- `get_model(name, n_classes, **kwargs)`

**Previously tried and dropped:** see `docs/REJECTED.md`. Do not re-add to the registry without reading that file first.

## Training Phases

1. **Warmup** -- CE loss only for `warmup_epochs`. Cached via `base_model_id` hash.
2. **Constraint Optimization** -- CE + transductive constraint loss.
   - Lambda ratchet: increments by `lambda_step` per epoch until satisfied, then freezes.
   - Linear rho schedule: `rho += rho_step` each epoch until first satisfaction, then frozen.
   - KL regularization: `alpha_kl * D_KL(current || warmup)`.
   - CE saturation skip: stops Phase 1 when train accuracy >= 0.995 for 2 checks.
   - Lambda toggle: zeroes lambdas when satisfied, restores when violated, with oscillation detection.
   - Convergence: early stops when constraints satisfied for 5 consecutive epochs.
   - Optimizer reset at satisfaction: Adam optimizer state is reset at first satisfaction to break post-satisfaction descent momentum (essential per component ablation).
3. **Post-hoc Adjustment** -- Flip borderline predictions to enforce hard count limits.
   - Global adjustment first, then local per-group, then re-verify global.

## Loss Formulation

```
L_total = L_ce + lambda_g * L_global + lambda_l * L_local + alpha_kl * L_kl
```

- **L_global / L_local**: Rational saturation `E/(E+K)` + bounded quadratic `rho * (E/K)^2 / (1 + (E/K)^2)`
- **L_kl**: KL divergence vs cached warmup predictions
- Soft counts (differentiable) vs hard counts (argmax for verification)

## Performance Optimizations

- AMP (BF16 on Ampere+ GPUs, FP16 with GradScaler on older)
- cudnn.benchmark for fixed-size inputs
- Fused Adam optimizer (with fallback)
- set_to_none=True for zero_grad
- Two-pass constraint computation (no_grad counts + grad accumulation)

## Directory Structure

```
main.py                 # Experiment orchestrator (sequential + parallel GPU)
run_experiments.sh      # Server experiment runner
setup_server.sh         # Server environment setup
docs/                   # README.md, THESIS_CONTEXT.md
src/
  config_generators/    generate_configs.py
  experiments/          run_experiment.py, run_heuristic.py
  losses/               transductive_loss.py (MulticlassTransductiveLoss)
  models/
    model_factory.py    unified registry
    imagery/            mobilenetv3.py, mobilenetv2.py, regnet.py, shufflenet.py
  training/             trainer.py, constraints.py, metrics.py, logging.py,
                        schedulers.py, model_cache.py
  utils/                data_loader.py, filesystem_manager.py, error_handler.py,
                        posthoc_adjustment.py, inference.py
  evaluation/           generate_all.py, thesis_figures.py, training_curves.py,
                        experiment_comparison.py, evaluate_statistical_significance.py,
                        visualize_stat_significance.py
data/tissuemnist/       images + labels as .npy (not in git)
data/dermmnist/         images + labels as .npy (not in git)
data/aider/             prep scripts only (raw on server)
model_cache/            cached warmup .pt files (not in git, auto-created)
archive_experiments/    completed DermMNIST results + analysis
results/
  pending_runs/         {constraint}/{model}/{variation}/ -- experiments to run
```

## How to Run

```bash
# Generate configs (interactive menu)
python -m src.config_generators.generate_configs

# Or programmatically
python -c "from src.config_generators.generate_configs import generate_configs, save_configs; save_configs(generate_configs('tralo', round='round4', dataset_mode='tissuemnist'), output_dir='results/pending_runs')"

# Run all pending experiments
python main.py
```

## Results

Each experiment directory contains: `config.json`, `training_log.csv`, `final_predictions.csv`, `evaluation_metrics.csv`

## Known Limitation

**Soft/hard count mismatch**: Loss optimizes soft counts (sum of probabilities) but satisfaction uses hard counts (argmax). Post-hoc adjustment closes this gap.
