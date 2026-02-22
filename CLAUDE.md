# OptimizationLoss

Thesis project: train neural networks to satisfy **transductive prediction-count constraints** using soft constraint optimization, and compare against a greedy heuristic baseline.

## Dataset

**Adult Income** (UCI) — binary classification: income >50K (class 1) vs <=50K (class 0).
- Train: `data/adult/train_dataset_cleaned.csv` (~30K samples)
- Test: `data/adult/test_dataset_cleaned.csv` (~15K samples)
- Group column: `race` (5 groups: White, Black, Asian-Pac-Islander, Amer-Indian-Eskimo, Other)
- Target column: `income`
- Columns configured in `config/experiment_config.py`

## Pipeline

```
main.py                                  # Find pending experiments, dispatch via subprocess
  -> src/experiments/run_experiment.py    # Optimization approach (CE warmup + constraint loss)
  -> src/experiments/run_heuristic.py     # Heuristic baseline (greedy top-K allocation)
```

**Config generation:** `src/config_generators/generate_configs.py` creates experiment configs for each constraint pair. Run interactively or import `generate_configs()`.

**Analysis:** `src/analysis/generate_all.py` recomputes metrics and generates comparison charts.

## Training Phases (Optimization Approach)

1. **Warmup** — CE loss only, runs until accuracy saturates (patience-based early stop, min 50 epochs, max 500). Cached via `base_model_id` hash so all constraint pairs share one warmup.

2. **Constraint Optimization** — CE + transductive constraint loss (up to 350 epochs).
   - Lambda ratchet: starts at 0.005, increments by 0.001/epoch until constraints satisfied, then freezes.
   - Adaptive ALM: rho starts at 0.5, multiplied by 1.5 every 25 epochs.
   - KL-divergence regularization: `alpha_kl * D_KL(current || warmup)` to prevent catastrophic drift.
   - Early stop when global + local constraints all satisfied for 5 consecutive epochs.

3. **Post-hoc Adjustment** — After training, flip borderline predictions (lowest confidence in constrained class) to enforce hard count limits. Applied globally first, then per-group locally.

## Loss Formulation

```
L_total = L_ce + lambda_g * L_global + lambda_l * L_local + alpha_kl * L_kl
```

- **L_global / L_local**: Rational saturation `E/(E+K)` + ALM penalty `(rho/2)*(E/K)^2`, where `E = relu(soft_count - limit)` and `K = limit`.
- **L_kl**: KL divergence between current softmax outputs and cached warmup-model predictions.
- Soft counts = sum of softmax probabilities (differentiable). Hard counts = argmax (used for satisfaction check only).

## Directory Structure

```
src/
  config_generators/  generate_configs.py — experiment grid
  experiments/        run_experiment.py, run_heuristic.py
  losses/             transductive_loss.py — MulticlassTransductiveLoss
  models/             ft_transformer.py, basic_nn.py, tabular_resnet.py, model_factory.py
  training/           trainer.py, constraints.py, metrics.py, logging.py, schedulers.py, model_cache.py
  utils/              data_loader.py, filesystem_manager.py, error_handler.py, posthoc_adjustment.py
  analysis/           generate_all.py, training_curves.py, experiment_comparison.py
config/               experiment_config.py (TARGET_COLUMN, GROUP_COLUMN)
data/adult/           train/test CSVs (not in git)
model_cache/          cached warmup .pt files (not in git)
results/              experiment outputs organized by methodology/model/constraint
```

## Key Hyperparameters (defaults in generate_configs.py)

| Param | Value | Notes |
|-------|-------|-------|
| lr | 0.001 | Warmup phase learning rate |
| lr_constraint | 0.00001 | Constraint phase learning rate |
| hidden_dims | [128, 64] | FTTransformer dimensions |
| dropout | 0.3 | |
| batch_size | 64 | |
| warmup_epochs | 50 | Minimum (actual determined by saturation) |
| constraint_epochs | 350 | Maximum constraint phase |
| lambda_global/local | 0.005 | Initial constraint weight |
| lambda_step | 0.001 | Per-epoch lambda increment |
| initial_rho | 0.5 | ALM penalty coefficient |
| alpha_kl | 1.0 | KL divergence weight |

## Constraint Pairs

Defined in `generate_configs.py` as `(local_fraction, global_fraction)` of test set size:
```
(0.9, 0.8), (0.9, 0.5), (0.8, 0.7), (0.8, 0.2),
(0.7, 0.5), (0.6, 0.5), (0.5, 0.3), (0.4, 0.2)
```

## Results Path Structure

```
results/{dataset_mode}/{methodology}/{model}/constraint_{local}_{global}/{regime}/{variation}/
  config.json              # Full experiment config + results
  training_log.csv         # Per-epoch loss/constraint tracking
  final_predictions.csv    # y_true, y_pred, probabilities, groups
  evaluation_metrics.csv   # Accuracy, F1, precision, recall, etc.
```

## How to Run

```bash
# Generate configs
python -m src.config_generators.generate_configs

# Run all pending experiments
python main.py

# Run a single experiment
python -m src.experiments.run_experiment results/binary/our_approach/FTTransformer/constraint_0.5_0.3/standard/default/config.json

# Run a single heuristic
python -m src.experiments.run_heuristic results/binary/heuristic/FTTransformer/constraint_0.5_0.3/standard/default/config.json
```

## Known Limitation

**Soft/hard count mismatch**: The loss function optimizes soft counts (sum of probabilities) for differentiability, but constraint satisfaction is verified using hard counts (argmax predictions). This is fundamental to differentiable constraint optimization — the post-hoc adjustment step exists to close this gap.
