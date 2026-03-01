# OptimizationLoss

Thesis project: train neural networks to satisfy **transductive prediction-count constraints** using soft constraint optimization, and compare against a greedy heuristic baseline.

## Datasets

### TissueMNIST (active — imagery)
- **Source**: MedMNIST collection, kidney cortex cell images (236K total, subsampled to 12K)
- **Classes**: 8 — CDI, CDS, CST, EPI, GE (constrained), PTC, STR, TUB
- **Images**: 28×28 grayscale upscaled to 224×224, auto-converted to 3-channel via `_ensure_3channel()`
- **Splits**: train 9,600 / test 2,400 (stratified 80/20)
- **Constrained class**: GE (class 4, 7.1%)
- **Group column**: `synth_group` (synthetic)
- **Data**: `data/tissuemnist/` (not in git), downloaded via `python data/tissuemnist/download_data.py`

### DermMNIST (archived — imagery)
- **Source**: MedMNIST collection, derived from HAM10000 (10,015 skin lesion images)
- **Classes**: 7 — AKIEC, BCC, BKL, DF, MEL (constrained), NV, VASC
- **Images**: 64×64×3 RGB, normalized [0, 1], channels-first `(N, 3, 64, 64)`
- **Splits**: train 7,007 / val 1,003 / test 2,005
- **Data**: `data/dermmnist/` (not in git)
- **Results**: archived in `archive_experiments/dermmnist/`

### Adult Income (legacy — tabular)
- **Classes**: 2 — income ≤50K (0) vs >50K (1, constrained)
- **Group column**: `race` (5 groups)
- **Data**: `data/adult/` (not in git)

## Pipeline

```
main.py                                  # Find pending experiments, dispatch via subprocess
  -> src/experiments/run_experiment.py    # Optimization approach (CE warmup + constraint loss)
  -> src/experiments/run_heuristic.py     # Heuristic baseline (greedy top-K allocation)
```

**Config generation:** `src/config_generators/generate_configs.py` — defines dataset configs, model grids, hyperparams. Run interactively or import `generate_configs()`.

**Evaluation & Analysis:** `src/evaluation/generate_all.py` — recomputes metrics and generates comparison charts. `src/evaluation/thesis_figures.py` — thesis-quality PDF figures.

## Models

### Tabular (`src/models/tabular/`)
- `BasicNN` — MLP with BatchNorm, ReLU, Dropout
- `FTTransformer` — Feature Tokenizer + Transformer Encoder
- `TabularResNet` — Residual blocks for tabular data
- Input: `(B, input_dim)` flat feature vectors → logits `(B, n_classes)`

### Imagery (`src/models/imagery/`)
- `ResNet18` — torchvision ResNet18 with custom head (pretrained)
- `MobileNetV3` — torchvision MobileNetV3-Large with custom head (~5.4M params, pretrained)
- Input: `(B, 3, H, W)` image tensors → logits `(B, n_classes)`

Registry in `src/models/model_factory.py`: `get_model(name, n_classes, **kwargs)` dispatches to correct type. `is_imagery_model(name)` for type checking.

## Training Phases (Optimization Approach)

1. **Warmup** — CE loss only for `warmup_epochs` (fixed count). Cached via `base_model_id` hash.

2. **Constraint Optimization** — CE + transductive constraint loss.
   - Lambda ratchet: starts at 0.005, +0.001/epoch until satisfied, then freezes.
   - Adaptive ALM: rho × 1.5 every 25 epochs.
   - KL-divergence regularization: `alpha_kl * D_KL(current || warmup)`.
   - Early stop when constraints satisfied for 5 consecutive epochs.

3. **Post-hoc Adjustment** — Flip borderline predictions to enforce hard count limits.

## Loss Formulation

```
L_total = L_ce + lambda_g * L_global + lambda_l * L_local + alpha_kl * L_kl
```

- **L_global / L_local**: Rational saturation `E/(E+K)` + ALM penalty `(rho/2)*(E/K)^2`
- **L_kl**: KL divergence vs cached warmup predictions
- Soft counts (differentiable) vs hard counts (argmax for verification)

## Directory Structure

```
src/
  config_generators/  generate_configs.py — dataset configs, model grid, hyperparams
  experiments/        run_experiment.py, run_heuristic.py
  losses/             transductive_loss.py — MulticlassTransductiveLoss
  models/
    model_factory.py  — unified registry (tabular + imagery)
    tabular/          basic_nn.py, ft_transformer.py, tabular_resnet.py
    imagery/          resnet.py (ResNet18Classifier), mobilenetv3.py (MobileNetV3Classifier)
  training/           trainer.py, constraints.py, metrics.py, logging.py, schedulers.py, model_cache.py
  utils/              data_loader.py, filesystem_manager.py, error_handler.py, posthoc_adjustment.py
  evaluation/         generate_all.py, training_curves.py, experiment_comparison.py,
                      thesis_figures.py, evaluate_statistical_significance.py,
                      visualize_stat_significance.py
run_experiments.sh    server experiment runner
setup_server.sh       server environment setup
data/tissuemnist/     images + labels as .npy (not in git)
data/dermmnist/       images + labels as .npy (not in git)
model_cache/          cached warmup .pt files (not in git)
archive_experiments/  completed DermMNIST results + analysis (dermmnist/)
results/
  completed_runs/     {constraint}/{model}/{methodology}/{variation}/ — finished experiments
  pending_runs/       {constraint}/{model}/{variation}/ — experiments to run
```

## How to Run

```bash
# Download DermMNIST data (first time only)
python data/dermmnist/download_data.py

# Generate configs (interactive menu)
python -m src.config_generators.generate_configs

# Or programmatically:
python -c "from src.config_generators.generate_configs import generate_configs, save_configs; save_configs(generate_configs('our_approach', round='round4'), output_dir='results/pending_runs')"

# Run all pending experiments
python main.py
```

## Results Path Structure

```
results/completed_runs/{constraint}/{model}/our_approach/{variation}/
results/completed_runs/{constraint}/{model}/heuristic/{variation}/
results/pending_runs/{constraint}/{model}/{variation}/
  Each contains: config.json, training_log.csv, final_predictions.csv, evaluation_metrics.csv
```

## Known Limitation

**Soft/hard count mismatch**: Loss optimizes soft counts (sum of probabilities) but satisfaction uses hard counts (argmax). Post-hoc adjustment closes this gap.
