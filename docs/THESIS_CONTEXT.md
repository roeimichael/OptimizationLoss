# Thesis Project Context: Transductive Prediction-Count Constraint Optimization

This document provides background context for AI assistants working on this thesis project. It covers the research problem, the novel loss function, experimental history, key findings, known issues, and lessons learned.

---

## The Research Problem

In classification tasks, a model's predictions are usually evaluated only on accuracy. But in real-world deployment — medical diagnostics, resource allocation, hiring — there are **hard limits on how many samples can be assigned to a given class**. For example: a dermatology clinic can only biopsy 50 lesions per day, so a skin cancer classifier must predict "melanoma" for at most 50 patients, while still maximizing diagnostic accuracy.

This project proposes a **soft constraint optimization approach** that trains neural networks to respect **prediction-count constraints** — limits on how many test samples may be classified into a specific class — at both global (whole test set) and local (per-subgroup) levels.

### Constraint Types

- **Global constraint**: "Predict melanoma for at most K samples total"
- **Local constraint**: "Within each demographic subgroup (e.g., body location), predict melanoma for at most K_g samples"
- Constraints are specified as percentages of the true class count: e.g., (local=60%, global=50%) means the model may predict MEL for at most 60% of true MELs per group, and 50% of true MELs globally

### Why This Is Hard

The core difficulty is that prediction counts are discrete (argmax), but neural network training requires differentiable objectives. Our approach bridges this gap using **soft counts** (sum of class probabilities) as a differentiable proxy for **hard counts** (argmax predictions). This creates a known soft/hard mismatch that requires post-hoc correction.

---

## The Novel Loss Function

```
L_total = L_CE + lambda_g * L_global + lambda_l * L_local + alpha_kl * L_kl
```

### Components

1. **L_CE (Cross-Entropy)**: Standard classification loss on training set. Drives accuracy.

2. **L_global (Global Constraint Loss)**: Penalizes exceeding the global prediction count limit.
   - Uses **rational saturation**: `E/(E+K)` where E = max(0, soft_count - K). Bounded in [0,1), prevents gradient explosion.
   - Plus **bounded ALM quadratic**: `(E/K)^2 / (1 + (E/K)^2)`. Also bounded, with adaptive penalty factor rho.
   - Combined: `sat + rho * quad`, bounded to ~[0, 1+rho).

3. **L_local (Local Constraint Loss)**: Same formulation as L_global but computed per demographic subgroup. Ensures fairness within groups.

4. **L_KL (KL Divergence Regularization)**: `D_KL(current_proba || warmup_proba)`. Prevents the model from distorting its prediction distribution too far from the CE-only baseline during constraint optimization. Controlled by `alpha_kl`.

### Key Design Choices in the Loss

- **Rational saturation** was chosen over raw ReLU penalties because unbounded penalties caused gradient explosion in early experiments.
- **Bounded ALM** accelerates convergence beyond what saturation alone achieves, while remaining numerically stable.
- **KL regularization** was added after observing that aggressive constraint optimization sometimes collapsed predictions into few classes. It acts as an anchor to the warmup model's distribution.

### Training Phases

1. **Warmup Phase** (typically 50 epochs): Train with CE loss only. Cache the model weights for reuse across constraint variations. This cached model serves as the KL reference distribution.

2. **Constraint Phase** (up to 500 epochs): Add constraint losses with a **lambda ratchet** mechanism:
   - Start lambda at a small value (0.005-0.01)
   - Increase by `lambda_step` (0.002) each epoch where constraints are violated
   - Once constraints are satisfied, freeze lambda (never decrease — see Known Issues)
   - Adaptive ALM: multiply rho by 1.5 every 25 epochs
   - Early stopping: halt when constraints satisfied for 5 consecutive epochs, or after 100 epochs of stagnation

3. **Post-hoc Adjustment**: After training, flip borderline predictions to enforce hard count limits. This closes the soft/hard count gap:
   - Over limit: reassign lowest-confidence constrained-class predictions to next-best class
   - Under limit: reassign highest-confidence non-constrained predictions to constrained class
   - Applied globally first, then per-group

### Heuristic Baseline (Comparison Method)

A **greedy top-K allocation** baseline that uses the same warmup model but skips constraint training entirely:
- Sort samples by predicted probability for the constrained class
- Assign top-K to constrained class (respecting global and local limits)
- Assign remaining samples to their highest-probability unconstrained class

---

## Datasets

### DermMNIST (Primary, Medical Imaging)
- **Source**: HAM10000 via MedMNIST, 10,015 skin lesion images (224x224 RGB)
- **Classes**: 7 — AKIEC (3.2%), BCC (5.1%), BKL (11%), DF (1.1%), MEL (11.1%), NV (66.9%), VASC (1.4%)
- **Constrained class**: MEL (melanoma, class 4) — clinically the most important to detect but also the one where over-prediction wastes biopsy resources
- **Group column**: `loc_group` — body location mapped to 3 groups: torso (~52%), extremity (~37%), head_neck (~11%)
- **Splits**: 5 independent stratified 80/20 splits (seeds 43-47), each with 8,012 train / 2,003 test
- **Status**: Extensive experiments completed; archived in `archive_experiments/dermmnist_good/`

### TissueMNIST (Secondary, Medical Imaging)
- **Source**: MedMNIST kidney cortex microscopy, subsampled from 236K to 12K images
- **Classes**: 8 — CDI, CDP, CT, DCT, GE (constrained, 7.1%), INT, PTC, PTS
- **Images**: 28x28 grayscale upscaled to 224x224, auto-converted to 3-channel for pretrained models
- **Group column**: `synth_group` — synthetic binary (no real demographics available)
- **Splits**: Single stratified 80/20 split, 9,600 train / 2,400 test
- **Status**: Active development dataset

### OULAD (Archived, Tabular)
- **Source**: Open University Learning Analytics Dataset
- **Task**: Student outcome prediction with constrained class = high-risk students
- **Group column**: Race (5 groups)
- **Status**: 443 experiments completed. Served as initial validation. Results archived in `archive_experiments/`

---

## Models Tested

### Imagery (Current Focus)
| Model | Params | Notes |
|-------|--------|-------|
| ResNet18 | ~11M | Consistently underperformed MobileNetV3. Dropped from active experiments. |
| MobileNetV3-Large | ~5.4M | Best accuracy-to-size ratio. Primary model for DermMNIST. |
| EfficientNetB0 | ~4M | Pending full evaluation. Included in server experiments. |
| ResNet50 | ~25M | Tested briefly. No significant advantage over lighter models for this task. |

### Tabular (Archived, OULAD only)
| Model | Notes |
|-------|-------|
| BasicNN (MLP) | Simplest. Won on relaxed constraints with heuristic. |
| TabularResNet | Best for tralo on strict constraints. |
| FTTransformer | Middle performance. No clear advantage for this task. |

All imagery models use torchvision pretrained weights with a custom classification head (Dropout -> Linear).

---

## Key Experimental Findings

### What Worked

1. **Our approach dominates on strict constraints**: On OULAD with constraints (0.5, 0.3) and (0.8, 0.2), our approach beat all baselines by +13-14%. The tighter the constraint, the bigger our advantage.

2. **Pretrained backbones are essential**: ImageNet pretraining gives +10-11% accuracy boost on DermMNIST with no constraint degradation. Always use pretrained=True for imagery.

3. **KL regularization helps stability**: `alpha_kl=0.5` prevents distribution collapse during aggressive constraint optimization. Without it, models sometimes predict only 2-3 classes.

4. **Rho=5.0 speeds convergence**: Higher initial ALM penalty (rho=5.0 vs default 1.0) pushes constraint satisfaction much faster without accuracy loss.

5. **The lambda ratchet is robust**: Monotonically increasing lambda reliably drives constraints to satisfaction across all tested configurations.

6. **Warmup model caching saves significant time**: 50-epoch warmup models are cached by a hash of (model, lr, dropout, batch_size, warmup_epochs, pretrained, dataset). Shared across all constraint variations and both methods.

7. **Post-hoc adjustment reliably closes the soft/hard gap**: Even when soft counts are borderline, flipping 1-5 predictions suffices.

### What Failed or Underperformed

1. **Heuristic wins on relaxed constraints**: When constraints are loose (e.g., 0.9/0.8), the heuristic baseline achieves better accuracy because it makes fewer unnecessary prediction changes. Our approach over-constrains.

2. **ResNet18 consistently underperformed MobileNetV3** on DermMNIST despite having more parameters. Dropped from the active experiment grid.

3. **Very small local constraints are problematic**: When a local constraint is <5 samples, soft count approximation breaks down. Sum of probabilities across a subgroup can exceed the limit even when argmax predictions satisfy it. This was a major issue on OULAD (small test set) and is less problematic on DermMNIST (larger groups).

4. **Lambda never decays after satisfaction**: Once lambdas are frozen at their ratcheted values, the model is "locked in" — it can't explore the constraint boundary to improve accuracy further. This causes post-convergence accuracy degradation. Multiple fixes were proposed (simple decay, sustained decay, utilization-target) but not implemented yet.

5. **Sex-based grouping showed no difference from loc_group** on DermMNIST: both produced identical heuristic results because the constraint pair (0.5, 0.3) made global constraint always bind first, making local constraints non-binding regardless of group structure.

6. **CE saturation**: With pretrained models on DermMNIST, train accuracy hits 0.995+ within a few epochs of constraint phase, making CE gradients near-zero. A workaround skips CE computation when train accuracy exceeds 0.995.

### Constraint Pair Insights

- When `local% > global%` (e.g., 0.5/0.3): global constraint binds first, local is non-binding. Heuristic and tralo produce very similar local-group distributions.
- When `local% < global%` (e.g., 0.3/0.8): local constraint is the binding one. This is where per-group optimization matters most.
- Equal constraints (0.3/0.3 or 0.8/0.8): both bind, creating the most complex optimization landscape.

### Current Pending Experiments (480 configs, DermMNIST)

4 scenarios (single_MEL, single_NV, multi_MEL_BKL, multi_MEL_BCC_VASC) x 6 constraint pairs (L30_G30, L30_G80, L80_G30, L70_G50, L50_G70, L80_G80) x 2 models (MobileNetV3, EfficientNetB0) x 2 methods (tralo, heuristic) x 5 slices.

These are designed to test:
- Single vs multi-class constraints
- Local-binding vs global-binding vs equal constraint pairs
- Constraining the majority class (NV, 67%) vs minority class (MEL, 11%)

---

## Architecture and Code Structure

```
main.py                     # Orchestrator: finds pending configs, dispatches to GPU workers
src/
  experiments/
    run_experiment.py       # Our approach: warmup + constraint optimization + post-hoc
    run_heuristic.py        # Heuristic: warmup + greedy top-K allocation
  losses/
    transductive_loss.py    # MulticlassTransductiveLoss (the novel loss)
  models/
    model_factory.py        # Registry: get_model(name) -> model instance
    imagery/                # ResNet18, MobileNetV3, EfficientNetB0
    tabular/                # BasicNN, TabularResNet, FTTransformer
  training/
    trainer.py              # ConstraintTrainer: two-phase training loop
    constraints.py          # Compute global/local constraint limits from data
    model_cache.py          # Warmup model caching by hash ID
    metrics.py              # Accuracy, F1, precision, recall, ECE, Brier
    schedulers.py           # LR schedule (warmup_lr -> constraint_lr step)
  utils/
    data_loader.py          # Load .npy images + metadata, compute constraints
    posthoc_adjustment.py   # Flip borderline predictions to enforce hard limits
  config_generators/
    generate_configs.py     # Hyperparameter grids and config generation
  evaluation/
    thesis_figures.py       # Publication-quality PDF figures
    experiment_comparison.py # Cross-experiment analysis
```

### Key Hyperparameters (Current Defaults)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| lr | 1e-4 | Warmup phase learning rate |
| lr_constraint | 5e-6 | Constraint phase learning rate (much lower) |
| warmup_epochs | 50 | CE-only training before constraints |
| constraint_epochs | 500 | Max epochs for constraint phase |
| lambda_global/local | 0.01 | Initial constraint loss weights |
| lambda_step | 0.002 | Ratchet increment per violated epoch |
| initial_rho | 5.0 | ALM quadratic penalty starting value |
| alpha_kl | 0.5 | KL divergence regularization strength |
| dropout | 0.3 | Dropout rate in classification head |
| batch_size | 64 | Training batch size |
| pretrained | True | Use ImageNet pretrained backbone |

---

## Known Limitations and Open Questions

1. **Soft/hard count mismatch**: Fundamental to the approach. Soft counts (differentiable) drive optimization, hard counts (discrete) determine real satisfaction. Post-hoc adjustment handles this but adds a non-differentiable step.

2. **Lambda never decays**: After first satisfaction, lambda stays frozen at its ratcheted value. This prevents the model from recovering accuracy post-convergence. Fix is designed but not yet implemented.

3. **Transductive setting**: Constraints require access to the full test set during training. This is a strong assumption — the model sees test features (but not labels) during training. This is standard in transductive learning but limits deployment to batch prediction settings.

4. **Class imbalance**: DermMNIST is heavily imbalanced (NV=67% vs DF=1.1%). Constraining the majority class (NV) is a harder optimization problem than constraining a minority class (MEL).

5. **Synthetic groups in TissueMNIST**: The binary group column is artificial. Real fairness constraints would use real demographics, but BBBC051 doesn't provide them.

---

## Reproducibility Notes

- All experiments use deterministic seeds where specified
- Warmup models are cached and reused across constraint variations (shared between tralo and heuristic via separate cache IDs with `_heuristic` suffix)
- 5 stratified slices with fixed seeds (43-47) enable statistical significance testing
- Config files are self-contained JSON: every experiment can be reproduced from its config.json alone
- The pipeline supports safe stop/restart: completed experiments are skipped on rerun (status field in config.json)
