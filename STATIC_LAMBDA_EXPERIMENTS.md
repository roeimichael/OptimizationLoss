# Static Lambda Experimental Methodology

## Overview

This document describes the **static lambda** experimental methodology, a parallel approach to the adaptive lambda methodology. In static lambda experiments, lambda values remain constant throughout training, allowing us to explore the effect of different fixed constraint weights.

## Key Differences from Adaptive Methodology

| Aspect | Adaptive (`our_approach`) | Static Lambda |
|--------|--------------------------|---------------|
| **Warmup Phase** | 250 epochs (CE only) | 0 epochs (no warmup) |
| **Lambda Behavior** | Increases adaptively when constraints violated | Remains constant throughout |
| **Training Epochs** | Up to 10,000 (early stopping) | Fixed 300 epochs |
| **Initial Lambda** | 0.01 (both global and local) | Configurable (0.001 to 10.0) |
| **Lambda Step** | 0.01 (incremental increase) | N/A (static) |
| **Early Stopping** | Yes (when constraints satisfied) | Optional (runs full 300 epochs) |
| **Failure Handling** | Always succeeds eventually | Fails if constraints not met |
| **Experiment Duration** | ~10-30 minutes | ~3 minutes |
| **Model Caching** | Uses warmup cache | No caching (trains from scratch) |

---

## Configuration Breakdown

### Total Experiments: 84
**Calculation:** 3 models × 4 constraint pairs × 7 lambda configurations = 84 experiments

---

## 1. Neural Network Architectures (3 tabular models)

Same as adaptive methodology:

| Model | Architecture Type | Key Feature | Complexity |
|-------|------------------|-------------|------------|
| **BasicNN** | Multi-Layer Perceptron | Standard feedforward baseline | Low |
| **TabularResNet** | Residual Network | Skip connections for tabular data | Medium |
| **FTTransformer** | Transformer | Feature tokenization + self-attention | High |

---

## 2. Constraint Pairs (4 configurations)

Same as adaptive methodology:

| Constraint | Local % | Global % | Interpretation |
|-----------|---------|----------|----------------|
| **(0.9, 0.8)** | 0.9 | 0.8 | [Soft, Soft] - Both permissive |
| **(0.3, 0.8)** | 0.3 | 0.8 | [Hard, Soft] - Local restrictive, Global permissive |
| **(0.8, 0.3)** | 0.8 | 0.3 | [Soft, Hard] - Local permissive, Global restrictive |
| **(0.3, 0.3)** | 0.3 | 0.3 | [Hard, Hard] - Both restrictive |

---

## 3. Lambda Value Combinations (7 configurations)

### Symmetric Lambda Configurations (global = local)

| Configuration | λ_global | λ_local | Description |
|--------------|----------|---------|-------------|
| **very_low** | 0.001 | 0.001 | Very weak constraint enforcement |
| **low** | 0.01 | 0.01 | Weak enforcement (adaptive baseline) |
| **medium** | 0.1 | 0.1 | Medium constraint enforcement |
| **high** | 1.0 | 1.0 | Strong constraint enforcement |
| **very_high** | 10.0 | 10.0 | Very strong enforcement |

**Design Rationale:**
- **very_low (0.001):** Tests minimal constraint influence
- **low (0.01):** Matches adaptive methodology's starting value
- **medium (0.1):** Tests moderate constant pressure
- **high (1.0):** Tests strong constant enforcement
- **very_high (10.0):** Tests if very high values can achieve convergence

### Asymmetric Lambda Configurations (global ≠ local)

| Configuration | λ_global | λ_local | Description |
|--------------|----------|---------|-------------|
| **global_heavy** | 1.0 | 0.01 | Prioritize global constraints |
| **local_heavy** | 0.01 | 1.0 | Prioritize local constraints |

**Design Rationale:**
- Tests whether asymmetric enforcement affects convergence
- Explores trade-offs between global vs local satisfaction
- Useful for understanding constraint interaction dynamics

---

## 4. Hyperparameters

All experiments use consistent hyperparameters for fair comparison:

| Parameter | Value | Notes |
|-----------|-------|-------|
| **epochs** | 300 | Fixed duration |
| **warmup_epochs** | 0 | No warmup phase |
| **lr** | 0.001 | Fixed learning rate |
| **batch_size** | 64 | Standard batch size |
| **dropout** | 0.3 | Regularization |
| **hidden_dims** | [128, 64] | Network architecture |
| **constraint_threshold** | 1e-6 | Satisfaction threshold |

---

## 5. Experiment Organization

Results are organized separately from adaptive methodology:

```
results/
├── our_approach/          # Adaptive lambda experiments
│   └── [untouched]
└── static_lambda/         # Static lambda experiments (NEW)
    ├── BasicNN/
    │   ├── constraint_0.9_0.8/
    │   │   └── lambda_search/
    │   │       ├── very_low/      # λ=0.001
    │   │       ├── low/            # λ=0.01
    │   │       ├── medium/         # λ=0.1
    │   │       ├── high/           # λ=1.0
    │   │       ├── very_high/      # λ=10.0
    │   │       ├── global_heavy/   # λ_g=1.0, λ_l=0.01
    │   │       └── local_heavy/    # λ_g=0.01, λ_l=1.0
    │   ├── constraint_0.3_0.8/
    │   ├── constraint_0.8_0.3/
    │   └── constraint_0.3_0.3/
    ├── TabularResNet/
    │   └── [same structure]
    └── FTTransformer/
        └── [same structure]
```

Each leaf directory contains:
- `config.json` - Full experiment configuration
- `training_log.csv` - Per-epoch training metrics
- `final_predictions.csv` - Model predictions (if constraints met)
- `evaluation_metrics.csv` - Performance metrics (if constraints met)

---

## 6. Experiment Outcomes

### Successful Experiments
- **Status:** `'completed'`
- **Indicators:**
  - Both global and local constraints satisfied after 300 epochs
  - Full training log available
  - Final predictions and metrics saved
  - `config['results']['constraints_satisfied'] = True`

### Failed Experiments
- **Status:** `'constraints_not_met'`
- **Indicators:**
  - Constraints NOT satisfied after 300 epochs
  - Training log available (shows failure trajectory)
  - No final predictions (constraints not met)
  - `config['results']['constraints_satisfied'] = False`
  - Includes diagnostic information:
    - `global_satisfied`: Boolean
    - `local_satisfied`: Boolean
    - `final_global_loss`: Float
    - `final_local_loss`: Float

---

## 7. Running Experiments

### Generate Configurations

```bash
python src/utils/generate_static_lambda_configs.py
```

This creates 84 experiment configurations in `results/static_lambda/`.

### Run Single Experiment

```bash
python run_static_lambda_experiment.py results/static_lambda/BasicNN/constraint_0.9_0.8/lambda_search/medium/config.json
```

### Batch Run All Experiments

```bash
find results/static_lambda -name 'config.json' | while read config; do
    python run_static_lambda_experiment.py "$config"
done
```

### Parallel Execution

```bash
find results/static_lambda -name 'config.json' | parallel -j 4 python run_static_lambda_experiment.py {}
```

---

## 8. Expected Results and Analysis

### Research Questions

1. **Lambda Sensitivity:**
   - Which lambda values enable constraint satisfaction in 300 epochs?
   - Is there a "sweet spot" for static lambda values?

2. **Comparison with Adaptive:**
   - Does adaptive lambda outperform optimal static lambda?
   - What is the minimum effective static lambda value?

3. **Constraint Interaction:**
   - Do asymmetric lambdas affect convergence?
   - Are global or local constraints harder to satisfy?

4. **Model Differences:**
   - Do different architectures require different lambda values?
   - Which models are most robust to lambda variation?

### Analysis Metrics

For each (model, constraint) pair, analyze:
- **Success rate** by lambda value
- **Convergence speed** (epochs to satisfaction)
- **Final performance** (accuracy, F1)
- **Lambda threshold** (minimum lambda for success)

### Comparison Framework

| Aspect | Metric | Expected Outcome |
|--------|--------|------------------|
| **Convergence** | Success rate by λ | Higher λ → higher success rate |
| **Performance** | Accuracy vs λ | Optimal λ balances constraints & accuracy |
| **Efficiency** | Training time | Static: ~3 min vs Adaptive: ~15 min |
| **Robustness** | Success across constraints | Some λ values work universally |

---

## 9. Timeline Estimate

Assuming average training time of 3 minutes per experiment:

| Configuration | Experiments | Time (serial) | Time (4 parallel) | Time (8 parallel) |
|--------------|-------------|---------------|-------------------|-------------------|
| **Static Lambda** | 84 | 4.2 hours | 1.05 hours | 31.5 minutes |
| Adaptive (`our_approach`) | 36 | 18 hours | 4.5 hours | 2.25 hours |

**Note:** Static lambda experiments are ~6× faster per experiment than adaptive methodology.

---

## 10. Advantages of Static Lambda Methodology

1. **Speed:** 300 epochs vs up to 10,000 epochs
2. **Simplicity:** No adaptive logic, easier to analyze
3. **Explicit Control:** Test specific lambda hypotheses
4. **Failure Information:** Learn which configurations don't work
5. **Parallel Comparison:** Compare against adaptive approach
6. **Lambda Sensitivity Analysis:** Map out constraint satisfaction landscape

---

## 11. Limitations

1. **Higher Failure Rate:** Some configurations will not converge
2. **No Warmup:** Trains from random initialization
3. **Fixed Duration:** May need more epochs for some configurations
4. **No Model Caching:** Cannot reuse warmup models

---

## 12. Next Steps After Running Experiments

### 1. Success Rate Analysis
```python
# Count successes per lambda value
import json
from pathlib import Path

results = {}
for config_path in Path('results/static_lambda').rglob('config.json'):
    config = json.load(open(config_path))
    lambda_val = config['variation_name']
    status = config['status']
    results[lambda_val] = results.get(lambda_val, {'success': 0, 'fail': 0})
    if status == 'completed':
        results[lambda_val]['success'] += 1
    elif status == 'constraints_not_met':
        results[lambda_val]['fail'] += 1
```

### 2. Compare Against Adaptive
- Which methodology achieves better final accuracy?
- Does adaptive lambda always succeed where static fails?
- What is the optimal static lambda equivalent to adaptive?

### 3. Constraint-Specific Analysis
- Which constraint pairs are hardest to satisfy?
- Do [Hard, Hard] constraints require higher lambda values?
- Are local or global constraints the bottleneck?

### 4. Model Architecture Analysis
- Do complex models (FTTransformer) need different lambdas?
- Are simple models (BasicNN) more sensitive to lambda values?

---

## Summary

The static lambda methodology provides a complementary approach to adaptive lambda experiments, enabling:
- **Faster experimentation** (300 epochs vs 10,000)
- **Explicit lambda exploration** (test specific values)
- **Failure analysis** (understand which configurations don't work)
- **Comparative evaluation** (static vs adaptive performance)

By running both methodologies, we can determine whether the complexity of adaptive lambda adjustment provides benefits over carefully chosen static values, and identify the optimal lambda configuration for constraint-based student dropout prediction.
