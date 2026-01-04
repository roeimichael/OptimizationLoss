# Project Structure

## Running Experiments

### Baseline Model (Original)
```bash
python experiments/run_experiments.py
```
- Uses: `config/experiment_config.py`
- Model: Simple neural network with BatchNorm + ReLU + Dropout
- Results saved to: `results/` (as configured in experiment_config.py)

### Enhanced Model (Improved Architecture)
```bash
python experiments/run_enhanced_experiments.py
```
- Uses: `config/enhanced_model_config.py`
- Model: Enhanced neural network with residual connections + GELU activation + LayerNorm
- Results saved to: `results/enhanced_gelu_residual/`

## Configuration Files

### config/experiment_config.py
Baseline model configuration:
- `CONSTRAINTS`: List of (local_percent, global_percent) tuples
- `NN_CONFIGS`: Model architectures to test (arch_deep, dropout_high, etc.)
- `TRAINING_PARAMS`: Epochs, batch size, learning rate
- `RESULTS_DIR`: Where to save results (default: ./results)

### config/enhanced_model_config.py
Enhanced model configuration:
- Same structure as experiment_config.py
- Additional `MODEL_PARAMS` dict:
  - `model_type`: 'enhanced'
  - `use_residual`: True/False
  - `use_attention`: True/False
  - `activation`: 'gelu', 'silu', 'mish', or 'relu'
- `RESULTS_DIR`: Automatically set to separate folder

## Pre-trained Model Caching

Both experiment scripts use automatic model caching:

1. **First run**: Trains 250 warmup epochs, saves model to `models/trained_models/`
2. **Subsequent runs**: Loads pre-trained model, skips warmup (saves ~40-50% time)
3. **Separate caches**: Baseline and enhanced models have separate cache files

Cache files are named: `warmup_{model_type}_d{input_dim}_h{hidden_dims}_drop{dropout}_{hash}.pt`

## Evaluation Scripts

All located in `evaluation/` folder:

### evaluation/comprehensive_results_analysis.py
Complete analysis of all experiment results:
- Filters failed experiments
- Per-course constraint satisfaction
- Class prediction distributions
- Constraint adherence visualizations

### evaluation/compare_model_variants.py
Compares baseline vs enhanced model performance:
- Auto-detects variant folders in results/
- Side-by-side comparison plots
- Performance improvement metrics

### evaluation/analyze_by_constraints.py
Groups results by constraint configuration (used for multi-constraint experiments)

### evaluation/analyze_top5.py
Ranks top 5 performing configurations (used for single-constraint experiments)

## Project Layout

```
OptimizationLoss/
├── config/
│   ├── experiment_config.py          # Baseline model config
│   └── enhanced_model_config.py      # Enhanced model config
├── experiments/
│   ├── run_experiments.py            # Run baseline experiments
│   └── run_enhanced_experiments.py   # Run enhanced model experiments
├── evaluation/
│   ├── comprehensive_results_analysis.py
│   ├── compare_model_variants.py
│   ├── analyze_by_constraints.py
│   └── analyze_top5.py
├── src/
│   ├── models/
│   │   ├── neural_network.py         # Baseline model
│   │   └── neural_network_enhanced.py # Enhanced models
│   ├── training/
│   │   ├── trainer.py                # Training logic with pre-trained model support
│   │   ├── metrics.py
│   │   └── logging.py
│   ├── losses/
│   │   └── transductive_loss.py      # Constraint-based loss
│   └── ...
├── models/
│   └── trained_models/               # Pre-trained model cache
├── results/
│   ├── baseline/                     # Baseline experiment results
│   ├── enhanced_gelu_residual/       # Enhanced model results
│   └── ...
└── data/
    ├── dataset.csv
    ├── dataset_train.csv
    └── dataset_test.csv
```

## Workflow

1. **Run baseline experiments:**
   ```bash
   python experiments/run_experiments.py
   ```
   - First run: Trains from scratch, saves warmup model
   - Subsequent runs: Loads pre-trained model, skips warmup
   - Results saved to `results/` (or as configured)

2. **Run enhanced model experiments:**
   ```bash
   python experiments/run_enhanced_experiments.py
   ```
   - Uses enhanced architecture (residual + GELU)
   - Separate pre-trained cache
   - Results saved to `results/enhanced_gelu_residual/`

3. **Analyze results:**
   ```bash
   python evaluation/comprehensive_results_analysis.py
   python evaluation/compare_model_variants.py
   ```

## Creating New Model Variants

To test different model architectures:

1. Create new config file: `config/enhanced_attention_config.py`
2. Set `MODEL_PARAMS`:
   ```python
   MODEL_PARAMS = {
       'model_type': 'enhanced',
       'use_residual': True,
       'use_attention': True,  # Enable attention
       'activation': 'gelu'
   }
   RESULTS_DIR = "./results/enhanced_with_attention"
   ```
3. Copy `run_enhanced_experiments.py` and update import to use new config
4. Run experiments - automatic caching will work for this variant too

## Notes

- All experiment scripts are config-based (no command-line arguments)
- Pre-trained models are architecture-specific (different dims = different cache)
- Model variants have separate caches (no conflicts)
- Evaluation scripts auto-detect and compare all variants in results/
