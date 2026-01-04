# Thesis Project Structure - Systematic Experiment Framework

This project implements a systematic approach to deep learning experiments for student dropout prediction, organized across 4 key dimensions.

## Project Architecture

### Directory Structure

```
OptimizationLoss/
├── models/                      # Model architectures
│   ├── basic_nn.py             # Simple feedforward network
│   ├── resnet56.py             # ResNet-inspired with residual connections
│   ├── densenet121.py          # DenseNet-inspired with dense connections
│   ├── inception_v3.py         # Inception-inspired multi-scale features
│   ├── vgg19.py                # VGG-inspired deep sequential network
│   └── model_factory.py        # Factory for model creation
│
├── experiments/                 # Training methodologies
│   ├── our_approach.py         # Transductive + Constraint-based (MAIN)
│   ├── benchmark.py            # Baseline comparison (placeholder)
│   ├── train_then_optimize.py  # Two-stage approach (placeholder)
│   └── hybrid.py               # Hybrid methodology (placeholder)
│
├── utils/                       # Utilities
│   └── filesystem_manager.py   # Experiment path management
│
├── generate_configs.py         # Generate all experiment configurations
├── run_all_experiments.py      # Execute all experiments
│
└── results/                     # Experiment results (auto-generated)
    └── methodology/model/constraint/base_model_id/
        ├── config.json         # Experiment configuration
        ├── final_predictions.csv
        ├── evaluation_metrics.csv
        └── .complete           # Completion marker
```

## The 4 Experimental Dimensions

### 1. Methodologies
- **our_approach** (Priority): Transductive learning with constraint-based optimization
- benchmark (Future): Traditional baseline approach
- train_then_optimize (Future): Two-stage training
- hybrid (Future): Combined methodology

### 2. Models
- **BasicNN**: Simple feedforward network (baseline)
- **ResNet56**: Residual connections for deeper learning
- **DenseNet121**: Dense connections for feature reuse
- **InceptionV3**: Multi-scale feature extraction
- **VGG19**: Deep sequential architecture

### 3. Constraints
Testing 8 different constraint configurations (dropout%, enrolled%):
- (0.9, 0.8)
- (0.9, 0.5)
- (0.8, 0.7)
- (0.8, 0.2)
- (0.7, 0.5)
- (0.6, 0.5)
- (0.5, 0.3)
- (0.4, 0.2)

### 4. Hyperparameter Regimes
- **Standard**: Default hyperparameters
- **LR Test**: 5 different learning rates [0.0001, 0.0005, 0.001, 0.005, 0.01]
- **Dropout Test**: 5 different dropout rates [0.1, 0.2, 0.3, 0.4, 0.5]
- **Batch Test**: 5 different batch sizes [32, 64, 128, 256, 512]

## Workflow

### Step 1: Generate Experiment Configurations

```bash
python generate_configs.py
```

This script:
- Creates all possible combinations across the 4 dimensions
- Generates a unique configuration file for each experiment
- Creates the complete `results/` directory structure
- Calculates `base_model_id` for weight sharing (model + hyperparams, excluding constraints)
- Saves a summary report: `experiment_plan_summary.txt`

**Output**:
- Total experiments = Methodologies × Models × Constraints × Hyperparameter Variations
- Current setup: 1 × 5 × 8 × (1 + 5 + 5 + 5) = **640 experiments**

### Step 2: Run All Experiments

```bash
python run_all_experiments.py
```

Options:
```bash
# Run with defaults (resume from last run)
python run_all_experiments.py

# Re-run all experiments (including completed)
python run_all_experiments.py --no-resume

# Run only first 10 experiments
python run_all_experiments.py --max 10
```

This script:
- Scans `results/` for all experiment configurations
- Skips completed experiments (marked with `.complete` file)
- Loads data once (shared across all experiments)
- Executes each pending experiment
- Saves results to the experiment folder

## Key Concepts

### Base Model ID
Each experiment has a `base_model_id` computed from:
- Model architecture
- Training hyperparameters (LR, dropout, batch size, hidden dims)

**Excludes**: Constraints (which don't affect base model training)

This allows:
- Pre-training a model once with warmup
- Reusing the same pre-trained weights across different constraints
- Significant time savings

### Experiment Configuration Example

```json
{
  "methodology": "our_approach",
  "model_name": "ResNet56",
  "constraint": [0.5, 0.3],
  "hyperparam_regime": "lr_test",
  "hyperparams": {
    "lr": 0.001,
    "dropout": 0.3,
    "batch_size": 64,
    "hidden_dims": [128, 64],
    "epochs": 10000,
    "lambda_global": 0.01,
    "lambda_local": 0.01,
    "max_lambda_global": 0.5,
    "max_lambda_local": 0.5,
    "gradient_clip": 1.0,
    "warmup_epochs": 250
  },
  "base_model_id": "ResNet56_a1b2c3d4e5f6",
  "experiment_path": "results/our_approach/ResNet56/constraint_0.5_0.3/ResNet56_a1b2c3d4e5f6",
  "status": "pending"
}
```

### Results Structure

Each experiment folder contains:
```
results/our_approach/ResNet56/constraint_0.5_0.3/ResNet56_hash123/
├── config.json               # Full experiment configuration
├── final_predictions.csv     # Model predictions
├── evaluation_metrics.csv    # Accuracy, precision, recall, F1
├── training_log.csv          # Epoch-by-epoch training stats
└── .complete                 # Marker file indicating completion
```

## Training Stability Features

All experiments include:
- **Lambda weight capping**: Prevents constraint loss from dominating
- **Gradient clipping**: Prevents gradient explosion
- **Learning rate scheduling**: Adapts to training plateaus
- **Warmup period**: 250 epochs of pure cross-entropy before constraints

## Adding New Experiments

### Add a New Model
1. Create `models/new_model.py` with class `NewModel(nn.Module)`
2. Add import to `models/model_factory.py`
3. Add to `MODEL_REGISTRY` dict
4. Add `'NewModel'` to `MODELS` list in `generate_configs.py`
5. Re-run `python generate_configs.py`

### Add a New Methodology
1. Create `experiments/new_methodology.py` with function `train_with_new_methodology()`
2. Add to `METHODOLOGY_FUNCTIONS` dict in `run_all_experiments.py`
3. Add `'new_methodology'` to `METHODOLOGIES` list in `generate_configs.py`
4. Re-run `python generate_configs.py`

### Add More Constraints
1. Edit `CONSTRAINTS` list in `generate_configs.py`
2. Re-run `python generate_configs.py`

### Add New Hyperparameter Regime
1. Edit `HYPERPARAM_REGIMES` dict in `generate_configs.py`
2. Re-run `python generate_configs.py`

## Monitoring Progress

Check how many experiments are complete:
```bash
find results -name ".complete" | wc -l
```

Check total experiments:
```bash
find results -name "config.json" | wc -l
```

Find pending experiments:
```bash
find results -name "config.json" -type f | while read f; do
  dir=$(dirname "$f")
  if [ ! -f "$dir/.complete" ]; then
    echo "$dir"
  fi
done
```

## Tips for Thesis Work

1. **Start Small**: Run with `--max 10` to test the pipeline
2. **Check Results Early**: Review first few experiments before running all 640
3. **Use Resume**: Always run with resume enabled to avoid re-running completed experiments
4. **Monitor Resources**: Track GPU memory, disk space, and time estimates
5. **Backup Regularly**: Results folder can get large - backup periodically

## Expected Timeline

With each experiment taking ~5-15 minutes:
- 640 experiments × 10 minutes average = ~6400 minutes = **~107 hours**
- Running on GPU: ~4-5 days continuous
- Running in batches: 1-2 weeks

## Next Steps

1. Generate configurations: `python generate_configs.py`
2. Review summary: `cat experiment_plan_summary.txt`
3. Test with small batch: `python run_all_experiments.py --max 5`
4. Run all experiments: `python run_all_experiments.py`
5. Analyze results: Use scripts in `evaluation/` folder
