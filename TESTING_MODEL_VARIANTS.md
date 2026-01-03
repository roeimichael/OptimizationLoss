# Testing Model Variants - Guide

This guide explains how to organize and test different model variants while preserving your existing results.

## Folder Structure

The new system organizes results by model variant:

```
results/
├── baseline/                          # Your current results (moved here)
│   ├── constraint_0.4_0.2/
│   ├── constraint_0.5_0.3/
│   ├── ...
│   └── nn_results.json
├── enhanced_gelu_residual/            # Enhanced model with GELU + residual
│   ├── constraint_0.4_0.2/
│   ├── ...
│   └── nn_results.json
├── enhanced_with_attention/           # Enhanced model with attention
│   ├── constraint_0.4_0.2/
│   ├── ...
│   └── nn_results.json
└── variant_comparison/                # Comparison analysis
    ├── model_variant_comparison.png
    ├── variant_constraint_heatmap.png
    └── MODEL_VARIANT_COMPARISON.md
```

## Step-by-Step Usage

### Step 1: Preserve Existing Results

Move your current results to a baseline folder:

```bash
python experiments/move_baseline_results.py
```

This will:
- Move all constraint folders to `results/baseline/`
- Preserve your current nn_results.json
- Keep analysis folders intact

### Step 2: Test Enhanced Models

Run experiments with different model variants:

#### Basic Enhanced Model (GELU + Residual)
```bash
python experiments/run_experiments_with_variants.py \
    --variant enhanced_gelu_residual \
    --use-residual \
    --activation gelu
```

#### Enhanced Model with Attention
```bash
python experiments/run_experiments_with_variants.py \
    --variant enhanced_with_attention \
    --use-residual \
    --use-attention \
    --activation gelu
```

#### Enhanced Model with SiLU Activation
```bash
python experiments/run_experiments_with_variants.py \
    --variant enhanced_silu \
    --use-residual \
    --activation silu
```

#### Custom Variant
```bash
python experiments/run_experiments_with_variants.py \
    --variant my_custom_model \
    --activation mish
```

### Step 3: Compare Results

After running multiple variants, compare them:

```bash
python experiments/compare_model_variants.py
```

This generates:
- **model_variant_comparison.png**: Overall comparison charts
- **variant_constraint_heatmap.png**: Performance across all constraints
- **MODEL_VARIANT_COMPARISON.md**: Detailed comparison report
- **variant_comparison_data.csv**: Raw comparison data

### Step 4: Analyze Specific Variant

You can run the comprehensive analysis on a specific variant:

```bash
cd results/enhanced_gelu_residual
python ../../comprehensive_results_analysis.py
```

## Variant Naming Conventions

Use descriptive names for your variants:

- `baseline` - Original model (current results)
- `enhanced_gelu_residual` - Enhanced with GELU + residual connections
- `enhanced_with_attention` - Enhanced with attention mechanism
- `enhanced_silu` - Enhanced with SiLU activation
- `enhanced_full` - All enhancements enabled
- `ensemble_3models` - Ensemble of 3 models
- `custom_arch_512x4` - Custom architecture description

## Command-Line Options

### run_experiments_with_variants.py

```
--variant NAME          Model variant name (default: baseline)
--use-residual          Enable residual connections
--use-attention         Enable self-attention mechanism
--activation {relu,gelu,silu,mish}  Activation function (default: relu)
```

## Example Workflow

### Complete Testing Workflow

```bash
# 1. Preserve existing results
python experiments/move_baseline_results.py

# 2. Test basic enhanced model
python experiments/run_experiments_with_variants.py \
    --variant enhanced_basic \
    --use-residual \
    --activation gelu

# 3. Test with attention
python experiments/run_experiments_with_variants.py \
    --variant enhanced_attention \
    --use-residual \
    --use-attention \
    --activation gelu

# 4. Compare all variants
python experiments/compare_model_variants.py

# 5. Review results
cat results/variant_comparison/MODEL_VARIANT_COMPARISON.md
```

## Advanced Usage

### Testing Multiple Configurations

Create a testing script:

```bash
#!/bin/bash

variants=(
    "enhanced_gelu_residual --use-residual --activation gelu"
    "enhanced_silu_residual --use-residual --activation silu"
    "enhanced_mish_residual --use-residual --activation mish"
    "enhanced_gelu_attention --use-residual --use-attention --activation gelu"
)

for variant_cmd in "${variants[@]}"; do
    python experiments/run_experiments_with_variants.py $variant_cmd
done

python experiments/compare_model_variants.py
```

### Manual Model Testing

For more control, you can use the enhanced model directly:

```python
from src.models.neural_network_enhanced import NeuralNetClassifierEnhanced

model = NeuralNetClassifierEnhanced(
    input_dim=X_train.shape[1],
    hidden_dims=[512, 512, 256, 256, 128, 64],
    n_classes=3,
    dropout=0.3,
    use_residual=True,
    use_attention=True,
    activation='gelu'
)

# Train and save results with custom variant name
```

## Interpreting Results

### Key Metrics to Compare

1. **Average Accuracy**: Overall performance across all constraints
2. **Best Accuracy**: Peak performance achieved
3. **Avg Improvement**: How much better than baseline
4. **Training Time**: Computational cost

### What to Look For

- **Positive improvements** (+2-5%): Model enhancements are working
- **Consistent improvements**: Works across multiple constraints
- **Training time vs accuracy tradeoff**: Is the improvement worth the cost?

### Expected Improvements

Based on MODEL_IMPROVEMENTS.md:

| Enhancement | Expected Improvement |
|-------------|---------------------|
| GELU + Residual | +2-4% |
| + Attention | +1-3% additional |
| + Better training | +1-2% additional |
| Ensemble (3-5 models) | +2-4% additional |

**Total expected**: +5-15% over baseline (75% → 80-90%)

## Troubleshooting

### "No variants found"
- Make sure you've run experiments with `--variant` parameter
- Check that `results/{variant_name}/nn_results.json` exists

### "Results already exist"
- Use different variant names for different experiments
- Or delete the existing variant folder to re-run

### Import errors
- Make sure you're running from the project root directory
- Ensure src/ is in your Python path

## Notes

- The baseline folder preserves your original results
- Each variant runs independently
- You can delete any variant folder to re-run experiments
- Comparison script automatically detects all valid variant folders
- Analysis folders (constraint_analysis, comprehensive_analysis) are preserved
