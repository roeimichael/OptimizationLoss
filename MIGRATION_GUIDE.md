# Migration Guide: Old → New Structure

## What Changed?

The project has been completely restructured into a professional Python package layout.

### Old Structure (Deprecated)
```
OptimizationLoss/
├── nn_experiments/        # All code was here
│   ├── config.py
│   ├── trainer.py
│   ├── model.py
│   └── ...
├── example_usage.py       # Old demos
├── loss.py               # Old loss function
└── ...                   # Scattered files
```

### New Structure (Current)
```
OptimizationLoss/
├── src/                   # Organized source code
│   ├── data/             # Data handling
│   ├── models/           # Neural networks
│   ├── losses/           # Loss functions
│   ├── training/         # Training logic
│   └── utils/            # Utilities
├── config/               # Configuration
├── experiments/          # Experiment runners
├── tests/                # Tests
└── docs/                 # Documentation
```

## How to Run Experiments

### Old Way (nn_experiments/)
```bash
cd nn_experiments
python run_experiments.py
```

### New Way (Current)
```bash
cd experiments
python run_experiments.py
```

## Import Changes

### Old Imports (from nn_experiments/)
```python
from config import *
from trainer import train_model_transductive
from model import NeuralNetClassifier
from transductive_loss import MulticlassTransductiveLoss
```

### New Imports (Current)
```python
from config.experiment_config import *
from src.training import train_model_transductive
from src.models import NeuralNetClassifier
from src.losses import MulticlassTransductiveLoss
```

## File Mapping

| Old Location | New Location |
|-------------|-------------|
| `nn_experiments/config.py` | `config/experiment_config.py` |
| `nn_experiments/trainer.py` | `src/training/trainer.py` |
| `nn_experiments/model.py` | `src/models/neural_network.py` |
| `nn_experiments/transductive_loss.py` | `src/losses/transductive_loss.py` |
| `nn_experiments/data_loader.py` | `src/data/data_loader.py` |
| `nn_experiments/constraints.py` | `src/training/constraints.py` |
| `nn_experiments/visualization.py` | `src/utils/visualization.py` |
| `nn_experiments/run_experiments.py` | `experiments/run_experiments.py` |
| `nn_experiments/test_*.py` | `tests/test_*.py` |

## What Was Removed?

✅ `example_usage.py` - Old demo (outdated)
✅ `loss.py` - Old loss implementation (replaced)
✅ `main.py` - Empty file
✅ `setup.py` - Not needed  
✅ `test_validation.py` - Outdated tests
✅ `config_template.yaml` - Unused

## Benefits of New Structure

1. **Clear Organization** - Everything has a logical place
2. **Professional** - Follows Python package best practices
3. **Scalable** - Easy to add new components
4. **Clean Imports** - Proper package structure
5. **Easy Navigation** - Find what you need quickly

## Quick Start with New Structure

```bash
# 1. Navigate to project root
cd OptimizationLoss

# 2. Install dependencies (if needed)
pip install -r requirements.txt

# 3. Run experiments
cd experiments
python run_experiments.py

# 4. Results appear in results/ folder
ls ../results/
```

## Documentation

- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - Full structure explanation
- **[README.md](README.md)** - Project overview and quick start
- **[docs/USAGE_GUIDE.md](docs/USAGE_GUIDE.md)** - Detailed usage
- **[docs/VISUALIZATION_GUIDE.md](docs/VISUALIZATION_GUIDE.md)** - Understanding plots

## Need Help?

The `nn_experiments/` folder still exists for reference, but **all new development uses the `src/` structure**. 

If you have scripts that import from `nn_experiments/`, update them to use the new imports from `src/`.
