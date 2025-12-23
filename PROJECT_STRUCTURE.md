# Project Structure

```
OptimizationLoss/
│
├── src/                                    # Source code package
│   ├── __init__.py
│   │
│   ├── data/                              # Data handling
│   │   ├── __init__.py
│   │   ├── data_loader.py                 # Load and preprocess CSV data
│   │   └── dataset.py                     # PyTorch Dataset wrapper
│   │
│   ├── models/                            # Neural network models
│   │   ├── __init__.py
│   │   └── neural_network.py              # NeuralNetClassifier (3-class)
│   │
│   ├── losses/                            # Loss functions
│   │   ├── __init__.py
│   │   └── transductive_loss.py           # MulticlassTransductiveLoss
│   │
│   ├── training/                          # Training utilities
│   │   ├── __init__.py
│   │   ├── trainer.py                     # Main training loop with adaptive lambdas
│   │   └── constraints.py                 # Compute global/local constraints
│   │
│   └── utils/                             # Utilities
│       ├── __init__.py
│       └── visualization.py               # Plot generation
│
├── config/                                 # Configuration
│   └── experiment_config.py               # All experiment parameters
│
├── experiments/                            # Experiment runners
│   └── run_experiments.py                 # Main experiment script
│
├── tests/                                  # Test scripts
│   └── test_hard_predictions.py           # Verify prediction counting
│
├── docs/                                   # Documentation
│   ├── README.md                          # Old README (archived)
│   ├── USAGE_GUIDE.md                     # Usage instructions
│   └── VISUALIZATION_GUIDE.md             # Plot interpretation
│
├── results/                                # Generated outputs (gitignored)
│   ├── global_constraints.png
│   ├── local_constraints.png
│   ├── losses.png
│   └── lambda_evolution.png
│
├── README.md                               # Main project README
├── requirements.txt                        # Python dependencies
├── .gitignore                             # Git ignore rules
├── LICENSE                                # License file
└── PROJECT_STRUCTURE.md                   # This file
```

## Module Responsibilities

### src/data/
- **data_loader.py**: Load CSV, handle preprocessing, train-test split
- **dataset.py**: PyTorch Dataset wrapper (if needed)

### src/models/
- **neural_network.py**: Neural network architecture (3-layer MLP)

### src/losses/
- **transductive_loss.py**: Transductive loss with rational saturation formula

### src/training/
- **trainer.py**: Training loop, adaptive lambda adjustment, early stopping
- **constraints.py**: Compute constraint values from data distribution

### src/utils/
- **visualization.py**: Generate plots for constraints, losses, lambda evolution

### config/
- **experiment_config.py**: All hyperparameters, constraints, paths

### experiments/
- **run_experiments.py**: Main entry point to run all experiments

### tests/
- **test_hard_predictions.py**: Unit tests for loss function

## Import Examples

```python
# From experiment runner
from config.experiment_config import *
from src.data import load_and_preprocess_data, split_data
from src.training import train_model_transductive, compute_global_constraints
from src.models import NeuralNetClassifier
from src.losses import MulticlassTransductiveLoss
from src.utils import create_all_visualizations
```

## Running Experiments

```bash
cd experiments
python run_experiments.py
```

## Clean Structure Benefits

✅ Clear separation of concerns  
✅ Easy to find components  
✅ Scalable for future additions  
✅ Professional package structure  
✅ Easy imports from anywhere  
