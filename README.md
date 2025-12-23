# Transductive Saturation Loss for Student Dropout Prediction

Neural network training framework with transductive learning and adaptive constraint satisfaction for student dropout and academic success prediction.

## Features

✅ **Soft Predictions** - Uses probability sums for proper gradient flow  
✅ **Adaptive Lambda Weighting** - Automatically increases constraint pressure until satisfied  
✅ **Transductive Learning** - Uses unlabeled test data structure during training  
✅ **Rational Saturation Loss** - E/(E+K) formula for bounded constraint violations  
✅ **Comprehensive Visualizations** - Automatic plots for constraints, losses, and lambda evolution

## Project Structure

```
OptimizationLoss/
├── src/                          # Source code
│   ├── data/                     # Data loading and preprocessing
│   ├── models/                   # Neural network architectures  
│   ├── losses/                   # Loss functions
│   ├── training/                 # Training logic and constraints
│   └── utils/                    # Visualization utilities
├── config/                       # Configuration files
├── experiments/                  # Experiment runners
├── tests/                        # Test scripts
├── docs/                         # Documentation
├── results/                      # Training outputs
└── requirements.txt              # Dependencies
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run experiments
cd experiments
python run_experiments.py
```

## Documentation

- **[Usage Guide](docs/USAGE_GUIDE.md)** - Detailed instructions
- **[Visualization Guide](docs/VISUALIZATION_GUIDE.md)** - Understanding plots

## License

See LICENSE file for details.
