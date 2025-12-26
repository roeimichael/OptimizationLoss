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
├── data/                         # Dataset directory (place CSV here)
├── experiments/                  # Experiment runners
├── tests/                        # Test scripts
├── docs/                         # Documentation
├── results/                      # Training outputs
└── requirements.txt              # Dependencies
```

## Quick Start

```bash
# 1. Install PyTorch with CUDA support (for GPU acceleration)
# Option A: Use the installation script (recommended)
bash install_pytorch_cuda.sh

# Option B: Manual installation
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# For CPU only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 2. Install other dependencies
pip install -r requirements.txt

# 3. Place dataset in data/ directory
# Copy your dataset CSV file to: data/dataset.csv
# See data/README.md for details

# 4. Run experiments
cd experiments
python run_experiments.py
```

## Documentation

- **[Usage Guide](docs/USAGE_GUIDE.md)** - Detailed instructions
- **[Visualization Guide](docs/VISUALIZATION_GUIDE.md)** - Understanding plots

## License

See LICENSE file for details.
