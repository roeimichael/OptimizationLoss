# Transductive Rational Saturation Loss

A PyTorch implementation of a custom loss function for constrained portfolio optimization with rational saturation penalties.

## Overview

This repository implements a novel loss function designed for financial portfolio optimization under structural constraints. The loss uses a **Rational Saturation** approach to ensure that both global target constraints and granular feature constraints are mapped to a strictly normalized range [0, 1), preventing gradient dominance issues during training.

## Mathematical Formulation

The total loss function is defined as:

```
L_total = BCE(y, ŷ) + λ₁ · L_target + λ₂ · L_sector
```

Where:
- **BCE**: Binary Cross-Entropy for prediction accuracy
- **L_target**: Global portfolio size constraint
- **L_sector**: Per-sector concentration constraint

### Constraint Terms

**Target Loss (Rational Saturation):**
```
E_target = ReLU(Σŷᵢ - K_target)
L_target = E_target / (E_target + K_target)
```

**Sector Loss (Rational Saturation):**
```
E_sector^(j) = ReLU((S·ŷ)_j - K_feat)
L_sector = (1/M) · Σ[E_sector^(j) / (E_sector^(j) + K_feat)]
```

### Parameters

- **N**: Total number of stocks (universe size)
- **M**: Number of unique market sectors
- **ŷ**: Prediction vector (N × 1)
- **K_target**: Maximum desired number of stocks in portfolio
- **K_feat**: Maximum stocks allowed per sector
- **S**: Binary sector matrix (M × N) mapping stocks to sectors
- **λ₁, λ₂**: Penalty weights for constraint enforcement

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+

### Setup

```bash
# Clone the repository
git clone https://github.com/roeimichael/OptimizationLoss.git
cd OptimizationLoss

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
import torch
from loss import TransductivePortfolioLoss

# Define parameters
N = 100  # Universe size
M = 5    # Number of sectors
K_TARGET = 20  # Max portfolio size
K_FEAT = 5     # Max stocks per sector

# Create sector matrix (M × N)
sector_matrix = torch.zeros(M, N)
stocks_per_sector = N // M
for i in range(M):
    start_idx = i * stocks_per_sector
    end_idx = start_idx + stocks_per_sector if i < M - 1 else N
    sector_matrix[i, start_idx:end_idx] = 1.0

# Initialize loss function
loss_fn = TransductivePortfolioLoss(
    k_target=K_TARGET,
    k_feat=K_FEAT,
    sector_matrix=sector_matrix,
    lambda_target=1.0,
    lambda_sector=1.0,
    use_bce=True,
    average_sectors=True
)

# Forward pass
y_hat = torch.sigmoid(torch.randn(N, 1, requires_grad=True))
y_true = torch.randint(0, 2, (N, 1)).float()

loss = loss_fn(y_hat, y_true)
loss.backward()
```

### Configuration Options

**Constructor Parameters:**
- `k_target` (int): Maximum portfolio size constraint
- `k_feat` (int): Maximum stocks per sector constraint
- `sector_matrix` (Tensor): Binary mapping matrix (M × N)
- `lambda_target` (float): Weight for global constraint penalty (default: 1.0)
- `lambda_sector` (float): Weight for sector constraint penalty (default: 1.0)
- `use_bce` (bool): Enable BCE prediction loss (default: False)
- `average_sectors` (bool): Use 1/M averaging for sector loss (default: True)

### Running Validation Suite

```bash
# Using make
make test

# Or directly
python test_validation.py
```

Expected output:
```
============================================================
TRANSDUCTIVE RATIONAL SATURATION LOSS - VALIDATION SUITE
============================================================
Test 1: PASSED - Loss: 0.000000
Test 2: PASSED - Loss: 0.XXXXXX
Test 3: PASSED - Loss: 0.XXXXXX
Test 4: PASSED - Loss: 0.XXXXXX (Max: 2.000000)
Test 5: PASSED - Initial Loss: X.XXXXXX, Final Loss: X.XXXXXX
============================================================
ADDITIONAL TEST - Full Loss with BCE
============================================================
Full Loss (BCE + Constraints): X.XXXXXX
Constraint Loss Only: X.XXXXXX
Test 6: PASSED - BCE integration validated
============================================================
ALL TESTS PASSED
============================================================
```

## Validation Test Cases

The implementation includes 6 comprehensive test cases:

1. **Compliant Portfolio**: Verifies loss = 0 when all constraints are satisfied
2. **Global Violation**: Tests penalty when total portfolio size exceeds K_target
3. **Sector Imbalance**: Tests penalty when single sector exceeds K_feat
4. **Saturation Stress Test**: Verifies loss bounds when all stocks are selected
5. **Gradient Flow**: Confirms loss decreases under SGD optimization
6. **BCE Integration**: Validates full loss with prediction component

## Integration with Training Loop

```python
import torch.nn as nn
import torch.optim as optim

# Your model
model = YourPortfolioModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Loss function
loss_fn = TransductivePortfolioLoss(
    k_target=20,
    k_feat=5,
    sector_matrix=sector_matrix,
    lambda_target=1.0,
    lambda_sector=1.0,
    use_bce=True
)

# Training loop
for epoch in range(num_epochs):
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()

        predictions = model(batch_x)
        loss = loss_fn(predictions, batch_y)

        loss.backward()
        optimizer.step()
```

## Dynamic Penalty Adjustment

For advanced usage, implement adaptive penalty scaling as described in Algorithm 1:

```python
lambda_target = 0.0
lambda_sector = 0.0
delta_step = 0.1

for epoch in range(num_epochs):
    # Train on historical data
    for batch_x, batch_y in train_loader:
        # ... training code ...
        pass

    # Evaluate constraints on test universe
    with torch.no_grad():
        y_test = model(X_test)
        constraint_loss = loss_fn(y_test)

    # Adjust penalties
    if constraint_loss > 0:
        lambda_target += delta_step
        lambda_sector += delta_step

    # Update loss function with new penalties
    loss_fn.lambda_target = lambda_target
    loss_fn.lambda_sector = lambda_sector
```

## Features

- **Gradient-Safe**: Rational saturation ensures bounded gradients
- **Vectorized**: No loops, fully optimized with torch.matmul
- **Flexible**: Toggle BCE and averaging independently
- **Tested**: Comprehensive edge case validation
- **Production-Ready**: Stable epsilon handling for numerical safety

## Project Structure

```
OptimizationLoss/
├── loss.py                           # Core loss function module
├── test_validation.py                # Validation test suite
├── example_usage.py                  # Complete training example
├── requirements.txt                  # Python dependencies
├── setup.py                          # Package installation
├── config_template.yaml              # Configuration template
├── Makefile                          # Development commands
├── README.md                         # This file
├── LICENSE                           # MIT License
└── .gitignore                        # Git exclusions
```

## Requirements

See `requirements.txt` for full dependency list:
- torch>=2.0.0
- tqdm>=4.60.0

## Citation

If you use this loss function in your research, please cite:

```bibtex
@article{transductive_saturation_loss,
  title={Constrained Portfolio Optimization: Loss Formulation with Rational Saturation},
  author={Your Name},
  year={2025}
}
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## References

- Rational Saturation Formula: Prevents unbounded growth via E/(E+K) normalization
- Split-Stream Training: Separate accuracy and constraint optimization passes
- Dynamic Penalty Adjustment: Adaptive λ scaling based on constraint violations

## Contact

For questions or issues, please open an issue on GitHub.
