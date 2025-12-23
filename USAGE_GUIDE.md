# Quick Usage Guide

## New Project Structure

The codebase has been refactored for better modularity:

```
loss.py                 → Core loss function (import this!)
test_validation.py      → Validation test suite
example_usage.py        → Training example
```

## How to Use the Loss Function

### 1. Import the Loss Function

```python
from loss import TransductivePortfolioLoss
```

### 2. Create Your Model

```python
import torch
import torch.nn as nn

class MyPortfolioModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)
```

### 3. Set Up the Loss Function

```python
# Create sector matrix
N = 100  # Number of stocks
M = 5    # Number of sectors
sector_matrix = torch.zeros(M, N)

# Fill sector matrix (example: evenly distributed)
stocks_per_sector = N // M
for i in range(M):
    start = i * stocks_per_sector
    end = start + stocks_per_sector if i < M - 1 else N
    sector_matrix[i, start:end] = 1.0

# Initialize loss
loss_fn = TransductivePortfolioLoss(
    k_target=20,           # Max 20 stocks in portfolio
    k_feat=5,              # Max 5 stocks per sector
    sector_matrix=sector_matrix,
    lambda_target=1.0,     # Global constraint weight
    lambda_sector=1.0,     # Sector constraint weight
    use_bce=True,          # Include prediction loss
    average_sectors=True   # Average sector violations
)
```

### 4. Training Loop

```python
import torch.optim as optim

model = MyPortfolioModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()

        predictions = model(batch_x)
        loss = loss_fn(predictions, batch_y)

        loss.backward()
        optimizer.step()
```

## Running Tests

### Validation Suite

```bash
# Using make
make test

# Or directly
python test_validation.py
```

### Example Training

```bash
# Using make
make example

# Or directly
python example_usage.py
```

## Configuration Options

### Loss Function Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k_target` | int | Required | Maximum portfolio size |
| `k_feat` | int | Required | Maximum stocks per sector |
| `sector_matrix` | Tensor | Required | Binary matrix (M × N) |
| `lambda_target` | float | 1.0 | Global constraint weight |
| `lambda_sector` | float | 1.0 | Sector constraint weight |
| `use_bce` | bool | False | Include BCE prediction loss |
| `average_sectors` | bool | True | Average sector violations (1/M factor) |

### Forward Pass

```python
# Constraint-only mode (use_bce=False)
loss = loss_fn(predictions)

# Full mode with BCE (use_bce=True)
loss = loss_fn(predictions, ground_truth)
```

## Common Use Cases

### 1. Training with Both Accuracy and Constraints

```python
loss_fn = TransductivePortfolioLoss(
    k_target=20,
    k_feat=5,
    sector_matrix=sector_matrix,
    use_bce=True  # ← Enable BCE
)

# In training loop
loss = loss_fn(predictions, targets)
```

### 2. Constraint Validation Only

```python
loss_fn = TransductivePortfolioLoss(
    k_target=20,
    k_feat=5,
    sector_matrix=sector_matrix,
    use_bce=False  # ← Disable BCE
)

# Check constraint violations
with torch.no_grad():
    constraint_loss = loss_fn(predictions)
    if constraint_loss < 0.01:
        print("Constraints satisfied!")
```

### 3. Dynamic Penalty Adjustment

```python
# Start with low penalties
loss_fn = TransductivePortfolioLoss(
    k_target=20,
    k_feat=5,
    sector_matrix=sector_matrix,
    lambda_target=0.1,
    lambda_sector=0.1
)

# During training
for epoch in range(num_epochs):
    # ... train ...

    # Check constraints on test set
    with torch.no_grad():
        test_pred = model(X_test)
        constraint_violation = loss_fn(test_pred)

    # Increase penalties if violated
    if constraint_violation > 0:
        loss_fn.lambda_target += 0.1
        loss_fn.lambda_sector += 0.1
```

## File Descriptions

### `loss.py` (1.4 KB)
- **Purpose**: Core loss function module
- **Contains**: `TransductivePortfolioLoss` class
- **Use**: Import this in your projects
- **Standalone**: No dependencies on other project files

### `test_validation.py` (4.1 KB)
- **Purpose**: Comprehensive validation suite
- **Tests**: 6 test cases covering edge conditions
- **Use**: Run to verify loss function correctness
- **Imports**: `loss.py`

### `example_usage.py` (3.5 KB)
- **Purpose**: Complete training example
- **Includes**: Model definition, training loop, validation
- **Use**: Reference for integration
- **Imports**: `loss.py`

### `transductive_saturation_loss.py` (DEPRECATED)
- **Status**: Backward compatibility wrapper
- **Use**: Redirects to new modules
- **Warning**: Shows deprecation notice

## Migration Guide

If you were using the old structure:

**Before:**
```python
from transductive_saturation_loss import TransductivePortfolioLoss
```

**After:**
```python
from loss import TransductivePortfolioLoss
```

That's it! The API is identical.

## Need Help?

- Run `make test` to verify your installation
- Run `make example` to see a complete training example
- Check `README.md` for detailed documentation
- Review `config_template.yaml` for configuration options
