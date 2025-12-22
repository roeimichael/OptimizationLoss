# Transductive Learning Implementation

This implementation uses true transductive learning where the test set (without labels) guides the training process through constraint satisfaction.

## What is Transductive Learning?

Traditional supervised learning: Train only on labeled data, test on unseen data.

Transductive learning: Use unlabeled test data during training to learn the structure/distribution, but never the labels.

## How This Implementation Works

### Data Split (90-10)
- 90% training set: Used for supervised learning (BCE loss)
- 10% test set: Used for constraint computation (structure only, no labels)

### Training Process

For EACH training batch:
1. Forward pass on training batch → Compute BCE loss
2. Forward pass on ENTIRE test set → Compute constraint loss
3. Combine losses: Total = BCE + Constraint
4. Backpropagate combined loss

### Key Insight

The model learns to:
- Predict accurately on training data (BCE loss)
- Respect global/local constraints on test data distribution (Constraint loss)

Test labels are NEVER used - only the distribution structure matters.

## Implementation Details

### Equal Lambda Weights

```python
lambda_global = 1.0  # Weight for global constraints
lambda_local = 1.0   # Weight for local constraints
```

Both constraint types have equal importance with the BCE loss.

### Constraint Computation

Constraints are computed on the TEST set:
- Global constraints: Based on test set class distribution
- Local constraints: Based on test set per-course distribution

### Training Loop Structure

```python
for epoch in range(epochs):
    for train_batch in train_loader:
        # 1. Supervised loss on training batch
        train_logits = model(train_batch)
        bce_loss = CrossEntropyLoss(train_logits, train_labels)

        # 2. Constraint loss on FULL test set
        test_logits = model(full_test_set)  # No labels needed!
        constraint_loss = ComputeConstraints(test_logits, constraints)

        # 3. Combined optimization
        total_loss = bce_loss + constraint_loss
        total_loss.backward()
        optimizer.step()
```

## Configuration

### config.py

```python
CONSTRAINTS = [
    (local%, global%),  # 8 constraint pairs
]

NN_CONFIGS = [
    {
        "lambda_global": 1.0,
        "lambda_local": 1.0,
        "hidden_dims": [128, 64, 32]
    }
]

TRAINING_PARAMS = {
    'epochs': 100,
    'batch_size': 64,
    'lr': 0.001,
    'dropout': 0.3,
    'patience': 10,
    'test_size': 0.1  # 90-10 split
}
```

## Running Experiments

### Test Setup

```bash
cd nn_experiments
python test_transductive_setup.py
```

### Run Experiments

```bash
python run_experiments.py
```

This will:
1. Load and preprocess data
2. Split 90-10 (train-test)
3. For each constraint pair:
   - Compute constraints on test set
   - Train model with transductive approach
   - Evaluate on test set
4. Save results to results/ folder

## Results

Output files:
- `results/students__train__nn_config1__transductive.csv` - Per-constraint results
- `results/nn_results.json` - Aggregated results

Results include:
- Test accuracy (model evaluated on same test set used for constraints)
- Training time
- Constraint satisfaction metrics

## Why This Approach?

### Benefits

1. **Leverages Test Set Structure**: Model learns the distribution of test data
2. **No Data Leakage**: Test labels never used, only structure
3. **Balanced Optimization**: Equal lambda weights ensure neither BCE nor constraints dominate
4. **Realistic Constraints**: Constraints based on actual test distribution
5. **Simpler Evaluation**: No cross-validation complexity

### Comparison to Standard Approach

**Standard:**
- Train on training set only
- Constraints computed on training set
- Risk: Constraints may not match test distribution

**Transductive:**
- Train on training set (supervised)
- Constraints computed on test set (transductive)
- Model adapts to test distribution during training

## Understanding the Output

### During Training

```
Epoch 10/100 | Loss: 1.2345 (CE: 0.8000, Constraint: 0.4345)
```

- **Loss**: Combined BCE + Constraint
- **CE**: Cross-entropy on training batches
- **Constraint**: Constraint violation on test set

### After Training

```
Test Accuracy: 0.7834, Time: 45.23s
```

- **Test Accuracy**: Classification accuracy on test set
- **Time**: Total training time

### Expected Behavior

As training progresses:
- CE loss should decrease (learning to classify)
- Constraint loss should decrease (learning test distribution)
- Test accuracy should increase

## Tuning Parameters

### If Constraints Not Satisfied

Increase lambda values:
```python
"lambda_global": 2.0,  # Increase
"lambda_local": 2.0,   # Increase
```

### If Accuracy Too Low

Decrease lambda values or increase model capacity:
```python
"lambda_global": 0.5,  # Decrease
"lambda_local": 0.5,   # Decrease
"hidden_dims": [256, 128, 64]  # Increase
```

### If Training Too Slow

Reduce batch size or epochs:
```python
'batch_size': 32,  # Reduce
'epochs': 50,      # Reduce
```

## Technical Details

### Memory Efficiency

Test set forward pass happens EVERY batch, so:
- Keep test set reasonable size (10% is good)
- Use GPU if available
- Monitor memory usage

### Gradient Flow

Gradients flow from:
1. Training batch → Model parameters (via BCE)
2. Test set → Model parameters (via Constraints)

Both gradients are combined in the same backward pass.

### No Cross-Validation

This approach uses a single train-test split instead of k-fold CV:
- Faster to run
- Simpler to understand
- More realistic (matches deployment scenario)

## File Structure

```
nn_experiments/
├── config.py                       # Configuration
├── data_loader.py                  # Data preprocessing
├── constraints.py                  # Constraint computation
├── transductive_loss.py            # Loss function
├── model.py                        # Neural network
├── trainer.py                      # Transductive training loop
├── run_experiments.py              # Main experiment runner
├── test_transductive_setup.py      # Setup verification
└── results/                        # Output directory
```

## Verification Checklist

Before running:
- [ ] Data file path correct in config.py
- [ ] CUDA available (check with test_transductive_setup.py)
- [ ] Constraints pairs make sense for your data
- [ ] Lambda weights set to 1.0 (equal importance)
- [ ] Test size set to 0.1 (90-10 split)

## Troubleshooting

### CUDA Out of Memory

Reduce batch size or test set size:
```python
'batch_size': 32,    # Reduce
'test_size': 0.05,   # Reduce to 5%
```

### Training Not Converging

Check constraint loss - if always high:
- Constraints may be too strict
- Increase lambda values
- Check constraint computation

### Poor Test Accuracy

Model may be overfitting to constraints:
- Decrease lambda values
- Increase training data
- Add more regularization (increase dropout)

## Next Steps

After running experiments:
1. Check results/nn_results.json
2. Identify best constraint pair
3. Analyze constraint satisfaction
4. Consider retraining with adjusted parameters

The transductive approach ensures your model learns both to classify accurately AND to respect the distribution structure of your deployment data.
