# Neural Network Model Improvements

## Current Performance Ceiling

Based on comprehensive analysis:
- Best result: 74.66% accuracy (lambda_high @ constraint 0.9, 0.8)
- Performance plateaus around 74-75%
- Current model is basic feedforward with BatchNorm + ReLU + Dropout

## Proposed Improvements

### 1. Residual Connections (Skip Connections)

**What**: Direct connections that skip layers (like ResNet)
**Why**:
- Helps gradient flow in deeper networks
- Prevents degradation as networks get deeper
- Allows network to learn identity mapping

**Expected Impact**: +1-3% accuracy for deeper architectures

### 2. Better Normalization - LayerNorm

**What**: Normalizes across features instead of batch dimension
**Why**:
- More stable for small batch sizes
- Better for sequential/attention models
- Less sensitive to batch statistics

**Expected Impact**: +0.5-1.5% accuracy, more stable training

### 3. Advanced Activation Functions - GELU

**What**: Gaussian Error Linear Unit (used in BERT, GPT)
**Why**:
- Smoother gradient flow
- Better for deeper networks
- Empirically better than ReLU for many tasks

**Expected Impact**: +0.5-2% accuracy

**Alternatives**: SiLU (Swish), Mish

### 4. Self-Attention Mechanism

**What**: Allows model to focus on important features
**Why**:
- Captures feature interactions
- Learns which features are important for each sample
- Can help with course-specific patterns

**Expected Impact**: +1-3% accuracy (if patterns exist)

### 5. Better Weight Initialization

**What**: Xavier/Glorot initialization
**Why**:
- Prevents vanishing/exploding gradients
- Faster convergence
- More stable training

**Expected Impact**: Faster training, more consistent results

### 6. Mixup Data Augmentation

**What**: Mix training samples with weighted combinations
**Why**:
- Regularization technique
- Reduces overfitting
- Smoother decision boundaries

**Expected Impact**: +0.5-2% accuracy, better generalization

### 7. Ensemble Methods

**What**: Combine predictions from multiple models
**Why**:
- Reduces variance
- Captures different patterns
- More robust predictions

**Expected Impact**: +2-4% accuracy (proven technique)

### 8. Uncertainty Estimation

**What**: Use dropout at test time to estimate prediction confidence
**Why**:
- Know when model is uncertain
- Can help with constraint satisfaction
- Better for production deployment

**Expected Impact**: Better constraint adherence, more reliable predictions

## Usage Examples

### Basic Enhanced Model (Recommended Starting Point)

```python
from src.models.neural_network_enhanced import NeuralNetClassifierEnhanced

model = NeuralNetClassifierEnhanced(
    input_dim=X_train.shape[1],
    hidden_dims=[256, 256, 128, 64],
    n_classes=3,
    dropout=0.3,
    use_residual=True,      # Enable residual connections
    use_attention=False,     # Start without attention
    activation='gelu'        # Better than ReLU
)
```

### With Attention (For More Complex Patterns)

```python
model = NeuralNetClassifierEnhanced(
    input_dim=X_train.shape[1],
    hidden_dims=[256, 256, 128],
    n_classes=3,
    dropout=0.3,
    use_residual=True,
    use_attention=True,      # Enable self-attention
    activation='gelu'
)
```

### With Mixup (Better Generalization)

```python
from src.models.neural_network_enhanced import NeuralNetClassifierMixup

model = NeuralNetClassifierMixup(
    input_dim=X_train.shape[1],
    hidden_dims=[256, 256, 128, 64],
    n_classes=3,
    dropout=0.3,
    use_residual=True,
    activation='gelu',
    mixup_alpha=0.2          # Mixup strength
)

# In training loop:
mixed_x, y_a, y_b, lam = model.mixup_data(X_batch, y_batch)
outputs = model(mixed_x)
loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
```

### Ensemble (Best Performance)

```python
from src.models.neural_network_enhanced import EnsembleClassifier

models = [
    NeuralNetClassifierEnhanced(input_dim, [256, 128, 64], activation='gelu'),
    NeuralNetClassifierEnhanced(input_dim, [512, 256, 128], activation='silu'),
    NeuralNetClassifierEnhanced(input_dim, [256, 256, 128], use_attention=True)
]

# Train each model separately
for model in models:
    train(model, ...)

# Use ensemble for prediction
ensemble = EnsembleClassifier(models)
predictions = ensemble(X_test)
```

### With Uncertainty Estimation

```python
from src.models.neural_network_enhanced import NeuralNetClassifierWithUncertainty

model = NeuralNetClassifierWithUncertainty(
    input_dim=X_train.shape[1],
    hidden_dims=[256, 256, 128, 64],
    n_classes=3,
    dropout=0.3,
    use_residual=True,
    num_samples=10           # Monte Carlo samples for uncertainty
)

# Get predictions with uncertainty
predictions, uncertainty = model(X_test, return_uncertainty=True)

# Use uncertainty for constraint satisfaction
# Adjust predictions for high-uncertainty samples
```

## Additional Training Improvements

### 1. Learning Rate Scheduling

```python
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

# Cosine annealing
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

# Or One Cycle (often better)
scheduler = OneCycleLR(
    optimizer,
    max_lr=0.01,
    epochs=epochs,
    steps_per_epoch=len(train_loader)
)
```

### 2. Gradient Clipping

```python
# In training loop, after loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

### 3. Label Smoothing

```python
# Instead of CrossEntropyLoss
class LabelSmoothingLoss(nn.Module):
    def __init__(self, n_classes, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.n_classes = n_classes

    def forward(self, pred, target):
        log_prob = F.log_softmax(pred, dim=-1)
        smooth_target = torch.zeros_like(log_prob)
        smooth_target.fill_(self.smoothing / (self.n_classes - 1))
        smooth_target.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        loss = (-smooth_target * log_prob).sum(dim=-1).mean()
        return loss
```

### 4. Better Optimizer

```python
# AdamW with weight decay
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01,
    betas=(0.9, 0.999)
)
```

## Recommended Testing Strategy

### Phase 1: Basic Improvements (Quick Wins)
1. Test GELU activation vs ReLU
2. Add residual connections
3. Try LayerNorm instead of BatchNorm
4. Test Xavier initialization

**Expected**: +2-4% accuracy improvement

### Phase 2: Architecture Search
1. Test different hidden dimensions with residuals
2. Try attention mechanism
3. Test different dropout rates with new architecture

**Expected**: +1-3% additional improvement

### Phase 3: Training Improvements
1. Add learning rate scheduling
2. Add gradient clipping
3. Try label smoothing
4. Test different optimizers

**Expected**: +1-2% additional improvement

### Phase 4: Advanced Techniques
1. Try Mixup augmentation
2. Build 3-5 model ensemble
3. Add uncertainty estimation

**Expected**: +2-4% additional improvement

## Total Expected Improvement

Conservative estimate: **+5-10% accuracy**
Optimistic estimate: **+8-15% accuracy**

This could push your best result from ~75% to **80-90%** with the right combination.

## Configuration Recommendations

Based on your current results showing arch_deep performs best:

```python
# Recommended configuration for best performance
config = {
    'hidden_dims': [512, 512, 256, 256, 128, 64],
    'dropout': 0.3,
    'use_residual': True,
    'use_attention': True,
    'activation': 'gelu',
    'learning_rate': 0.001,
    'optimizer': 'adamw',
    'weight_decay': 0.01,
    'scheduler': 'cosine',
    'gradient_clip': 1.0,
    'label_smoothing': 0.1
}
```

## Next Steps

1. Start with basic enhanced model (residual + GELU)
2. Run experiments on your constraint settings
3. Compare against current best (74.66%)
4. If improvement seen, add attention
5. Then try ensemble of best 3-5 configurations
6. Finally, add training improvements

Would you like me to create a training script that incorporates these improvements?
