# Process Flow Documentation

This document provides a comprehensive, step-by-step explanation of the entire execution flow when running `experiments/run_experiments.py`.

## Table of Contents

1. [Overview](#overview)
2. [Execution Flow Diagram](#execution-flow-diagram)
3. [Phase-by-Phase Breakdown](#phase-by-phase-breakdown)
4. [Function Reference](#function-reference)
5. [Data Flow](#data-flow)

---

## Overview

The experiment script orchestrates the complete machine learning pipeline:
- Data loading
- Constraint computation
- Model training with transductive constraints
- Prediction and evaluation
- Results saving and visualization

**Total execution time**: ~1-10 minutes per experiment (depends on epochs until convergence)

---

## Execution Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    run_experiments.py::main()                   │
└─────────────────────────────────────────────────────────────────┘
                                 │
                ┌────────────────┴────────────────┐
                │                                  │
                ▼                                  ▼
┌───────────────────────────┐      ┌──────────────────────────────┐
│  PHASE 1: DATA LOADING    │      │  PHASE 2: CONSTRAINT SETUP   │
│  load_presplit_data()     │──────▶  compute_global_constraints()│
│  Returns: X_train, X_test │      │  compute_local_constraints() │
│  y_train, y_test, dfs     │      │  Returns: constraint values  │
└───────────────────────────┘      └──────────────────────────────┘
                                                  │
                                                  ▼
                                    ┌──────────────────────────────┐
                                    │  PHASE 3: TRAINING           │
                                    │  train_model_transductive()  │
                                    │  ┌──────────────────────┐    │
                                    │  │ prepare_training_data│    │
                                    │  │ initialize_model     │    │
                                    │  │ Training Loop:       │    │
                                    │  │   train_single_epoch │    │
                                    │  │   update_lambda      │    │
                                    │  │   log_progress       │    │
                                    │  │   visualizations     │    │
                                    │  └──────────────────────┘    │
                                    │  Returns: model, scaler      │
                                    └──────────────────────────────┘
                                                  │
                                                  ▼
                                    ┌──────────────────────────────┐
                                    │  PHASE 4: EVALUATION         │
                                    │  predict()                   │
                                    │  evaluate_accuracy()         │
                                    │  save_results()              │
                                    └──────────────────────────────┘
```

---

## Phase-by-Phase Breakdown

### PHASE 0: Initialization

**Location**: `run_experiments.py::main()` (lines 33-52)

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

**Functions Called**: None (setup only)

**What Happens**:
1. Detect GPU availability (CUDA)
2. Load configuration from `config/experiment_config.py`:
   - `TRAIN_PATH`, `TEST_PATH`
   - `TARGET_COLUMN`
   - `CONSTRAINTS` list
   - `NN_CONFIGS` list
   - `TRAINING_PARAMS` dict
3. Create results directory

**Data State**: Empty, just configuration loaded

---

### PHASE 1: Data Loading

**Location**: `run_experiments.py::main()` (lines 38-50)

#### 1.1 Load Pre-Split Data

**Function**: `load_presplit_data()`
**Module**: `src/data/data_loader.py` (lines 7-30)
**Called From**: `run_experiments.py` line 41

```python
X_train, X_test, y_train, y_test, train_df, test_df = load_presplit_data(
    TRAIN_PATH, TEST_PATH, TARGET_COLUMN
)
```

**Internal Steps**:
```python
def load_presplit_data(train_path, test_path, target_column):
    # 1. Load CSV files
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # 2. Separate features from target
    y_train = train_df[target_column]      # Labels
    X_train = train_df.drop([target_column], axis=1)  # Features

    y_test = test_df[target_column]
    X_test = test_df.drop([target_column], axis=1)

    return X_train, X_test, y_train, y_test, train_df, test_df
```

**Returns**:
- `X_train`: Training features (DataFrame) - e.g., 400 rows × 35 columns
- `X_test`: Test features (DataFrame) - e.g., 442 rows × 35 columns
- `y_train`: Training labels (Series) - e.g., 400 values (0=Dropout, 1=Enrolled, 2=Graduate)
- `y_test`: Test labels (Series) - e.g., 442 values
- `train_df`: Full training dataframe (includes target + Course column)
- `test_df`: Full test dataframe (includes target + Course column)

**Data State After**: Raw data loaded, not scaled/preprocessed yet

#### 1.2 Combine for Constraints

**Location**: `run_experiments.py` lines 46-50

```python
full_df = pd.concat([train_df, test_df], ignore_index=True)
groups = full_df['Course'].unique()
```

**What Happens**:
- Combine train + test to compute realistic constraints
- Extract unique course IDs (e.g., [2, 3, 4, 5, ...])

**Data State After**: Full dataset ready for constraint computation

---

### PHASE 2: Constraint Computation

**Location**: `run_experiments.py::main()` (lines 65-74, inside experiment loop)

#### 2.1 Compute Global Constraints

**Function**: `compute_global_constraints()`
**Module**: `src/training/constraints.py` (lines 4-23)
**Called From**: `run_experiments.py` line 70

```python
global_constraint = compute_global_constraints(full_df, TARGET_COLUMN, global_percent)
```

**Internal Steps**:
```python
def compute_global_constraints(data, target_column, percentage):
    n_classes = 3
    constraint = np.zeros(n_classes)

    # 1. Count how many students in each class
    items = data[target_column].value_counts()
    # Example: Dropout=142, Enrolled=79, Graduate=221

    # 2. Compute constraint as percentage/10 of actual count
    for class_id in items.index:
        constraint[int(class_id)] = np.round(items[class_id] * percentage / 10)
    # If percentage=0.8 (80%), Dropout constraint = 142 * 0.8 / 10 ≈ 11.4 → 11

    # 3. Graduate class unconstrained
    constraint[2] = 1e10  # Effectively infinite

    return constraint.tolist()
```

**Example Input**:
- `data`: 842 students (Dropout=142, Enrolled=79, Graduate=221)
- `percentage`: 0.8

**Example Output**:
```python
[11.0, 6.0, 1e10]  # [Dropout≤11, Enrolled≤6, Graduate=∞]
```

**Purpose**: Limit total predictions across all students

#### 2.2 Compute Local Constraints

**Function**: `compute_local_constraints()`
**Module**: `src/training/constraints.py` (lines 26-61)
**Called From**: `run_experiments.py` line 71

```python
local_constraint = compute_local_constraints(full_df, TARGET_COLUMN, local_percent, groups)
```

**Internal Steps**:
```python
def compute_local_constraints(data, target_column, percentage, groups):
    n_classes = 3
    local_constraint = {}

    for group in groups:
        # 1. Skip course 1 (data issues)
        if group == 1:
            continue

        # 2. Get data for this course
        data_group = data[data['Course'] == group]

        # 3. Skip empty courses
        if len(data_group) == 0:
            continue

        # 4. Count students per class in this course
        constraint = np.zeros(n_classes)
        items = data_group[target_column].value_counts()

        # 5. Compute constraint (same formula as global)
        for class_id in items.index:
            constraint[int(class_id)] = np.round(items[class_id] * percentage / 10)

        # 6. Graduate unconstrained
        constraint[2] = 1e10

        local_constraint[group] = constraint.tolist()

    return local_constraint
```

**Example Input**:
- `groups`: [2, 3, 4, 5, ...]
- Course 2 has: Dropout=8, Enrolled=5, Graduate=22
- `percentage`: 0.6

**Example Output**:
```python
{
    2: [0.0, 0.0, 1e10],  # Course 2: Dropout≤0, Enrolled≤0
    3: [1.0, 1.0, 1e10],  # Course 3: Dropout≤1, Enrolled≤1
    4: [2.0, 1.0, 1e10],  # Course 4: Dropout≤2, Enrolled≤1
    ...
}
```

**Purpose**: Limit predictions per course/group

**Data State After**: Constraints computed, ready for training

---

### PHASE 3: Model Training

**Location**: `run_experiments.py::main()` (lines 76-95)

#### 3.0 Data Preparation for Training

**Location**: `run_experiments.py` lines 76-78

```python
X_train_clean = X_train.drop("Course", axis=1)
X_test_clean = X_test.drop("Course", axis=1)
groups_test = X_test["Course"]
```

**What Happens**:
- Remove "Course" column from features (not used for prediction)
- Keep Course column separately for constraint grouping

#### 3.1 Main Training Function Call

**Function**: `train_model_transductive()`
**Module**: `src/training/trainer.py` (lines 204-344)
**Called From**: `run_experiments.py` line 81

```python
model, scaler, training_time, history = train_model_transductive(
    X_train_clean, y_train,
    X_test_clean, groups_test,
    global_constraint, local_constraint,
    lambda_global=0.01,
    lambda_local=0.01,
    hidden_dims=[128, 64, 32],
    epochs=1000,
    batch_size=32,
    lr=0.001,
    dropout=0.3,
    device=device,
    constraint_dropout_pct=0.8,
    constraint_enrolled_pct=0.6
)
```

**This function orchestrates the entire training process. Let's break it down:**

---

#### 3.2 Prepare Training Data

**Function**: `prepare_training_data()`
**Module**: `src/training/trainer.py` (lines 26-60)
**Called From**: `train_model_transductive()` line 236

```python
train_loader, X_test_tensor, group_ids_test, scaler = prepare_training_data(
    X_train, y_train, X_test, groups_test, batch_size, device
)
```

**Internal Steps**:
```python
def prepare_training_data(X_train, y_train, X_test, groups_test, batch_size, device):
    # 1. Standardize features (mean=0, std=1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Fit on train
    X_test_scaled = scaler.transform(X_test)        # Transform test

    # 2. Encode labels if needed (strings → integers)
    if y_train.dtype == 'O' or isinstance(y_train.iloc[0], str):
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
    else:
        y_train_encoded = y_train.values

    # 3. Create PyTorch dataset
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled),
        torch.LongTensor(y_train_encoded)
    )

    # 4. Create data loader for batching
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 5. Convert test data to tensors
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    group_ids_test = torch.LongTensor(groups_test.values).to(device)

    return train_loader, X_test_tensor, group_ids_test, scaler
```

**Returns**:
- `train_loader`: DataLoader yielding batches of (features, labels)
- `X_test_tensor`: Test features as GPU tensor (442 × 34)
- `group_ids_test`: Course IDs as GPU tensor (442,)
- `scaler`: Fitted StandardScaler (for later predictions)

**Data State After**: Data scaled, batched, on GPU, ready for training

---

#### 3.3 Initialize Model and Optimizer

**Function**: `initialize_model_and_optimizer()`
**Module**: `src/training/trainer.py` (lines 63-102)
**Called From**: `train_model_transductive()` line 241

```python
model, criterion_ce, criterion_constraint, optimizer = \
    initialize_model_and_optimizer(
        input_dim=34,
        hidden_dims=[128, 64, 32],
        dropout=0.3,
        lr=0.001,
        device=device,
        global_constraint=[142, 85, 1e10],
        local_constraint={2: [10, 6, 1e10], ...},
        lambda_global=0.01,
        lambda_local=0.01
    )
```

**Internal Steps**:

##### 3.3.1 Create Neural Network

**Class**: `NeuralNetClassifier`
**Module**: `src/models/neural_network.py` (lines 5-22)

```python
model = NeuralNetClassifier(
    input_dim=34,
    hidden_dims=[128, 64, 32],
    n_classes=3,
    dropout=0.3
).to(device)
```

**Network Architecture**:
```python
Sequential(
  # Layer 1
  Linear(34 → 128)
  BatchNorm1d(128)
  ReLU()
  Dropout(0.3)

  # Layer 2
  Linear(128 → 64)
  BatchNorm1d(64)
  ReLU()
  Dropout(0.3)

  # Layer 3
  Linear(64 → 32)
  BatchNorm1d(32)
  ReLU()
  Dropout(0.3)

  # Output Layer
  Linear(32 → 3)  # 3 classes: Dropout, Enrolled, Graduate
)
```

**Forward Pass**:
```python
def forward(x):
    return self.network(x)  # Returns logits (raw scores)
```

##### 3.3.2 Create Cross-Entropy Loss

```python
criterion_ce = nn.CrossEntropyLoss()
```

**Purpose**: Standard supervised classification loss for labeled training data

##### 3.3.3 Create Constraint Loss

**Class**: `MulticlassTransductiveLoss`
**Module**: `src/losses/transductive_loss.py` (lines 5-121)

```python
criterion_constraint = MulticlassTransductiveLoss(
    global_constraints=[142, 85, 1e10],
    local_constraints={2: [10, 6, 1e10], ...},
    lambda_global=0.01,
    lambda_local=0.01,
    use_ce=False
).to(device)
```

**Internal Initialization**:
```python
def __init__(...):
    # 1. Store lambda weights
    self.lambda_global = 0.01
    self.lambda_local = 0.01

    # 2. Register global constraints as buffer
    self.register_buffer('global_constraints',
                        torch.tensor([142, 85, 1e10]))

    # 3. Register local constraints (per course)
    for group_id, constraints in local_constraints.items():
        buffer_name = f'local_constraint_{group_id}'
        self.register_buffer(buffer_name, torch.tensor(constraints))
        self.local_constraint_dict[group_id] = buffer_name
    # Creates: local_constraint_2, local_constraint_3, ...

    # 4. Initialize satisfaction flags
    self.global_constraints_satisfied = False
    self.local_constraints_satisfied = False
```

##### 3.3.4 Create Optimizer

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

**Returns**:
- `model`: Neural network (on GPU)
- `criterion_ce`: Cross-entropy loss function
- `criterion_constraint`: Constraint loss function
- `optimizer`: Adam optimizer

**Data State After**: Model initialized, ready to train

---

#### 3.4 Training Loop

**Location**: `train_model_transductive()` lines 286-329

```python
for epoch in range(epochs):  # epochs = 1000
    # 3.4.1 Train single epoch
    avg_ce, avg_global, avg_local = train_single_epoch(...)

    # 3.4.2 Update lambda weights
    update_lambda_weights(avg_global, avg_local, criterion_constraint, epoch)

    # 3.4.3 Log and print progress (every 3 epochs)
    if (epoch + 1) % 3 == 0:
        compute_prediction_statistics(...)
        log_progress_to_csv(...)
        if epoch >= WARMUP_EPOCHS:
            print_progress_from_csv(...)

    # 3.4.4 Check early stopping
    if constraints_satisfied:
        break
```

Let's detail each sub-function:

---

##### 3.4.1 Train Single Epoch

**Function**: `train_single_epoch()`
**Module**: `src/training/trainer.py` (lines 140-201)
**Called From**: Training loop line 287

```python
avg_ce, avg_global, avg_local = train_single_epoch(
    model, train_loader, criterion_ce, criterion_constraint,
    optimizer, X_test_tensor, group_ids_test, device
)
```

**Internal Steps - Per Batch**:
```python
def train_single_epoch(...):
    model.train()
    epoch_loss_ce = 0.0

    for batch_features, batch_labels in train_loader:
        # Step 1: Move batch to GPU
        batch_features = batch_features.to(device)  # (32, 34)
        batch_labels = batch_labels.to(device)      # (32,)

        optimizer.zero_grad()

        # Step 2: Forward pass on LABELED training data
        train_logits = model(batch_features)  # (32, 3)
        loss_ce = criterion_ce(train_logits, batch_labels)

        # Step 3: Forward pass on UNLABELED test data (for constraints)
        model.eval()  # Turn off dropout for stable predictions
        test_logits = model(X_test_tensor)  # (442, 3)
        model.train()  # Turn back on

        # Step 4: Compute constraint losses
        _, _, loss_global, loss_local = criterion_constraint(
            test_logits, y_true=None, group_ids=group_ids_test
        )

        # Step 5: Combine losses
        loss = loss_ce + λ_global * loss_global + λ_local * loss_local

        # Step 6: Backpropagation
        loss.backward()
        optimizer.step()

        epoch_loss_ce += loss_ce.item()

    # Step 7: Compute final constraint losses for reporting
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_tensor)
        _, _, loss_global, loss_local = criterion_constraint(
            test_logits, y_true=None, group_ids=group_ids_test
        )
    model.train()

    return avg_ce, loss_global.item(), loss_local.item()
```

**Constraint Loss Computation Details**:

When `criterion_constraint()` is called, it executes:

```python
def forward(logits, y_true=None, group_ids=None):
    # 1. Convert logits to probabilities
    y_proba = F.softmax(logits, dim=1)  # (442, 3)

    # 2. Compute global constraint loss
    L_target = _compute_global_constraint_loss(y_proba, device)

    # 3. Compute local constraint loss
    L_feat = _compute_local_constraint_loss(y_proba, group_ids, device)

    # 4. Return all losses
    return L_total, L_pred, L_target, L_feat
```

**Global Constraint Loss**:
```python
def _compute_global_constraint_loss(y_proba, device):
    # 1. Get hard predictions for satisfaction checking
    y_hard = torch.argmax(y_proba, dim=1)  # (442,) - discrete classes

    L_target = 0.0

    for class_id in [0, 1, 2]:  # Dropout, Enrolled, Graduate
        K = global_constraints[class_id]  # e.g., 142 for Dropout

        if K > 1e9:  # Skip unconstrained classes
            continue

        # 2. Count hard predictions
        hard_count = (y_hard == class_id).sum()  # e.g., 150

        # 3. Check if constraint violated
        if hard_count > K:  # 150 > 142? YES
            # 4. Compute loss using SOFT predictions (differentiable)
            predicted_count = y_proba[:, class_id].sum()  # e.g., 148.5

            # 5. Rational saturation loss
            E = max(0, predicted_count - K)  # 148.5 - 142 = 6.5
            loss = E / (E + K + eps)  # 6.5 / (6.5 + 142) ≈ 0.044

            L_target += loss

    # 6. Average across violated constraints
    if num_violations > 0:
        L_target = L_target / num_violations

    # 7. Set satisfaction flag
    self.global_constraints_satisfied = (no violations)

    return L_target
```

**Local Constraint Loss** (similar logic per course):
```python
def _compute_local_constraint_loss(y_proba, group_ids, device):
    y_hard = torch.argmax(y_proba, dim=1)
    L_feat = 0.0

    for group_id in [2, 3, 4, ...]:  # Each course
        # 1. Filter data for this course
        in_group = (group_ids == group_id)
        group_proba = y_proba[in_group]  # e.g., (35, 3) for course 2
        group_hard = y_hard[in_group]    # (35,)

        # 2. Get constraints for this course
        K = local_constraints[group_id]  # e.g., [10, 6, 1e10]

        # 3. Check each class (same logic as global)
        for class_id in [0, 1, 2]:
            hard_count = (group_hard == class_id).sum()
            if hard_count > K[class_id]:
                # Compute loss...
                L_feat += loss

    return L_feat
```

**Returns**:
- `avg_ce`: Average cross-entropy loss over all batches
- `avg_global`: Global constraint loss
- `avg_local`: Local constraint loss

---

##### 3.4.2 Update Lambda Weights

**Function**: `update_lambda_weights()`
**Module**: `src/training/trainer.py` (lines 105-137)
**Called From**: Training loop line 292

```python
update_lambda_weights(avg_global, avg_local, criterion_constraint, epoch)
```

**Internal Logic**:
```python
def update_lambda_weights(avg_global, avg_local, criterion_constraint, epoch):
    # 1. During warmup, force lambdas to 0
    if epoch < WARMUP_EPOCHS:  # epoch < 250
        criterion_constraint.set_lambda(lambda_global=0.0, lambda_local=0.0)
        return False

    # 2. After warmup, increase lambda if constraints violated
    lambda_updated = False

    if avg_global > CONSTRAINT_THRESHOLD:  # > 1e-6
        new_lambda_global = criterion_constraint.lambda_global + LAMBDA_STEP
        criterion_constraint.set_lambda(lambda_global=new_lambda_global)
        lambda_updated = True

    if avg_local > CONSTRAINT_THRESHOLD:
        new_lambda_local = criterion_constraint.lambda_local + LAMBDA_STEP
        criterion_constraint.set_lambda(lambda_local=new_lambda_local)
        lambda_updated = True

    return lambda_updated
```

**Example Evolution**:
```
Epoch   0-249: λ_global=0.00, λ_local=0.00  (warmup)
Epoch 250:     λ_global=0.00, λ_local=0.00  (constraints kick in)
Epoch 251:     λ_global=0.01, λ_local=0.01  (violation detected)
Epoch 254:     λ_global=0.02, λ_local=0.01  (global still violated)
Epoch 257:     λ_global=0.03, λ_local=0.01
...
Epoch 300:     λ_global=0.15, λ_local=0.07  (plateau - satisfied)
```

---

##### 3.4.3 Compute and Log Statistics

**Every 3 Epochs** (epoch 3, 6, 9, ...):

**Step 1: Compute Prediction Statistics**

**Function**: `compute_prediction_statistics()`
**Module**: `src/training/metrics.py` (lines 14-46)

```python
global_counts, local_counts, global_soft_counts, local_soft_counts = \
    compute_prediction_statistics(model, X_test_tensor, group_ids_test)
```

**Internal Steps**:
```python
def compute_prediction_statistics(model, X_test_tensor, group_ids_test):
    model.eval()
    with torch.no_grad():
        # 1. Forward pass
        test_logits = model(X_test_tensor)  # (442, 3)

        # 2. Get hard predictions
        test_preds = torch.argmax(test_logits, dim=1)  # (442,)

        # 3. Get soft predictions
        test_proba = F.softmax(test_logits, dim=1)  # (442, 3)

        # 4. Global hard counts
        global_counts = {}
        for class_id in [0, 1, 2]:
            count = (test_preds == class_id).sum().item()
            global_counts[class_id] = count
        # Example: {0: 140, 1: 81, 2: 221}

        # 5. Global soft counts
        global_soft_counts = {}
        for class_id in [0, 1, 2]:
            soft_count = test_proba[:, class_id].sum().item()
            global_soft_counts[class_id] = soft_count
        # Example: {0: 141.23, 1: 82.45, 2: 218.32}

        # 6. Local (per-course) hard counts
        local_counts = {}
        for group_id in unique_groups:
            group_mask = (group_ids_test == group_id)
            group_preds = test_preds[group_mask]

            course_counts = {}
            for class_id in [0, 1, 2]:
                count = (group_preds == class_id).sum().item()
                course_counts[class_id] = count
            local_counts[group_id] = course_counts
        # Example: {2: {0: 8, 1: 5, 2: 22}, 3: {...}, ...}

        # 7. Local soft counts (same logic)
        local_soft_counts = {...}

    model.train()
    return global_counts, local_counts, global_soft_counts, local_soft_counts
```

**Step 2: Log to CSV**

**Function**: `log_progress_to_csv()`
**Module**: `src/training/logging.py` (lines 131-187)

```python
log_progress_to_csv(csv_log_path, epoch, avg_global, avg_local, avg_ce,
                   global_counts, local_counts, global_soft_counts, local_soft_counts,
                   lambda_global, lambda_local, global_constraints,
                   global_satisfied, local_satisfied)
```

**Internal Steps**:
```python
def log_progress_to_csv(...):
    # 1. Create header if file doesn't exist
    if not file_exists:
        header = ['Epoch', 'L_pred_CE', 'L_target_Global', 'L_feat_Local',
                 'Lambda_Global', 'Lambda_Local', 'Global_Satisfied', 'Local_Satisfied',
                 'Limit_Dropout', 'Limit_Enrolled', 'Limit_Graduate',
                 'Hard_Dropout', 'Hard_Enrolled', 'Hard_Graduate',
                 'Soft_Dropout', 'Soft_Enrolled', 'Soft_Graduate',
                 'Excess_Dropout', 'Excess_Enrolled', 'Excess_Graduate',
                 'Course_ID', 'Course_Hard_Dropout', ...]
        writer.writerow(header)

    # 2. Compute excess violations
    excess_dropout = max(0, soft_dropout - limit_dropout)

    # 3. Get tracked course data
    if TRACKED_COURSE_ID in local_counts:
        course_hard = [local_counts[TRACKED_COURSE_ID][i] for i in range(3)]

    # 4. Write row
    row = [epoch+1, avg_ce, avg_global, avg_local, ...]
    writer.writerow(row)
```

**CSV Output Example**:
```csv
Epoch,L_pred_CE,L_target_Global,L_feat_Local,Lambda_Global,Lambda_Local,...
3,0.654321,0.045231,0.012345,0.00,0.00,0,0,142,85,inf,140,81,221,141.23,82.45,218.32,0.00,0.00,0.00,2,8,5,22,8.12,5.34,21.54
6,0.632145,0.041234,0.010987,0.00,0.00,0,0,142,85,inf,139,82,221,140.56,82.89,218.55,...
...
```

**Step 3: Print to Console** (only after warmup)

**Function**: `print_progress_from_csv()`
**Module**: `src/training/logging.py` (lines 242-278)

Only executes if `epoch >= WARMUP_EPOCHS` (250)

```python
if epoch >= WARMUP_EPOCHS:
    print_progress_from_csv(csv_log_path, criterion_constraint)
```

**Output Example**:
```
================================================================================
Epoch 252
================================================================================
L_target (Global):  0.024531
L_feat (Local):     0.013245
L_pred (CE):        0.456789

────────────────────────────────────────────────────────────────────────────────
GLOBAL CONSTRAINTS vs PREDICTIONS (Hard vs Soft)
────────────────────────────────────────────────────────────────────────────────
Class        Limit    Hard     Soft       Diff     Status
────────────────────────────────────────────────────────────────────────────────
Dropout      142      140      141.23     1.23     ✓ OK
Enrolled     85       81       82.45      1.45     ✓ OK
Graduate     ∞        221      218.32     -2.68    N/A
────────────────────────────────────────────────────────────────────────────────
Total                 442      442.00

Current Lambda Weights: λ_global=0.02, λ_local=0.01
Constraint Status: Global=✓, Local=✓
================================================================================
```

---

##### 3.4.4 Check Early Stopping

**Location**: Training loop lines 321-329

```python
if criterion_constraint.global_constraints_satisfied and \
   criterion_constraint.local_constraints_satisfied:
    print("✓ ALL CONSTRAINTS SATISFIED at epoch {epoch + 1}!")
    break
```

**What Happens**:
- If both flags are True (set in constraint loss computation)
- Stop training early
- Saves time if constraints satisfied before 1000 epochs

---

#### 3.5 Post-Training: Load History and Visualize

**Location**: `train_model_transductive()` lines 331-342

**Step 1: Load History from CSV**

**Function**: `load_history_from_csv()`
**Module**: `src/training/logging.py` (lines 281-313)

```python
history = load_history_from_csv(csv_log_path)
```

**Returns**:
```python
{
    'epochs': [3, 6, 9, 12, ...],
    'loss_ce': [0.654, 0.632, ...],
    'loss_global': [0.045, 0.041, ...],
    'loss_local': [0.012, 0.010, ...],
    'lambda_global': [0.00, 0.00, 0.01, ...],
    'lambda_local': [0.00, 0.00, 0.01, ...],
    'global_predictions': [
        {0: 140, 1: 81, 2: 221},  # Epoch 3
        {0: 139, 1: 82, 2: 221},  # Epoch 6
        ...
    ],
    'local_predictions': [
        {2: [8, 5, 22]},  # Epoch 3, Course 2
        {2: [8, 5, 22]},  # Epoch 6, Course 2
        ...
    ]
}
```

**Step 2: Create Visualizations**

**Function**: `create_all_visualizations()`
**Module**: `src/utils/visualization.py` (lines 251-295)

```python
create_all_visualizations(history, global_constraints, local_constraints,
                         output_dir=experiment_folder)
```

**Internal Calls**:
```python
def create_all_visualizations(...):
    # 1. Global constraints plot
    plot_global_constraints(history, global_constraints,
                           save_path='results/.../global_constraints.png')

    # 2. Local constraints plot (tracked course only)
    plot_local_constraints(history, local_constraints,
                          save_path='results/.../local_constraints.png')

    # 3. Loss components plot
    plot_losses(history, save_path='results/.../losses.png')

    # 4. Lambda evolution plot
    plot_lambda_evolution(history, save_path='results/.../lambda_evolution.png')
```

**Outputs Created**:
- `global_constraints.png`: 4 lines (3 predictions + constraints)
- `local_constraints.png`: Single course view
- `losses.png`: 2×2 grid (L_target, L_feat, L_pred, L_total)
- `lambda_evolution.png`: Lambda weights over time

**Returns from `train_model_transductive()`**:
```python
return model, scaler, training_time, history
```

---

### PHASE 4: Evaluation and Results Saving

**Location**: `run_experiments.py::main()` (lines 97-109)

#### 4.1 Make Predictions

**Function**: `predict()`
**Module**: `src/training/trainer.py` (lines 347-368)
**Called From**: `run_experiments.py` line 97

```python
y_test_pred = predict(model, scaler, X_test_clean, device)
```

**Internal Steps**:
```python
def predict(model, scaler, X_test, device):
    model.eval()

    # 1. Scale test data (same scaler as training)
    X_test_scaled = scaler.transform(X_test)

    # 2. Convert to tensor
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)

    # 3. Forward pass
    with torch.no_grad():
        logits = model(X_test_tensor)  # (442, 3)
        predictions = torch.argmax(logits, dim=1).cpu().numpy()  # (442,)

    return predictions  # [0, 2, 1, 2, 0, ...]
```

**Returns**: NumPy array of predicted classes (442,)

#### 4.2 Evaluate Accuracy

**Function**: `evaluate_accuracy()`
**Module**: `src/training/metrics.py` (lines 117-126)
**Called From**: `run_experiments.py` line 98

```python
accuracy = evaluate_accuracy(y_test.values, y_test_pred)
```

**Internal Steps**:
```python
def evaluate_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)
```

**Example**:
```python
y_true = [0, 1, 2, 2, 0, 1, ...]  # Ground truth
y_pred = [0, 1, 2, 1, 0, 1, ...]  # Predictions
accuracy = 0.7421  # 74.21% correct
```

#### 4.3 Save Results to CSV

**Function**: `save_results()`
**Module**: `run_experiments.py` (lines 19-30)
**Called From**: `run_experiments.py` line 101

```python
save_results(results_file, config_name, "transductive", accuracy,
            local_percent, global_percent, training_time)
```

**Internal Steps**:
```python
def save_results(results_file, model_name, method_name, accuracy,
                local_percent, global_percent, training_time):
    output = f"{model_name},{method_name},{accuracy},{local_percent},{global_percent},{training_time}"

    # Create header if new file
    if not os.path.exists(results_file):
        headlines = "model_name,method_name,accuracy,local_percent,global_percent,training_time"
        f.write(headlines + "\n")

    # Append results
    f.write(output + "\n")
```

**Output File**: `results/students__train__nn_config1__transductive.csv`

**Content Example**:
```csv
model_name,method_name,accuracy,local_percent,global_percent,training_time
nn_config1,transductive,0.7421,0.8,0.6,125.34
nn_config1,transductive,0.7389,0.7,0.5,142.67
```

#### 4.4 Save Results to JSON

**Location**: `run_experiments.py` lines 113-115

```python
results_json_path = f"{RESULTS_DIR}/nn_results.json"
with open(results_json_path, 'w') as f:
    json.dump(all_results, f, indent=4)
```

**Output File**: `results/nn_results.json`

**Content Example**:
```json
{
    "nn_config1_transductive": {
        "(0.8, 0.6)": {
            "accuracy": 0.7421,
            "training_time": 125.34
        },
        "(0.7, 0.5)": {
            "accuracy": 0.7389,
            "training_time": 142.67
        }
    }
}
```

---

## Function Reference

### Complete Function Call Hierarchy

```
main()
├── load_presplit_data()
│   └── Returns: X_train, X_test, y_train, y_test, train_df, test_df
│
├── compute_global_constraints()
│   └── Returns: [142.0, 85.0, 1e10]
│
├── compute_local_constraints()
│   └── Returns: {2: [10, 6, 1e10], 3: [...], ...}
│
├── train_model_transductive()
│   ├── prepare_training_data()
│   │   └── Returns: train_loader, X_test_tensor, group_ids_test, scaler
│   │
│   ├── initialize_model_and_optimizer()
│   │   ├── NeuralNetClassifier.__init__()
│   │   ├── MulticlassTransductiveLoss.__init__()
│   │   └── Returns: model, criterion_ce, criterion_constraint, optimizer
│   │
│   ├── FOR EACH EPOCH:
│   │   ├── train_single_epoch()
│   │   │   ├── FOR EACH BATCH:
│   │   │   │   ├── model.forward() [on train batch]
│   │   │   │   ├── criterion_ce()
│   │   │   │   ├── model.forward() [on test data]
│   │   │   │   ├── criterion_constraint.forward()
│   │   │   │   │   ├── _compute_global_constraint_loss()
│   │   │   │   │   └── _compute_local_constraint_loss()
│   │   │   │   ├── loss.backward()
│   │   │   │   └── optimizer.step()
│   │   │   └── Returns: avg_ce, avg_global, avg_local
│   │   │
│   │   ├── update_lambda_weights()
│   │   │   └── criterion_constraint.set_lambda()
│   │   │
│   │   ├── IF (epoch + 1) % 3 == 0:
│   │   │   ├── compute_prediction_statistics()
│   │   │   ├── log_progress_to_csv()
│   │   │   └── IF epoch >= WARMUP_EPOCHS:
│   │   │       └── print_progress_from_csv()
│   │   │
│   │   └── IF constraints_satisfied:
│   │       └── BREAK
│   │
│   ├── load_history_from_csv()
│   ├── create_all_visualizations()
│   │   ├── plot_global_constraints()
│   │   ├── plot_local_constraints()
│   │   ├── plot_losses()
│   │   └── plot_lambda_evolution()
│   │
│   └── Returns: model, scaler, training_time, history
│
├── predict()
│   └── Returns: y_test_pred
│
├── evaluate_accuracy()
│   └── Returns: accuracy
│
└── save_results()
    └── Writes CSV and JSON files
```

---

## Data Flow

### Data Shape Transformations

```
CSV Files
  ↓
[400 rows × 36 columns] train.csv
[442 rows × 36 columns] test.csv
  ↓ load_presplit_data()
  ↓
X_train: [400 × 35] (features only, no Course column for model)
y_train: [400]      (labels)
X_test:  [442 × 35]
y_test:  [442]
  ↓ prepare_training_data()
  ↓
X_train_scaled: [400 × 35] (standardized)
X_test_scaled:  [442 × 35]
  ↓
train_loader: Batches of ([32 × 35], [32]) (features, labels)
X_test_tensor: [442 × 35] GPU tensor
  ↓ model.forward()
  ↓
train_logits: [32 × 3]  (batch predictions)
test_logits:  [442 × 3] (all test predictions)
  ↓ softmax()
  ↓
test_proba: [442 × 3] (probabilities per class)
  ↓ argmax()
  ↓
test_preds: [442] (hard predictions: 0, 1, or 2)
  ↓
global_counts: {0: 140, 1: 81, 2: 221}
local_counts:  {2: {0: 8, 1: 5, 2: 22}, ...}
```

### Constraint Flow

```
Full Dataset (train + test combined)
  ↓ compute_global_constraints()
  ↓
Global Constraints: [142, 85, 1e10]
  ↓
  ↓ compute_local_constraints()
  ↓
Local Constraints: {2: [10, 6, 1e10], 3: [...], ...}
  ↓
  ↓ MulticlassTransductiveLoss.__init__()
  ↓
Registered as buffers (moved to GPU with model)
  ↓
  ↓ During training: _compute_global_constraint_loss()
  ↓
Compare predictions vs constraints → compute loss
  ↓
Loss gradients → backprop → update model weights
```

---

## Summary: Complete Execution Timeline

| Time | Phase | Key Functions | Data State |
|------|-------|---------------|------------|
| **0s** | **Initialization** | `main()` | Config loaded |
| **0-1s** | **Data Loading** | `load_presplit_data()` | Raw data in memory |
| **1-2s** | **Constraints** | `compute_global_constraints()`<br>`compute_local_constraints()` | Constraints computed |
| **2-3s** | **Preparation** | `prepare_training_data()`<br>`initialize_model_and_optimizer()` | Model initialized, data on GPU |
| **3s-5min** | **Training Loop** | `train_single_epoch()` ×250<br>(warmup) | Model learning classification |
| **5min-8min** | **Constraint Training** | `train_single_epoch()` ×50<br>`update_lambda_weights()`<br>`log_progress_to_csv()` | Constraints being satisfied |
| **8min** | **Early Stop** | Check satisfaction | Training complete (epoch 300) |
| **8min-9min** | **Visualization** | `load_history_from_csv()`<br>`create_all_visualizations()` | 4 PNG files created |
| **9min** | **Evaluation** | `predict()`<br>`evaluate_accuracy()` | Final accuracy computed |
| **9min** | **Save Results** | `save_results()` | CSV + JSON written |

**Total**: ~9 minutes for one experiment configuration

---

## Key Insights

### Why This Architecture?

1. **Transductive Learning**: Uses unlabeled test data during training (not just evaluation)
2. **Dual Objectives**: Balances classification accuracy (CE loss) + constraint satisfaction
3. **Adaptive Weighting**: Lambda increases when constraints violated → stronger pressure
4. **Warmup Period**: Learn classification first (250 epochs), then apply constraints
5. **Hard/Soft Split**: Check satisfaction with hard predictions, compute loss with soft (differentiable)

### Critical Design Decisions

1. **Eval mode for constraints**: Dropout off when computing constraint losses → stable
2. **Gradient flow**: No `torch.no_grad()` on constraint forward pass → gradients reach model
3. **Rational saturation loss**: `E/(E+K)` prevents unbounded growth, smooth gradients
4. **CSV-centric logging**: Single source of truth for all metrics
5. **Early stopping**: Don't waste time if constraints satisfied early

---

## Next Steps

- See [USAGE_GUIDE.md](USAGE_GUIDE.md) for how to run experiments
- See [VISUALIZATION_GUIDE.md](VISUALIZATION_GUIDE.md) for interpreting results
- Check source code for implementation details

---

**Document Version**: 1.0
**Last Updated**: Based on codebase as of latest commit
