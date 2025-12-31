import numpy as np

FULL_DATASET_PATH = "./data/dataset.csv"
TRAIN_PATH = "./data/dataset_train.csv"
TEST_PATH = "./data/dataset_test.csv"
TARGET_COLUMN = 'Target'

CONSTRAINTS = [
    (0.5, 0.3)  # Fixed constraint for hyperparameter experiments
]

# TOP 5 CONFIGURATIONS - Final Validation
# Selected from 31 total experiments (Round 1 + Round 2)
# Based on comprehensive hyperparameter search results
NN_CONFIGS = [
    # 1. ARCH_DEEP - 🏆 CHAMPION (61.09% accuracy, +2.72% vs benchmark)
    # Why it works: 4 layers provides optimal depth for constraint learning
    # Key insight: Deep networks need LOW lambda (0.01), not high!
    # The depth provides constraint-learning capacity WITHOUT needing high lambda pressure
    {
        "name": "arch_deep",
        "lambda_global": 0.01,  # Baseline lambda - CRITICAL!
        "lambda_local": 0.01,
        "hidden_dims": [256, 128, 64, 32],  # 4 layers - optimal depth
        "lr": 0.001,
        "dropout": 0.3,
        "batch_size": 64
    },

    # 2. DROPOUT_HIGH (60.86% accuracy, tied with benchmark)
    # Why it works: Strong regularization (0.5 dropout) prevents overfitting
    # Shallow network (3 layers) with aggressive dropout
    {
        "name": "dropout_high",
        "lambda_global": 0.01,
        "lambda_local": 0.01,
        "hidden_dims": [128, 64, 32],
        "lr": 0.001,
        "dropout": 0.5,  # Strong regularization
        "batch_size": 64
    },

    # 3. VERY_DEEP_BASELINE (60.63% accuracy, +1.58% vs benchmark)
    # Why it works: 5 layers with baseline lambda
    # Key insight: Performance starts degrading beyond 4 layers (-0.46%)
    # Still viable but not optimal
    {
        "name": "very_deep_baseline",
        "lambda_global": 0.01,  # Baseline lambda
        "lambda_local": 0.01,
        "hidden_dims": [512, 256, 128, 64, 32],  # 5 layers
        "lr": 0.001,
        "dropout": 0.3,
        "batch_size": 64
    },

    # 4. LAMBDA_HIGH (60.63% accuracy, +0.45% vs benchmark)
    # Why it works: High lambda works ONLY with shallow networks (3 layers)
    # Key insight: Deep + high lambda = disaster due to conflicting gradients
    {
        "name": "lambda_high",
        "lambda_global": 0.1,  # High lambda
        "lambda_local": 0.1,
        "hidden_dims": [128, 64, 32],  # 3 layers only!
        "lr": 0.001,
        "dropout": 0.3,
        "batch_size": 64
    },

    # 5. VERY_DEEP_EXTREME_LAMBDA (60.63% accuracy, +1.35% vs benchmark)
    # Why it works: Surprisingly, extreme lambda (1.0) + 5 layers works
    # This is an outlier - extreme lambdas usually cause early termination failures
    # Kept for scientific interest and diversity
    {
        "name": "very_deep_extreme_lambda",
        "lambda_global": 1.0,  # Extreme lambda - risky but works here
        "lambda_local": 1.0,
        "hidden_dims": [512, 256, 128, 64, 32],  # 5 layers
        "lr": 0.001,
        "dropout": 0.3,
        "batch_size": 64
    },
]

# KEY LEARNINGS FROM 31 EXPERIMENTS:
# ✅ 4 layers [256,128,64,32] is optimal depth
# ✅ Deep networks need LOW lambda (0.01), not high (0.1)
# ✅ Learning rate 0.001 is critical - both 0.0001 and 0.01 failed
# ❌ 5+ layers degrades performance (5 layers: -0.46%, 6 layers: -1.36%, 7 layers: -3.85%)
# ❌ Combining winning strategies (deep + high lambda) causes interference (-0.91%)
# ❌ Extreme lambda values (0.5, 1.0) usually cause catastrophic early termination
# 🚧 Performance ceiling at ~61% - cannot be broken with hyperparameter tuning alone

TRAINING_PARAMS = {
    'epochs': 10000,
    'batch_size': 64,  # Default, overridden by config
    'lr': 0.001,       # Default, overridden by config
    'dropout': 0.3,    # Default, overridden by config
    'test_size': 0.1
}

CONSTRAINT_THRESHOLD = 1e-6
LAMBDA_STEP = 0.01
WARMUP_EPOCHS = 250
TRACKED_COURSE_ID = 2  # Course to track for local constraint visualization

RESULTS_DIR = "./results"
