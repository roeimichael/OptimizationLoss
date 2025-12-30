import numpy as np

FULL_DATASET_PATH = "./data/dataset.csv"
TRAIN_PATH = "./data/dataset_train.csv"
TEST_PATH = "./data/dataset_test.csv"
TARGET_COLUMN = 'Target'

CONSTRAINTS = [
    (0.5, 0.3)  # Fixed constraint for hyperparameter experiments
]

# Hyperparameter configurations for (0.5, 0.3) constraint experiments
# Each config tests a specific aspect of the model
NN_CONFIGS = [
    # 1. BASELINE
    {
        "name": "baseline",
        "lambda_global": 0.01,
        "lambda_local": 0.01,
        "hidden_dims": [128, 64, 32],
        "lr": 0.001,
        "dropout": 0.3,
        "batch_size": 64
    },

    # 2-5. LAMBDA EXPERIMENTS (Constraint Pressure)
    {
        "name": "lambda_low",
        "lambda_global": 0.001,
        "lambda_local": 0.001,
        "hidden_dims": [128, 64, 32],
        "lr": 0.001,
        "dropout": 0.3,
        "batch_size": 64
    },
    {
        "name": "lambda_high",
        "lambda_global": 0.1,
        "lambda_local": 0.1,
        "hidden_dims": [128, 64, 32],
        "lr": 0.001,
        "dropout": 0.3,
        "batch_size": 64
    },
    {
        "name": "lambda_favor_global",
        "lambda_global": 0.1,
        "lambda_local": 0.01,
        "hidden_dims": [128, 64, 32],
        "lr": 0.001,
        "dropout": 0.3,
        "batch_size": 64
    },
    {
        "name": "lambda_favor_local",
        "lambda_global": 0.01,
        "lambda_local": 0.1,
        "hidden_dims": [128, 64, 32],
        "lr": 0.001,
        "dropout": 0.3,
        "batch_size": 64
    },

    # 6-8. ARCHITECTURE EXPERIMENTS
    {
        "name": "arch_shallow",
        "lambda_global": 0.01,
        "lambda_local": 0.01,
        "hidden_dims": [64, 32],
        "lr": 0.001,
        "dropout": 0.3,
        "batch_size": 64
    },
    {
        "name": "arch_deep",
        "lambda_global": 0.01,
        "lambda_local": 0.01,
        "hidden_dims": [256, 128, 64, 32],
        "lr": 0.001,
        "dropout": 0.3,
        "batch_size": 64
    },
    {
        "name": "arch_wide",
        "lambda_global": 0.01,
        "lambda_local": 0.01,
        "hidden_dims": [256, 128, 64],
        "lr": 0.001,
        "dropout": 0.3,
        "batch_size": 64
    },

    # 9-10. LEARNING RATE EXPERIMENTS
    {
        "name": "lr_slow",
        "lambda_global": 0.01,
        "lambda_local": 0.01,
        "hidden_dims": [128, 64, 32],
        "lr": 0.0001,
        "dropout": 0.3,
        "batch_size": 64
    },
    {
        "name": "lr_fast",
        "lambda_global": 0.01,
        "lambda_local": 0.01,
        "hidden_dims": [128, 64, 32],
        "lr": 0.01,
        "dropout": 0.3,
        "batch_size": 64
    },

    # 11-12. DROPOUT EXPERIMENTS (Regularization)
    {
        "name": "dropout_low",
        "lambda_global": 0.01,
        "lambda_local": 0.01,
        "hidden_dims": [128, 64, 32],
        "lr": 0.001,
        "dropout": 0.1,
        "batch_size": 64
    },
    {
        "name": "dropout_high",
        "lambda_global": 0.01,
        "lambda_local": 0.01,
        "hidden_dims": [128, 64, 32],
        "lr": 0.001,
        "dropout": 0.5,
        "batch_size": 64
    },

    # 13-14. BATCH SIZE EXPERIMENTS
    {
        "name": "batch_small",
        "lambda_global": 0.01,
        "lambda_local": 0.01,
        "hidden_dims": [128, 64, 32],
        "lr": 0.001,
        "dropout": 0.3,
        "batch_size": 32
    },
    {
        "name": "batch_large",
        "lambda_global": 0.01,
        "lambda_local": 0.01,
        "hidden_dims": [128, 64, 32],
        "lr": 0.001,
        "dropout": 0.3,
        "batch_size": 128
    },

    # 15-16. COMBINED OPTIMIZATIONS
    {
        "name": "optimized_v1",
        "lambda_global": 0.1,
        "lambda_local": 0.1,
        "hidden_dims": [256, 128, 64, 32],
        "lr": 0.001,
        "dropout": 0.3,
        "batch_size": 64
    },
    {
        "name": "optimized_v2",
        "lambda_global": 0.01,
        "lambda_local": 0.1,
        "hidden_dims": [256, 128, 64],
        "lr": 0.0001,
        "dropout": 0.2,
        "batch_size": 32
    },
]

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
