import numpy as np

FULL_DATASET_PATH = "./data/dataset.csv"
TRAIN_PATH = "./data/dataset_train.csv"
TEST_PATH = "./data/dataset_test.csv"
TARGET_COLUMN = 'Target'

CONSTRAINTS = [
    (0.5, 0.3)  # Fixed constraint for hyperparameter experiments
]

# ROUND 2: Advanced configurations based on Round 1 insights
# Key findings: Deep architectures (+2.72%), High lambdas (+0.45%), lr=0.001 is critical
NN_CONFIGS = [
    # 1. BEST COMBINED (Deep + High Lambda)
    # Combines the two winning strategies from Round 1
    {
        "name": "best_combined",
        "lambda_global": 0.1,
        "lambda_local": 0.1,
        "hidden_dims": [256, 128, 64, 32],
        "lr": 0.001,
        "dropout": 0.3,
        "batch_size": 64
    },

    # 2-4. VERY DEEP ARCHITECTURES (5 layers)
    # Test if more depth improves constraint learning
    {
        "name": "very_deep_baseline",
        "lambda_global": 0.01,
        "lambda_local": 0.01,
        "hidden_dims": [512, 256, 128, 64, 32],
        "lr": 0.001,
        "dropout": 0.3,
        "batch_size": 64
    },
    {
        "name": "very_deep_high_lambda",
        "lambda_global": 0.1,
        "lambda_local": 0.1,
        "hidden_dims": [512, 256, 128, 64, 32],
        "lr": 0.001,
        "dropout": 0.3,
        "batch_size": 64
    },
    {
        "name": "very_deep_very_high_lambda",
        "lambda_global": 0.5,
        "lambda_local": 0.5,
        "hidden_dims": [512, 256, 128, 64, 32],
        "lr": 0.001,
        "dropout": 0.3,
        "batch_size": 64
    },

    # 5-6. ULTRA DEEP ARCHITECTURES (6 layers)
    # Push the depth limit to see if even more capacity helps
    {
        "name": "ultra_deep_baseline",
        "lambda_global": 0.01,
        "lambda_local": 0.01,
        "hidden_dims": [512, 256, 128, 64, 32, 16],
        "lr": 0.001,
        "dropout": 0.3,
        "batch_size": 64
    },
    {
        "name": "ultra_deep_high_lambda",
        "lambda_global": 0.1,
        "lambda_local": 0.1,
        "hidden_dims": [512, 256, 128, 64, 32, 16],
        "lr": 0.001,
        "dropout": 0.3,
        "batch_size": 64
    },

    # 7-9. DEEP + VERY HIGH LAMBDA (Aggressive constraint pressure)
    # Test if much stronger constraint pressure helps with deep networks
    {
        "name": "deep_very_high_lambda",
        "lambda_global": 0.5,
        "lambda_local": 0.5,
        "hidden_dims": [256, 128, 64, 32],
        "lr": 0.001,
        "dropout": 0.3,
        "batch_size": 64
    },
    {
        "name": "deep_extreme_lambda",
        "lambda_global": 1.0,
        "lambda_local": 1.0,
        "hidden_dims": [256, 128, 64, 32],
        "lr": 0.001,
        "dropout": 0.3,
        "batch_size": 64
    },
    {
        "name": "very_deep_extreme_lambda",
        "lambda_global": 1.0,
        "lambda_local": 1.0,
        "hidden_dims": [512, 256, 128, 64, 32],
        "lr": 0.001,
        "dropout": 0.3,
        "batch_size": 64
    },

    # 10-11. WIDE DEEP ARCHITECTURES
    # Test if width + depth together provide more capacity
    {
        "name": "wide_deep_baseline",
        "lambda_global": 0.01,
        "lambda_local": 0.01,
        "hidden_dims": [512, 256, 128, 64],
        "lr": 0.001,
        "dropout": 0.3,
        "batch_size": 64
    },
    {
        "name": "wide_deep_high_lambda",
        "lambda_global": 0.1,
        "lambda_local": 0.1,
        "hidden_dims": [512, 256, 128, 64],
        "lr": 0.001,
        "dropout": 0.3,
        "batch_size": 64
    },

    # 12-13. DEEP + DROPOUT VARIATIONS
    # Test if deeper networks need different regularization
    {
        "name": "deep_low_dropout",
        "lambda_global": 0.1,
        "lambda_local": 0.1,
        "hidden_dims": [256, 128, 64, 32],
        "lr": 0.001,
        "dropout": 0.2,
        "batch_size": 64
    },
    {
        "name": "very_deep_high_dropout",
        "lambda_global": 0.1,
        "lambda_local": 0.1,
        "hidden_dims": [512, 256, 128, 64, 32],
        "lr": 0.001,
        "dropout": 0.4,
        "batch_size": 64
    },

    # 14. DEEP + SMALLER BATCH (More gradient noise for exploration)
    {
        "name": "deep_small_batch_high_lambda",
        "lambda_global": 0.1,
        "lambda_local": 0.1,
        "hidden_dims": [256, 128, 64, 32],
        "lr": 0.001,
        "dropout": 0.3,
        "batch_size": 32
    },

    # 15. MEGA DEEP (7 layers - maximum complexity)
    {
        "name": "mega_deep",
        "lambda_global": 0.1,
        "lambda_local": 0.1,
        "hidden_dims": [1024, 512, 256, 128, 64, 32, 16],
        "lr": 0.001,
        "dropout": 0.35,
        "batch_size": 64
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
