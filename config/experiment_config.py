import numpy as np

FULL_DATASET_PATH = "./data/dataset.csv"
TRAIN_PATH = "./data/dataset_train.csv"
TEST_PATH = "./data/dataset_test.csv"
TARGET_COLUMN = 'Target'

CONSTRAINTS = [
    (0.5, 0.3)
]

NN_CONFIGS = [
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
        "name": "dropout_high",
        "lambda_global": 0.01,
        "lambda_local": 0.01,
        "hidden_dims": [128, 64, 32],
        "lr": 0.001,
        "dropout": 0.5,
        "batch_size": 64
    },
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
        "name": "lambda_high",
        "lambda_global": 0.1,
        "lambda_local": 0.1,
        "hidden_dims": [128, 64, 32],
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
]

TRAINING_PARAMS = {
    'epochs': 10000,
    'batch_size': 64,
    'lr': 0.001,
    'dropout': 0.3,
    'test_size': 0.1
}

CONSTRAINT_THRESHOLD = 1e-6
LAMBDA_STEP = 0.01
WARMUP_EPOCHS = 250
TRACKED_COURSE_ID = 2

RESULTS_DIR = "./results"
