import numpy as np

DATA_PATH = "./Students_ Dropout_and_Academic_Success_with_costs.csv"
TARGET_COLUMN = 'Target'

CONSTRAINTS = [
    (0.9, 0.8), (0.9, 0.5), (0.8, 0.7), (0.8, 0.2),
    (0.7, 0.5), (0.6, 0.5), (0.5, 0.3), (0.4, 0.2)
]

NN_CONFIGS = [
    {"lambda_global": 1.0, "lambda_local": 0.5, "hidden_dims": [128, 64, 32]},
    {"lambda_global": 1.5, "lambda_local": 1.0, "hidden_dims": [128, 64, 32]},
    {"lambda_global": 0.5, "lambda_local": 0.3, "hidden_dims": [64, 32]}
]

TRAINING_PARAMS = {
    'epochs': 30,
    'batch_size': 64,
    'lr': 0.001,
    'dropout': 0.3,
    'patience': 10,
    'k_folds': 9
}

RESULTS_DIR = "./results"
