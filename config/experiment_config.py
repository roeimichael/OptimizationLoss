import numpy as np

FULL_DATASET_PATH = "./data/dataset.csv"
TRAIN_PATH = "./data/dataset_train.csv"
TEST_PATH = "./data/dataset_test.csv"
TARGET_COLUMN = 'Target'

CONSTRAINTS = [
    # (0.9, 0.8), (0.9, 0.5), (0.8, 0.7), (0.8, 0.2),
    # (0.7, 0.5), (0.6, 0.5), (0.4, 0.2)
    # (0.5, 0.3)
    (0.8, 0.6)
]

NN_CONFIGS = [
    {"lambda_global": 0.01, "lambda_local": 0.01, "hidden_dims": [128, 64, 32]}
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
WARMUP_EPOCHS = 50

RESULTS_DIR = "./results"

