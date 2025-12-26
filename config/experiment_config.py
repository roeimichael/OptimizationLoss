
import os

DATA_PATH = "./data/dataset.csv"
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CONFIG_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'dataset.csv')
TARGET_COLUMN = 'Target'

CONSTRAINTS = [
    (0.9, 0.8), (0.9, 0.5), (0.8, 0.7), (0.8, 0.2),
    (0.7, 0.5), (0.6, 0.5), (0.5, 0.3), (0.4, 0.2)
]

NN_CONFIGS = [
    {"lambda_global": 1.0, "lambda_local": 1.0, "hidden_dims": [128, 64, 32]}
]

TRAINING_PARAMS = {
    'epochs': 10000,  # High limit - training stops when constraints are satisfied
    'batch_size': 64,
    'lr': 0.001,
    'dropout': 0.3,
    'patience': 10,
    'test_size': 0.1
}

RESULTS_DIR = "./results"
