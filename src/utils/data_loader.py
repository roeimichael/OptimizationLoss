"""Data loading and constraint computation for experiments."""

import logging

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from config.experiment_config import TARGET_COLUMN, GROUP_COLUMN
from src.training.constraints import compute_global_constraints, compute_local_constraints

log = logging.getLogger(__name__)

DATASET_PATHS = {
    'binary': {
        'train': 'data/adult/train_dataset_cleaned.csv',
        'test': 'data/adult/test_dataset_cleaned.csv',
    },
}


def load_presplit_data(train_path, test_path):
    """Load pre-split train and test CSVs."""
    return pd.read_csv(train_path), pd.read_csv(test_path)


def encode_categorical_features(train_df, test_df):
    """LabelEncode all object columns."""
    train_enc, test_enc = train_df.copy(), test_df.copy()
    for col in train_df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        train_enc[col] = le.fit_transform(train_df[col].astype(str))
        test_enc[col] = test_df[col].astype(str).map(
            lambda x, _le=le: _le.transform([x])[0] if x in _le.classes_ else -1
        )
    return train_enc, test_enc


def load_experiment_data(config):
    """Load data and compute constraints. Returns 8-tuple."""
    dataset_mode = config.get('dataset_mode', 'binary')
    paths = DATASET_PATHS[dataset_mode]

    train_df, test_df = load_presplit_data(paths['train'], paths['test'])
    train_df, test_df = encode_categorical_features(train_df, test_df)

    if dataset_mode == 'binary':
        num_classes, constrained_class = 2, 1
    else:
        num_classes, constrained_class = 5, 4

    local_percent, global_percent = config['constraint']

    global_constraint = compute_global_constraints(
        test_df, TARGET_COLUMN, global_percent,
        constrained_class=constrained_class, num_classes=num_classes)
    local_constraint = compute_local_constraints(
        test_df, TARGET_COLUMN, local_percent, GROUP_COLUMN,
        constrained_class=constrained_class, num_classes=num_classes)

    log.info("mode=%s classes=%d constrained=%d global=%s local_groups=%d",
             dataset_mode, num_classes, constrained_class,
             global_constraint, len(local_constraint))

    drop_cols = [TARGET_COLUMN, GROUP_COLUMN]
    return (train_df.drop(columns=drop_cols), test_df.drop(columns=drop_cols),
            train_df[TARGET_COLUMN], test_df[TARGET_COLUMN],
            test_df[GROUP_COLUMN], global_constraint, local_constraint, num_classes)
