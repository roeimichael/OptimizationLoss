"""Data loading utilities for constraint-based learning."""

import pandas as pd
from typing import Tuple, Dict, Any

from config.experiment_config import TRAIN_PATH, TEST_PATH, TARGET_COLUMN, GROUP_COLUMN
from src.training.constraints import compute_global_constraints, compute_local_constraints


def load_presplit_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load pre-split train and test datasets."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def load_experiment_data(config: Dict[str, Any]):
    """Load data and compute constraints for experiment."""
    print("\nLoading dataset...")
    train_df, test_df = load_presplit_data(TRAIN_PATH, TEST_PATH)

    local_percent, global_percent = config['constraint']

    # Compute constraints (can specify unlimited classes if needed)
    unlimited_classes = config.get('unlimited_classes', [])
    global_constraint = compute_global_constraints(test_df, TARGET_COLUMN, global_percent, unlimited_classes)
    local_constraint = compute_local_constraints(test_df, TARGET_COLUMN, local_percent, GROUP_COLUMN, unlimited_classes)

    print(f"Global constraint: {global_constraint}")
    print(f"Local constraints: {len(local_constraint)} groups")

    drop_cols = [TARGET_COLUMN, GROUP_COLUMN]
    y_train = train_df[TARGET_COLUMN]
    X_train_clean = train_df.drop(columns=drop_cols)
    y_test = test_df[TARGET_COLUMN]
    groups_test = test_df[GROUP_COLUMN]
    X_test_clean = test_df.drop(columns=drop_cols)

    return X_train_clean, X_test_clean, y_train, y_test, groups_test, global_constraint, local_constraint
