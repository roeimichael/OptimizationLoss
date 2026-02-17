"""Data loading utilities for constraint-based learning."""

import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.preprocessing import LabelEncoder

from config.experiment_config import TRAIN_PATH, TEST_PATH, TARGET_COLUMN, GROUP_COLUMN
from src.training.constraints import compute_global_constraints, compute_local_constraints
from src.utils.error_handler import logger


@logger()
def load_presplit_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load pre-split train and test datasets."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


@logger()
def encode_categorical_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Encode categorical features using LabelEncoder."""
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()

    categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()

    for col in categorical_cols:
        le = LabelEncoder()
        train_encoded[col] = le.fit_transform(train_df[col].astype(str))
        test_encoded[col] = test_df[col].astype(str).map(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )

    return train_encoded, test_encoded


@logger()
def load_experiment_data(config: Dict[str, Any]):
    """Load data and compute constraints for experiment.

    Labels are converted to binary: 0 (no churn) vs 1 (churn).
    This is the ONLY place where label conversion happens.
    """
    print("\nLoading dataset...")
    train_df, test_df = load_presplit_data(TRAIN_PATH, TEST_PATH)
    train_df, test_df = encode_categorical_features(train_df, test_df)

    # Convert to binary classification:
    # Original labels 1-3 → 0 (no churn), 4-5 → 1 (churn)
    train_df[TARGET_COLUMN] = (train_df[TARGET_COLUMN] >= 4).astype(int)
    test_df[TARGET_COLUMN] = (test_df[TARGET_COLUMN] >= 4).astype(int)

    local_percent, global_percent = config['constraint']

    # Constraint computation: class 1 (churn) gets a percentage limit
    # Class 0 is unlimited
    global_constraint = compute_global_constraints(
        test_df, TARGET_COLUMN, global_percent
    )
    local_constraint = compute_local_constraints(
        test_df, TARGET_COLUMN, local_percent, GROUP_COLUMN
    )

    # Validation: print label range and distribution
    print(f"[LABELS] Train range: {train_df[TARGET_COLUMN].min()}-{train_df[TARGET_COLUMN].max()}")
    print(f"[LABELS] Test range: {test_df[TARGET_COLUMN].min()}-{test_df[TARGET_COLUMN].max()}")
    print(f"[LABELS] Train distribution: {dict(train_df[TARGET_COLUMN].value_counts().sort_index())}")
    print(f"[LABELS] Test distribution: {dict(test_df[TARGET_COLUMN].value_counts().sort_index())}")
    print(f"Global constraint: {global_constraint}")
    print(f"Local constraints: {len(local_constraint)} groups")

    drop_cols = [TARGET_COLUMN, GROUP_COLUMN]
    y_train = train_df[TARGET_COLUMN]
    X_train_clean = train_df.drop(columns=drop_cols)
    y_test = test_df[TARGET_COLUMN]
    groups_test = test_df[GROUP_COLUMN]
    X_test_clean = test_df.drop(columns=drop_cols)

    return X_train_clean, X_test_clean, y_train, y_test, groups_test, global_constraint, local_constraint
