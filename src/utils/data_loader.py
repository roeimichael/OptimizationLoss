import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.preprocessing import LabelEncoder

from config.experiment_config import TRAIN_PATH, TEST_PATH, TARGET_COLUMN
from src.training.constraints import compute_global_constraints, compute_local_constraints


def load_presplit_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return train_df, test_df


def encode_categorical_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Encode categorical features using LabelEncoder."""
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()

    # Find categorical columns
    categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()

    # Encode each categorical column
    for col in categorical_cols:
        le = LabelEncoder()
        # Fit on train data
        train_encoded[col] = le.fit_transform(train_df[col].astype(str))
        # Transform test data, handling unseen categories
        test_encoded[col] = test_df[col].astype(str).map(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )

    return train_encoded, test_encoded


def load_experiment_data(config: Dict[str, Any]):
    print("\nLoading dataset...")
    train_df, test_df = load_presplit_data(TRAIN_PATH, TEST_PATH)
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    # Encode categorical features before computing constraints
    train_df, test_df = encode_categorical_features(train_df, test_df)

    local_percent, global_percent = config['constraint']
    groups = test_df['Course'].unique()
    global_constraint = compute_global_constraints(test_df, TARGET_COLUMN, global_percent)
    local_constraint = compute_local_constraints(test_df, TARGET_COLUMN, local_percent, groups)
    print(f"Global constraint: {global_constraint}")
    print(f"Local constraints: {len(local_constraint)} courses")

    drop_cols = [TARGET_COLUMN, 'Course']
    y_train = train_df[TARGET_COLUMN]
    X_train_clean = train_df.drop(columns=drop_cols)
    y_test = test_df[TARGET_COLUMN]
    groups_test = test_df['Course']
    X_test_clean = test_df.drop(columns=drop_cols)
    return X_train_clean, X_test_clean, y_train, y_test, groups_test, global_constraint, local_constraint
