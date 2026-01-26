"""Data preprocessing utilities for constraint-based learning."""

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_and_preprocess(train_path: str, test_path: str, target_column: str = 'Target'):
    """Load train and test datasets and perform basic preprocessing."""
    print(f"Loading train data from: {train_path}")
    train_df = pd.read_csv(train_path)
    print(f"Train shape: {train_df.shape}")

    print(f"Loading test data from: {test_path}")
    test_df = pd.read_csv(test_path)
    print(f"Test shape: {test_df.shape}")

    # Remove unnamed index columns if present
    for df in [train_df, test_df]:
        unnamed_cols = [col for col in df.columns if col.startswith('Unnamed')]
        if unnamed_cols:
            df.drop(columns=unnamed_cols, inplace=True)

    # Encode target column if it's not numerical
    if train_df[target_column].dtype == 'object':
        print(f"\nEncoding target column '{target_column}' to numerical")
        le = LabelEncoder()
        train_df[target_column] = le.fit_transform(train_df[target_column])
        test_df[target_column] = le.transform(test_df[target_column])
        mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        print(f"Target mapping: {mapping}")

    print(f"\nTrain class distribution:\n{train_df[target_column].value_counts().sort_index()}")
    print(f"\nTest class distribution:\n{test_df[target_column].value_counts().sort_index()}")

    return train_df, test_df
