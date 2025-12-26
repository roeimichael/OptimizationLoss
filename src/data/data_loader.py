import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold


def load_and_preprocess_data(data_path, target_column):
    df = pd.read_csv(data_path)

    # Drop cost_matrix if it exists
    if 'cost_matrix' in df.columns:
        df = df.drop(columns=['cost_matrix'])

    # Drop any columns that look like "Unnamed" (e.g., index columns from CSVs)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Drop all rows with any None/NaN values
    df = df.dropna()

    # Identify categorical columns
    cats = [c for c in df.columns if df[c].dtypes == 'object']

    # Remove target_column from the list of features to encode if present
    if target_column in cats:
        cats.remove(target_column)

    # Encode categorical features
    for col in cats:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # Encode the target column
    le = LabelEncoder()
    df[target_column] = le.fit_transform(df[target_column].astype(str))
    df[target_column] = df[target_column].astype(float)

    return df


def split_data(df, target_column, test_size=0.1, random_state=42):
    y = df[target_column]
    X = df.drop(labels=[target_column], axis=1)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=False
    )

    return X_train_val, X_test, y_train_val, y_test


def load_presplit_data(train_path, test_path, target_column):
    """
    Load pre-split train and test datasets.

    Args:
        train_path: Path to training CSV file
        test_path: Path to test CSV file
        target_column: Name of target column

    Returns:
        X_train, X_test, y_train, y_test: Split features and labels
    """
    # Load pre-split datasets (already preprocessed)
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Separate features and target
    y_train = train_df[target_column]
    X_train = train_df.drop(labels=[target_column], axis=1)

    y_test = test_df[target_column]
    X_test = test_df.drop(labels=[target_column], axis=1)

    return X_train, X_test, y_train, y_test


def get_stratified_folds(X, y, n_splits=9, random_state=42):
    skfolds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return skfolds.split(X, y)