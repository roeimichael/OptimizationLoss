import pandas as pd
from sklearn.model_selection import StratifiedKFold

def load_presplit_data(train_path, test_path, target_column):

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