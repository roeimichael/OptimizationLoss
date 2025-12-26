import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold


def load_and_preprocess_data(data_path, target_column):
    df = pd.read_csv(data_path)
    if 'cost_matrix' in df.columns:
        df = df.drop(columns=['cost_matrix'])
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.dropna()
    cats = [c for c in df.columns if df[c].dtypes == 'object']
    if target_column in cats:
        cats.remove(target_column)
    for col in cats:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    le = LabelEncoder()
    df[target_column] = le.fit_transform(df[target_column].astype(str))
    df[target_column] = df[target_column].astype(float)

    return df

def load_presplit_data(train_path, test_path, target_column):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    y_train = train_df[target_column]
    X_train = train_df.drop(labels=[target_column], axis=1)
    y_test = test_df[target_column]
    X_test = test_df.drop(labels=[target_column], axis=1)
    return X_train, X_test, y_train, y_test


def get_stratified_folds(X, y, n_splits=9, random_state=42):
    skfolds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return skfolds.split(X, y)