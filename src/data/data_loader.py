import pandas as pd


def load_presplit_data(train_path, test_path, target_column):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    y_train = train_df[target_column]
    X_train = train_df.drop(labels=[target_column], axis=1)

    y_test = test_df[target_column]
    X_test = test_df.drop(labels=[target_column], axis=1)

    return X_train, X_test, y_train, y_test, train_df, test_df
