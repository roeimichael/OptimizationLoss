import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings('ignore')

MAX_MISSING_VALUES_PER_ROW = 3
OUTLIER_PERCENTILES = (1, 99)
IDENTIFIERS_TO_DROP = ['customer_id', 'Name', 'security_no']


def load_and_clean_data(train_path: str, test_path: str) -> tuple:
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    placeholders = ['?', 'Error', 'xxxxxxxx', np.nan, -999]
    train = train.replace(placeholders, np.nan)
    test = test.replace(placeholders, np.nan)

    for df in [train, test]:
        df['avg_frequency_login_days'] = pd.to_numeric(df['avg_frequency_login_days'], errors='coerce')

    return train, test


def filter_low_quality_rows(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    train_missing = train.isnull().sum(axis=1)
    test_missing = test.isnull().sum(axis=1)

    train_clean = train[train_missing <= MAX_MISSING_VALUES_PER_ROW].reset_index(drop=True)
    test_clean = test[test_missing <= MAX_MISSING_VALUES_PER_ROW].reset_index(drop=True)

    return train_clean, test_clean


def process_target(df: pd.DataFrame, target_col: str = 'churn_risk_score') -> pd.DataFrame:
    neg1_count = (df[target_col] == -1).sum()
    neg1_pct = 100 * neg1_count / len(df)

    if neg1_pct > 10:
        df[target_col] = df[target_col].replace(-1, 6)
    else:
        df = df[df[target_col] != -1].reset_index(drop=True)

    return df


def remove_outliers(df: pd.DataFrame, num_cols: list) -> pd.DataFrame:
    df_clean = df.copy()

    for col in num_cols:
        if col in df_clean.columns:
            lower, upper = df_clean[col].quantile([OUTLIER_PERCENTILES[0]/100, OUTLIER_PERCENTILES[1]/100])
            df_clean[col] = df_clean[col].clip(lower, upper)

    return df_clean


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['has_referral'] = (
        (df['referral_id'].notna()) | (df['joined_through_referral'] != 'No')
    ).astype(int)

    df['joining_date'] = pd.to_datetime(df['joining_date'], errors='coerce')
    ref_date = df['joining_date'].max()
    df['days_since_joining'] = (ref_date - df['joining_date']).dt.days

    df['last_visit_hour'] = df['last_visit_time'].astype(str).str[:2].astype(float)

    membership_map = {
        'No Membership': 0,
        'Basic Membership': 1,
        'Silver Membership': 2,
        'Gold Membership': 3,
        'Premium Membership': 4,
        'Platinum Membership': 5
    }
    df['membership_tier'] = df['membership_category'].map(membership_map)

    df = df.drop(columns=['referral_id', 'joined_through_referral', 'joining_date',
                         'last_visit_time', 'membership_category'])

    return df


def impute_remaining_missing(df: pd.DataFrame, num_cols: list, cat_cols: list) -> pd.DataFrame:
    df = df.copy()

    if any(df[num_cols].isnull().any()):
        imputer = KNNImputer(n_neighbors=5)
        df[num_cols] = imputer.fit_transform(df[num_cols])

    for col in cat_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)

    return df


def load_and_preprocess(train_path: str, test_path: str, save_cleaned: bool = True,
                        train_val_split: float = 0.8) -> tuple:
    import os
    from sklearn.model_selection import train_test_split

    train, test = load_and_clean_data(train_path, test_path)

    for df in [train, test]:
        for col in IDENTIFIERS_TO_DROP:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

    train = engineer_features(train)
    test = engineer_features(test)
    train = process_target(train)

    NUMERIC_COLS = [
        'age', 'days_since_last_login', 'avg_time_spent',
        'avg_transaction_value', 'avg_frequency_login_days',
        'points_in_wallet', 'days_since_joining',
        'last_visit_hour', 'membership_tier', 'has_referral'
    ]
    CATEGORICAL_COLS = [
        'gender', 'region_category', 'preferred_offer_types',
        'medium_of_operation', 'internet_option',
        'used_special_discount', 'offer_application_preference',
        'past_complaint', 'complaint_status', 'feedback'
    ]

    NUMERIC_COLS = [c for c in NUMERIC_COLS if c in train.columns]
    CATEGORICAL_COLS = [c for c in CATEGORICAL_COLS if c in train.columns]

    train_full, _ = filter_low_quality_rows(train, test)
    train_full = remove_outliers(train_full, NUMERIC_COLS)
    train_full = impute_remaining_missing(train_full, NUMERIC_COLS, CATEGORICAL_COLS)

    all_features = NUMERIC_COLS + CATEGORICAL_COLS

    # Split train into train/validation (80/20)
    train_data, val_data = train_test_split(
        train_full,
        test_size=(1 - train_val_split),
        random_state=42,
        stratify=train_full['churn_risk_score']
    )

    train_cleaned = train_data[all_features + ['churn_risk_score']].copy().reset_index(drop=True)
    val_cleaned = val_data[all_features + ['churn_risk_score']].copy().reset_index(drop=True)

    if save_cleaned:
        train_dir = os.path.dirname(train_path)
        train_cleaned_path = os.path.join(train_dir, 'train_dataset_cleaned.csv')
        test_cleaned_path = os.path.join(train_dir, 'test_dataset_cleaned.csv')
        train_cleaned.to_csv(train_cleaned_path, index=False)
        val_cleaned.to_csv(test_cleaned_path, index=False)
        print(f"Saved: {train_cleaned_path} ({len(train_cleaned)} samples)")
        print(f"Saved: {test_cleaned_path} ({len(val_cleaned)} samples)")

    return train_cleaned, val_cleaned


if __name__ == "__main__":
    load_and_preprocess('data/churn/train_dataset.csv', 'data/churn/test_dataset.csv')
