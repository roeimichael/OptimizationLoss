"""
Churn Dataset Preprocessing
============================
Cleans and preprocesses churn prediction datasets.

Processing steps:
1. Drop pure identifiers (customer_id, Name, security_no)
2. Convert high-missing columns to informative features
3. Filter low-quality rows (>3 missing values)
4. Remove outliers (1st/99th percentile clipping)
5. Impute remaining missing (KNN for numeric, mode for categorical)

Usage:
    from src.utils.preprocess_data import load_and_preprocess
    train_df, test_df = load_and_preprocess('data/churn/train_dataset.csv',
                                              'data/churn/test_dataset.csv')
"""

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
MAX_MISSING_VALUES_PER_ROW = 3  # Drop rows with >3 missing values
OUTLIER_PERCENTILES = (1, 99)   # Clip to 1st/99th percentiles
RANDOM_SEED = 42

# Columns to drop (identifiers - no predictive value)
IDENTIFIERS_TO_DROP = ['customer_id', 'Name', 'security_no']

# Note: region_category and medium_of_operation have lower importance (5-6% of average)
# but are kept as they still provide some signal. Missing values (14.8%, 14.5%)
# will be filled with mode imputation.

# ============================================================================
# LOAD & INITIAL CLEANING
# ============================================================================
def load_and_clean_data(train_path: str, test_path: str) -> tuple:
    """Load data and replace placeholders."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    print(f"Initial shapes - Train: {train.shape}, Test: {test.shape}")

    # Replace placeholders with NaN
    placeholders = ['?', 'Error', 'xxxxxxxx', np.nan, -999]
    train = train.replace(placeholders, np.nan)
    test = test.replace(placeholders, np.nan)

    # Convert avg_frequency_login_days to numeric
    for df in [train, test]:
        df['avg_frequency_login_days'] = pd.to_numeric(df['avg_frequency_login_days'], errors='coerce')

    return train, test

# ============================================================================
# ROW QUALITY FILTERING
# ============================================================================
def filter_low_quality_rows(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    """
    Drop rows with too many missing values.

    [THESIS NOTE] Prioritizes data quality over quantity. Rows with >3 missing
    values likely have fundamental data collection issues.
    """
    print("\\n=== Filtering Low-Quality Rows ===")

    # Count missing values per row
    train_missing = train.isnull().sum(axis=1)
    test_missing = test.isnull().sum(axis=1)

    print(f"Train - Rows with >3 missing: { (train_missing > MAX_MISSING_VALUES_PER_ROW).sum() }")
    print(f"Test  - Rows with >3 missing: { (test_missing > MAX_MISSING_VALUES_PER_ROW).sum() }")

    # Filter
    train_clean = train[train_missing <= MAX_MISSING_VALUES_PER_ROW].reset_index(drop=True)
    test_clean = test[test_missing <= MAX_MISSING_VALUES_PER_ROW].reset_index(drop=True)

    print(f"After filtering - Train: {train_clean.shape}, Test: {test_clean.shape}")

    return train_clean, test_clean

# ============================================================================
# TARGET HANDLING
# ============================================================================
def process_target(df: pd.DataFrame, target_col: str = 'churn_risk_score') -> tuple:
    """
    Handle -1 target values based on prevalence.

    [THESIS NOTE] Data-driven decision: if -1 > 10% of data, treat as separate class.
    Otherwise, remove those rows to avoid ambiguity.
    """
    print("\\n=== Target Processing ===")

    neg1_count = (df[target_col] == -1).sum()
    total_count = len(df)
    neg1_pct = 100 * neg1_count / total_count

    print(f"-1 values: {neg1_count}/{total_count} ({neg1_pct:.1f}%)")

    if neg1_pct > 10:
        print("→ Keeping -1 as separate class (6 total classes)")
        df[target_col] = df[target_col].replace(-1, 6)  # Map to 6 for clean classes
    else:
        print("→ Removing -1 rows (<10%, likely data errors)")
        df = df[df[target_col] != -1].reset_index(drop=True)

    return df

# ============================================================================
# OUTLIER REMOVAL
# ============================================================================
def remove_outliers(df: pd.DataFrame, num_cols: list) -> pd.DataFrame:
    """
    Aggressively filter extreme outliers.

    [THESIS NOTE] Clips numeric values to 1st/99th percentiles to remove
    implausible business values (e.g., negative time spent).
    """
    print("\\n=== Outlier Filtering (1st/99th percentiles) ===")

    df_clean = df.copy()
    n_rows_before = len(df)

    for col in num_cols:
        if col in df_clean.columns:
            lower, upper = df_clean[col].quantile([OUTLIER_PERCENTILES[0]/100, OUTLIER_PERCENTILES[1]/100])
            n_outliers = ((df_clean[col] < lower) | (df_clean[col] > upper)).sum()
            df_clean[col] = df_clean[col].clip(lower, upper)
            print(f"{col}: clipped {n_outliers} outliers")

    n_rows_after = len(df_clean)
    print(f"Total rows after outlier filtering: {n_rows_after} (dropped {n_rows_before - n_rows_after} extreme rows)")

    return df_clean

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Smart feature engineering - convert high-missing columns to informative features.
    Eliminates 48.3% + 14.7% missing by creating single binary indicator.
    """
    df = df.copy()

    # Binary referral feature (combines referral_id + joined_through_referral)
    # referral_id: 48.3% missing → is NOT null = referred
    # joined_through_referral: 14.7% missing → not "No" = referred
    df['has_referral'] = (
        (df['referral_id'].notna()) | (df['joined_through_referral'] != 'No')
    ).astype(int)

    # Date features
    df['joining_date'] = pd.to_datetime(df['joining_date'], errors='coerce')
    ref_date = df['joining_date'].max()
    df['days_since_joining'] = (ref_date - df['joining_date']).dt.days

    # Time feature
    df['last_visit_hour'] = df['last_visit_time'].astype(str).str[:2].astype(float)

    # Natural membership ordinal encoding
    membership_map = {
        'No Membership': 0,
        'Basic Membership': 1,
        'Silver Membership': 2,
        'Gold Membership': 3,
        'Premium Membership': 4,
        'Platinum Membership': 5
    }
    df['membership_tier'] = df['membership_category'].map(membership_map)

    # Drop original columns that were converted
    df = df.drop(columns=['referral_id', 'joined_through_referral', 'joining_date',
                         'last_visit_time', 'membership_category'])

    return df

# ============================================================================
# IMPUTATION (Limited)
# ============================================================================
def impute_remaining_missing(df: pd.DataFrame, num_cols: list, cat_cols: list) -> pd.DataFrame:
    """
    KNN imputation for numeric, mode for categorical (only 1-2 missing per row).

    [THESIS NOTE] KNN preserves relationships between features better than median.
    Only applied after aggressive row filtering, so minimal artificial data.
    """
    print("\\n=== KNN Imputation for Remaining Missing Values ===")

    df = df.copy()

    # Numeric KNN imputation
    if any(df[num_cols].isnull().any()):
        imputer = KNNImputer(n_neighbors=5)
        df[num_cols] = imputer.fit_transform(df[num_cols])

    # Categorical mode imputation
    for col in cat_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)

    missing_summary = df.isnull().sum().sum()
    print(f"Remaining missing values after imputation: {missing_summary}")

    return df

# ============================================================================
# MAIN PIPELINE
# ============================================================================
def load_and_preprocess(train_path: str, test_path: str, save_cleaned: bool = True) -> tuple:
    """
    Load, clean, and preprocess churn datasets.

    Args:
        train_path: Path to training dataset CSV
        test_path: Path to test dataset CSV
        save_cleaned: If True, save cleaned datasets to same folder with '_cleaned' suffix

    Returns:
        (train_df, test_df): Cleaned dataframes with target column
    """
    import os

    print("=" * 70)
    print("CHURN DATASET PREPROCESSING")
    print("=" * 70)

    # Load and clean
    train, test = load_and_clean_data(train_path, test_path)

    # Drop identifiers
    for df in [train, test]:
        for col in IDENTIFIERS_TO_DROP:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

    # Feature engineering
    train = engineer_features(train)
    test = engineer_features(test)

    # Process target
    train = process_target(train)

    # Define feature columns
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

    # Ensure columns exist
    NUMERIC_COLS = [c for c in NUMERIC_COLS if c in train.columns]
    CATEGORICAL_COLS = [c for c in CATEGORICAL_COLS if c in train.columns]

    # Filter low-quality rows
    train, test = filter_low_quality_rows(train, test)

    # Remove outliers
    train = remove_outliers(train, NUMERIC_COLS)

    # Imputation
    train = impute_remaining_missing(train, NUMERIC_COLS, CATEGORICAL_COLS)
    test = impute_remaining_missing(test, NUMERIC_COLS, CATEGORICAL_COLS)

    # Prepare final dataframes (features + target)
    all_features = NUMERIC_COLS + CATEGORICAL_COLS
    train_cleaned = train[all_features + ['churn_risk_score']].copy()
    test_cleaned = test[all_features].copy()

    # Summary
    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)
    print(f"Train: {len(train_cleaned):,} samples × {len(all_features)} features")
    print(f"Test:  {len(test_cleaned):,} samples × {len(all_features)} features")
    print(f"Features: {len(NUMERIC_COLS)} numeric, {len(CATEGORICAL_COLS)} categorical")
    print(f"Missing values: {train_cleaned.isnull().sum().sum() + test_cleaned.isnull().sum().sum()}")

    print("\nTarget distribution:")
    for cls, count in train_cleaned['churn_risk_score'].value_counts().sort_index().items():
        pct = 100 * count / len(train_cleaned)
        print(f"  Class {int(cls)}: {count:,} ({pct:.1f}%)")

    # Save cleaned datasets
    if save_cleaned:
        train_dir = os.path.dirname(train_path)
        train_cleaned_path = os.path.join(train_dir, 'train_dataset_cleaned.csv')
        test_cleaned_path = os.path.join(train_dir, 'test_dataset_cleaned.csv')

        train_cleaned.to_csv(train_cleaned_path, index=False)
        test_cleaned.to_csv(test_cleaned_path, index=False)

        print(f"\n✓ Saved cleaned datasets:")
        print(f"  {train_cleaned_path}")
        print(f"  {test_cleaned_path}")

    print("=" * 70 + "\n")

    return train_cleaned, test_cleaned


# Usage
if __name__ == "__main__":
    train_df, test_df = load_and_preprocess(
        'data/churn/train_dataset.csv',
        'data/churn/test_dataset.csv',
        save_cleaned=True
    )
