"""
Standalone Preprocessing Analysis for Churn Dataset
====================================================
Runs preprocessing with detailed tracking without package imports.
"""

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings('ignore')

# Configuration from preprocess_data.py
MAX_MISSING_VALUES_PER_ROW = 3
OUTLIER_PERCENTILES = (1, 99)
COLS_TO_DROP = ['customer_id', 'Name', 'security_no', 'referral_id']

def load_and_clean_data(train_path: str, test_path: str) -> tuple:
    """Load data and replace placeholders."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # Replace placeholders with NaN
    placeholders = ['?', 'Error', 'xxxxxxxx', np.nan, -999]
    train = train.replace(placeholders, np.nan)
    test = test.replace(placeholders, np.nan)

    # Convert avg_frequency_login_days to numeric
    for df in [train, test]:
        df['avg_frequency_login_days'] = pd.to_numeric(df['avg_frequency_login_days'], errors='coerce')

    return train, test

def filter_low_quality_rows(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    """Drop rows with too many missing values."""
    train_missing = train.isnull().sum(axis=1)
    test_missing = test.isnull().sum(axis=1)

    train_clean = train[train_missing <= MAX_MISSING_VALUES_PER_ROW].reset_index(drop=True)
    test_clean = test[test_missing <= MAX_MISSING_VALUES_PER_ROW].reset_index(drop=True)

    return train_clean, test_clean

def process_target(df: pd.DataFrame, target_col: str = 'churn_risk_score') -> tuple:
    """Handle -1 target values based on prevalence."""
    neg1_count = (df[target_col] == -1).sum()
    total_count = len(df)
    neg1_pct = 100 * neg1_count / total_count

    if neg1_pct > 10:
        df[target_col] = df[target_col].replace(-1, 6)
    else:
        df = df[df[target_col] != -1].reset_index(drop=True)

    return df

def remove_outliers(df: pd.DataFrame, num_cols: list) -> pd.DataFrame:
    """Clip numeric values to 1st/99th percentiles."""
    df_clean = df.copy()

    for col in num_cols:
        if col in df_clean.columns:
            lower, upper = df_clean[col].quantile([OUTLIER_PERCENTILES[0]/100, OUTLIER_PERCENTILES[1]/100])
            df_clean[col] = df_clean[col].clip(lower, upper)

    return df_clean

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering."""
    df = df.copy()

    df['has_referral'] = (df['joined_through_referral'] != 'No').astype(int)
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

    df = df.drop(columns=['joined_through_referral', 'joining_date',
                         'last_visit_time', 'membership_category'])

    return df

def impute_remaining_missing(df: pd.DataFrame, num_cols: list, cat_cols: list) -> pd.DataFrame:
    """KNN imputation for numeric, mode for categorical."""
    df = df.copy()

    if any(df[num_cols].isnull().any()):
        imputer = KNNImputer(n_neighbors=5)
        df[num_cols] = imputer.fit_transform(df[num_cols])

    for col in cat_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)

    return df

def analyze_preprocessing():
    """Run preprocessing with detailed tracking and analysis."""

    print("=" * 80)
    print("COMPREHENSIVE PREPROCESSING ANALYSIS")
    print("=" * 80)

    # Load raw data
    train_path = 'data/churn/train_dataset.csv'
    test_path = 'data/churn/test_dataset.csv'

    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)

    print(f"\n📊 INITIAL DATA OVERVIEW")
    print(f"{'='*80}")
    print(f"Train: {train_raw.shape[0]:,} rows × {train_raw.shape[1]} columns")
    print(f"Test:  {test_raw.shape[0]:,} rows × {test_raw.shape[1]} columns")

    # Count placeholder values BEFORE replacing
    placeholders = ['?', 'Error', 'xxxxxxxx', -999]
    train_placeholders = 0
    test_placeholders = 0

    for placeholder in placeholders:
        train_placeholders += (train_raw == placeholder).sum().sum()
        test_placeholders += (test_raw == placeholder).sum().sum()

    # Track missing values before
    train_missing_before = train_raw.isnull().sum().sum()
    test_missing_before = test_raw.isnull().sum().sum()

    print(f"\n🔍 PLACEHOLDER & MISSING VALUE DETECTION")
    print(f"{'='*80}")
    print(f"Train placeholders (?, Error, xxxxxxxx, -999): {train_placeholders:,}")
    print(f"Test placeholders (?, Error, xxxxxxxx, -999):  {test_placeholders:,}")
    print(f"Train explicit NaN values:                      {train_missing_before:,}")
    print(f"Test explicit NaN values:                       {test_missing_before:,}")
    print(f"\nTotal bad values to handle: {train_placeholders + test_placeholders + train_missing_before + test_missing_before:,}")

    # Check missing values per column initially
    print(f"\n📋 TOP 15 COLUMNS WITH MISSING/PLACEHOLDER VALUES (Train)")
    print(f"{'='*80}")

    # Count both missing and placeholders per column
    train_issues = train_raw.isnull().sum()
    for placeholder in placeholders:
        train_issues = train_issues + (train_raw == placeholder).sum()

    train_issues_sorted = train_issues.sort_values(ascending=False).head(15)
    for col, count in train_issues_sorted.items():
        pct = 100 * count / len(train_raw)
        print(f"{col:45s}: {count:5,} ({pct:5.1f}%)")

    # Analyze target distribution
    print(f"\n🎯 TARGET DISTRIBUTION (Before Processing)")
    print(f"{'='*80}")
    target_counts = train_raw['churn_risk_score'].value_counts().sort_index()
    for val, count in target_counts.items():
        pct = 100 * count / len(train_raw)
        print(f"Class {val:2d}: {count:6,} ({pct:5.1f}%)")

    # Count -1 values in target
    neg1_count = (train_raw['churn_risk_score'] == -1).sum()
    neg1_pct = 100 * neg1_count / len(train_raw)
    print(f"\n⚠️  Target -1 values: {neg1_count:,} ({neg1_pct:.1f}%) - ", end="")
    if neg1_pct > 10:
        print("Will be KEPT as separate class")
    else:
        print("Will be REMOVED (<10%)")

    # Analyze rows with excessive missing values
    train_missing_per_row = train_raw.isnull().sum(axis=1)
    test_missing_per_row = test_raw.isnull().sum(axis=1)

    print(f"\n🧹 ROW QUALITY ASSESSMENT")
    print(f"{'='*80}")
    train_bad_rows = (train_missing_per_row > 3).sum()
    test_bad_rows = (test_missing_per_row > 3).sum()
    print(f"Train - Rows with >3 missing: {train_bad_rows:,} ({100*train_bad_rows/len(train_raw):.1f}%) - WILL BE DELETED")
    print(f"Test  - Rows with >3 missing: {test_bad_rows:,} ({100*test_bad_rows/len(test_raw):.1f}%) - WILL BE DELETED")

    # Distribution of missing values per row
    print(f"\nDistribution of missing values per row (Train):")
    for n_missing in range(0, min(11, train_missing_per_row.max() + 1)):
        count = (train_missing_per_row == n_missing).sum()
        pct = 100 * count / len(train_raw)
        marker = " ← WILL DELETE" if n_missing > 3 else " ← WILL KEEP"
        print(f"  {n_missing} missing: {count:6,} ({pct:5.1f}%){marker}")

    # Run preprocessing
    print(f"\n{'='*80}")
    print("RUNNING PREPROCESSING PIPELINE...")
    print(f"{'='*80}\n")

    # Step 1: Load and clean
    train, test = load_and_clean_data(train_path, test_path)
    print(f"Step 1: Replaced placeholders with NaN")

    # Step 2: Filter low-quality rows
    train_before_filter = len(train)
    test_before_filter = len(test)
    train, test = filter_low_quality_rows(train, test)
    train_deleted = train_before_filter - len(train)
    test_deleted = test_before_filter - len(test)
    print(f"Step 2: Deleted {train_deleted:,} train rows, {test_deleted:,} test rows (>3 missing)")

    # Step 3: Process target
    train_before_target = len(train)
    train = process_target(train)
    target_deleted = train_before_target - len(train)
    print(f"Step 3: Processed target (-1 handling): deleted {target_deleted:,} rows")

    # Step 4: Feature engineering
    train = engineer_features(train)
    test = engineer_features(test)
    print(f"Step 4: Engineered features")

    # Step 5: Define columns
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

    # Step 6: Remove outliers (count clipped values)
    print(f"\nStep 5: Outlier clipping (1st/99th percentiles):")
    total_clipped = 0
    for col in NUMERIC_COLS:
        if col in train.columns:
            lower, upper = train[col].quantile([0.01, 0.99])
            n_outliers = ((train[col] < lower) | (train[col] > upper)).sum()
            total_clipped += n_outliers
            if n_outliers > 0:
                print(f"  {col:30s}: {n_outliers:5,} values clipped")
    print(f"  Total outliers clipped: {total_clipped:,}")

    train = remove_outliers(train, NUMERIC_COLS)

    # Step 7: Drop identifiers
    cols_to_drop_train = [c for c in COLS_TO_DROP if c in train.columns]
    cols_to_drop_test = [c for c in COLS_TO_DROP if c in test.columns]
    train = train.drop(columns=cols_to_drop_train)
    test = test.drop(columns=cols_to_drop_test)
    print(f"\nStep 6: Dropped identifier columns: {cols_to_drop_train}")

    # Step 8: Count missing before imputation
    train_missing_before_impute = train[NUMERIC_COLS + CATEGORICAL_COLS].isnull().sum().sum()
    test_missing_before_impute = test[NUMERIC_COLS + CATEGORICAL_COLS].isnull().sum().sum()

    print(f"\nStep 7: KNN Imputation")
    print(f"  Missing values before imputation:")
    print(f"    Train: {train_missing_before_impute:,}")
    print(f"    Test:  {test_missing_before_impute:,}")
    print(f"    Total: {train_missing_before_impute + test_missing_before_impute:,} values TO BE IMPUTED")

    # Detailed missing by column before imputation
    print(f"\n  Missing by column (Train, before imputation):")
    for col in NUMERIC_COLS + CATEGORICAL_COLS:
        missing = train[col].isnull().sum()
        if missing > 0:
            print(f"    {col:40s}: {missing:5,} ({100*missing/len(train):5.1f}%)")

    train = impute_remaining_missing(train, NUMERIC_COLS, CATEGORICAL_COLS)
    test = impute_remaining_missing(test, NUMERIC_COLS, CATEGORICAL_COLS)

    # Extract final data
    X_train = train[NUMERIC_COLS + CATEGORICAL_COLS]
    y_train = train['churn_risk_score']
    X_test = test[NUMERIC_COLS + CATEGORICAL_COLS]

    # Verify no missing values remain
    train_missing_after = X_train.isnull().sum().sum()
    test_missing_after = X_test.isnull().sum().sum()

    print(f"\n  Missing values after imputation:")
    print(f"    Train: {train_missing_after:,}")
    print(f"    Test:  {test_missing_after:,}")

    # Final summary
    print(f"\n{'='*80}")
    print(f"📉 DATA TRANSFORMATION SUMMARY")
    print(f"{'='*80}")

    train_rows_lost = len(train_raw) - len(y_train)
    test_rows_lost = len(test_raw) - len(X_test)
    train_retention = 100 * len(y_train) / len(train_raw)
    test_retention = 100 * len(X_test) / len(test_raw)

    print(f"\n1. ROWS DELETED:")
    print(f"   Train: {len(train_raw):,} → {len(y_train):,} (deleted {train_rows_lost:,}, retained {train_retention:.1f}%)")
    print(f"   Test:  {len(test_raw):,} → {len(X_test):,} (deleted {test_rows_lost:,}, retained {test_retention:.1f}%)")

    print(f"\n2. VALUES REPLACED/IMPUTED:")
    total_bad_values = train_placeholders + test_placeholders + train_missing_before + test_missing_before
    values_imputed = train_missing_before_impute + test_missing_before_impute
    print(f"   Placeholders found and replaced: {train_placeholders + test_placeholders:,}")
    print(f"   Values imputed by KNN/mode:      {values_imputed:,}")
    print(f"   Outliers clipped:                {total_clipped:,}")
    print(f"   Total bad values handled:        {total_bad_values:,}")

    # Final target distribution
    print(f"\n3. FINAL TARGET DISTRIBUTION:")
    final_target = y_train.value_counts().sort_index()
    for val, count in final_target.items():
        pct = 100 * count / len(y_train)
        print(f"   Class {int(val):2d}: {count:6,} ({pct:5.1f}%)")
    print(f"   Total classes: {len(final_target)}")

    # Feature info
    print(f"\n4. FEATURE ENGINEERING:")
    print(f"   Numeric features:     {len(NUMERIC_COLS)}")
    print(f"   Categorical features: {len(CATEGORICAL_COLS)}")
    print(f"   Total features:       {X_train.shape[1]}")

    # Data quality score
    print(f"\n{'='*80}")
    print(f"📈 OVERALL DATA QUALITY ASSESSMENT")
    print(f"{'='*80}")

    retention_score = (train_retention + test_retention) / 2
    missing_resolution_rate = 100 * (1 - (train_missing_after + test_missing_after) / max(1, total_bad_values))

    print(f"\n✓ Data Retention Rate:        {retention_score:.1f}%")
    print(f"✓ Missing Value Resolution:   {missing_resolution_rate:.1f}%")
    print(f"✓ Final Train Size:           {len(y_train):,} samples")
    print(f"✓ Final Test Size:            {len(X_test):,} samples")
    print(f"✓ Outliers Handled:           {total_clipped:,} values clipped")
    print(f"✓ Values Imputed:             {values_imputed:,}")

    # Class imbalance
    class_balance = final_target.max() / final_target.min()
    print(f"✓ Class Balance Ratio:        {class_balance:.2f}x")

    # Quality insights
    print(f"\n💡 QUALITY INSIGHTS:")
    print(f"{'='*80}")

    if retention_score < 85:
        print(f"⚠️  Significant data loss ({100-retention_score:.1f}% deleted)")
    elif retention_score < 95:
        print(f"⚠️  Moderate data loss ({100-retention_score:.1f}% deleted)")
    else:
        print(f"✓ Excellent data retention ({100-retention_score:.1f}% deleted)")

    if missing_resolution_rate < 95:
        print(f"⚠️  Some bad values may remain")
    else:
        print(f"✓ Excellent data cleaning (all bad values resolved)")

    if class_balance > 10:
        print(f"⚠️  High class imbalance ({class_balance:.1f}x) - consider class weights")
    elif class_balance > 3:
        print(f"⚠️  Moderate class imbalance ({class_balance:.1f}x)")
    else:
        print(f"✓ Good class balance ({class_balance:.1f}x)")

    imputation_rate = 100 * values_imputed / len(y_train)
    if imputation_rate > 5:
        print(f"⚠️  High imputation rate ({imputation_rate:.1f}% of final dataset)")
    elif imputation_rate > 2:
        print(f"⚠️  Moderate imputation ({imputation_rate:.1f}% of final dataset)")
    else:
        print(f"✓ Low imputation rate ({imputation_rate:.1f}% of final dataset)")

    print(f"\n{'='*80}")
    print("PREPROCESSING ANALYSIS COMPLETE")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    analyze_preprocessing()
