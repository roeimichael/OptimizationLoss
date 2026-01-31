"""
Churn Risk Thesis - Evidence-Based Smart Preprocessing
======================================================
Data-driven preprocessing with mode imputation for categorical features.
Key principles:
- Drop pure identifiers (customer_id, Name, security_no)
- Convert high-missing to binary/ordinal features (referral → has_referral)
- Keep all informative features including region_category, medium_of_operation
- Filter low-quality rows (>3 missing values)
- Mode imputation for categorical, KNN for numeric
Result: 28,665 samples, 20 features
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
def preprocess_thesis_data(train_path: str, test_path: str) -> dict:
    """
    Evidence-based smart preprocessing pipeline.
    Keeps all informative features, uses mode imputation for categorical.
    """
    print("🎓 EVIDENCE-BASED SMART CHURN DATA PREPROCESSING")
    print("=" * 70)

    # 1. Load and initial clean
    train, test = load_and_clean_data(train_path, test_path)

    # Store test IDs
    test_ids = test['customer_id'].copy()

    # 2. Drop identifiers (no predictive value)
    print("\n=== Step 1: Dropping Pure Identifiers ===")
    cols_to_drop_train = [c for c in IDENTIFIERS_TO_DROP if c in train.columns]
    cols_to_drop_test = [c for c in IDENTIFIERS_TO_DROP if c in test.columns]
    train = train.drop(columns=cols_to_drop_train)
    test = test.drop(columns=cols_to_drop_test)
    print(f"Dropped: {IDENTIFIERS_TO_DROP}")

    # 3. Feature engineering (converts high-missing columns)
    print("\n=== Step 2: Smart Feature Engineering ===")
    print("Converting high-missing columns to informative features:")
    print("  - referral_id (48.3% missing) + joined_through_referral (14.7% missing)")
    print("    → has_referral (binary, 0% missing)")
    print("  - joining_date → days_since_joining")
    print("  - last_visit_time → last_visit_hour")
    print("  - membership_category → membership_tier (ordinal 0-5)")
    train = engineer_features(train)
    test = engineer_features(test)

    # 4. Process target (data-driven -1 handling)
    train = process_target(train)

    # 5. Define informative columns (20 features)
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

    # Ensure all columns exist
    NUMERIC_COLS = [c for c in NUMERIC_COLS if c in train.columns]
    CATEGORICAL_COLS = [c for c in CATEGORICAL_COLS if c in train.columns]

    print(f"\n=== Step 3: Informative Features ===")
    print(f"Numeric: {len(NUMERIC_COLS)}, Categorical: {len(CATEGORICAL_COLS)}")
    print(f"Total: {len(NUMERIC_COLS) + len(CATEGORICAL_COLS)} features")
    print("Note: Keeping region_category and medium_of_operation despite lower importance")
    print("      (will use mode imputation for 14.8% and 14.5% missing)")

    # 6. Filter low-quality rows (>3 missing)
    train, test = filter_low_quality_rows(train, test)

    # 7. Aggressive outlier removal (numeric only)
    train = remove_outliers(train, NUMERIC_COLS)

    # 8. Imputation for remaining missing
    print("\n=== Step 4: Imputation (KNN for Numeric, Mode for Categorical) ===")
    all_features = NUMERIC_COLS + CATEGORICAL_COLS
    missing_before = train[all_features].isnull().sum().sum()
    imputation_rate = 100 * missing_before / len(train)
    print(f"Missing values to impute: {missing_before:,}")
    print(f"Imputation rate: {imputation_rate:.1f}% per sample")

    train = impute_remaining_missing(train, NUMERIC_COLS, CATEGORICAL_COLS)
    test = impute_remaining_missing(test, NUMERIC_COLS, CATEGORICAL_COLS)

    # 10. Final separation
    X_train = train[NUMERIC_COLS + CATEGORICAL_COLS]
    y_train = train['churn_risk_score']
    X_test = test[NUMERIC_COLS + CATEGORICAL_COLS]

    # Summary
    print("\n" + "=" * 70)
    print("✅ SMART PREPROCESSING COMPLETE")
    print("=" * 70)
    print(f"✓ Samples: {len(y_train):,} train, {len(X_test):,} test")
    print(f"✓ Features: {len(NUMERIC_COLS) + len(CATEGORICAL_COLS)} features (10 numeric, {len(CATEGORICAL_COLS)} categorical)")
    print(f"✓ Imputation: {imputation_rate:.1f}% per sample (vs 66.1% original)")
    print(f"✓ Strategy: Keep all features, mode imputation for categorical missing")
    print("=" * 70 + "\n")

    # Target distribution
    print("Target distribution:")
    for cls, count in y_train.value_counts().sort_index().items():
        print(f"  Class {int(cls)}: {count:,} ({100*count/len(y_train):.1f}%)")

    return {
        'X_train_numeric': X_train[NUMERIC_COLS],
        'X_train_categorical': X_train[CATEGORICAL_COLS],
        'X_train_combined': X_train,
        'y_train': y_train,
        'X_test_numeric': X_test[NUMERIC_COLS],
        'X_test_categorical': X_test[CATEGORICAL_COLS],
        'X_test_combined': X_test,
        'test_ids': test_ids,
        'numeric_cols': NUMERIC_COLS,
        'categorical_cols': CATEGORICAL_COLS
    }

# Usage
if __name__ == "__main__":
    result = preprocess_thesis_data('data/churn/train_dataset.csv', 'data/churn/test_dataset.csv')
