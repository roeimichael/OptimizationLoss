"""
Smart Column Analysis: Separate Informative from Meaningless
=============================================================
Identify which columns are:
1. IDENTIFIERS (drop entirely)
2. CONVERTIBLE (turn into binary/simple features)
3. INFORMATIVE (keep and handle missing values carefully)
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def load_raw_data():
    """Load raw data."""
    train = pd.read_csv('data/churn/train_dataset.csv')
    test = pd.read_csv('data/churn/test_dataset.csv')

    # Replace placeholders
    placeholders = ['?', 'Error', 'xxxxxxxx', np.nan, -999]
    train = train.replace(placeholders, np.nan)
    test = test.replace(placeholders, np.nan)

    # Convert to numeric
    for df in [train, test]:
        df['avg_frequency_login_days'] = pd.to_numeric(df['avg_frequency_login_days'], errors='coerce')

    # Remove -1 target
    train = train[train['churn_risk_score'] != -1].reset_index(drop=True)

    return train, test

def analyze_column_types(train):
    """Categorize columns by type and usefulness."""
    print("\n" + "="*80)
    print("COLUMN CATEGORIZATION ANALYSIS")
    print("="*80)

    columns_analysis = {}

    # Analyze each column
    for col in train.columns:
        if col == 'churn_risk_score':
            continue

        missing_pct = 100 * train[col].isnull().sum() / len(train)
        n_unique = train[col].nunique()
        dtype = train[col].dtype

        columns_analysis[col] = {
            'missing_pct': missing_pct,
            'n_unique': n_unique,
            'dtype': dtype,
            'category': None,
            'recommendation': None
        }

        # Categorize
        if col in ['customer_id', 'Name', 'security_no']:
            columns_analysis[col]['category'] = 'IDENTIFIER'
            columns_analysis[col]['recommendation'] = 'DROP - Pure identifier, no predictive value'

        elif col == 'referral_id':
            columns_analysis[col]['category'] = 'CONVERTIBLE'
            columns_analysis[col]['recommendation'] = 'CONVERT to binary has_referral (is not NaN)'

        elif col == 'joined_through_referral':
            columns_analysis[col]['category'] = 'CONVERTIBLE'
            columns_analysis[col]['recommendation'] = 'CONVERT to binary has_referral (not "No")'

        elif col in ['joining_date', 'last_visit_time']:
            columns_analysis[col]['category'] = 'CONVERTIBLE'
            columns_analysis[col]['recommendation'] = f'CONVERT to derived features (already done)'

        elif col == 'membership_category':
            columns_analysis[col]['category'] = 'CONVERTIBLE'
            columns_analysis[col]['recommendation'] = 'CONVERT to ordinal membership_tier (already done)'

        # Behavioral/informative columns
        elif col in ['age', 'days_since_last_login', 'avg_time_spent', 'avg_transaction_value',
                     'avg_frequency_login_days', 'points_in_wallet']:
            columns_analysis[col]['category'] = 'INFORMATIVE_NUMERIC'
            columns_analysis[col]['recommendation'] = f'KEEP - Critical behavioral metric ({missing_pct:.1f}% missing)'

        elif col in ['gender', 'preferred_offer_types', 'internet_option',
                     'used_special_discount', 'offer_application_preference',
                     'past_complaint', 'complaint_status', 'feedback']:
            columns_analysis[col]['category'] = 'INFORMATIVE_CATEGORICAL'
            columns_analysis[col]['recommendation'] = f'KEEP - Valuable categorical ({missing_pct:.1f}% missing)'

        # Questionable columns
        elif col in ['region_category', 'medium_of_operation']:
            if missing_pct > 10:
                columns_analysis[col]['category'] = 'QUESTIONABLE'
                columns_analysis[col]['recommendation'] = f'CONSIDER DROPPING - High missing ({missing_pct:.1f}%) and moderate value'
            else:
                columns_analysis[col]['category'] = 'INFORMATIVE_CATEGORICAL'
                columns_analysis[col]['recommendation'] = f'KEEP - Categorical ({missing_pct:.1f}% missing)'

    return columns_analysis

def print_column_analysis(columns_analysis):
    """Print categorized columns."""
    print("\n" + "="*80)
    print("CATEGORIZED COLUMNS")
    print("="*80)

    # Group by category
    categories = {}
    for col, info in columns_analysis.items():
        cat = info['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((col, info))

    # Print each category
    for cat in ['IDENTIFIER', 'CONVERTIBLE', 'INFORMATIVE_NUMERIC', 'INFORMATIVE_CATEGORICAL', 'QUESTIONABLE']:
        if cat not in categories:
            continue

        print(f"\n{cat}:")
        print("-" * 80)

        for col, info in sorted(categories[cat], key=lambda x: x[1]['missing_pct'], reverse=True):
            print(f"  {col:40s} | Missing: {info['missing_pct']:5.1f}% | Unique: {info['n_unique']:5d}")
            print(f"    → {info['recommendation']}")

def smart_preprocessing(train, test):
    """Apply smart preprocessing based on column analysis."""
    print("\n" + "="*80)
    print("SMART PREPROCESSING")
    print("="*80)

    print("\n1. DROPPING IDENTIFIERS...")
    identifiers = ['customer_id', 'Name', 'security_no']
    train_clean = train.drop(columns=identifiers)
    test_clean = test.drop(columns=[c for c in identifiers if c in test.columns])
    print(f"   Dropped: {identifiers}")

    print("\n2. FEATURE ENGINEERING (Binary conversions)...")

    # has_referral (from both referral_id and joined_through_referral)
    train_clean['has_referral'] = (
        (train_clean['referral_id'].notna()) |
        (train_clean['joined_through_referral'] != 'No')
    ).astype(int)
    test_clean['has_referral'] = (
        (test_clean['referral_id'].notna()) |
        (test_clean['joined_through_referral'] != 'No')
    ).astype(int)
    print("   Created: has_referral (binary)")

    # Date features
    train_clean['joining_date'] = pd.to_datetime(train_clean['joining_date'], errors='coerce')
    test_clean['joining_date'] = pd.to_datetime(test_clean['joining_date'], errors='coerce')

    ref_date_train = train_clean['joining_date'].max()
    ref_date_test = test_clean['joining_date'].max()

    train_clean['days_since_joining'] = (ref_date_train - train_clean['joining_date']).dt.days
    test_clean['days_since_joining'] = (ref_date_test - test_clean['joining_date']).dt.days
    print("   Created: days_since_joining")

    # Time feature
    train_clean['last_visit_hour'] = train_clean['last_visit_time'].astype(str).str[:2].astype(float)
    test_clean['last_visit_hour'] = test_clean['last_visit_time'].astype(str).str[:2].astype(float)
    print("   Created: last_visit_hour")

    # Membership tier
    membership_map = {
        'No Membership': 0,
        'Basic Membership': 1,
        'Silver Membership': 2,
        'Gold Membership': 3,
        'Premium Membership': 4,
        'Platinum Membership': 5
    }
    train_clean['membership_tier'] = train_clean['membership_category'].map(membership_map)
    test_clean['membership_tier'] = test_clean['membership_category'].map(membership_map)
    print("   Created: membership_tier (ordinal)")

    # Drop original columns that were converted
    to_drop_converted = ['referral_id', 'joined_through_referral', 'joining_date',
                         'last_visit_time', 'membership_category']
    train_clean = train_clean.drop(columns=to_drop_converted)
    test_clean = test_clean.drop(columns=[c for c in to_drop_converted if c in test_clean.columns])
    print(f"   Dropped converted columns: {to_drop_converted}")

    print("\n3. EVALUATING QUESTIONABLE COLUMNS...")

    questionable = ['region_category', 'medium_of_operation']
    for col in questionable:
        if col in train_clean.columns:
            missing_pct = 100 * train_clean[col].isnull().sum() / len(train_clean)
            print(f"   {col}: {missing_pct:.1f}% missing")

            if missing_pct > 10:
                print(f"     → HIGH MISSING - Consider dropping")
            else:
                print(f"     → OK - Keep")

    return train_clean, test_clean

def evaluate_after_smart_cleaning(train_clean, test_clean):
    """Evaluate missing values after smart cleaning."""
    print("\n" + "="*80)
    print("AFTER SMART CLEANING - REMAINING MISSING VALUES")
    print("="*80)

    # Define informative columns
    INFORMATIVE_NUMERIC = [
        'age', 'days_since_last_login', 'avg_time_spent',
        'avg_transaction_value', 'avg_frequency_login_days',
        'points_in_wallet', 'days_since_joining',
        'last_visit_hour', 'membership_tier', 'has_referral'
    ]

    INFORMATIVE_CATEGORICAL = [
        'gender', 'preferred_offer_types', 'internet_option',
        'used_special_discount', 'offer_application_preference',
        'past_complaint', 'complaint_status', 'feedback'
    ]

    # Filter to existing columns
    INFORMATIVE_NUMERIC = [c for c in INFORMATIVE_NUMERIC if c in train_clean.columns]
    INFORMATIVE_CATEGORICAL = [c for c in INFORMATIVE_CATEGORICAL if c in train_clean.columns]

    all_informative = INFORMATIVE_NUMERIC + INFORMATIVE_CATEGORICAL

    print(f"\nInformative columns: {len(all_informative)}")
    print(f"  Numeric: {len(INFORMATIVE_NUMERIC)}")
    print(f"  Categorical: {len(INFORMATIVE_CATEGORICAL)}")

    # Check missing in informative columns only
    print(f"\nMissing values in INFORMATIVE columns:")
    print("-" * 80)

    total_missing_train = 0
    total_missing_test = 0

    for col in all_informative:
        missing_train = train_clean[col].isnull().sum()
        missing_test = test_clean[col].isnull().sum() if col in test_clean.columns else 0

        if missing_train > 0 or missing_test > 0:
            pct_train = 100 * missing_train / len(train_clean)
            pct_test = 100 * missing_test / len(test_clean) if col in test_clean.columns else 0

            col_type = "NUM" if col in INFORMATIVE_NUMERIC else "CAT"
            print(f"  [{col_type}] {col:40s}: Train {missing_train:5d} ({pct_train:5.1f}%), Test {missing_test:5d} ({pct_test:5.1f}%)")

            total_missing_train += missing_train
            total_missing_test += missing_test

    print("-" * 80)
    print(f"TOTAL missing in informative columns:")
    print(f"  Train: {total_missing_train:,} values")
    print(f"  Test:  {total_missing_test:,} values")

    # Per-row missing analysis
    print(f"\n" + "="*80)
    print("PER-ROW MISSING VALUE DISTRIBUTION (Informative columns only)")
    print("="*80)

    missing_per_row_train = train_clean[all_informative].isnull().sum(axis=1)
    missing_per_row_test = test_clean[all_informative].isnull().sum(axis=1) if len(test_clean) > 0 else pd.Series()

    print(f"\nTrain:")
    for n_missing in range(0, min(6, missing_per_row_train.max() + 1)):
        count = (missing_per_row_train == n_missing).sum()
        pct = 100 * count / len(train_clean)
        print(f"  {n_missing} missing: {count:6,} ({pct:5.1f}%)")

    if len(missing_per_row_test) > 0:
        print(f"\nTest:")
        for n_missing in range(0, min(6, missing_per_row_test.max() + 1)):
            count = (missing_per_row_test == n_missing).sum()
            pct = 100 * count / len(test_clean)
            print(f"  {n_missing} missing: {count:6,} ({pct:5.1f}%)")

    # Strategy recommendation
    print(f"\n" + "="*80)
    print("IMPUTATION STRATEGY RECOMMENDATION")
    print("="*80)

    # Filter rows with >3 missing
    train_good_quality = train_clean[missing_per_row_train <= 3].reset_index(drop=True)
    test_good_quality = test_clean[missing_per_row_test <= 3].reset_index(drop=True) if len(test_clean) > 0 else test_clean

    print(f"\nAfter filtering rows with >3 missing:")
    print(f"  Train: {len(train_clean):,} → {len(train_good_quality):,} ({100*len(train_good_quality)/len(train_clean):.1f}% retained)")
    if len(test_clean) > 0:
        print(f"  Test:  {len(test_clean):,} → {len(test_good_quality):,} ({100*len(test_good_quality)/len(test_clean):.1f}% retained)")

    # Remaining missing in good-quality rows
    remaining_missing_train = train_good_quality[all_informative].isnull().sum().sum()
    remaining_missing_test = test_good_quality[all_informative].isnull().sum().sum() if len(test_good_quality) > 0 else 0

    imputation_rate = 100 * remaining_missing_train / len(train_good_quality) if len(train_good_quality) > 0 else 0

    print(f"\nRemaining missing values (in good-quality rows):")
    print(f"  Train: {remaining_missing_train:,}")
    print(f"  Test:  {remaining_missing_test:,}")
    print(f"  Imputation rate: {imputation_rate:.1f}% values per sample")

    print(f"\n" + "="*80)
    print("RECOMMENDATION:")
    print("="*80)

    if imputation_rate < 10:
        print(f"✅ EXCELLENT: Imputation rate is {imputation_rate:.1f}%")
        print(f"   → This is DEFENSIBLE in thesis (< 10%)")
        print(f"   → Use KNN imputation for remaining {remaining_missing_train:,} values")
    elif imputation_rate < 20:
        print(f"✓ GOOD: Imputation rate is {imputation_rate:.1f}%")
        print(f"   → This is ACCEPTABLE in thesis (< 20%)")
        print(f"   → Use KNN imputation for remaining {remaining_missing_train:,} values")
    elif imputation_rate < 30:
        print(f"⚠️  MODERATE: Imputation rate is {imputation_rate:.1f}%")
        print(f"   → Consider dropping more columns or rows")
        print(f"   → Or accept moderate imputation with good justification")
    else:
        print(f"❌ HIGH: Imputation rate is {imputation_rate:.1f}%")
        print(f"   → Need to drop more columns or use complete case analysis")

    return train_good_quality, test_good_quality, all_informative

def main():
    print("="*80)
    print("SMART COLUMN ANALYSIS - SEPARATE INFORMATIVE FROM MEANINGLESS")
    print("="*80)

    # Load data
    train, test = load_raw_data()
    print(f"\nInitial: Train={len(train):,}, Test={len(test):,}")

    # Analyze columns
    columns_analysis = analyze_column_types(train)
    print_column_analysis(columns_analysis)

    # Apply smart preprocessing
    train_clean, test_clean = smart_preprocessing(train, test)

    # Evaluate what remains
    train_final, test_final, informative_cols = evaluate_after_smart_cleaning(train_clean, test_clean)

    print(f"\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nFinal dataset (after filtering >3 missing):")
    print(f"  Train: {len(train_final):,} samples")
    print(f"  Test:  {len(test_final):,} samples")
    print(f"  Features: {len(informative_cols)}")

    # Show which columns remain
    print(f"\nInformative columns to use:")
    for col in informative_cols:
        print(f"  - {col}")

if __name__ == "__main__":
    main()
