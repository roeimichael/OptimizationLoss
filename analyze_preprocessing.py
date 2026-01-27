"""
Comprehensive Preprocessing Analysis for Churn Dataset
======================================================
This script runs the preprocessing pipeline and provides detailed statistics about:
- Values replaced (placeholders → NaN)
- Rows deleted (quality filtering)
- Values imputed (KNN/mode)
- Outliers clipped
- Overall data quality metrics
"""

import numpy as np
import pandas as pd
import sys
sys.path.append('/home/user/OptimizationLoss')
from src.utils.preprocess_data import *

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
    print(f"Train: {train_raw.shape[0]} rows × {train_raw.shape[1]} columns")
    print(f"Test:  {test_raw.shape[0]} rows × {test_raw.shape[1]} columns")

    # Track missing values before
    train_missing_before = train_raw.isnull().sum().sum()
    test_missing_before = test_raw.isnull().sum().sum()

    # Count placeholder values
    placeholders = ['?', 'Error', 'xxxxxxxx', -999]
    train_placeholders = 0
    test_placeholders = 0

    for placeholder in placeholders:
        train_placeholders += (train_raw == placeholder).sum().sum()
        test_placeholders += (test_raw == placeholder).sum().sum()

    print(f"\n🔍 PLACEHOLDER DETECTION")
    print(f"{'='*80}")
    print(f"Train placeholders found: {train_placeholders:,}")
    print(f"Test placeholders found:  {test_placeholders:,}")
    print(f"Total placeholders:       {train_placeholders + test_placeholders:,}")

    # Check missing values per column initially
    print(f"\n📋 MISSING VALUES BY COLUMN (Top 10 - Train)")
    print(f"{'='*80}")
    missing_cols = train_raw.isnull().sum().sort_values(ascending=False).head(10)
    for col, count in missing_cols.items():
        pct = 100 * count / len(train_raw)
        print(f"{col:40s}: {count:5d} ({pct:5.1f}%)")

    # Analyze target distribution
    print(f"\n🎯 TARGET DISTRIBUTION (Before Processing)")
    print(f"{'='*80}")
    target_counts = train_raw['churn_risk_score'].value_counts().sort_index()
    for val, count in target_counts.items():
        pct = 100 * count / len(train_raw)
        print(f"Class {val:2d}: {count:6d} ({pct:5.1f}%)")

    # Count -1 values in target
    neg1_count = (train_raw['churn_risk_score'] == -1).sum()
    neg1_pct = 100 * neg1_count / len(train_raw)
    print(f"\n⚠️  Target -1 values: {neg1_count:,} ({neg1_pct:.1f}%)")

    # Analyze rows with excessive missing values
    train_missing_per_row = train_raw.isnull().sum(axis=1)
    test_missing_per_row = test_raw.isnull().sum(axis=1)

    print(f"\n🧹 ROW QUALITY ASSESSMENT")
    print(f"{'='*80}")
    print(f"Train - Rows with >3 missing: {(train_missing_per_row > 3).sum():,} ({100*(train_missing_per_row > 3).sum()/len(train_raw):.1f}%)")
    print(f"Test  - Rows with >3 missing: {(test_missing_per_row > 3).sum():,} ({100*(test_missing_per_row > 3).sum()/len(test_raw):.1f}%)")

    # Distribution of missing values per row
    print(f"\nDistribution of missing values per row (Train):")
    for n_missing in range(0, min(11, train_missing_per_row.max() + 1)):
        count = (train_missing_per_row == n_missing).sum()
        pct = 100 * count / len(train_raw)
        print(f"  {n_missing} missing: {count:6d} ({pct:5.1f}%)")

    # Now run the preprocessing pipeline
    print(f"\n{'='*80}")
    print("RUNNING PREPROCESSING PIPELINE")
    print(f"{'='*80}")

    result = preprocess_thesis_data(train_path, test_path)

    # Extract processed data
    X_train = result['X_train_combined']
    y_train = result['y_train']
    X_test = result['X_test_combined']

    # Calculate data loss
    print(f"\n📉 DATA REDUCTION SUMMARY")
    print(f"{'='*80}")
    train_rows_lost = len(train_raw) - len(y_train)
    test_rows_lost = len(test_raw) - len(X_test)
    train_retention = 100 * len(y_train) / len(train_raw)
    test_retention = 100 * len(X_test) / len(test_raw)

    print(f"Train: {len(train_raw):,} → {len(y_train):,} (lost {train_rows_lost:,}, retained {train_retention:.1f}%)")
    print(f"Test:  {len(test_raw):,} → {len(X_test):,} (lost {test_rows_lost:,}, retained {test_retention:.1f}%)")

    # Final target distribution
    print(f"\n🎯 FINAL TARGET DISTRIBUTION")
    print(f"{'='*80}")
    final_target = y_train.value_counts().sort_index()
    for val, count in final_target.items():
        pct = 100 * count / len(y_train)
        print(f"Class {int(val):2d}: {count:6d} ({pct:5.1f}%)")
    print(f"Total classes: {len(final_target)}")

    # Check for remaining missing values
    train_missing_after = X_train.isnull().sum().sum()
    test_missing_after = X_test.isnull().sum().sum()

    print(f"\n✅ MISSING VALUE RESOLUTION")
    print(f"{'='*80}")
    print(f"Train missing values: {train_missing_before:,} → {train_missing_after:,}")
    print(f"Test missing values:  {test_missing_before:,} → {test_missing_after:,}")
    print(f"Total resolved:       {(train_missing_before + test_missing_before) - (train_missing_after + test_missing_after):,}")

    # Analyze numeric vs categorical features
    print(f"\n📊 FEATURE BREAKDOWN")
    print(f"{'='*80}")
    print(f"Numeric features:     {len(result['numeric_cols'])}")
    print(f"Categorical features: {len(result['categorical_cols'])}")
    print(f"Total features:       {X_train.shape[1]}")

    print(f"\nNumeric features:")
    for col in result['numeric_cols']:
        print(f"  - {col}")

    print(f"\nCategorical features:")
    for col in result['categorical_cols']:
        print(f"  - {col}")

    # Analyze numeric feature ranges
    print(f"\n📐 NUMERIC FEATURE RANGES (After Outlier Clipping)")
    print(f"{'='*80}")
    X_train_numeric = result['X_train_numeric']
    for col in result['numeric_cols']:
        min_val = X_train_numeric[col].min()
        max_val = X_train_numeric[col].max()
        mean_val = X_train_numeric[col].mean()
        median_val = X_train_numeric[col].median()
        print(f"{col:30s}: [{min_val:8.2f}, {max_val:8.2f}] mean={mean_val:8.2f} median={median_val:8.2f}")

    # Analyze categorical feature cardinality
    print(f"\n🏷️  CATEGORICAL FEATURE CARDINALITY")
    print(f"{'='*80}")
    X_train_cat = result['X_train_categorical']
    for col in result['categorical_cols']:
        n_unique = X_train_cat[col].nunique()
        print(f"{col:40s}: {n_unique:3d} unique values")

    # Overall quality assessment
    print(f"\n{'='*80}")
    print("📈 DATA QUALITY ASSESSMENT")
    print(f"{'='*80}")

    # Calculate quality score
    retention_score = (train_retention + test_retention) / 2
    missing_resolution = 100 * (1 - (train_missing_after + test_missing_after) / max(1, train_missing_before + test_missing_before))

    print(f"\n✓ Data Retention Rate:     {retention_score:.1f}%")
    print(f"✓ Missing Value Resolution: {missing_resolution:.1f}%")
    print(f"✓ Final Train Size:        {len(y_train):,} samples")
    print(f"✓ Final Test Size:         {len(X_test):,} samples")
    print(f"✓ Feature Count:           {X_train.shape[1]} features")
    print(f"✓ Target Balance:          {final_target.min()}-{final_target.max()} samples per class")

    # Quality recommendations
    print(f"\n💡 QUALITY INSIGHTS")
    print(f"{'='*80}")

    if retention_score < 90:
        print(f"⚠️  Data retention is {retention_score:.1f}% - significant data loss")
    else:
        print(f"✓ Good data retention: {retention_score:.1f}%")

    if missing_resolution < 95:
        print(f"⚠️  Some missing values remain unresolved")
    else:
        print(f"✓ Excellent missing value resolution: {missing_resolution:.1f}%")

    # Class imbalance check
    class_balance = final_target.max() / final_target.min()
    if class_balance > 10:
        print(f"⚠️  High class imbalance detected: {class_balance:.1f}x ratio")
    elif class_balance > 3:
        print(f"⚠️  Moderate class imbalance: {class_balance:.1f}x ratio")
    else:
        print(f"✓ Reasonable class balance: {class_balance:.1f}x ratio")

    print(f"\n{'='*80}")
    print("PREPROCESSING ANALYSIS COMPLETE")
    print(f"{'='*80}\n")

    return result

if __name__ == "__main__":
    result = analyze_preprocessing()
