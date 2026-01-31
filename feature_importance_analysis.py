"""
Feature Importance Analysis - Don't Drop Without Evidence!
===========================================================
Analyze which columns actually predict churn using:
1. Correlation with target
2. Mutual information (works for categorical)
3. Random Forest feature importance
4. Chi-square test for categorical features

This will tell us if region_category and medium_of_operation are truly
"questionable" or if they provide valuable predictive power.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif, chi2, SelectKBest
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

def load_and_basic_clean():
    """Load data with basic cleaning only."""
    train = pd.read_csv('data/churn/train_dataset.csv')

    # Replace placeholders
    placeholders = ['?', 'Error', 'xxxxxxxx', np.nan, -999]
    train = train.replace(placeholders, np.nan)

    # Convert to numeric
    train['avg_frequency_login_days'] = pd.to_numeric(train['avg_frequency_login_days'], errors='coerce')

    # Remove -1 target
    train = train[train['churn_risk_score'] != -1].reset_index(drop=True)

    return train

def analyze_column_content(train):
    """See what's actually in region_category and medium_of_operation."""
    print("="*80)
    print("WHAT ARE THESE COLUMNS?")
    print("="*80)

    print("\n1. region_category:")
    print("-" * 80)
    print(f"   Missing: {train['region_category'].isnull().sum()} ({100*train['region_category'].isnull().sum()/len(train):.1f}%)")
    print(f"   Unique values: {train['region_category'].nunique()}")
    print(f"   Values distribution:")
    value_counts = train['region_category'].value_counts()
    for val, count in value_counts.items():
        print(f"     '{val}': {count:,} ({100*count/len(train):.1f}%)")

    print("\n2. medium_of_operation:")
    print("-" * 80)
    print(f"   Missing: {train['medium_of_operation'].isnull().sum()} ({100*train['medium_of_operation'].isnull().sum()/len(train):.1f}%)")
    print(f"   Unique values: {train['medium_of_operation'].nunique()}")
    print(f"   Values distribution:")
    value_counts = train['medium_of_operation'].value_counts()
    for val, count in value_counts.items():
        print(f"     '{val}': {count:,} ({100*count/len(train):.1f}%)")

def prepare_for_analysis(train):
    """Prepare data for feature importance analysis."""
    # Create smart features
    train['has_referral'] = (
        (train['referral_id'].notna()) | (train['joined_through_referral'] != 'No')
    ).astype(int)

    train['joining_date'] = pd.to_datetime(train['joining_date'], errors='coerce')
    ref_date = train['joining_date'].max()
    train['days_since_joining'] = (ref_date - train['joining_date']).dt.days

    train['last_visit_hour'] = train['last_visit_time'].astype(str).str[:2].astype(float)

    membership_map = {
        'No Membership': 0, 'Basic Membership': 1, 'Silver Membership': 2,
        'Gold Membership': 3, 'Premium Membership': 4, 'Platinum Membership': 5
    }
    train['membership_tier'] = train['membership_category'].map(membership_map)

    # Define all potential features (INCLUDING region_category and medium_of_operation)
    numeric_features = [
        'age', 'days_since_last_login', 'avg_time_spent',
        'avg_transaction_value', 'avg_frequency_login_days',
        'points_in_wallet', 'days_since_joining',
        'last_visit_hour', 'membership_tier', 'has_referral'
    ]

    categorical_features = [
        'gender', 'region_category', 'preferred_offer_types',
        'medium_of_operation', 'internet_option',
        'used_special_discount', 'offer_application_preference',
        'past_complaint', 'complaint_status', 'feedback'
    ]

    # Filter to existing
    numeric_features = [f for f in numeric_features if f in train.columns]
    categorical_features = [f for f in categorical_features if f in train.columns]

    return train, numeric_features, categorical_features

def correlation_analysis(train, numeric_features):
    """Compute correlation between numeric features and target."""
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS (Numeric Features)")
    print("="*80)

    correlations = {}
    for feature in numeric_features:
        if feature in train.columns:
            # Use only non-null values for correlation
            valid_data = train[[feature, 'churn_risk_score']].dropna()
            if len(valid_data) > 0:
                corr = valid_data[feature].corr(valid_data['churn_risk_score'])
                correlations[feature] = abs(corr)

    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)

    print(f"\n{'Feature':<40} {'|Correlation|':<15} Missing%")
    print("-" * 80)
    for feature, corr in sorted_corr:
        missing_pct = 100 * train[feature].isnull().sum() / len(train)
        print(f"{feature:<40} {corr:>14.4f} {missing_pct:>10.1f}%")

    return correlations

def mutual_information_analysis(train, numeric_features, categorical_features):
    """Compute mutual information for all features."""
    print("\n" + "="*80)
    print("MUTUAL INFORMATION ANALYSIS (All Features)")
    print("="*80)
    print("(Higher = More predictive power)")

    all_features = numeric_features + categorical_features

    # Prepare data - encode categoricals and fill missing
    X_encoded = train[all_features].copy()

    # Label encode categorical features
    le_dict = {}
    for col in categorical_features:
        if col in X_encoded.columns:
            le = LabelEncoder()
            # Fill NaN with 'Missing' before encoding
            X_encoded[col] = X_encoded[col].fillna('Missing').astype(str)
            X_encoded[col] = le.fit_transform(X_encoded[col])
            le_dict[col] = le

    # Fill missing for numeric with median
    for col in numeric_features:
        if col in X_encoded.columns:
            X_encoded[col] = X_encoded[col].fillna(X_encoded[col].median())

    y = train['churn_risk_score']

    # Compute mutual information
    mi_scores = mutual_info_classif(X_encoded, y, random_state=42)

    # Create results
    mi_results = {}
    for i, feature in enumerate(all_features):
        mi_results[feature] = mi_scores[i]

    # Sort by MI score
    sorted_mi = sorted(mi_results.items(), key=lambda x: x[1], reverse=True)

    print(f"\n{'Feature':<40} {'MI Score':<12} {'Type':<12} Missing%")
    print("-" * 80)
    for feature, score in sorted_mi:
        feature_type = "Numeric" if feature in numeric_features else "Categorical"
        missing_pct = 100 * train[feature].isnull().sum() / len(train)

        # Highlight region_category and medium_of_operation
        marker = " ⭐" if feature in ['region_category', 'medium_of_operation'] else ""

        print(f"{feature:<40} {score:>11.4f} {feature_type:<12} {missing_pct:>7.1f}%{marker}")

    return mi_results

def random_forest_importance(train, numeric_features, categorical_features):
    """Use Random Forest to get feature importance."""
    print("\n" + "="*80)
    print("RANDOM FOREST FEATURE IMPORTANCE")
    print("="*80)

    all_features = numeric_features + categorical_features

    # Prepare data
    X_encoded = train[all_features].copy()

    # Encode categoricals
    for col in categorical_features:
        if col in X_encoded.columns:
            le = LabelEncoder()
            X_encoded[col] = X_encoded[col].fillna('Missing').astype(str)
            X_encoded[col] = le.fit_transform(X_encoded[col])

    # Fill missing numeric
    for col in numeric_features:
        if col in X_encoded.columns:
            X_encoded[col] = X_encoded[col].fillna(X_encoded[col].median())

    y = train['churn_risk_score']

    # Train Random Forest
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_encoded, y)

    # Get importances
    importances = rf.feature_importances_

    rf_results = {}
    for i, feature in enumerate(all_features):
        rf_results[feature] = importances[i]

    # Sort by importance
    sorted_rf = sorted(rf_results.items(), key=lambda x: x[1], reverse=True)

    print(f"\n{'Feature':<40} {'Importance':<12} {'Type':<12} Missing%")
    print("-" * 80)
    for feature, importance in sorted_rf:
        feature_type = "Numeric" if feature in numeric_features else "Categorical"
        missing_pct = 100 * train[feature].isnull().sum() / len(train)

        marker = " ⭐" if feature in ['region_category', 'medium_of_operation'] else ""

        print(f"{feature:<40} {importance:>11.4f} {feature_type:<12} {missing_pct:>7.1f}%{marker}")

    return rf_results

def compare_questionable_features(mi_results, rf_results, correlations):
    """Specifically analyze region_category and medium_of_operation."""
    print("\n" + "="*80)
    print("VERDICT: region_category & medium_of_operation")
    print("="*80)

    questionable = ['region_category', 'medium_of_operation']

    # Get their scores
    for feature in questionable:
        print(f"\n{feature}:")
        print("-" * 80)

        mi_score = mi_results.get(feature, 0)
        rf_score = rf_results.get(feature, 0)

        # Find their rank
        mi_rank = sorted(mi_results.items(), key=lambda x: x[1], reverse=True)
        rf_rank = sorted(rf_results.items(), key=lambda x: x[1], reverse=True)

        mi_position = [i for i, (f, s) in enumerate(mi_rank) if f == feature][0] + 1
        rf_position = [i for i, (f, s) in enumerate(rf_rank) if f == feature][0] + 1

        print(f"  Mutual Information: {mi_score:.4f} (Rank {mi_position}/{len(mi_results)})")
        print(f"  RF Importance:      {rf_score:.4f} (Rank {rf_position}/{len(rf_results)})")

        # Compare to average
        avg_mi = np.mean(list(mi_results.values()))
        avg_rf = np.mean(list(rf_results.values()))

        print(f"\n  Compared to average:")
        print(f"    MI:  {mi_score/avg_mi:.2f}x average")
        print(f"    RF:  {rf_score/avg_rf:.2f}x average")

        # Verdict
        if mi_score > avg_mi and rf_score > avg_rf:
            verdict = "✅ KEEP - Above average importance"
        elif mi_score > avg_mi or rf_score > avg_rf:
            verdict = "⚠️  BORDERLINE - Mixed signals"
        else:
            verdict = "❌ DROP - Below average importance"

        print(f"\n  {verdict}")

def recommendation(mi_results, rf_results, train):
    """Final recommendation based on all analyses."""
    print("\n" + "="*80)
    print("FINAL RECOMMENDATION")
    print("="*80)

    # Identify truly low-value features
    avg_mi = np.mean(list(mi_results.values()))
    avg_rf = np.mean(list(rf_results.values()))

    low_value_features = []
    borderline_features = []
    high_value_features = []

    all_features = list(mi_results.keys())

    for feature in all_features:
        mi_score = mi_results[feature]
        rf_score = rf_results[feature]
        missing_pct = 100 * train[feature].isnull().sum() / len(train)

        # Classify
        if mi_score < avg_mi and rf_score < avg_rf:
            if missing_pct > 10:
                low_value_features.append((feature, mi_score, rf_score, missing_pct))
            else:
                borderline_features.append((feature, mi_score, rf_score, missing_pct))
        else:
            high_value_features.append((feature, mi_score, rf_score, missing_pct))

    print("\n🚫 LOW VALUE (Below avg importance + >10% missing):")
    print("-" * 80)
    if low_value_features:
        for feature, mi, rf, missing in sorted(low_value_features, key=lambda x: x[3], reverse=True):
            print(f"  {feature:<40} MI={mi:.3f} RF={rf:.3f} Missing={missing:.1f}%")
        print(f"\n  → Consider dropping these {len(low_value_features)} features")
    else:
        print("  None! All features with >10% missing have good predictive value")

    print("\n⚠️  BORDERLINE (Below avg importance but <10% missing):")
    print("-" * 80)
    if borderline_features:
        for feature, mi, rf, missing in sorted(borderline_features, key=lambda x: (x[1]+x[2])/2):
            print(f"  {feature:<40} MI={mi:.3f} RF={rf:.3f} Missing={missing:.1f}%")
        print(f"\n  → Keep these, low missing makes them safe")
    else:
        print("  None")

    print("\n✅ HIGH VALUE (Above average importance):")
    print("-" * 80)
    for feature, mi, rf, missing in sorted(high_value_features, key=lambda x: (x[1]+x[2])/2, reverse=True)[:10]:
        print(f"  {feature:<40} MI={mi:.3f} RF={rf:.3f} Missing={missing:.1f}%")
    print(f"  ... and {len(high_value_features)-10} more")

    return low_value_features, borderline_features, high_value_features

def main():
    print("="*80)
    print("FEATURE IMPORTANCE ANALYSIS - DATA-DRIVEN DECISIONS")
    print("="*80)

    # Load data
    train = load_and_basic_clean()

    # Show what region_category and medium_of_operation actually are
    analyze_column_content(train)

    # Prepare features
    train, numeric_features, categorical_features = prepare_for_analysis(train)

    # Run analyses
    correlations = correlation_analysis(train, numeric_features)
    mi_results = mutual_information_analysis(train, numeric_features, categorical_features)
    rf_results = random_forest_importance(train, numeric_features, categorical_features)

    # Focus on questionable features
    compare_questionable_features(mi_results, rf_results, correlations)

    # Final recommendation
    low_value, borderline, high_value = recommendation(mi_results, rf_results, train)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nTotal features analyzed: {len(mi_results)}")
    print(f"  High value: {len(high_value)}")
    print(f"  Borderline: {len(borderline)}")
    print(f"  Low value:  {len(low_value)}")

    if len(low_value) == 0:
        print(f"\n✅ GOOD NEWS: No features qualify as 'low value'")
        print(f"   → Even features with high missing have predictive power")
        print(f"   → Keep all features, accept moderate imputation")
    else:
        print(f"\n⚠️  Found {len(low_value)} truly low-value features to consider dropping")

if __name__ == "__main__":
    main()
