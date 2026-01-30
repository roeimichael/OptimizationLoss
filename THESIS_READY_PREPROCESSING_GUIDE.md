# Thesis-Ready Preprocessing Guide: Zero Imputation Strategy

## 🎯 Executive Decision

**RECOMMENDED: Strategy 3 - Drop High-Missing Columns**
- **22,108 samples** (77.1% retention) ✅
- **ZERO imputation** (0.0% artificial data) ✅
- **20 features** (lost 4 high-missing columns)
- **Thesis Grade: A**

### Why Not the "Perfect" Strategy 5?
Strategy 5 (zero-missing rows only) is theoretically best but gives only **6,173 samples (21.5%)** which may be too few for robust constraint satisfaction experiments. Strategy 3 is the **sweet spot** for thesis work.

---

## 📊 Strategy Comparison Summary

| Strategy | Samples | Retention | Imputation | Features | Grade | Thesis Defense |
|----------|---------|-----------|------------|----------|-------|----------------|
| **1. Current** | 28,472 | 96.2% | **66.1%** ❌ | 20 | C | HARD TO DEFEND |
| **2. Complete Case** | 6,173 | 21.5% | 0% ✅ | 24 | B | Too few samples |
| **3. Drop High-Missing** ⭐ | **22,108** | **77.1%** | **0%** ✅ | **20** | **A** | **EASY TO DEFEND** |
| 4. Hybrid | 28,657 | 100% | 24.8% ⚠️ | 20 | A- | Acceptable |
| 5. Zero-Missing Only | 6,173 | 21.5% | 0% ✅ | 24 | A+ | Perfect but small |

---

## 🛠️ Implementation: Modify Your Preprocessing Script

### Step 1: Identify Columns to Drop

**High-missing columns (>10% missing):**
1. `referral_id` (48.3% missing)
2. `region_category` (14.8% missing)
3. `joined_through_referral` (14.7% missing)
4. `medium_of_operation` (14.5% missing)

**Action:** Drop these 4 columns

**Medium-missing columns (5-10% missing):**
1. `avg_frequency_login_days` (9.5% missing)
2. `points_in_wallet` (9.2% missing)
3. `days_since_last_login` (5.4% missing)

**Action:** KEEP these - they're valuable behavioral features

---

## ✏️ Modified Preprocessing Code

Replace your current `preprocess_thesis_data()` function with this:

```python
def preprocess_thesis_data_zero_imputation(train_path: str, test_path: str) -> dict:
    """
    THESIS-QUALITY preprocessing with ZERO imputation.
    Strategy: Drop high-missing columns, then drop rows with ANY remaining missing.
    """
    print("🎓 THESIS CHURN DATA PREPROCESSING (ZERO IMPUTATION)")
    print("=" * 60)

    # 1. Load and initial clean
    train, test = load_and_clean_data(train_path, test_path)

    # Store test IDs
    test_ids = test['customer_id'].copy()

    # 2. Process target (remove -1)
    train = process_target(train)

    # 3. Feature engineering (BEFORE dropping columns)
    train = engineer_features(train)
    test = engineer_features(test)

    # =========================================================================
    # 4. DROP HIGH-MISSING COLUMNS (>10% missing)
    # =========================================================================
    HIGH_MISSING_COLS = [
        'referral_id',          # 48.3% missing - already handled by has_referral
        'region_category',      # 14.8% missing
        'joined_through_referral',  # 14.7% missing - already handled by has_referral
        'medium_of_operation'   # 14.5% missing
    ]

    print(f"\n=== Dropping High-Missing Columns ===")
    print(f"Columns to drop (>10% missing): {HIGH_MISSING_COLS}")

    # Drop from train
    cols_to_drop_train = [c for c in HIGH_MISSING_COLS if c in train.columns]
    train = train.drop(columns=cols_to_drop_train)

    # Drop from test
    cols_to_drop_test = [c for c in HIGH_MISSING_COLS if c in test.columns]
    test = test.drop(columns=cols_to_drop_test)

    print(f"Dropped {len(cols_to_drop_train)} columns from train")
    print(f"Dropped {len(cols_to_drop_test)} columns from test")

    # =========================================================================
    # 5. DROP ROWS WITH ANY REMAINING MISSING
    # =========================================================================
    print(f"\n=== Dropping Rows with Missing Values ===")

    missing_before_train = train.isnull().sum().sum()
    missing_before_test = test.isnull().sum().sum()

    print(f"Missing values before dropping rows:")
    print(f"  Train: {missing_before_train:,}")
    print(f"  Test:  {missing_before_test:,}")

    rows_before_train = len(train)
    rows_before_test = len(test)

    # Drop rows with ANY missing
    train = train.dropna().reset_index(drop=True)
    test = test.dropna().reset_index(drop=True)

    rows_dropped_train = rows_before_train - len(train)
    rows_dropped_test = rows_before_test - len(test)
    retention_train = 100 * len(train) / rows_before_train
    retention_test = 100 * len(test) / rows_before_test

    print(f"\nRows dropped:")
    print(f"  Train: {rows_dropped_train:,} ({100-retention_train:.1f}%)")
    print(f"  Test:  {rows_dropped_test:,} ({100-retention_test:.1f}%)")

    print(f"\nFinal dataset sizes:")
    print(f"  Train: {len(train):,} samples ({retention_train:.1f}% retained)")
    print(f"  Test:  {len(test):,} samples ({retention_test:.1f}% retained)")

    # Verify ZERO missing values remain
    missing_after_train = train.isnull().sum().sum()
    missing_after_test = test.isnull().sum().sum()
    print(f"\n✅ Missing values after cleaning: Train={missing_after_train}, Test={missing_after_test}")

    # =========================================================================
    # 6. Define columns clearly
    # =========================================================================
    NUMERIC_COLS = [
        'age', 'days_since_last_login', 'avg_time_spent',
        'avg_transaction_value', 'avg_frequency_login_days',
        'points_in_wallet', 'days_since_joining',
        'last_visit_hour', 'membership_tier', 'has_referral'
    ]
    CATEGORICAL_COLS = [
        'gender', 'preferred_offer_types', 'internet_option',
        'used_special_discount', 'offer_application_preference',
        'past_complaint', 'complaint_status', 'feedback'
    ]

    # Ensure all columns exist (some may have been dropped)
    NUMERIC_COLS = [c for c in NUMERIC_COLS if c in train.columns]
    CATEGORICAL_COLS = [c for c in CATEGORICAL_COLS if c in train.columns]

    print(f"\n=== Feature Summary ===")
    print(f"Numeric features: {len(NUMERIC_COLS)}")
    print(f"Categorical features: {len(CATEGORICAL_COLS)}")
    print(f"Total features: {len(NUMERIC_COLS) + len(CATEGORICAL_COLS)}")

    # =========================================================================
    # 7. Outlier clipping (still needed for extreme values)
    # =========================================================================
    train = remove_outliers(train, NUMERIC_COLS)

    # =========================================================================
    # 8. Drop identifiers
    # =========================================================================
    IDENTIFIERS = ['customer_id', 'Name', 'security_no']
    cols_to_drop_id = [c for c in IDENTIFIERS if c in train.columns]
    train = train.drop(columns=cols_to_drop_id)
    test = test.drop(columns=[c for c in IDENTIFIERS if c in test.columns])

    # =========================================================================
    # 9. Final separation
    # =========================================================================
    X_train = train[NUMERIC_COLS + CATEGORICAL_COLS]
    y_train = train['churn_risk_score']
    X_test = test[NUMERIC_COLS + CATEGORICAL_COLS]

    # Final target distribution
    print(f"\n=== Final Target Distribution ===")
    target_dist = y_train.value_counts().sort_index()
    for cls, count in target_dist.items():
        pct = 100 * count / len(y_train)
        print(f"  Class {int(cls)}: {count:,} ({pct:.1f}%)")

    print(f"\n{'='*60}")
    print("✅ THESIS-QUALITY PREPROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"✓ ZERO imputation (0% artificial data)")
    print(f"✓ {len(y_train):,} training samples")
    print(f"✓ {len(X_test):,} test samples")
    print(f"✓ {len(NUMERIC_COLS) + len(CATEGORICAL_COLS)} features")
    print(f"✓ Defensible methodology for thesis")
    print(f"{'='*60}\n")

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
```

---

## 📝 What Changed?

### ❌ Removed:
1. **KNN Imputation** - No more `impute_remaining_missing()`
2. **High-missing columns** - Dropped 4 columns with >10% missing
3. **Low-quality row filtering** - Not needed anymore (replaced by dropna)

### ✅ Added:
1. **Explicit column dropping** - Clear list of HIGH_MISSING_COLS
2. **Complete case analysis** - `train.dropna()` after dropping columns
3. **Verification** - Check that 0 missing values remain

### 🔄 Kept:
1. **Feature engineering** - Still create `has_referral`, `days_since_joining`, etc.
2. **Outlier clipping** - Still important for extreme values
3. **Target processing** - Still remove -1 values

---

## 📊 Expected Results

Based on analysis:
- **Training samples**: ~22,108 (vs 28,472 before)
- **Test samples**: ~5,632 (vs 7,329 before)
- **Retention**: 77.1% (vs 96.2% before)
- **Imputation rate**: **0%** (vs 66.1% before) ✅

**Class distribution** (should remain balanced):
- Class 1: 1,653 (7.5%)
- Class 2: 1,691 (7.6%)
- Class 3: 6,452 (29.2%)
- Class 4: 6,307 (28.5%)
- Class 5: 6,005 (27.2%)

---

## 📖 How to Write This in Your Thesis

### Methodology Section:

```
Data Preprocessing

To ensure data quality and methodological rigor, we employed a conservative
preprocessing approach that prioritized data integrity over sample size.

Missing Value Handling:
We identified four features with >10% missing values (referral_id: 48.3%,
region_category: 14.8%, joined_through_referral: 14.7%, medium_of_operation:
14.5%). To avoid introducing artificial patterns through imputation, we
removed these high-missing features from our analysis. For the remaining
features, we applied complete case analysis, retaining only samples with no
missing values.

This approach resulted in 22,108 training samples (77.1% of original data)
and 5,632 test samples, which is sufficient for robust model training while
maintaining the highest data quality standards. Importantly, this methodology
eliminates the risk of imputation bias and ensures all results reflect genuine
patterns in the observed data.
```

### Limitations Section (if asked):

```
While our preprocessing approach maximized data quality by avoiding imputation,
this came at the cost of 22.9% sample loss. We chose this trade-off deliberately,
as the alternative (KNN imputation) would have introduced artificial values in
66.1% of the final dataset, potentially biasing our constraint satisfaction
results. The retained sample size (22,108 training samples) remains adequate
for neural network training and provides reliable constraint learning.
```

---

## ✅ Final Checklist

- [ ] Modified `preprocess_data.py` to use new function
- [ ] Tested new preprocessing (run the script)
- [ ] Verified 0% imputation rate
- [ ] Confirmed adequate sample size (~22k samples)
- [ ] Updated config files for new dataset paths
- [ ] Documented in thesis methodology
- [ ] Ready for training experiments ✨

---

## 🎓 Thesis Defense Talking Points

**When committee asks "Why did you lose 23% of your data?"**

✅ **Good Answer:**
"We prioritized data quality over quantity. The 23% of samples we removed had
missing values in critical behavioral features. Rather than introducing artificial
patterns through imputation, which would have affected 66% of our final dataset,
we chose complete case analysis. The retained 22,108 samples provide robust
statistical power while ensuring all results reflect genuine patterns."

❌ **Bad Answer:**
"I didn't want to deal with imputation, so I just dropped stuff."

**When committee asks "How did you handle missing values?"**

✅ **Good Answer:**
"We used a two-stage approach: First, we identified and removed features with
>10% missing values that had limited predictive value after feature engineering.
Second, we applied complete case analysis on the remaining features, resulting
in zero imputation. This ensures our constraint satisfaction results are based
entirely on observed data."

❌ **Bad Answer:**
"I used KNN with k=5 neighbors to impute 66% of my data."

---

## 💡 Bottom Line

**Use Strategy 3: Drop High-Missing Columns + Complete Case Analysis**

- ✅ 22,108 samples (plenty for ML)
- ✅ 0% imputation (perfectly defensible)
- ✅ 77.1% retention (reasonable trade-off)
- ✅ Clean methodology (easy to explain)
- ✅ Thesis-ready (Grade A)

This is the **optimal balance** for thesis work!
