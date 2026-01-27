# Churn Dataset Preprocessing Quality Report

## Executive Summary

The preprocessing pipeline demonstrates **excellent data retention (97.6%)** with **complete missing value resolution (100%)**. However, there are some concerns about the **high imputation rate (66.1%)** and **moderate class imbalance (3.9x)** that should be considered.

---

## 📊 Dataset Overview

### Initial Data
- **Training Set**: 29,593 rows × 25 columns
- **Test Set**: 7,399 rows × 24 columns
- **Total Samples**: 36,992

### Final Data (After Preprocessing)
- **Training Set**: 28,472 rows × 20 features (96.2% retained)
- **Test Set**: 7,329 rows × 20 features (99.1% retained)
- **Overall Retention**: 97.6%

---

## 🔍 Data Quality Issues Detected

### 1. Placeholder & Missing Values
**Total bad values found: 43,357**

| Category | Train | Test | Total |
|----------|-------|------|-------|
| Placeholders (?, Error, xxxxxxxx, -999) | 27,319 | 6,879 | 34,198 |
| Explicit NaN values | 7,308 | 1,851 | 9,159 |
| **Total** | **34,627** | **8,730** | **43,357** |

**Most problematic columns:**
1. `referral_id`: 14,294 issues (48.3%)
2. `region_category`: 4,366 issues (14.8%)
3. `joined_through_referral`: 4,339 issues (14.7%)
4. `medium_of_operation`: 4,291 issues (14.5%)
5. `avg_frequency_login_days`: 2,792 issues (9.4%)
6. `points_in_wallet`: 2,714 issues (9.2%)

### 2. Target Distribution (Before Processing)

| Class | Count | Percentage |
|-------|-------|------------|
| -1 (Unknown) | 927 | 3.1% |
| 1 (Low Risk) | 2,146 | 7.3% |
| 2 | 2,193 | 7.4% |
| 3 | 8,337 | 28.2% |
| 4 | 8,165 | 27.6% |
| 5 (High Risk) | 7,825 | 26.4% |

**Decision**: Remove -1 values (< 10% threshold) → Deleted 927 rows

### 3. Row Quality
- **Train rows with >3 missing**: 0 (0.0%) ✓
- **Test rows with >3 missing**: 0 (0.0%) ✓

**Distribution of missing values per row (Train):**
- 0 missing: 22,712 (76.7%)
- 1 missing: 6,459 (21.8%)
- 2 missing: 417 (1.4%)
- 3 missing: 5 (0.0%)

---

## 🛠️ Preprocessing Steps Applied

### Step 1: Placeholder Replacement
**Replaced 34,198 placeholder values** (?, Error, xxxxxxxx, -999) with NaN

### Step 2: Low-Quality Row Filtering
- **Threshold**: >3 missing values per row
- **Train**: Deleted 202 rows (0.7%)
- **Test**: Deleted 70 rows (0.9%)
- **Rationale**: ✓ Good - preserves 99%+ of data while removing truly bad rows

### Step 3: Target Processing
- **Deleted 919 rows** with target=-1 (3.1% of original data)
- **Rationale**: ✓ Appropriate - data-driven decision (<10% threshold)

### Step 4: Feature Engineering
**Created 4 new features:**
1. `has_referral` (binary: 0/1)
2. `days_since_joining` (calculated from joining_date)
3. `last_visit_hour` (extracted from last_visit_time)
4. `membership_tier` (ordinal encoding: 0-5)

**Dropped 4 original columns** (joining_date, last_visit_time, membership_category, joined_through_referral)

### Step 5: Outlier Clipping (1st/99th Percentiles)

**Total outliers clipped: 3,189 values**

| Feature | Values Clipped | % of Data |
|---------|----------------|-----------|
| days_since_last_login | 486 | 1.7% |
| avg_time_spent | 570 | 2.0% |
| avg_transaction_value | 570 | 2.0% |
| avg_frequency_login_days | 518 | 1.8% |
| points_in_wallet | 520 | 1.8% |
| days_since_joining | 525 | 1.8% |

**Assessment**: ✓ Reasonable - ~2% outliers per feature is expected

### Step 6: Identifier Removal
**Dropped**: customer_id, Name, security_no, referral_id

### Step 7: KNN Imputation

**Total values imputed: 18,825**

| Column | Train Missing | % Missing |
|--------|---------------|-----------|
| days_since_last_login | 1,488 | 5.2% |
| avg_frequency_login_days | 2,603 | 9.1% |
| points_in_wallet | 2,530 | 8.9% |
| region_category | 4,106 | 14.4% |
| preferred_offer_types | 206 | 0.7% |
| medium_of_operation | 4,018 | 14.1% |

**Method**: KNN (k=5) for numeric, mode for categorical

---

## 📈 Final Results

### Data Transformation Summary

| Metric | Value |
|--------|-------|
| **Rows Deleted** | 1,191 (3.2%) |
| **Rows Retained** | 35,801 (96.8%) |
| **Placeholders Replaced** | 34,198 |
| **Values Imputed** | 18,825 |
| **Outliers Clipped** | 3,189 |
| **Total Bad Values Handled** | 43,357 |

### Final Target Distribution

| Class | Count | Percentage | Balance |
|-------|-------|------------|---------|
| 1 (Low Risk) | 2,129 | 7.5% | Min |
| 2 | 2,179 | 7.7% | |
| 3 | 8,278 | 29.1% | **Max** |
| 4 | 8,121 | 28.5% | |
| 5 (High Risk) | 7,765 | 27.3% | |

**Class Balance Ratio**: 3.89x (max/min)

### Feature Breakdown

| Type | Count | Features |
|------|-------|----------|
| **Numeric** | 10 | age, days_since_last_login, avg_time_spent, avg_transaction_value, avg_frequency_login_days, points_in_wallet, days_since_joining, last_visit_hour, membership_tier, has_referral |
| **Categorical** | 10 | gender, region_category, preferred_offer_types, medium_of_operation, internet_option, used_special_discount, offer_application_preference, past_complaint, complaint_status, feedback |
| **Total** | 20 | |

---

## 💡 Quality Assessment & Recommendations

### ✅ Strengths

1. **Excellent Data Retention (97.6%)**
   - Only 2.4% of data deleted - very conservative approach
   - Good balance between quality and quantity

2. **Complete Missing Value Resolution (100%)**
   - All placeholder and missing values properly handled
   - No NaN values remain in final dataset

3. **Appropriate Target Handling**
   - Data-driven decision on -1 values (<10% → remove)
   - Resulted in clean 5-class problem

4. **Smart Feature Engineering**
   - Created meaningful features (days_since_joining, has_referral)
   - Natural ordinal encoding for membership tiers

5. **Conservative Outlier Treatment**
   - Only clipped 1st/99th percentiles (~2% per feature)
   - Preserves data distribution while removing implausible values

### ⚠️ Concerns & Recommendations

#### 1. **HIGH IMPUTATION RATE (66.1%)**
**Concern**: 18,825 values imputed out of 28,472 final samples = 0.66 imputed values per sample

**Analysis**:
- This is HIGH and could introduce artificial patterns
- Mainly concentrated in 6 features:
  - `region_category`: 14.4% missing
  - `medium_of_operation`: 14.1% missing
  - `avg_frequency_login_days`: 9.1% missing
  - `points_in_wallet`: 8.9% missing

**Recommendations**:
1. **Consider dropping high-missing features** if they're not critical (region_category, medium_of_operation)
2. **Create missingness indicators** - add binary flags for "was_imputed" which can be informative
3. **Compare model performance** with/without imputed features
4. **Document in thesis** - be transparent about 66.1% imputation rate

#### 2. **MODERATE CLASS IMBALANCE (3.9x)**
**Concern**: Class 1 (2,129 samples) vs Class 3 (8,278 samples) = 3.89x ratio

**Recommendations**:
1. **Use class weights** in loss function: `weight = n_samples / (n_classes * class_count)`
2. **Consider stratified sampling** to ensure all classes represented in validation
3. **Monitor per-class metrics** - don't rely only on overall accuracy
4. **May need to adjust constraints** - small classes might be harder to satisfy

#### 3. **Potential Information Leakage in KNN Imputation**
**Concern**: KNN imputation uses k=5 neighbors - could leak information from test to train or vice versa

**Recommendations**:
1. **Fit KNN only on train** data, then transform test
2. **Verify** that train/test imputation is done separately
3. **Current code** seems to impute train and test separately ✓

#### 4. **Referral Feature Handling**
**Concern**: 48.3% of referral_id was missing, converted to binary has_referral

**Analysis**: This is actually good - converted high-cardinality feature with many missing values into simple binary indicator

**Recommendation**: ✓ Current approach is solid

---

## 🎯 Overall Quality Score

| Metric | Score | Assessment |
|--------|-------|------------|
| **Data Retention** | 97.6% | ✅ Excellent |
| **Missing Resolution** | 100% | ✅ Excellent |
| **Outlier Handling** | ~2% clipped | ✅ Good |
| **Class Balance** | 3.9x ratio | ⚠️ Moderate |
| **Imputation Rate** | 66.1% | ⚠️ High |
| **Feature Engineering** | 20 features | ✅ Good |

### Overall Grade: **B+ (Very Good with Concerns)**

**Summary**: The preprocessing is well-executed with excellent retention and complete cleaning. The main concern is the high imputation rate (66.1%) which could introduce artificial patterns. Consider dropping or flagging high-missing features, and definitely use class weights for the imbalanced target.

---

## 📋 Recommendations for Training

### 1. Use Class Weights
```python
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced',
                                     classes=np.unique(y_train),
                                     y=y_train)
```

### 2. Monitor Per-Class Metrics
- Don't rely only on overall accuracy
- Track precision, recall, F1 for each class
- Pay special attention to minority class (Class 1)

### 3. Consider Feature Selection
- Test model with/without high-missing features
- `region_category` (14.4% missing) and `medium_of_operation` (14.1% missing) might hurt more than help

### 4. Validate Constraint Satisfaction
- With class imbalance, constraints on minority classes will be harder to satisfy
- May need to adjust constraint percentages or use unlimited constraints for minority classes

### 5. Document Imputation in Thesis
- Be transparent about 66.1% imputation rate
- Discuss potential impact on model reliability
- Consider this as limitation in discussion section

---

## 🔬 Additional Analysis Suggestions

1. **Correlation Analysis**: Check if imputed values correlate with target differently than real values
2. **Distribution Comparison**: Compare distributions of imputed vs non-imputed subsets
3. **Ablation Study**: Train models with/without imputed features to measure impact
4. **Feature Importance**: See if imputed features rank high - if so, interpret with caution

---

## ✅ Final Verdict

**The preprocessing is fundamentally sound** with good practices:
- Conservative row filtering
- Appropriate placeholder handling
- Smart feature engineering
- Complete missing value resolution

**However**, the **high imputation rate is a significant concern** that should be:
1. Documented transparently in your thesis
2. Addressed through feature selection or missingness indicators
3. Validated through ablation studies

**Recommendation**: Proceed with training, but keep the imputation rate in mind when interpreting results. Consider it a limitation rather than a fatal flaw.
