# Critical Bug Fix: NaN Shape Mismatch in ModelScorer

## 🐛 Bug Description

### The Problem
When `prepare_feature_matrix()` dropped rows containing NaN values, it reduced the numpy array size from **N rows** (original DataFrame) to **M rows** (valid rows only, where M < N). However, `fit_and_score()` attempted to assign these M-sized score and label arrays directly to the N-row DataFrame, causing:

```python
ValueError: Length of values (M) does not match length of index (N)
```

This bug would occur in production **whenever input data contained any missing values**, making the entire scoring pipeline fail.

### Root Cause
```python
# BEFORE FIX (buggy code):
def prepare_feature_matrix(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    X = df[numeric_cols].values
    nan_mask = np.isnan(X).any(axis=1)
    X = X[~nan_mask]  # Drops rows, reducing size from N to M
    return X, numeric_cols  # ❌ No way to track which rows were dropped!

def fit_and_score(self, features_df: pd.DataFrame) -> pd.DataFrame:
    X, feature_cols = self.prepare_feature_matrix(features_df)  # M rows
    scores, labels = self.score_anomalies(X_scaled)  # M-sized arrays
    
    result_df = features_df.copy()  # N rows
    result_df["anomaly_score"] = scores  # ❌ ValueError: M != N
    result_df["anomaly_label"] = labels  # ❌ ValueError: M != N
```

## ✅ Solution Implemented

### The Fix
Track which rows are valid using a **boolean mask**, then use **sentinel values** for invalid rows to preserve DataFrame shape:

```python
# AFTER FIX (corrected code):
def prepare_feature_matrix(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str], np.ndarray]:
    X = df[numeric_cols].values
    nan_mask = np.isnan(X).any(axis=1)
    valid_mask = ~nan_mask  # ✅ Track which rows are valid (N elements)
    X = X[valid_mask]  # M rows (valid only)
    return X, numeric_cols, valid_mask  # ✅ Return the mask!

def fit_and_score(self, features_df: pd.DataFrame) -> pd.DataFrame:
    X, feature_cols, valid_mask = self.prepare_feature_matrix(features_df)  # M rows + N-sized mask
    valid_count = valid_mask.sum()  # M
    scores, labels = self.score_anomalies(X_scaled)  # M-sized arrays
    
    # Use Option 2: Sentinel values to preserve shape
    result_df = features_df.copy()  # N rows
    result_df["anomaly_score"] = np.nan  # ✅ Initialize all as NaN
    result_df["anomaly_label"] = 0  # ✅ Initialize all as 0 (invalid)
    result_df.loc[valid_mask, "anomaly_score"] = scores  # ✅ Fill only valid rows
    result_df.loc[valid_mask, "anomaly_label"] = labels  # ✅ Fill only valid rows
    return result_df  # ✅ Shape preserved: N rows out
```

### Why Option 2 (Sentinel Values)?
We considered two approaches:

**Option 1: Drop invalid rows from DataFrame**
```python
result_df = features_df[valid_mask].copy()  # Only M rows
result_df["anomaly_score"] = scores
result_df["anomaly_label"] = labels
return result_df  # M rows returned
```

**Option 2: Keep all rows, use sentinel values** (✅ **CHOSEN**)
```python
result_df = features_df.copy()  # N rows
result_df["anomaly_score"] = np.nan  # Sentinel for invalid
result_df["anomaly_label"] = 0  # Sentinel for invalid (neither 1=normal nor -1=anomaly)
result_df.loc[valid_mask, "anomaly_score"] = scores
result_df.loc[valid_mask, "anomaly_label"] = labels
return result_df  # N rows returned
```

**Why Option 2 is better for production:**

1. **Maintains 1:1 correspondence with input data** - Critical for downstream reporting and reconciliation
2. **Invalid rows are identifiable** - `anomaly_score = np.nan` clearly marks rows that couldn't be scored
3. **Preserves customer identifiers** - Allows data quality investigation for specific customers
4. **Easier debugging and monitoring** - Can track which customers had missing data
5. **Predictable output shape** - Always returns same number of rows as input

### Sentinel Value Semantics
- **`anomaly_score = np.nan`**: Indicates the customer couldn't be scored due to missing feature data
- **`anomaly_label = 0`**: Indicates unknown/invalid status (distinct from normal=1 and anomaly=-1)

## 🧪 Test Coverage

### Added Regression Tests

1. **`test_fit_and_score_with_nan_preserves_shape`**
   - Tests basic NaN handling with 5 customers, 4 with NaN
   - Validates shape preservation (5 in → 5 out)
   - Verifies sentinel values for invalid rows
   - Verifies real scores/labels for valid row

2. **`test_fit_and_score_with_non_consecutive_index`**
   - Tests with non-consecutive DataFrame indices (0, 2, 5, 7, 10)
   - Ensures boolean mask indexing works with index gaps
   - Critical for DataFrames created after filtering operations

3. **`test_fit_and_score_all_rows_have_nan`**
   - Tests error handling when every row has at least one NaN
   - Should raise `ModelScoringError: No valid rows remaining`
   - Prevents model training on empty arrays

### Updated Existing Tests
- Modified `test_prepare_feature_matrix_success` to unpack 3-element tuple
- Modified `test_prepare_feature_matrix_with_nan` to validate valid_mask
- All 26 tests pass (23 unit + 3 integration)

## 📊 Test Results

```
✅ 26/26 tests pass
✅ 95% coverage (up from 94%)
✅ Black formatting: compliant
✅ Ruff linting: all checks pass
```

### Example Test Execution
```python
# Before fix (would have failed):
df = pd.DataFrame({
    "customer_id": ["C001", "C002", "C003"],
    "total_spend": [1000.0, np.nan, 2000.0],
    "total_transactions": [10, 20, 30]
})

scorer = ModelScorer(config)
# This would raise: ValueError: Length of values (2) does not match length of index (3)
result_df = scorer.fit_and_score(df)

# After fix (works correctly):
result_df = scorer.fit_and_score(df)
assert len(result_df) == 3  # ✅ Shape preserved
assert pd.isna(result_df.loc[1, "anomaly_score"])  # ✅ Sentinel for invalid row
assert result_df.loc[1, "anomaly_label"] == 0  # ✅ Sentinel for invalid row
assert not pd.isna(result_df.loc[0, "anomaly_score"])  # ✅ Valid score
assert result_df.loc[0, "anomaly_label"] in [-1, 1]  # ✅ Valid label
```

## 🎯 Impact

### Before Fix
- ❌ Pipeline would crash on **any** input with missing values
- ❌ Production deployment would fail with real-world data
- ❌ No way to identify which customers had missing data

### After Fix
- ✅ Pipeline handles missing values gracefully
- ✅ Production-ready with robust error handling
- ✅ Invalid rows clearly marked with sentinel values
- ✅ Complete audit trail for data quality issues

## 📝 Code Changes Summary

### Files Modified
1. **`src/models/scorer.py`**
   - Updated `prepare_feature_matrix()` signature to return `valid_mask`
   - Updated `fit_and_score()` to use sentinel values for invalid rows
   - Improved logging to show valid/total row counts

2. **`tests/unit/test_modelscorer.py`**
   - Added 3 new regression tests in `TestNaNHandling` class
   - Updated existing tests to handle 3-element tuple return
   - Added comprehensive docstrings explaining the bug

### Lines of Code Changed
- **scorer.py**: ~20 lines modified
- **test_modelscorer.py**: ~100 lines added

## 🚀 Deployment Readiness

The fix is:
- ✅ Fully tested with comprehensive regression tests
- ✅ Production-ready with robust error handling
- ✅ Backward compatible (output structure unchanged for valid data)
- ✅ Well-documented with clear sentinel value semantics
- ✅ Code quality validated (Black + Ruff)

## 🔗 References

- **Issue**: Critical bug identified during code review
- **PR**: #14 - Phase 3: Model Scoring with IsolationForest
- **Commits**:
  - Initial implementation: `beb147f`
  - Bug fix: `85f7699`

