# Claude Project Context

**Project**: Financial Health Prediction - Zindi Competition
**Last Updated**: 2026-03-12
**Status**: Hyperparameter Tuning Phase

---

## Project Overview

This is a machine learning competition project for predicting financial health of small businesses across Africa. The task is to classify businesses into three categories (Low, Medium, High financial health) using business and owner characteristics.

### Competition Details
- **Platform**: Zindi
- **Challenge**: Data.org Financial Health Prediction
- **Task**: 3-class classification (imbalanced)
- **Evaluation Metric**: F1 Macro Score
- **Current Baseline**: 0.8002 (default XGBoost)
- **Goal**: Improve through hyperparameter tuning (target: 0.82-0.85)

### Dataset Characteristics
- **Training samples**: 9,618
- **Test samples**: 2,405
- **Features**: 42 (after preprocessing)
- **Target classes**:
  - Low: 6,280 samples (65.3%)
  - Medium: 2,868 samples (29.8%)
  - High: 470 samples (4.9%) ← **SEVERE IMBALANCE**

---

## Project Structure Overview

```
financial_health_prediction/
│
├── claude.md                           # This file - project context for Claude
├── HYPERPARAMETER_TUNING_NOTES.md     # Complete hyperparameter tuning documentation
│
├── financial-health-zindi-gbms.ipynb  # Main analysis notebook
│   ├── EDA section
│   ├── FastAI preprocessing
│   ├── Baseline XGBoost model
│   ├── Hyperparameter tuning (in progress)
│   └── Submission generation
│
├── Data files/
│   ├── Train.csv                      # Original training data (9,618 rows × 38 cols)
│   ├── Test.csv                       # Original test data (2,405 rows × 37 cols)
│   ├── SampleSubmission.csv           # Submission format template
│   └── VariableDefinitions.csv        # Feature descriptions
│
├── Submissions/
│   ├── submission_fastai.csv          # FastAI neural network predictions
│   ├── submission.csv                 # Basic XGBoost predictions
│   └── submission_tuned_xgboost.csv   # (To be generated) Tuned XGBoost
│
└── .ipynb_checkpoints/                # Jupyter auto-saves
```

---

## Tech Stack and Key Dependencies

### Core ML Libraries
```python
# Data manipulation
pandas>=1.5.0
numpy>=1.23.0

# Machine Learning
xgboost>=2.0.0          # Primary model (gradient boosting)
lightgbm>=4.0.0         # Alternative GBM (imported but not used yet)
catboost>=1.2.0         # Alternative GBM (imported but not used yet)
scikit-learn>=1.3.0     # Preprocessing, metrics, model selection

# Deep Learning
fastai>=2.7.0           # High-level DL library, used for tabular preprocessing
torch>=2.0.0            # PyTorch (fastai dependency)

# Hyperparameter Optimization
scikit-optimize>=0.9.0  # Bayesian optimization (BayesSearchCV)

# Utilities
tqdm                    # Progress bars
pathlib                 # Path handling
```

### Visualization (imported but minimal use)
```python
matplotlib>=3.7.0
seaborn>=0.12.0
```

### Environment
- **Python**: 3.10
- **Platform**: Linux (WSL2)
- **Jupyter**: Notebook environment

---

## Coding Style Preferences

### 1. FastAI Usage
**✅ USE: FastAI v2 API for tabular data preprocessing**

```python
from fastai.tabular.all import *

# Preprocessing pipeline
to = TabularPandas(
    df,
    procs=[Categorify, FillMissing, Normalize],  # FastAI processors
    cat_names=cat_names,
    cont_names=cont_names,
    y_names='Target',
    y_block=CategoryBlock(),
    splits=splits
)

# Access processed data
X_train = to.train.xs      # Preprocessed features
y_train = to.train.ys.values.ravel()  # Labels
```

**Why FastAI?**
- Handles categorical encoding automatically
- Creates missing value indicators
- Normalizes continuous features
- Maintains train/valid splits
- Integrated with PyTorch for neural networks

**⚠️ Note**: Use FastAI for preprocessing, but extract to numpy/pandas for XGBoost training

### 2. XGBoost Best Practices

**✅ ALWAYS handle class imbalance with sample weights:**

```python
from sklearn.utils.class_weight import compute_class_weight

# Calculate balanced weights
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)

# Create sample weights array
sample_weights = np.array([class_weights[int(i)] for i in y_train])

# Pass to XGBoost
model.fit(X_train, y_train, sample_weight=sample_weights)
```

**✅ Use scikit-learn API (not native XGBoost API):**

```python
# PREFER:
from xgboost import XGBClassifier
model = XGBClassifier(objective='multi:softprob', num_class=3, ...)

# AVOID:
import xgboost as xgb
dtrain = xgb.DMatrix(X_train, label=y_train)
model = xgb.train(params, dtrain, ...)  # Native API (harder to use with sklearn tools)
```

**Why sklearn API?**
- Works seamlessly with GridSearchCV, BayesSearchCV
- Compatible with sklearn metrics and cross-validation
- Simpler syntax
- Better integration with pipelines

**✅ Always use stratified cross-validation:**

```python
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Maintains 65:30:5 class distribution in every fold
```

### 3. Evaluation Metrics

**✅ ALWAYS use F1 Macro for this imbalanced problem:**

```python
from sklearn.metrics import f1_score, make_scorer

# For sklearn tools (GridSearchCV, etc.)
f1_scorer = make_scorer(
    lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro'),
    greater_is_better=True
)

# For manual evaluation
f1_macro = f1_score(y_true, y_pred, average='macro')
```

**❌ DON'T use accuracy** (misleading for imbalanced data)

### 4. Random State & Reproducibility

**✅ ALWAYS set random_state=42 for reproducibility:**

```python
# In all random operations:
RandomSplitter(valid_pct=0.2, seed=42)
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
XGBClassifier(random_state=42, ...)
GridSearchCV(..., random_state=42)
BayesSearchCV(..., random_state=42)
```

### 5. Code Organization

**✅ Prefer verbose, documented code over concise:**

```python
# GOOD: Clear and documented
class_weights = compute_class_weight(
    class_weight='balanced',          # Use balanced weighting formula
    classes=np.unique(y_train),        # All unique classes [0, 1, 2]
    y=y_train                          # Training labels
)
sample_weights = np.array([class_weights[int(i)] for i in y_train])

# AVOID: Too concise
sw = np.array([compute_class_weight('balanced', np.unique(y), y)[int(i)] for i in y])
```

**✅ Use meaningful variable names:**

```python
# GOOD
X_train_full = to.train.xs
y_train_full = to.train.ys.values.ravel()
val_predictions = model.predict(X_val)

# AVOID
X = to.train.xs
y = to.train.ys.values.ravel()
preds = model.predict(X2)
```

**✅ Add explanatory comments for complex operations:**

```python
# Calculate balanced class weights to handle severe imbalance
# Formula: weight[class] = total_samples / (n_classes × samples_in_class)
# Result: Low=0.51x, Medium=1.12x, High=6.82x
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
```

### 6. Hyperparameter Tuning Strategy

**✅ PREFER Bayesian Optimization for this project:**

```python
from skopt import BayesSearchCV
from skopt.space import Real, Integer

# Define search space with appropriate priors
search_spaces = {
    'learning_rate': Real(0.01, 0.3, prior='log-uniform'),  # Log scale for LR
    'max_depth': Integer(4, 10),                             # Discrete integers
    'reg_alpha': Real(1e-5, 10, prior='log-uniform'),       # Log scale for reg
    # ...
}
```

**Why Bayesian?**
- 10 hyperparameters to tune (large search space)
- F1 score expensive to compute
- Class imbalance needs smart search
- Competition requires best results

**✅ Use multi-stage Grid Search as fallback:**

```python
# Stage 1: Coarse grid (quick exploration)
# Stage 2: Refined grid (zoom into good regions)
# Stage 3: Fine grid (regularization tuning)
```

### 7. Notebook Structure Preferences

**✅ Use markdown cells for documentation:**
- Explain what each section does
- Include expected outputs
- Document findings and decisions

**✅ Print informative progress messages:**

```python
print("="*70)
print("BAYESIAN OPTIMIZATION CONFIGURATION")
print("="*70)
print(f"Total hyperparameters: {len(search_spaces)}")
print(f"Iterations: 50")
print(f"Expected runtime: ~1-2 hours")
```

**✅ Show results with context:**

```python
print(f"Best F1 Macro Score: {best_score:.6f}")
print(f"Improvement over baseline (0.8002): {best_score - 0.8002:+.6f}")
if best_score > 0.8002:
    improvement_pct = ((best_score / 0.8002) - 1) * 100
    print(f"({improvement_pct:+.2f}% relative improvement)")
```

### 8. Error Handling

**✅ Validate data shapes:**

```python
print(f"Training features shape: {X_train.shape}")  # Expected: (7695, 42)
print(f"Training labels shape: {y_train.shape}")    # Expected: (7695,)
print(f"Sample weights shape: {sample_weights.shape}")  # Expected: (7695,)
```

**✅ Check for common issues:**

```python
# Check for missing values in target
assert y_train.isna().sum() == 0, "Target has missing values!"

# Verify class labels are encoded
assert set(y_train.unique()) == {0, 1, 2}, f"Unexpected labels: {y_train.unique()}"

# Confirm sample weights align
assert len(sample_weights) == len(y_train), "Sample weights mismatch!"
```

---

## Current Progress

### ✅ Completed
1. **Data Loading**: Train, test, sample submission loaded
2. **EDA**: Class distribution analyzed, missing values identified
3. **Preprocessing**: FastAI TabularPandas pipeline implemented
   - Categorify (categorical encoding)
   - FillMissing (with indicators)
   - Normalize (continuous features)
4. **Baseline Models**:
   - FastAI neural network: F1 ≈ 0.75 (underfitted)
   - XGBoost (default params): F1 = 0.8002
5. **Documentation**:
   - Complete hyperparameter tuning notes created
   - All XGBoost parameters explained
   - Search methods compared

### 🔄 In Progress
1. **Hyperparameter Tuning**: Ready to implement
   - Grid Search (3-stage) prepared
   - Bayesian Optimization prepared
   - Class weights calculated

### 📋 To Do
1. **Run Hyperparameter Tuning**:
   - [ ] Execute Bayesian Optimization (50 iterations, ~1-2 hours)
   - [ ] OR Execute Grid Search (3 stages, ~50-65 minutes)
   - [ ] Validate on held-out set
   - [ ] Generate optimized submission

2. **If Time Permits**:
   - [ ] Feature engineering (ratios, interactions)
   - [ ] Ensemble models (XGBoost + LightGBM + CatBoost)
   - [ ] Stacking/blending
   - [ ] Calibration for probability predictions

---

## Key Decisions Made

### 1. Model Choice: XGBoost
**Reason**:
- Handles mixed feature types well
- Robust to missing values
- Fast training
- Excellent for tabular data
- Good baseline performance (0.8002)

**Alternatives considered**:
- LightGBM (imported, not used yet)
- CatBoost (imported, not used yet)
- FastAI neural network (tried, underperformed at 0.75)

### 2. Preprocessing: FastAI TabularPandas
**Reason**:
- Handles categorical features automatically
- Creates missing indicators
- Normalizes continuous features
- Integrates well with notebook workflow

**Note**: Extract to numpy/pandas for XGBoost (doesn't use FastAI dataloaders)

### 3. Metric: F1 Macro Score
**Reason**:
- Competition metric
- Treats all classes equally (perfect for imbalance)
- Prevents "predict majority class" trap

**Why not accuracy?**
- Misleading (65% accuracy = always predict Low!)

### 4. Class Imbalance Handling: Sample Weights
**Reason**:
- High class only 4.9% of data
- Must use weights or model ignores it
- Sample weights = 6.82× for High class

**Why not SMOTE?**
- XGBoost handles weights natively
- Simpler and faster
- No synthetic data artifacts

### 5. Hyperparameter Search: Bayesian Optimization
**Reason**:
- 10 hyperparameters to tune
- Large search space
- Expensive F1 evaluation
- Competition (need best results)

**Fallback**: 3-stage Grid Search if Bayesian has issues

---

## Important Notes for Claude

### When Working on This Project:

1. **ALWAYS use sample weights** when training XGBoost:
   ```python
   model.fit(X_train, y_train, sample_weight=sample_weights)
   ```

2. **ALWAYS use F1 Macro** for evaluation:
   ```python
   f1_score(y_true, y_pred, average='macro')
   ```

3. **ALWAYS use stratified CV** for imbalanced data:
   ```python
   StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
   ```

4. **Data shapes to expect**:
   - X_train: (7695, 42)
   - y_train: (7695,)
   - X_val: (1923, 42)
   - y_val: (1923,)
   - X_test: (2405, 42)

5. **Label encoding**:
   - Classes are encoded as: 0, 1, 2
   - Mapping stored in: `to.vocab` (e.g., ['High', 'Low', 'Medium'])
   - Convert predictions: `to.vocab[predictions]`

6. **Class weights** (recalculate if data changes):
   ```python
   class_weights ≈ [0.51, 1.12, 6.82]  # Low, Medium, High
   ```

7. **Baseline to beat**: F1 = 0.8002

8. **Target after tuning**: F1 ≈ 0.82-0.85

### Files to Reference:

- **`HYPERPARAMETER_TUNING_NOTES.md`**: Complete hyperparameter documentation
  - All XGBoost parameters explained
  - Search methods compared
  - Implementation examples
  - Quick reference commands

- **`financial-health-zindi-gbms.ipynb`**: Main working notebook
  - Contains preprocessed data (`to` object)
  - Baseline models
  - Ready for hyperparameter tuning

### Common Patterns in This Project:

```python
# Standard imports
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, make_scorer, classification_report
from sklearn.utils.class_weight import compute_class_weight
from skopt import BayesSearchCV
from skopt.space import Real, Integer

# Load preprocessed data
X_train = to.train.xs
y_train = to.train.ys.values.ravel()

# Calculate class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
sample_weights = np.array([class_weights[int(i)] for i in y_train])

# F1 scorer
f1_scorer = make_scorer(lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro'),
                        greater_is_better=True)

# Stratified CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Base model
model = xgb.XGBClassifier(objective='multi:softprob', num_class=3,
                          tree_method='hist', random_state=42, n_jobs=-1)

# Fit with weights
model.fit(X_train, y_train, sample_weight=sample_weights)

# Evaluate
val_preds = model.predict(X_val)
val_f1 = f1_score(y_val, val_preds, average='macro')
```

---

## Questions to Ask User Before Starting:

When resuming work on this project, consider asking:

1. **Which hyperparameter search method?**
   - Bayesian Optimization (recommended, ~1-2 hours)
   - Grid Search 3-stage (~50-65 min)
   - Random Search (~30 min, quick baseline)

2. **How much time available?**
   - Full tuning (50+ iterations)
   - Quick tuning (20-30 iterations)
   - Just test implementation

3. **Want to try ensemble?**
   - Single best XGBoost
   - Ensemble (XGBoost + LightGBM + CatBoost)

4. **Feature engineering desired?**
   - Use current 42 features
   - Create new features (ratios, interactions)

---

## Useful Commands

### View data summary:
```python
print(f"Train shape: {to.train.xs.shape}")
print(f"Valid shape: {to.valid.xs.shape}")
print(f"Class distribution: {pd.Series(to.train.ys.values.ravel()).value_counts()}")
```

### Quick model test:
```python
# Train quick model
quick_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
quick_model.fit(to.train.xs, to.train.ys.values.ravel())

# Evaluate
val_preds = quick_model.predict(to.valid.xs)
print(f"F1 Macro: {f1_score(to.valid.ys.values.ravel(), val_preds, average='macro'):.4f}")
```

### Generate submission:
```python
# Predict
test_preds = best_model.predict(test_dl.xs)
test_labels = to.vocab[test_preds]

# Save
submission = pd.DataFrame({'ID': test_df.index, 'Target': test_labels})
submission.to_csv('submission_tuned.csv', index=False)
```

---

**Last Session**: Comprehensive hyperparameter tuning theory and implementation prepared. Ready to execute tuning!

**Next Step**: Run Bayesian Optimization or Grid Search to improve baseline F1 score.
