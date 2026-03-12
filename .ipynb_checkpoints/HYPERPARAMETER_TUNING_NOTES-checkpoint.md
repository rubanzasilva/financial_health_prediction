# Hyperparameter Tuning Session Notes

**Date**: 2026-03-12
**Project**: Financial Health Prediction (Zindi Competition)
**Model**: XGBoost Classifier
**Metric**: F1 Macro Score

---

## Table of Contents
1. [Project Context](#project-context)
2. [The Class Imbalance Problem](#the-class-imbalance-problem)
3. [Hyperparameter Search Methods](#hyperparameter-search-methods)
4. [XGBoost Hyperparameters Explained](#xgboost-hyperparameters-explained)
5. [Grid Search vs Strategies](#grid-search-vs-strategies)
6. [Implementation Details](#implementation-details)
7. [Key Takeaways](#key-takeaways)

---

## Project Context

### Dataset
- **Training samples**: 9,618
- **Test samples**: 2,405
- **Features**: 42 (after preprocessing)
- **Target classes**: 3 (Low, Medium, High)

### Class Distribution (IMBALANCED!)
```
Low:    6,280 samples (65.3%)
Medium: 2,868 samples (29.8%)
High:     470 samples (4.9%)  ← RARE CLASS!
```

### Current Performance
- **Baseline F1 Macro**: 0.8002 (default XGBoost parameters)
- **Goal**: Improve through hyperparameter tuning
- **Expected improvement**: 0.82-0.85 (+2-5%)

### Why F1 Macro?
- **Accuracy is misleading** for imbalanced data
- Example: Model that always predicts "Low"
  - Accuracy: 65.3% (looks decent!)
  - F1 for High class: 0.0 (never predicts it!)
  - **Useless model but good accuracy!**

- **F1 Macro treats all classes equally**:
  ```
  F1_Macro = (F1_Low + F1_Medium + F1_High) / 3
  ```
  - Forces model to perform well on rare "High" class
  - Perfect for competition metrics that value all classes

---

## The Class Imbalance Problem

### Why Class Weights Are Essential

#### Without Class Weights ❌
```python
# XGBoost default: all samples weighted equally
loss = sum of all prediction errors

During training:
- Misclassify 100 "Low" samples: loss += 100
- Misclassify 10 "High" samples: loss += 10

Model thinking: "Low class is 65% of data, focus there!"
Result: Model predicts "Low" for everything
```

**Outcome**:
- High class F1: ~0.1 (rarely predicted)
- Medium class F1: ~0.5 (sometimes predicted)
- Low class F1: ~0.9 (always predicted)
- **F1 Macro: 0.5** (poor!)

#### With Class Weights ✅
```python
# Calculate balanced weights
class_weights = compute_class_weight('balanced',
                                     classes=np.unique(y),
                                     y=y)

Result:
  Low (class 0):    weight = 0.51x
  Medium (class 1): weight = 1.12x
  High (class 2):   weight = 6.82x  ← Each High sample counts as 6.82!

# Now loss function becomes:
loss = sum of (weight × prediction error)

During training:
- Misclassify 100 "Low" samples: loss += 100 × 0.51 = 51
- Misclassify 10 "High" samples: loss += 10 × 6.82 = 68.2

Model thinking: "High class errors hurt more! Must learn it!"
```

**Outcome**:
- High class F1: ~0.65 (properly learned!)
- Medium class F1: ~0.75
- Low class F1: ~0.85
- **F1 Macro: 0.75** (much better!)

### The Math Behind Class Weights

```python
weight[class] = total_samples / (n_classes × samples_in_class)

For your data:
  weight_High = 9618 / (3 × 470) = 6.82
  weight_Medium = 9618 / (3 × 2868) = 1.12
  weight_Low = 9618 / (3 × 6280) = 0.51
```

### How to Apply Class Weights

```python
from sklearn.utils.class_weight import compute_class_weight

# Calculate weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

# Map each sample to its class weight
sample_weights = np.array([class_weights[int(i)] for i in y_train])

# Pass to XGBoost
model.fit(X_train, y_train, sample_weight=sample_weights)
```

**Critical**: For multiclass, use `sample_weight` parameter, NOT `scale_pos_weight` (binary only)!

---

## Hyperparameter Search Methods

### The 3 Core Methods (Different Algorithms)

| Method | Algorithm | Learning | Best For |
|--------|-----------|----------|----------|
| **Grid Search** | Exhaustive enumeration | None (dumb) | Small search space (<100 combos) |
| **Random Search** | Random sampling | None (dumb) | Medium space (100-500 samples) |
| **Bayesian Optimization** | Gaussian Process + Acquisition | Yes (smart) | Large space, expensive evaluation |

### 1. Grid Search

**How it works**:
- You specify exact values for each hyperparameter
- Tests EVERY possible combination
- Deterministic (always same results)

**Example**:
```python
param_grid = {
    'max_depth': [4, 6, 8],         # 3 values
    'learning_rate': [0.05, 0.1],   # 2 values
    'n_estimators': [500, 1000]     # 2 values
}

# Total combinations: 3 × 2 × 2 = 12
# With 5-fold CV: 12 × 5 = 60 model fits
```

**Pros**:
- ✅ Guaranteed to test every combination
- ✅ Simple to understand
- ✅ Reproducible
- ✅ Good for small search spaces

**Cons**:
- ❌ Combinatorial explosion (grows exponentially)
- ❌ Wastes time on bad combinations
- ❌ Doesn't learn from results
- ❌ Can't handle continuous ranges well

**When to use**:
- ≤4 hyperparameters
- ≤3 values per hyperparameter
- Total combinations <100

### 2. Random Search

**How it works**:
- You specify distributions (not exact values)
- Randomly samples N combinations from those distributions
- Different results each run (unless same random seed)

**Example**:
```python
from scipy.stats import uniform, randint

param_distributions = {
    'max_depth': randint(4, 11),           # Randomly picks 4-10
    'learning_rate': uniform(0.01, 0.29),  # Randomly picks 0.01-0.3
    'n_estimators': randint(300, 1501)     # Randomly picks 300-1500
}

# You control budget: n_iter=100
# Tests 100 random combinations
# With 5-fold CV: 100 × 5 = 500 model fits
```

**Pros**:
- ✅ Much faster than grid for large spaces
- ✅ Explores more diversity than grid
- ✅ Works with continuous distributions
- ✅ Often finds good solutions quickly
- ✅ Bergstra & Bengio (2012): Often beats grid search!

**Cons**:
- ❌ Random/wasteful (doesn't learn from trials)
- ❌ Might miss optimal combination
- ❌ Less reproducible (random)
- ❌ No guarantee of coverage

**When to use**:
- Medium-large search space
- Want quick exploration
- 5-10 hyperparameters
- Budget: 100-150 iterations

**Key insight**:
```
Testing 60 random combinations often beats
testing a 60-combination grid because random
search explores more parameter space diversity.
```

### 3. Bayesian Optimization (RECOMMENDED for Your Case)

**How it works**:
1. Try a few random combinations (initialization)
2. Build probabilistic model (Gaussian Process) of: `F1_score = f(hyperparameters)`
3. Use acquisition function to pick next most promising point:
   - **Exploitation**: Try near known good regions
   - **Exploration**: Try unexplored regions
4. Test that combination, update model
5. Repeat steps 3-4

**Example**:
```python
from skopt import BayesSearchCV
from skopt.space import Real, Integer

search_spaces = {
    'max_depth': Integer(4, 10),
    'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
    'n_estimators': Integer(300, 1500)
}

# Intelligent search with budget: n_iter=50
# Tests 50 SMART combinations (learns from each)
# With 5-fold CV: 50 × 5 = 250 model fits
```

**Process Example**:
```
Iteration 1-5: Random exploration
  Trial 3: max_depth=4, lr=0.1, n_est=500 → F1=0.78
  Trial 5: max_depth=8, lr=0.05, n_est=1000 → F1=0.82 ← Good!

Iteration 6: Bayesian model thinks "depth=8, lr=0.05 worked well"
  → Try nearby: max_depth=7, lr=0.05, n_est=1100 → F1=0.83 ← Better!

Iteration 7: Also explore unexplored regions
  → Try: max_depth=5, lr=0.2, n_est=300 → F1=0.75 ← Not good

Iteration 8: Back to good region with variations
  → Try: max_depth=8, lr=0.03, n_est=1200 → F1=0.84 ← Best so far!

... continues intelligently balancing exploration vs exploitation
```

**Pros**:
- ✅ Most efficient (learns from previous trials)
- ✅ Finds better results faster than Grid/Random
- ✅ Works with continuous spaces
- ✅ Adaptive - focuses on promising regions
- ✅ Best for expensive evaluations (like F1 score with CV)

**Cons**:
- ❌ More complex to understand
- ❌ Can get stuck in local optima
- ❌ Small overhead building model (negligible for XGBoost)
- ❌ Requires extra library (`scikit-optimize`)

**When to use**:
- Large search space (>5 hyperparameters)
- Expensive model training
- Imbalanced data (needs smart search)
- Competition/production (need best results)
- Budget: 50-100 iterations

**Why Bayesian for Your Case**:
1. ✅ You have 10 hyperparameters to tune
2. ✅ F1 score is expensive to compute (needs full CV)
3. ✅ Class imbalance makes search tricky (rare class behavior)
4. ✅ Competition setting (need every 0.01 F1 improvement)
5. ✅ With 50 iterations, finds near-optimal in ~1-2 hours

### Comparison Table

| Aspect | Grid | Random | Bayesian |
|--------|------|--------|----------|
| **Speed** | ⭐ Slowest | ⭐⭐⭐⭐ Fast | ⭐⭐⭐ Medium |
| **Quality** | ⭐⭐⭐⭐⭐ Perfect | ⭐⭐⭐ Good | ⭐⭐⭐⭐⭐ Excellent |
| **Scalability** | ⭐ Poor | ⭐⭐⭐⭐ Good | ⭐⭐⭐⭐⭐ Excellent |
| **Complexity** | ⭐ Simple | ⭐ Simple | ⭐⭐⭐ Complex |
| **Your case** | 50-65 min | ~30 min | ~1-2 hours |

---

## XGBoost Hyperparameters Explained

### Official Documentation
📚 https://xgboost.readthedocs.io/en/release_3.2.0/parameter.html

### Parameter Categories

#### 1. Learning Task Parameters (What Problem?)

**`objective`** - The loss function to optimize
```python
# For your multiclass classification:
objective='multi:softprob'   # Returns probability matrix
# OR
objective='multi:softmax'    # Returns class labels

# Other options (NOT for your case):
'binary:logistic'            # Binary classification
'reg:squarederror'           # Regression (MSE)
'rank:ndcg'                  # Ranking
```

**`num_class`** - Number of classes (REQUIRED for multiclass)
```python
num_class=3  # Low, Medium, High
```

**`eval_metric`** - Metric for monitoring (doesn't affect training)
```python
eval_metric='mlogloss'    # Multiclass log loss (default for multi:softprob)
# Other options:
# 'merror'  - Multiclass error rate
# 'auc'     - Area under curve (binary only)
```

**`base_score`** - Initial prediction before boosting
```python
base_score=0.5   # Default
# Auto mode: uses mean for regression, log-odds for classification
# Usually leave at default
```

**`random_state`** (or `seed`) - Random seed
```python
random_state=42  # For reproducibility
```

---

#### 2. Tree Booster Parameters (How Trees Are Built)

**Priority 1: Tree Structure**

**`max_depth`** - Maximum tree depth
```python
max_depth=6  # Default

# Range: [0, ∞), where 0 = unlimited
# Typical values: [4-10]

# Visual:
max_depth=3:          max_depth=8:
   [root]                [root]
   /    \               /      \
 [L]    [R]         [deep tree with
                     many levels...]

# Lower (4): Simpler trees, less overfitting, faster
# Higher (10): Complex patterns, more overfitting, slower
```

**`min_child_weight`** - Minimum sum of instance weight in child
```python
min_child_weight=1  # Default

# Range: [0, ∞)
# Typical values: [1, 3, 5, 7]

# How it works:
# If a potential split creates child with weight < min_child_weight → DON'T SPLIT

# Example with sample_weights:
Node has: 5 High samples (weight=6.82 each) + 50 Low samples (weight=0.51 each)
Total weight = 5×6.82 + 50×0.51 = 34.1 + 25.5 = 59.6

If min_child_weight=30:
  - Potential split: left=10 (weight=15), right=45 (weight=44.6)
  - Left weight (15) < 30 → DON'T SPLIT (protects small groups)

# Lower (1): Allows tiny leaves, flexible, can overfit
# Higher (7): Forces larger leaves, conservative, better generalization
```

**`gamma`** (or `min_split_loss`) - Minimum loss reduction for split
```python
gamma=0  # Default

# Range: [0, ∞)
# Typical values: [0, 0.1, 0.3, 0.5]

# How it works:
# Split only if: loss_before - (loss_left + loss_right) > gamma

# Example with gamma=0.1:
Current node loss: 10.0
Potential split: left_loss=4.8, right_loss=5.1
Gain = 10.0 - (4.8 + 5.1) = 0.1 ✓ SPLIT (exactly at threshold)

Another split: left_loss=4.95, right_loss=5.0
Gain = 10.0 - (4.95 + 5.0) = 0.05 ✗ DON'T SPLIT (below threshold)

# 0: Split if ANY improvement
# 0.5: Split only if SUBSTANTIAL improvement
# Higher gamma = simpler trees = less overfitting
```

**Priority 1: Learning Rate**

**`learning_rate`** (or `eta`) - Step size shrinkage
```python
learning_rate=0.3  # Default

# Range: [0, 1]
# Typical values: [0.01, 0.05, 0.1, 0.3]

# How it works:
prediction = base_score + lr×tree1 + lr×tree2 + lr×tree3 + ...

# Example with lr=0.1:
base_score = 0.5
tree1 predicts: +2.0 → adds 0.1 × 2.0 = 0.2 → prediction = 0.7
tree2 predicts: +1.5 → adds 0.1 × 1.5 = 0.15 → prediction = 0.85
tree3 predicts: +0.8 → adds 0.1 × 0.8 = 0.08 → prediction = 0.93

# Low (0.01): Slow learning, need MANY trees (1000-2000), better generalization
# High (0.3): Fast learning, need FEW trees (300-500), risk overfitting

# Rule of thumb: Lower LR + more trees = better performance (but slower)
```

**`n_estimators`** - Number of boosting rounds
```python
n_estimators=100  # Default

# Range: [1, ∞)
# Typical values: [300, 500, 1000, 1500]

# Interaction with learning_rate:
lr=0.3, n_est=300  → Fast learning, fewer trees
lr=0.01, n_est=2000 → Slow learning, many trees (usually better)

# Use early_stopping_rounds to prevent overfitting:
model.fit(X, y, eval_set=[(X_val, y_val)], early_stopping_rounds=50)
# Stops if no improvement for 50 rounds
```

**Priority 2: Sampling (Prevents Overfitting)**

**`subsample`** - Row sampling per tree
```python
subsample=1.0  # Default (use all samples)

# Range: (0, 1]
# Typical values: [0.7, 0.8, 0.9]

# How it works:
You have 7695 training samples

subsample=1.0: Each tree sees all 7695 samples
subsample=0.8: Each tree sees random 80% (6156 samples)

Tree 1 might see: samples [1, 3, 4, 5, 7, 8, ...]
Tree 2 might see: samples [2, 3, 5, 6, 8, 9, ...] ← Different!

# Benefits:
✓ Prevents overfitting (like bagging)
✓ Faster training (less data per tree)
✓ Adds diversity to ensemble

# 1.0: All data, faster, might overfit
# 0.8: Recommended (good balance)
# 0.6: Much randomness, can underfit
```

**`colsample_bytree`** - Column sampling per tree
```python
colsample_bytree=1.0  # Default

# Range: (0, 1]
# Typical values: [0.7, 0.8, 0.9]

# How it works:
You have 42 features

colsample_bytree=1.0: Each tree can use all 42 features
colsample_bytree=0.8: Each tree uses random 80% (34 features)

Tree 1 uses: [owner_age, income, expenses, country, ...]
Tree 2 uses: [country, income, turnover, has_mobile_money, ...] ← Different!

# Benefits:
✓ Reduces correlation between trees
✓ Prevents over-reliance on strong features
✓ Like Random Forest's feature randomness

# 1.0: All features
# 0.8: Recommended
```

**`colsample_bylevel`** - Column sampling per tree level
```python
colsample_bylevel=1.0  # Default

# Range: (0, 1]
# Usually not tuned unless many features

# Difference from colsample_bytree:
colsample_bytree=0.8: Sample features ONCE per tree
colsample_bylevel=0.8: RE-sample features at EACH depth level

# Tree with colsample_bylevel=0.8:
Depth 0 (root): Uses random 80% of features to find best split
Depth 1: Uses NEW random 80% for both nodes
Depth 2: Uses ANOTHER NEW random 80%
... (more granular randomness)
```

**`colsample_bynode`** - Column sampling per node
```python
colsample_bynode=1.0  # Default

# Most granular level (rarely tuned)
# Re-samples features for EVERY SINGLE split decision

# Hierarchy:
colsample_bytree: Sample once per tree
colsample_bylevel: Sample once per depth level
colsample_bynode: Sample once per node ← Most random
```

**Priority 3: Regularization (Prevents Overfitting)**

**`reg_lambda`** (or `lambda`) - L2 regularization
```python
reg_lambda=1  # Default

# Range: [0, ∞)
# Typical values: [0.1, 1, 10, 100]

# Mathematical formula:
Objective = Loss + reg_lambda × Σ(leaf_weights²)

# Example with 3 leaves:
Leaf weights: [2.5, -1.8, 3.2]

reg_lambda=0: No penalty
reg_lambda=1: Penalty = 1 × (2.5² + 1.8² + 3.2²) = 19.53
reg_lambda=10: Penalty = 10 × 19.53 = 195.3

# Higher penalty → training pushes weights toward 0 → simpler model

# Effect:
# 0: No regularization, can overfit
# 1: Moderate (default)
# 10: Strong regularization, simpler model
```

**`reg_alpha`** (or `alpha`) - L1 regularization
```python
reg_alpha=0  # Default

# Range: [0, ∞)
# Typical values: [0, 0.1, 1, 10]

# Mathematical formula:
Objective = Loss + reg_alpha × Σ|leaf_weights|

# Example:
Leaf weights: [2.5, -1.8, 3.2]

reg_alpha=0: No penalty
reg_alpha=1: Penalty = 1 × (|2.5| + |-1.8| + |3.2|) = 7.5
reg_alpha=10: Penalty = 10 × 7.5 = 75

# L1 vs L2 difference:
L1 (alpha): Encourages SPARSE solutions (some weights → exactly 0)
L2 (lambda): Encourages SMALL weights (all shrunk but rarely exactly 0)

# When to use:
L1: Feature selection, interpretability
L2: General regularization, smoother predictions
```

**Priority 3: Imbalance Handling**

**`max_delta_step`** - Maximum leaf output change
```python
max_delta_step=0  # Default (no constraint)

# Range: [0, ∞)
# Typical values: [0, 1, 2, 5]

# How it works:
Without constraint (=0):
  Leaf for "High" class might output: +15.3 (extreme!)
  → Model overconfident on rare class

With max_delta_step=2:
  All leaf outputs capped at [-2, +2]
  Original: +15.3 → Clipped to: +2.0
  Original: -8.1 → Clipped to: -2.0

# Effect:
✓ Makes training more conservative/stable
✓ Prevents wild predictions on rare class
✓ Useful for severe class imbalance

# 0: No limit (default)
# 1-5: Cap outputs (if unstable training)
```

**System Parameters**

**`tree_method`** - Algorithm for building trees
```python
tree_method='auto'  # Default

# Options:
'exact':    Exact greedy (slow, optimal splits)
'approx':   Approximate (faster, nearly same quality)
'hist':     Histogram-based (FAST, recommended) ← Use this
'gpu_hist': GPU-accelerated histogram (10-100× faster)

# For your data (9k samples, 42 features): Use 'hist'
```

**`n_jobs`** - Number of parallel threads
```python
n_jobs=-1  # Use all CPU cores

# -1: Detect and use all cores
# 1: Single-threaded
# 4: Use 4 cores
```

**`verbosity`** - Message level
```python
verbosity=1  # Default (warnings)

# 0: Silent
# 1: Warnings
# 2: Info
# 3: Debug
```

---

#### 3. Classification-Specific Parameters

**`scale_pos_weight`** - Balance positive/negative (BINARY ONLY!)
```python
# For BINARY classification only:
scale_pos_weight = count(negative) / count(positive)

# Example:
Negative: 9000 samples
Positive: 1000 samples
scale_pos_weight = 9000/1000 = 9

# ⚠️ For MULTICLASS (your case): DON'T USE THIS!
# Use sample_weight parameter in .fit() instead
```

---

#### 4. Regression-Only Parameters (NOT for Your Case)

**Objective functions**:
```python
'reg:squarederror'      # L2 loss (MSE)
'reg:squaredlogerror'   # MSLE
'reg:gamma'             # Gamma regression
'reg:tweedie'           # Tweedie regression
'reg:pseudohubererror'  # Huber loss (robust to outliers)
```

**Special parameters**:
```python
tweedie_variance_power  # For Tweedie regression
huber_slope             # For Huber loss
quantile_alpha          # For quantile regression
aft_loss_distribution   # For survival analysis
```

---

### Parameter Tuning Priority

**Tier 1: Biggest Impact** (Tune first)
1. `learning_rate` + `n_estimators` (tune together!)
2. `max_depth`
3. `min_child_weight`
4. `subsample` + `colsample_bytree`

**Tier 2: Fine-Tuning**
5. `gamma`
6. `reg_alpha` + `reg_lambda`

**Tier 3: Optional**
7. `colsample_bylevel`
8. `max_delta_step` (if class imbalance issues)

---

### Good Starting Values for Your Case

```python
xgb.XGBClassifier(
    # Task
    objective='multi:softprob',
    num_class=3,
    eval_metric='mlogloss',

    # Tree structure
    max_depth=6,
    min_child_weight=3,
    gamma=0.1,

    # Learning
    learning_rate=0.05,
    n_estimators=1000,

    # Sampling
    subsample=0.8,
    colsample_bytree=0.8,

    # Regularization
    reg_alpha=0.1,
    reg_lambda=1,

    # Imbalance
    max_delta_step=1,

    # System
    tree_method='hist',
    random_state=42,
    n_jobs=-1
)
```

---

## Grid Search vs Strategies

### IMPORTANT CLARIFICATION

**Grid Search is ONE method, not multiple methods!**

What I showed you were different **STRATEGIES** for using Grid Search, not different types of Grid Search.

### Complete Taxonomy

```
HYPERPARAMETER SEARCH METHODS (Different algorithms)
│
├── Grid Search ← ONE METHOD (exhaustive enumeration)
│   │
│   ├── Strategy: Full Grid (test everything at once)
│   ├── Strategy: Coarse-to-Fine (multi-stage)
│   ├── Strategy: Fixed-Order (sequential tuning)
│   └── Strategy: Hierarchical (conditional params)
│
├── Random Search ← ONE METHOD (random sampling)
│
├── Bayesian Optimization ← ONE METHOD (Gaussian Process)
│
└── Others (Evolutionary, Gradient-based, etc.)
```

### Strategy 1: Naive Full Grid ❌

```python
# Test EVERYTHING at once
param_grid = {
    'max_depth': [4, 5, 6, 7, 8, 9, 10],        # 7 values
    'min_child_weight': [1, 2, 3, 4, 5, 6, 7],  # 7 values
    'learning_rate': [0.01, 0.03, 0.05, 0.1],   # 4 values
    'n_estimators': [300, 500, 700, 1000],      # 4 values
    'subsample': [0.7, 0.8, 0.9],               # 3 values
    'colsample_bytree': [0.7, 0.8, 0.9],        # 3 values
    'gamma': [0, 0.1, 0.3],                     # 3 values
    'reg_alpha': [0, 0.1, 1],                   # 3 values
    'reg_lambda': [1, 10, 50],                  # 3 values
    'max_delta_step': [0, 1, 2]                 # 3 values
}

# Total: 7×7×4×4×3×3×3×3×3×3 = 2,286,900 combinations!
# With 5-fold CV: 11,434,500 model fits
# Runtime: ~400 DAYS! ❌❌❌
```

**Problem**: Combinatorial explosion

### Strategy 2: Coarse-to-Fine (Multi-Stage) ✅

```python
# SAME GridSearchCV class, just used in stages!

# Stage 1: Wide net, coarse values
param_grid_stage1 = {
    'max_depth': [4, 6, 8],           # 3 values (not 7)
    'min_child_weight': [1, 3, 5],    # 3 values (not 7)
    'learning_rate': [0.05, 0.1],     # 2 values (not 4)
    'n_estimators': [500, 1000],      # 2 values (not 4)
    'subsample': [0.8, 1.0],          # 2 values
    # Fix others at defaults
    'colsample_bytree': [0.8],
    'gamma': [0],
    'reg_alpha': [0],
    'reg_lambda': [1],
    'max_delta_step': [0]
}
# Combinations: 3×3×2×2×2 = 72

grid1 = GridSearchCV(model, param_grid_stage1, ...)
grid1.fit(X, y)
# Runtime: ~15-20 min

# Stage 2: Refine around Stage 1 best
best_depth = grid1.best_params_['max_depth']  # e.g., 6

param_grid_stage2 = {
    'max_depth': [best_depth-1, best_depth, best_depth+1],  # [5,6,7]
    'min_child_weight': [2, 3, 4],    # Refined
    'learning_rate': [0.03, 0.05, 0.07],  # Refined
    'n_estimators': [700, 1000],
    'subsample': [0.8],  # Fix Stage 1 best
    # NOW add:
    'colsample_bytree': [0.7, 0.8, 0.9],
    'gamma': [0, 0.1, 0.3],
    # Still fix:
    'reg_alpha': [0],
    'reg_lambda': [1],
    'max_delta_step': [0]
}
# Combinations: 3×3×3×2×3×3 = 162

grid2 = GridSearchCV(model, param_grid_stage2, ...)  # SAME CLASS!
grid2.fit(X, y)
# Runtime: ~25-30 min

# Stage 3: Regularization only
param_grid_stage3 = {
    # Fix ALL structural params
    'max_depth': [grid2.best_params_['max_depth']],
    'min_child_weight': [grid2.best_params_['min_child_weight']],
    # ... fix all
    # NOW tune:
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 10, 50],
    'max_delta_step': [0, 1, 2]
}
# Combinations: 3×3×3 = 27

grid3 = GridSearchCV(model, param_grid_stage3, ...)  # SAME CLASS!
grid3.fit(X, y)
# Runtime: ~10-15 min

# TOTAL: 72 + 162 + 27 = 261 combinations
# Runtime: ~50-65 minutes ✅
```

**Key point**: It's still `GridSearchCV`! Just used strategically in stages.

### Why Multi-Stage Works

**Avoids bad combinations**:
```python
# Full grid wastes time on bad pairs:
lr=0.01, n_est=300  ❌ Too few trees, underfits
lr=0.3, n_est=2000  ❌ Too many trees, overfits

# Multi-stage learns:
Stage 1 finds: lr=0.05, n_est=1000 works well
Stage 2: Refines around that good region
Stage 3: Adds polish without re-testing bad regions
```

### Analogy: Tuning a Guitar

1. **Stage 1**: Rough tuning (get close to right pitch)
2. **Stage 2**: Medium tuning (get closer)
3. **Stage 3**: Fine tuning (perfect pitch)

### All Strategies Use Same Class

```python
from sklearn.model_selection import GridSearchCV

# All these use GridSearchCV:
grid_full = GridSearchCV(model, huge_grid, ...)      # Strategy 1
grid_stage1 = GridSearchCV(model, coarse_grid, ...)  # Strategy 2 - Step 1
grid_stage2 = GridSearchCV(model, fine_grid, ...)    # Strategy 2 - Step 2
grid_order1 = GridSearchCV(model, grid_a, ...)       # Strategy 3

# SAME method (Grid Search), different human strategies!
```

---

## Implementation Details

### Data Preparation

```python
# From FastAI TabularPandas preprocessing
X_train = to.train.xs                    # (7695, 42) features
y_train = to.train.ys.values.ravel()     # (7695,) labels [0,1,2]

X_val = to.valid.xs                      # (1923, 42) validation features
y_val = to.valid.ys.values.ravel()       # (1923,) validation labels

# Test data
X_test = test_dl.xs                      # (2405, 42) test features
```

### Class Weight Calculation

```python
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Calculate balanced weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

# Map each sample to its class weight
sample_weights = np.array([class_weights[int(i)] for i in y_train])

print(f"Class weights: {class_weights}")
# Output: [0.51, 1.12, 6.82]
```

### Cross-Validation Setup

```python
from sklearn.model_selection import StratifiedKFold

# Stratified K-Fold: maintains class distribution in every fold
cv = StratifiedKFold(
    n_splits=5,        # 5 folds
    shuffle=True,      # Randomly shuffle before splitting
    random_state=42    # Reproducible splits
)

# How it works:
# Fold 1: 80% train (6156 samples) → 20% val (1539 samples) → F1
# Fold 2: Different 80%/20% split → F1
# Fold 3: Different 80%/20% split → F1
# Fold 4: Different 80%/20% split → F1
# Fold 5: Different 80%/20% split → F1
# Average F1 across all 5 folds
```

### F1 Macro Scorer

```python
from sklearn.metrics import f1_score, make_scorer

def f1_macro_scorer(y_true, y_pred):
    """Calculate F1 Macro score for multiclass"""
    return f1_score(y_true, y_pred, average='macro')

# Wrap for sklearn compatibility
f1_scorer = make_scorer(
    f1_macro_scorer,
    greater_is_better=True  # Higher F1 = better
)
```

### Grid Search Implementation

```python
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

# Define parameter grid
param_grid = {
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [500, 1000],
    # ... other params
}

# Create base model
base_model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    tree_method='hist',
    random_state=42,
    n_jobs=-1
)

# Create GridSearchCV
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    scoring=f1_scorer,       # Optimize F1 Macro
    cv=cv,                   # 5-fold stratified CV
    n_jobs=-1,               # Parallel across folds
    verbose=2,               # Show progress
    return_train_score=True  # Check overfitting
)

# Fit with sample weights
grid_search.fit(X_train, y_train, sample_weight=sample_weights)

# Get results
print(f"Best F1: {grid_search.best_score_:.6f}")
print(f"Best params: {grid_search.best_params_}")
```

### Bayesian Optimization Implementation

```python
from skopt import BayesSearchCV
from skopt.space import Real, Integer

# Define search space
search_spaces = {
    'max_depth': Integer(4, 10, name='max_depth'),
    'learning_rate': Real(0.01, 0.3, prior='log-uniform', name='learning_rate'),
    'n_estimators': Integer(300, 1500, name='n_estimators'),
    'subsample': Real(0.6, 1.0, prior='uniform', name='subsample'),
    'colsample_bytree': Real(0.6, 1.0, prior='uniform', name='colsample_bytree'),
    'gamma': Real(0, 0.5, prior='uniform', name='gamma'),
    'reg_alpha': Real(1e-5, 10, prior='log-uniform', name='reg_alpha'),
    'reg_lambda': Real(1e-5, 100, prior='log-uniform', name='reg_lambda'),
    'max_delta_step': Integer(0, 5, name='max_delta_step'),
    'min_child_weight': Integer(1, 7, name='min_child_weight')
}

# Create Bayesian search
bayes_search = BayesSearchCV(
    estimator=base_model,
    search_spaces=search_spaces,
    n_iter=50,               # 50 intelligent trials
    scoring=f1_scorer,
    cv=cv,
    n_jobs=-1,
    verbose=2,
    random_state=42,
    return_train_score=True
)

# Fit
bayes_search.fit(X_train, y_train, sample_weight=sample_weights)

# Get results
print(f"Best F1: {bayes_search.best_score_:.6f}")
print(f"Best params: {bayes_search.best_params_}")
```

### Validation & Submission

```python
# Get best model
best_model = grid_search.best_estimator_  # or bayes_search.best_estimator_

# Retrain on full training data
best_model.fit(X_train, y_train, sample_weight=sample_weights)

# Validate on held-out set
val_predictions = best_model.predict(X_val)
val_f1 = f1_score(y_val, val_predictions, average='macro')
print(f"Validation F1: {val_f1:.6f}")

# Generate test predictions
test_predictions = best_model.predict(X_test)

# Map to original labels
test_pred_labels = to.vocab[test_predictions]

# Create submission
submission = pd.DataFrame({
    'ID': test_df.index,
    'Target': test_pred_labels
})

submission.to_csv('submission_tuned_xgboost.csv', index=False)
print("Submission saved!")
```

---

## Key Takeaways

### 1. Class Imbalance is Critical
- ✅ **Always use class weights** for imbalanced multiclass
- ✅ Formula: `weight = total_samples / (n_classes × class_samples)`
- ✅ Pass as `sample_weight` to `.fit()` method
- ❌ Don't use `scale_pos_weight` (binary only)

### 2. F1 Macro is Perfect for Your Case
- ✅ Treats all classes equally (High class counts same as Low)
- ✅ Prevents "predict everything as majority class" problem
- ✅ Aligns with competition metrics

### 3. Bayesian Optimization Recommended
- ✅ Best for your 10 hyperparameters
- ✅ Learns from expensive F1 score evaluations
- ✅ 50 iterations ≈ 1-2 hours
- ✅ Typically finds better solutions than Grid/Random

### 4. Grid Search Strategies
- ✅ Multi-stage reduces 2M combinations to 261
- ✅ 3 stages: Coarse → Medium → Fine
- ✅ Total time: ~50-65 minutes
- ❌ Don't do full grid (combinatorial explosion)

### 5. Hyperparameter Priority
**Tier 1** (Tune first):
1. `learning_rate` + `n_estimators`
2. `max_depth`
3. `min_child_weight`
4. `subsample` + `colsample_bytree`

**Tier 2** (Fine-tune):
5. `gamma`
6. `reg_alpha` + `reg_lambda`

**Tier 3** (Optional):
7. `max_delta_step` (if imbalance issues)

### 6. Expected Results
- Baseline: F1 = 0.8002
- After tuning: F1 ≈ 0.82-0.85
- Improvement: +2-5%

### 7. Next Steps
1. Run Bayesian Optimization (recommended)
   - OR Grid Search (3-stage)
   - OR Random Search (quick baseline)
2. Validate on held-out set
3. Generate submission
4. Submit to competition
5. If improvement, consider:
   - Feature engineering
   - Ensemble (XGBoost + LightGBM + CatBoost)
   - Stacking

---

## Quick Reference Commands

### Install Requirements
```bash
pip install scikit-optimize  # For Bayesian optimization
```

### Load Your Data
```python
# Your FastAI preprocessed data
X_train = to.train.xs
y_train = to.train.ys.values.ravel()
X_val = to.valid.xs
y_val = to.valid.ys.values.ravel()
```

### Calculate Class Weights
```python
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
sample_weights = np.array([class_weights[int(i)] for i in y_train])
```

### Run Bayesian Search (Copy-Paste Ready)
```python
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, make_scorer

# F1 scorer
f1_scorer = make_scorer(lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro'),
                        greater_is_better=True)

# CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Search space
search_spaces = {
    'max_depth': Integer(4, 10),
    'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
    'n_estimators': Integer(300, 1500),
    'subsample': Real(0.6, 1.0),
    'colsample_bytree': Real(0.6, 1.0),
    'gamma': Real(0, 0.5),
    'reg_alpha': Real(1e-5, 10, prior='log-uniform'),
    'reg_lambda': Real(1e-5, 100, prior='log-uniform'),
    'max_delta_step': Integer(0, 5),
    'min_child_weight': Integer(1, 7)
}

# Model
model = xgb.XGBClassifier(objective='multi:softprob', num_class=3,
                          tree_method='hist', random_state=42, n_jobs=-1)

# Bayesian search
bayes = BayesSearchCV(model, search_spaces, n_iter=50, scoring=f1_scorer,
                      cv=cv, n_jobs=-1, verbose=2, random_state=42)

# Fit
bayes.fit(X_train, y_train, sample_weight=sample_weights)

print(f"Best F1: {bayes.best_score_:.6f}")
print(f"Best params: {bayes.best_params_}")
```

---

## Resources

### Documentation
- XGBoost Parameters: https://xgboost.readthedocs.io/en/release_3.2.0/parameter.html
- scikit-learn GridSearchCV: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
- scikit-optimize BayesSearchCV: https://scikit-optimize.github.io/stable/modules/generated/skopt.BayesSearchCV.html

### Papers
- **Random Search**: Bergstra & Bengio (2012) - "Random Search for Hyper-Parameter Optimization"
- **Bayesian Optimization**: Snoek et al. (2012) - "Practical Bayesian Optimization of Machine Learning Algorithms"

### Key Concepts
- **Class Imbalance**: When classes have very different sample sizes
- **F1 Macro**: Average of per-class F1 scores (treats all classes equally)
- **Stratified CV**: Maintains class distribution in each fold
- **Sample Weights**: Per-sample weights to handle imbalance
- **Combinatorial Explosion**: Exponential growth of combinations

---

**End of Session Notes**

Last updated: 2026-03-12
