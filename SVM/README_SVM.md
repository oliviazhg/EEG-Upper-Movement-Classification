# SVM Experiment Pipeline

Complete implementation of Support Vector Machine (SVM) with RBF kernel for EEG-based movement classification.

## Overview

This pipeline implements a rigorous two-phase experimental protocol:

1. **Phase 1: Grid Search** - Find optimal hyperparameters (C, γ) on validation set
2. **Phase 2: Final Evaluation** - Evaluate best configuration across 5 independent trials

**Key Features:**
- Excludes "rest" class (11 movement classes only)
- Proper feature scaling (StandardScaler - required for SVM!)
- Checkpoint saving for resumability
- Bootstrap confidence intervals
- Comprehensive visualization

---

## File Structure

```
svm_experiment/
├── svm_config.py           # Configuration parameters
├── svm_train.py            # Training functions (grid search + final eval)
├── svm_analysis.py         # Statistical analysis and aggregation
├── svm_visualize.py        # Plotting functions
├── run_svm_experiment.py   # Main experiment script
└── README_SVM.md          # This file
```

---

## Experimental Protocol

### Grid Search (Phase 1)

**Parameters:**
- C (regularization): [0.1, 1, 10, 100]
- γ (kernel width): [0.001, 0.01, 0.1]
- Total configurations: 4 × 3 = 12

**Process:**
1. Scale features using StandardScaler (fit on train, transform on val)
2. For each (C, γ) combination:
   - Train SVM on training set (seed=0)
   - Evaluate on validation set
   - Record metrics and timing
   - Save checkpoint
3. Select best (C, γ) based on validation accuracy

**Outputs:**
- Validation accuracy grid (4×3)
- Training time per configuration
- Best hyperparameters

---

### Final Evaluation (Phase 2)

**Protocol:**
- Use best (C, γ) from Phase 1
- Run 5 independent trials with different seeds [0, 1, 2, 3, 4]
- Each trial uses same train/val/test split proportions but different random states

**Per Trial Metrics:**
- Training/validation/test accuracy
- Macro F1 scores
- Per-class F1 scores (11 classes)
- Confusion matrix (11×11)
- Number of support vectors (total and per-class)
- Training time
- Inference time per sample

**Aggregated Metrics:**
- Mean ± std for all metrics
- Bootstrap 95% confidence intervals for test accuracy and F1
- Average confusion matrix
- Per-class F1 statistics

---

## Usage

### Quick Start

```python
python run_svm_experiment.py
```

### With Custom Data

Replace the `load_data()` function in `run_svm_experiment.py`:

```python
def load_data():
    # Load your preprocessed EEG features
    # Expected shape: (n_samples, 240)
    # Features: mu/beta band power + mean + std from 60 channels
    
    X_train = ...  # Your training features
    y_train = ...  # Labels in range [1-11] (excluding rest=0)
    X_val = ...
    y_val = ...
    X_test = ...
    y_test = ...
    
    return X_train, y_train, X_val, y_val, X_test, y_test
```

### Configuration

Edit `svm_config.py` to modify:

```python
# Grid search parameters
GRID_SEARCH_CONFIG = {
    'C_values': [0.1, 1, 10, 100],
    'gamma_values': [0.001, 0.01, 0.1],
}

# Number of final trials
FINAL_EVAL_CONFIG = {
    'n_trials': 5,
    'seeds': [0, 1, 2, 3, 4],
}

# Output directories
OUTPUT_CONFIG = {
    'results_dir': 'results/svm',
    'figures_dir': 'figures/svm',
}
```

---

## Outputs

### Results Files (JSON)

**`grid_search_results.json`**
```json
{
  "C_values": [0.1, 1, 10, 100],
  "gamma_values": [0.001, 0.01, 0.1],
  "val_accuracy_grid": [[...], ...],
  "training_times": [[...], ...],
  "best_C": 10,
  "best_gamma": 0.01,
  "best_val_accuracy": 0.8542
}
```

**`trial_results.json`**
```json
{
  "trials": [
    {
      "C": 10,
      "gamma": 0.01,
      "seed": 0,
      "test_accuracy": 0.8234,
      "test_f1_macro": 0.8156,
      "test_f1_per_class": [...],
      "confusion_matrix": [[...], ...],
      "n_support_vectors": 542,
      "training_time_sec": 12.34
    },
    ...
  ]
}
```

**`aggregated_results.json`**
```json
{
  "test_accuracy_mean": 0.8245,
  "test_accuracy_ci_low": 0.8102,
  "test_accuracy_ci_high": 0.8388,
  "test_f1_macro_mean": 0.8167,
  "test_f1_per_class_mean": [...],
  "confusion_matrix_mean": [[...], ...],
  "n_support_vectors_mean": 538.2
}
```

### Figures

1. **`grid_search_heatmap.png`**
   - Validation accuracy vs (C, γ)
   - Annotated with accuracy values
   - Best configuration marked with star

2. **`final_performance.png`**
   - Bar plot of test accuracy
   - Error bars showing 95% CI
   - Best configuration in title

3. **`confusion_matrix.png`**
   - Normalized confusion matrix (11×11)
   - Averaged over 5 trials
   - Annotated with percentages

4. **`per_class_f1.png`**
   - Horizontal bar chart
   - Sorted by F1 score
   - Error bars showing std dev
   - Overall mean marked

5. **`training_time_heatmap.png`**
   - Training time vs (C, γ)
   - Helps identify computational trade-offs

### Checkpoints

During execution, checkpoints are saved to `checkpoints/svm/`:

- `grid_C{C}_gamma{gamma}.json` - After each grid configuration
- `trial_{n}_seed{seed}.json` - After each final trial
- `grid_search_complete_{timestamp}.json` - After grid search completes

**Resumability:** If experiments are interrupted, you can:
1. Load checkpoint files
2. Continue from where you left off
3. Skip completed configurations

---

## Critical Notes

### Feature Scaling

**SVM REQUIRES feature scaling!** The pipeline automatically applies StandardScaler:

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit on train
X_val_scaled = scaler.transform(X_val)          # Transform val
X_test_scaled = scaler.transform(X_test)        # Transform test
```

**Why it matters:**
- SVM uses distance-based computations (RBF kernel)
- Features with larger magnitudes dominate the distance calculation
- Example: If one feature is salary ($20k-$100k) and another is height (1.5-2.0m), salary completely dominates

### Data Splits

- Training: 70% (fit scaler and model)
- Validation: 15% (grid search only - never used for training)
- Test: 15% (final evaluation only - touched only in Phase 2)

**Important:** The same scaler fitted on training data in Phase 1 is used in Phase 2.

### Class Encoding

- Movement labels: 1-11 (rest=0 is excluded)
- No need to remap to 0-10 for sklearn
- Confusion matrix indices: 0-10 (corresponding to classes 1-11)

---

## Validation Checklist

Before running:
- [ ] Feature matrix shape: (n_samples, 240)
- [ ] Labels in range [1, 11] (no 0's)
- [ ] Train/val/test splits are stratified
- [ ] No data leakage between splits
- [ ] Features are raw (not pre-scaled)

After running:
- [ ] Grid search heatmap shows expected patterns
- [ ] Best config is not at grid boundary (suggests need for wider search)
- [ ] Training time scales reasonably with C
- [ ] Number of support vectors is reasonable (< 50% of training data)
- [ ] Confusion matrix is roughly symmetric
- [ ] Per-class F1 scores don't have extreme outliers

---

## Comparison to Random Forest

To compare SVM results to Random Forest:

```python
from svm_analysis import compare_to_baseline

# Load RF results
with open('results/random_forest/aggregated_results.json', 'r') as f:
    rf_results = json.load(f)

# Load SVM results  
with open('results/svm/aggregated_results.json', 'r') as f:
    svm_results = json.load(f)

# Statistical comparison
comparison = compare_to_baseline(svm_results, rf_results)

# Output:
# - Mean difference in accuracy
# - 95% CI for difference
# - Paired t-test results
# - Cohen's d effect size
```

---

## Expected Performance

Based on typical EEG movement classification:

**Grid Search:**
- Best validation accuracy: 70-85%
- Training time per config: 5-30 seconds (depends on data size)
- Support vectors: 30-50% of training data

**Final Evaluation:**
- Test accuracy: 65-80% (11-class problem)
- F1 macro: Similar to accuracy
- CI width: ±2-5% (with 5 trials)

**Common patterns:**
- Confusion between left/right movements
- Confusion between similar grasp types
- Better performance on reach directions than grasps

---

## Troubleshooting

### Low Accuracy

1. Check feature scaling is applied
2. Verify labels are correct (1-11, not 0-10)
3. Examine confusion matrix for systematic errors
4. Try wider grid search range
5. Check for data leakage

### High Variance

1. Increase number of trials (>5)
2. Check if different seeds give very different data splits
3. Examine per-trial confusion matrices
4. Consider class imbalance

### Long Training Time

1. Reduce training set size (for debugging)
2. Use coarser grid initially
3. Consider linear kernel for very large datasets
4. Check for duplicate samples

### Many Support Vectors

If >70% of training data becomes support vectors:
- Data may not be linearly separable in feature space
- Try different gamma values
- Consider feature engineering
- May indicate overfitting (increase C)

---

## Next Steps

After completing SVM experiments:

1. **Compare to Random Forest**
   - Use paired t-test
   - Look at where each model struggles

2. **Analyze Failures**
   - Which movements are hardest?
   - Are there consistent confusion patterns?
   - Do errors make sense anatomically?

3. **Feature Engineering**
   - Try different frequency bands
   - Add temporal features
   - Channel selection

4. **Advanced Methods**
   - Multi-kernel SVM
   - Ensemble with RF
   - Deep learning (CNN)

---

## References

- Course lecture: SYDE 522 Lecture 4 (SVM)
- Course lecture: SYDE 522 Lecture 5-6 (Kernel Trick)
- Cortes & Vapnik (1995). Support-vector networks
- Bishop (2006). Pattern Recognition and Machine Learning, Ch. 7

---

## Contact

For questions about this implementation, refer to:
- Course: SYDE 522 - Fundamentals of AI
- Project: EEG-Based Upper-Limb Movement Classification
