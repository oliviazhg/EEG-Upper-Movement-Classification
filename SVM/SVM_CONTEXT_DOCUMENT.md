# SVM Experiment Context Document

This document describes the exact experimental protocol and data structures for the SVM classification experiments.

## Model Overview

Support Vector Machine with Radial Basis Function kernel. Grid search over C (regularization) and γ (kernel width).

## Parameters to Test

### Phase 1: Grid Search (on validation set)

| Parameter | Values | Total Configs |
|-----------|--------|---------------|
| C | [0.1, 1, 10, 100] | 4 |
| gamma | [0.001, 0.01, 0.1] | 3 |
| **Grid search total** | **4 × 3 = 12** | **12** |

### Phase 2: Final Evaluation (on test set)

Best (C, γ) from grid search → 5 independent trials with different seeds

## Experimental Protocol

```python
# Data split
train: 70%, validation: 15%, test: 15%

# Phase 1: Grid search (use validation set)
for each (C, gamma) in grid:
    1. Train on training data (seed=0)
    2. Evaluate on validation data
    3. Record validation accuracy
    
# Phase 2: Best config evaluation
best_C, best_gamma = argmax(validation_accuracy)
for seed in [0, 1, 2, 3, 4]:
    1. Split data with stratification (seed)
    2. Train SVM with (best_C, best_gamma)
    3. Evaluate on test set
    4. Record metrics
```

## Data to Collect

### Grid Search Results

```python
grid_search_results = {
    'C_values': [0.1, 1, 10, 100],
    'gamma_values': [0.001, 0.01, 0.1],
    
    # Validation accuracy grid
    'val_accuracy_grid': np.ndarray,  # shape (4, 3)
    # Example:
    # [[0.78, 0.81, 0.76],   # C=0.1
    #  [0.82, 0.85, 0.79],   # C=1
    #  [0.84, 0.87, 0.82],   # C=10
    #  [0.83, 0.86, 0.81]]   # C=100
    
    # Best configuration
    'best_C': float,
    'best_gamma': float,
    'best_val_accuracy': float,
    
    # Training time per config
    'training_times': np.ndarray,  # shape (4, 3)
}
```

### Per Trial (5 total, best config only)

```python
trial_data = {
    # Configuration (same for all trials)
    'C': float,
    'gamma': float,
    'seed': int,
    
    # Performance
    'train_accuracy': float,
    'val_accuracy': float,
    'test_accuracy': float,
    
    'train_f1_macro': float,
    'val_f1_macro': float,
    'test_f1_macro': float,
    
    'test_f1_per_class': np.ndarray,  # shape (11,)
    
    # Confusion matrix
    'confusion_matrix': np.ndarray,  # shape (11, 11)
    
    # Model characteristics
    'n_support_vectors': int,
    'n_support_vectors_per_class': dict,  # {class: count}
    
    # Timing
    'training_time_sec': float,
    'inference_time_per_sample': float,
}
```

### Aggregated (best config)

```python
final_summary = {
    'C': float,
    'gamma': float,
    
    # Performance with CI
    'test_acc_mean': float,
    'test_acc_ci_low': float,
    'test_acc_ci_high': float,
    
    'test_f1_mean': float,
    'test_f1_ci_low': float,
    'test_f1_ci_high': float,
    
    # Per-class F1
    'test_f1_per_class_mean': np.ndarray,
    
    # Support vectors (mean)
    'n_support_vectors_mean': float,
    'n_support_vectors_std': float,
    
    # Average confusion
    'confusion_matrix_mean': np.ndarray,
}
```

## Validation Requirements

### 1. Grid Search Validation

```python
# Use validation set, not test set
# Only touch test set for final 5 trials
```

### 2. Statistical Testing

```python
# Bootstrap CI on final 5 trials
# Compare to other models with paired t-test
```

### 3. Scaling Check

```python
# SVM requires feature scaling!
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

## Required Plots

### 1. Heatmap: Validation accuracy vs (C, γ)

```
X-axis: gamma [0.001, 0.01, 0.1]
Y-axis: C [0.1, 1, 10, 100]
Color: Validation accuracy (%)
Annotations: Accuracy values
Mark: Best config with star/box
```

### 2. Final performance (bar plot with CI)

```
Single bar for best config
Y-axis: Test accuracy (%)
Error bar: 95% CI from 5 trials
```

### 3. Confusion Matrix (best config, averaged)

```
11×11 normalized confusion matrix
Averaged over 5 trials
Annotated with values
```

### 4. Per-class F1 (best config)

```
Horizontal bar chart
11 movement classes
Error bars showing std dev
Sorted by F1 score
```

## Implementation Skeleton

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

def grid_search_svm(X_train, y_train, X_val, y_val):
    """Phase 1: Grid search on validation set"""
    # Scale features (critical for SVM!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Grid search
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [0.001, 0.01, 0.1]
    }
    
    results = np.zeros((len(param_grid['C']), len(param_grid['gamma'])))
    
    for i, C in enumerate(param_grid['C']):
        for j, gamma in enumerate(param_grid['gamma']):
            svm = SVC(C=C, gamma=gamma, kernel='rbf')
            svm.fit(X_train_scaled, y_train)
            val_acc = svm.score(X_val_scaled, y_val)
            results[i, j] = val_acc
    
    # Find best
    best_idx = np.unravel_index(results.argmax(), results.shape)
    best_C = param_grid['C'][best_idx[0]]
    best_gamma = param_grid['gamma'][best_idx[1]]
    
    return {
        'val_accuracy_grid': results,
        'best_C': best_C,
        'best_gamma': best_gamma,
        'scaler': scaler,  # Save for later use
    }

def train_final_svm(X_train, y_train, X_test, y_test, 
                   C, gamma, scaler, seed=0):
    """Phase 2: Train with best params"""
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    svm = SVC(C=C, gamma=gamma, kernel='rbf', random_state=seed)
    svm.fit(X_train_scaled, y_train)
    
    y_test_pred = svm.predict(X_test_scaled)
    
    return {
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'test_f1_macro': f1_score(y_test, y_test_pred, average='macro'),
        'test_f1_per_class': f1_score(y_test, y_test_pred, average=None),
        'confusion_matrix': confusion_matrix(y_test, y_test_pred),
        'n_support_vectors': len(svm.support_),
    }
```

## Important Notes

### Classes
- **Exclude 'rest' class** - Only classify 11 movement types
- Labels should be 1-11 (not 0-10)

### Checkpoints
- Save after each grid configuration
- Save after each final trial
- Enables resuming interrupted experiments

### File Structure

```
results/svm/
├── grid_search_results.json
├── trial_results.json
└── aggregated_results.json

figures/svm/
├── grid_search_heatmap.png
├── final_performance.png
├── confusion_matrix.png
├── per_class_f1.png
└── training_time_heatmap.png

checkpoints/svm/
├── grid_C0.1_gamma0.001.json
├── grid_C0.1_gamma0.01.json
├── ...
├── trial_1_seed0.json
├── trial_2_seed1.json
└── ...
```

## Quick Start

```bash
# Run complete experiment
python run_svm_experiment.py

# Check results
ls results/svm/
ls figures/svm/

# Load and analyze
python
>>> import json
>>> with open('results/svm/aggregated_results.json') as f:
...     results = json.load(f)
>>> print(f"Test Acc: {results['test_accuracy_mean']:.4f}")
```
