# Random Forest Experiment Suite
## EEG-Based Upper-Limb Movement Classification

**Author:** Olivia Zheng  
**Course:** SYDE 522 - Fundamentals of Artificial Intelligence  
**Institution:** University of Waterloo

---

## Overview

This suite implements Random Forest classification for decoding 11 distinct upper-limb movements from 60-channel EEG signals. The implementation follows rigorous experimental protocols with proper statistical validation.

### Key Features

- ✅ **11-class classification** (excluding rest state)
- ✅ **240 hand-crafted features** (mu/beta power, mean, std)
- ✅ **Stratified train/val/test splits** (70/15/15)
- ✅ **5 independent trials** per configuration
- ✅ **Bootstrap confidence intervals** (95%)
- ✅ **Statistical significance testing** (Bonferroni-corrected)
- ✅ **Publication-quality figures** (300 DPI)

---

## Files in This Suite

### 1. `random_forest_experiment.py`
**Main experimental pipeline**

Implements the complete Random Forest classification experiment including:
- Feature extraction (240 features from 60 channels)
- Data splitting with stratification
- Model training and evaluation
- Statistical analysis with bootstrap CIs
- Automated figure generation

**Parameters tested:**
- `n_estimators`: 100, 200 trees
- `max_features`: sqrt(240)=15, 0.3×240=72

### 2. `test_random_forest.py`
**Testing script with synthetic data**

Generates realistic synthetic EEG data to verify the pipeline works correctly before running on real data.

Features:
- Synthetic data generation with class-specific patterns
- Diagnostic visualizations (time series, spectra, distributions)
- Quick test run with reduced parameter space

### 3. `analyze_random_forest_results.py`
**Post-hoc analysis and visualization**

Performs detailed statistical analysis and generates additional visualizations:
- Pairwise statistical comparisons with Bonferroni correction
- Cohen's d effect sizes
- Feature importance stability analysis
- Additional plots (learning curves, timing comparisons, etc.)

---

## Installation

### Requirements

```bash
pip install numpy scipy matplotlib seaborn pandas scikit-learn
```

Or use the provided requirements file:

```bash
pip install -r requirements_random_forest.txt
```

---

## Usage

### Quick Start (Synthetic Data Test)

```bash
# Test the pipeline with synthetic data
python test_random_forest.py
```

This will:
1. Generate synthetic EEG data (550 trials, 11 classes)
2. Create diagnostic plots
3. Run abbreviated experiment (2 configs × 2 seeds)
4. Verify the pipeline works correctly

**Expected output:**
- `./test_data/` - Synthetic data and visualizations
- `./test_results/random_forest/` - Test experiment results

### Real Data Experiment

#### Option 1: Using preprocessed .npz file

```bash
python random_forest_experiment.py --data_path ./preprocessed/rf_data.npz
```

#### Option 2: Using directory with .npy files

```bash
python random_forest_experiment.py --data_path ./data/
```

The directory should contain:
- `eeg_data.npy` - Shape: (n_trials, 60, 5000)
- `labels.npy` - Shape: (n_trials,) with values 0-11

**Note:** Class 0 (rest) will be automatically removed, leaving classes 0-10 for the 11 movements.

### Post-Hoc Analysis

```bash
# After running the main experiment
python analyze_random_forest_results.py --results_dir ./results/random_forest
```

---

## Data Format

### Input Data

The experiment expects EEG data in one of two formats:

#### Format 1: .npz file (recommended)
```python
data = np.load('rf_data.npz')
X = data['X']  # Shape: (n_trials, 60, 5000)
y = data['y']  # Shape: (n_trials,) with values 0-11
```

#### Format 2: Separate .npy files
```
data/
├── eeg_data.npy  # Shape: (n_trials, 60, 5000)
└── labels.npy    # Shape: (n_trials,) with values 0-11
```

### Data Specifications

- **Sampling rate:** 2500 Hz
- **Trial duration:** 2.0 seconds (5000 samples)
- **Channels:** 60 EEG channels
- **Classes:** 
  - 0: Rest (automatically removed)
  - 1-11: 11 upper-limb movements
    1. Forward reach
    2. Backward reach
    3. Left reach
    4. Right reach
    5. Up reach
    6. Down reach
    7. Power grasp
    8. Precision grasp
    9. Lateral grasp
    10. Pronation
    11. Supination

### Class Imbalance Handling

**IMPORTANT:** The dataset contains approximately:
- Rest (class 0): ~550 trials
- Each movement: ~50 trials

This creates a severe 11:1 imbalance. The experiment automatically removes rest to focus on classifying the 11 movements. After removal:
- Classes are renumbered 0-10
- Each class has ~50 trials
- Balanced classification task

---

## Feature Extraction

### 240 Features Total (4 per channel × 60 channels)

For each of the 60 EEG channels, we extract:

1. **Mu band power (8-12 Hz)** - 60 features
   - Bandpass filter → compute variance
   
2. **Beta band power (13-30 Hz)** - 60 features
   - Bandpass filter → compute variance
   
3. **Mean amplitude** - 60 features
   - Temporal average of raw signal
   
4. **Standard deviation** - 60 features
   - Temporal variability of raw signal

### Feature Vector Organization

```python
features = [
    mu_ch1, mu_ch2, ..., mu_ch60,      # Indices 0-59
    beta_ch1, beta_ch2, ..., beta_ch60, # Indices 60-119
    mean_ch1, mean_ch2, ..., mean_ch60, # Indices 120-179
    std_ch1, std_ch2, ..., std_ch60     # Indices 180-239
]
```

---

## Experimental Protocol

### Configurations Tested

| Config | n_estimators | max_features | Description |
|--------|-------------|--------------|-------------|
| 1 | 100 | sqrt(240)=15 | Standard, fewer trees |
| 2 | 100 | 0.3×240=72 | More features, fewer trees |
| 3 | 200 | sqrt(240)=15 | Standard, more trees |
| 4 | 200 | 0.3×240=72 | More features, more trees |

### Trial Protocol

For each configuration:
1. **5 independent trials** with random seeds [0, 1, 2, 3, 4]
2. **Stratified data split:**
   - Train: 70% (~385 trials)
   - Validation: 15% (~83 trials)
   - Test: 15% (~83 trials)
3. **Metrics collected per trial:**
   - Train/val/test accuracy
   - Train/val/test F1 (macro)
   - Per-class F1 scores
   - Confusion matrix
   - Feature importances
   - Training/inference time
   - Out-of-bag score

### Statistical Validation

1. **Bootstrap Confidence Intervals:**
   - 95% CI using 10,000 resamples
   - Applied to all performance metrics

2. **Pairwise Comparisons:**
   - Paired t-tests between all configurations
   - Bonferroni correction for multiple comparisons
   - Cohen's d effect sizes

3. **Feature Importance Stability:**
   - Spearman correlation between trials
   - Should be > 0.8 for stable features

---

## Output Structure

```
results/random_forest/
├── data/
│   ├── features.npy                    # Extracted features (n_trials, 240)
│   ├── labels.npy                      # Class labels (n_trials,)
│   ├── feature_names.pkl               # List of feature names
│   ├── results_n100_mfsqrt.pkl        # Results for config 1
│   ├── results_n100_mf72.pkl          # Results for config 2
│   ├── results_n200_mfsqrt.pkl        # Results for config 3
│   ├── results_n200_mf72.pkl          # Results for config 4
│   └── config_summaries.pkl            # Aggregated statistics
│
├── figures/
│   ├── parameter_comparison.png        # Bar plot with error bars
│   ├── feature_importance.png          # Top 20 features
│   ├── confusion_matrix.png            # Best config CM
│   ├── per_class_f1.png               # Per-class F1 scores
│   └── posthoc/                        # Additional analyses
│       ├── accuracy_distribution.png
│       ├── feature_category_contribution.png
│       ├── training_time_comparison.png
│       ├── learning_curves.png
│       └── statistical_significance_matrix.png
│
├── experiment_report.txt               # Summary report
├── posthoc_analysis_report.txt        # Detailed analysis
└── statistical_tests.csv              # Pairwise comparisons
```

---

## Interpreting Results

### Key Metrics

1. **Test Accuracy (with 95% CI)**
   - Primary metric for model comparison
   - Example: 0.645 [0.621, 0.668]
   - Interpretation: 64.5% accuracy, CI doesn't overlap with chance (9.1%)

2. **Macro F1 Score**
   - Accounts for class imbalance
   - Average F1 across all 11 classes
   - More informative than accuracy for multiclass problems

3. **Per-Class F1 Scores**
   - Identifies which movements are easier/harder to classify
   - Look for systematic patterns (e.g., reaches vs grasps)

4. **Confusion Matrix**
   - Shows common misclassifications
   - Diagonal = correct classifications
   - Look for confusion between similar movements

### Statistical Significance

- **p < 0.05 (Bonferroni-corrected):** Significant difference
- **Cohen's d interpretation:**
  - |d| < 0.2: Negligible
  - 0.2 ≤ |d| < 0.5: Small
  - 0.5 ≤ |d| < 0.8: Medium
  - |d| ≥ 0.8: Large

### Feature Importance

- **Top features:** Most discriminative for classification
- **Feature categories:** Which signal properties matter most?
  - Mu power: Related to motor preparation
  - Beta power: Related to motor execution
  - Mean/std: Capture other signal characteristics

---

## Comparison with Other Methods

### Expected Performance Hierarchy

Based on literature and course material:

1. **CNN (Deep Learning):** ~70-80%
   - Learns features automatically
   - Requires more data
   
2. **SVM with RBF Kernel:** ~65-75%
   - Sophisticated decision boundaries
   - Kernel trick provides flexibility
   
3. **Random Forest:** ~60-70%
   - Robust to overfitting
   - Good baseline performance
   
4. **CSP+LDA:** ~55-65%
   - Classic BCI approach
   - Strong for binary problems

### When to Use Random Forest

**Advantages:**
- Minimal hyperparameter tuning
- Built-in feature importance
- Robust to outliers and overfitting
- Fast training and inference
- Good baseline for comparison

**Limitations:**
- Cannot capture complex interactions as well as deep learning
- Fixed feature extraction (no learning)
- May underperform on very large datasets
- Interpretation limited to feature importance

---

## Troubleshooting

### Common Issues

#### 1. Memory Error During Feature Extraction

**Problem:** `MemoryError` when processing large datasets

**Solution:**
```python
# Process in batches
batch_size = 100
for i in range(0, len(eeg_data), batch_size):
    batch = eeg_data[i:i+batch_size]
    features_batch = extract_all_features(batch)
```

#### 2. Class Imbalance Warning

**Problem:** Warning about unbalanced classes

**Solution:** This is expected! The code uses `class_weight='balanced'` to handle this automatically. You can verify:
```python
from sklearn.utils.class_weight import compute_class_weight
weights = compute_class_weight('balanced', 
                               classes=np.unique(y_train),
                               y=y_train)
```

#### 3. Low Accuracy (<50%)

**Possible causes:**
- Rest class not removed properly (check class distribution)
- Features not normalized (some algorithms sensitive to scale)
- Data quality issues (artifacts not removed)
- Insufficient data per class

**Check:**
```python
# Verify class distribution
unique, counts = np.unique(labels, return_counts=True)
print(dict(zip(unique, counts)))

# Check feature ranges
print(f"Feature min: {features.min()}")
print(f"Feature max: {features.max()}")
```

#### 4. Inconsistent Results Across Seeds

**Problem:** Very high variance in accuracy across trials

**Possible causes:**
- Small dataset (high sampling variance)
- Unstable features (check feature importance correlation)

**Solution:** Increase number of trials or use more data

---

## Integration with Preprocessing Pipeline

If you have the preprocessing script from earlier conversations:

```python
# In preprocessing script, save data in compatible format:
np.savez_compressed(
    'rf_data.npz',
    X=eeg_data,  # Shape: (n_trials, 60, 5000)
    y=labels,    # Shape: (n_trials,)
    metadata={
        'sampling_rate': 2500,
        'duration': 2.0,
        'n_channels': 60
    }
)

# Then run Random Forest experiment:
# python random_forest_experiment.py --data_path rf_data.npz
```

---

## References

### Dataset
Zhang, Y., et al. (2020). "A dataset of multi-channel EEG and EMG during intended and imagined movements of the upper limbs and hands." *Scientific Data*, 7(1), 1-10.

### Methods
- Breiman, L. (2001). "Random forests." *Machine Learning*, 45(1), 5-32.
- Pfurtscheller, G., & Lopes da Silva, F. H. (1999). "Event-related EEG/MEG synchronization and desynchronization." *Clinical Neurophysiology*, 110(11), 1842-1857.

### Related Work
- Lotte, F., et al. (2018). "A review of classification algorithms for EEG-based brain–computer interfaces." *Journal of Neural Engineering*, 15(3), 031001.

---

## Contact

For questions or issues related to this code:
- **Author:** Olivia Zheng
- **Email:** o2zheng@uwaterloo.ca
- **Course:** SYDE 522, Fall 2025

---

## License

This code is provided for educational purposes as part of SYDE 522 coursework.

---

## Acknowledgments

- Prof. Terrence Stewart for course instruction
- Gigascience dataset authors for providing the data
- Scikit-learn developers for the Random Forest implementation