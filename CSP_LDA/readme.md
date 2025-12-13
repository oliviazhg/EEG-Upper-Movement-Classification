# CSP+LDA Experiment for EEG Movement Classification

This script implements the Common Spatial Patterns (CSP) + Linear Discriminant Analysis (LDA) pipeline for classifying 11 upper-limb movements from EEG signals.

## Overview

The experiment evaluates:
- **3 frequency bands**: mu (8-12 Hz), beta (13-30 Hz), combined (8-30 Hz)
- **2 CSP component counts**: 4, 6 components
- **5 random seeds** per configuration (for statistical robustness)
- **Total**: 30 trials (6 configurations × 5 seeds)

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install numpy pandas scipy scikit-learn mne matplotlib seaborn
```

## Data Format

The script expects preprocessed EEG data in one of these formats:

### Option 1: NumPy (.npz)
```python
# Save your data as:
np.savez('preprocessed_data.npz', 
         X=X,  # Shape: (n_trials, n_channels, n_timepoints)
         y=y)  # Shape: (n_trials,)
```

### Option 2: Pickle (.pkl)
```python
# Save your data as:
with open('preprocessed_data.pkl', 'wb') as f:
    pickle.dump({'X': X, 'y': y}, f)
```

### Data Specifications
- **X**: EEG trials
  - Shape: (n_trials, n_channels, n_timepoints)
  - Example: (1000, 60, 5000) for 1000 trials, 60 channels, 2 seconds at 2500 Hz
  - Data type: float32 or float64 (raw voltage values)
  
- **y**: Class labels
  - Shape: (n_trials,)
  - Values: 0-10 (11 movement classes)
  - Data type: int

## Usage

### Basic Usage
```bash
python csp_lda_experiment.py --data_path /path/to/preprocessed_data.npz
```

### Specify Output Directory
```bash
python csp_lda_experiment.py \
    --data_path /path/to/data.npz \
    --output_dir /path/to/results
```

### Example with Gigascience Dataset
```bash
python csp_lda_experiment.py \
    --data_path ./data/gigascience_preprocessed.npz \
    --output_dir ./results_csp_lda
```

## Output Structure

The script creates the following output structure:

```
results_csp_lda/
├── trial_results.pkl              # Raw trial data (all 30 trials)
├── trial_results.csv              # Trial results in CSV format
├── config_summaries.pkl           # Aggregated statistics per config
├── config_summaries.csv           # Config summaries in CSV
├── statistical_comparisons.csv    # Pairwise statistical tests
├── plot1_parameter_comparison.png # Bar plot: all configs
├── plot1_parameter_comparison.pdf
├── plot2_frequency_band_ablation.png  # Grouped bars: bands
├── plot2_frequency_band_ablation.pdf
├── plot3_confusion_matrix.png     # Best config confusion matrix
├── plot3_confusion_matrix.pdf
├── plot4_per_class_f1.png         # Per-class F1 scores
└── plot4_per_class_f1.pdf
```

## Key Features

### 1. Rigorous Experimental Protocol
- **Stratified splits**: Ensures balanced class distribution across train/val/test
- **5 independent trials** per configuration with different random seeds
- **95% confidence intervals** using bootstrap method
- **Statistical testing** with Bonferroni correction for multiple comparisons

### 2. Comprehensive Metrics
For each trial:
- Train/Val/Test accuracy
- Macro F1 score
- Per-class F1 scores
- Confusion matrix
- Training time
- Inference time per sample

### 3. Statistical Analysis
- Bootstrap confidence intervals (10,000 resamples)
- Paired t-tests between all configuration pairs
- Cohen's d effect sizes
- Bonferroni correction for multiple comparisons

### 4. Visualization
Four publication-quality plots:
1. **Parameter comparison**: All 6 configs with error bars
2. **Frequency band ablation**: Grouped bars comparing bands
3. **Confusion matrix**: Best performing configuration
4. **Per-class F1**: Movement-specific performance

## Customization

### Modify Experimental Parameters

Edit the `CSPLDAExperiment.__init__()` method:

```python
# Change frequency bands
self.frequency_bands = {
    'mu': (8, 12),
    'beta': (13, 30),
    'combined': (8, 30),
    'gamma': (30, 50)  # Add new band
}

# Change component counts
self.n_components_list = [4, 6, 8]  # Test more components

# Change number of trials
self.random_seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 10 trials

# Change data splits
self.train_ratio = 0.80  # 80% training
self.val_ratio = 0.10    # 10% validation
self.test_ratio = 0.10   # 10% testing
```

### Add Custom Metrics

In `train_csp_lda()` method, add your metrics:

```python
# Add precision/recall
from sklearn.metrics import precision_score, recall_score

results['test_precision'] = precision_score(
    y_test, y_test_pred, average='macro'
)
results['test_recall'] = recall_score(
    y_test, y_test_pred, average='macro'
)
```

### Modify CSP Parameters

In `train_csp_lda()` method:

```python
csp = CSP(
    n_components=n_components,
    reg=0.1,           # Add regularization
    log=True,          # Log-transform variance
    norm_trace=True,   # Normalize by trace
    random_state=seed
)
```

## Expected Runtime

For a dataset with ~1000 trials:
- **Per trial**: ~10-30 seconds (depends on n_channels and n_timepoints)
- **Total**: ~5-15 minutes for all 30 trials

Breakdown:
- Filtering: ~1-2 sec/trial
- CSP fitting: ~2-5 sec/trial
- LDA training: <1 sec/trial
- Evaluation: <1 sec/trial

## Troubleshooting

### Memory Issues
If you encounter memory errors:
```python
# Reduce data precision
X = X.astype(np.float32)

# Or downsample in time
X_downsampled = X[:, :, ::2]  # Keep every 2nd sample
```

### Slow Execution
If experiments are too slow:
```python
# Reduce number of trials
self.random_seeds = [0, 1, 2]  # Just 3 trials

# Or reduce bootstrap resamples
result = bootstrap([data], np.mean, n_resamples=1000)  # Instead of 10000
```

### Import Errors
Make sure MNE is properly installed:
```bash
pip install --upgrade mne
```

## Data Preprocessing (Not Included)

Before running this script, you need to preprocess your raw EEG data:

```python
import numpy as np
from scipy.signal import butter, filtfilt

# 1. Load raw data from Gigascience dataset
# 2. Extract trial epochs (e.g., -0.5 to 1.5 seconds around movement onset)
# 3. Remove bad channels/trials
# 4. Optionally apply baseline correction
# 5. Save as .npz or .pkl

# Example preprocessing:
def preprocess_gigascience_data(raw_data_path, output_path):
    # Load data
    # ... your loading code ...
    
    # Extract epochs around movement onset
    trial_duration = 2.0  # seconds
    fs = 2500  # Hz
    n_samples = int(trial_duration * fs)
    
    X = []  # (n_trials, n_channels, n_timepoints)
    y = []  # (n_trials,)
    
    # ... epoch extraction ...
    
    # Save
    np.savez(output_path, X=np.array(X), y=np.array(y))

preprocess_gigascience_data('raw_data/', 'preprocessed_data.npz')
```

## Citation

If you use this code, please cite:

```bibtex
@article{zhang2020upper,
  title={A dataset of multi-channel EEG and EMG during intended and imagined movements of the upper limbs and hands},
  author={Zhang, Yifan and others},
  journal={Scientific Data},
  volume={7},
  number={1},
  pages={1--10},
  year={2020}
}
```

## License

This code is provided for academic and research purposes. Please comply with your institution's policies and the Gigascience dataset license.

## Contact

For questions or issues, please contact:
- Olivia Zheng (o2zheng@uwaterloo.ca)