# EEG-Upper-Movement-Classification

# EEG Preprocessing Pipeline - Guide & Checklist

## Quick Start

```python
from eeg_preprocessing import EEGPreprocessor

# Initialize
preprocessor = EEGPreprocessor(
    data_dir='/path/to/your/data',
    output_dir='/path/to/output'
)

# Test on single file first
test_data = preprocessor.process_file(
    subject='sub1',
    session='session1', 
    movement_type='reaching_realMove',
    filter_low=8,
    filter_high=30
)

# If successful, process full dataset
all_processed = preprocessor.process_dataset()
preprocessor.save_preprocessed(all_processed)
```

## Output Files

After running, you'll have:
- `csp_lda_data.npz` - Trials (8-30 Hz) for CSP+LDA
  - Shape: (n_trials, n_motor_channels, 7500_samples)
- `ml_features_data.npz` - Features for Random Forest/SVM
  - Shape: (n_trials, 240_features) assuming ~60 motor channels
- `cnn_data.npz` - Trials (1-40 Hz) for CNN
  - Shape: (n_trials, n_motor_channels, 7500_samples)

## What the Pipeline Does

### 1. Channel Selection (Motor Cortex Only)
Selected channels based on standard motor BCI literature:
- **Primary motor (M1):** C3, C1, Cz, C2, C4, C5, C6
- **Premotor/SMA:** FC5, FC3, FC1, FCz, FC2, FC4, FC6
- **Somatosensory:** CP5, CP3, CP1, CPz, CP2, CP4, CP6
- **Extended motor:** FT7, FT8

This gives you ~20-25 channels instead of all 60 EEG channels.

**Why motor cortex only?**
- Reduces dimensionality while preserving motor-relevant signals
- Improves signal-to-noise ratio for movement classification
- Standard practice in motor BCI literature

### 2. Trial Extraction
- **Window:** -1.0s to +2.0s around movement trigger
- **Length:** 3 seconds = 7,500 samples at 2500 Hz
- **Includes:** Motor preparation (-1s to 0s) + execution (0s to +2s)

**Why this window?**
- Pre-movement period captures motor planning signals
- Post-movement captures execution and returns to baseline
- From your intro paper: movement preparation signals appear ~300ms before

### 3. Filtering

Different filters for different models based on lecture content:

**CSP+LDA (8-30 Hz):**
- Mu band (8-12 Hz): Motor-related oscillations
- Beta band (13-30 Hz): Movement execution/suppression
- From SYDE 552: Event-related desynchronization in these bands

**Random Forest/SVM (8-30 Hz):**
- Same as CSP since features are extracted from these bands
- Features: mu power, beta power, mean amplitude, std dev per channel

**CNN (1-40 Hz):**
- Broader filter lets CNN learn optimal frequency features
- From lecture: Deep learning can learn features automatically
- Includes low gamma (30-40 Hz) which may contain movement info

### 4. Feature Extraction (ML Models)

For each trial, extracts per-channel:
1. **Mu band power** (8-12 Hz) - Mean squared amplitude
2. **Beta band power** (13-30 Hz) - Mean squared amplitude  
3. **Mean amplitude** - Average signal level
4. **Standard deviation** - Signal variability

**Total features:** 4 × n_motor_channels ≈ 240 features

**Why these features?**
- Band power: Standard in motor BCI (from lectures)
- Mean/std: Capture additional signal characteristics
- Proven effective for classical ML (Random Forest, SVM)

## Validation Checklist

### Before Running on Full Dataset:

1. **Test Single File:**
```python
test = preprocessor.process_file('sub1', 'session1', 'reaching_realMove', 8, 30)
print(f"Trials shape: {test['trials'].shape}")
print(f"Labels: {np.unique(test['labels'])}")
print(f"Channels: {test['channel_names']}")
```

2. **Check Expected Outputs:**
- Number of motor channels: Should be 20-27 (depends on dataset)
- Number of trials per file: Varies, but typically 30-100
- Trial shape: (n_trials, n_channels, 7500)
- Classes: Should see all 11 movement classes across full dataset

3. **Verify Data Quality:**
```python
# Check for NaN/Inf
assert not np.any(np.isnan(test['trials']))
assert not np.any(np.isinf(test['trials']))

# Check reasonable voltage range (EEG typically ±100 µV)
print(f"Voltage range: {test['trials'].min():.2f} to {test['trials'].max():.2f} µV")
```

### After Preprocessing:

1. **Load and inspect:**
```python
data = np.load('preprocessed_eeg/csp_lda_data.npz', allow_pickle=True)
X = data['X']
y = data['y']

print(f"Total trials: {len(y)}")
print(f"Class distribution: {np.bincount(y.astype(int))}")
print(f"Trial shape: {X[0].shape}")
```

2. **Expected totals (full dataset):**
- 25 subjects × 3 sessions × 3 movement types × ~40 trials/file ≈ 9,000 trials
- 11 classes (6 reaches + 3 grasps + 2 twists)
- Approximately balanced (but verify)

## Potential Issues & Solutions

### Issue 1: Memory
**Problem:** Processing 9000 trials × 25 channels × 7500 samples ≈ 1.7 GB
**Solution:** Process in batches by subject/session

```python
# Process subjects in batches
for batch_start in range(0, 25, 5):
    subjects = [f'sub{i}' for i in range(batch_start+1, batch_start+6)]
    batch_data = preprocessor.process_dataset(subjects=subjects)
    # ... save batch ...
```

### Issue 2: Missing Files
**Problem:** Some files may be corrupted/missing
**Solution:** Pipeline gracefully skips and reports

Check the output for warnings like:
```
Warning: File not found: session2_sub15_twist_realMove.mat
```

### Issue 3: Class Imbalance
**Problem:** Some movement classes may have fewer trials
**Solution:** Check distribution before training

```python
unique, counts = np.unique(y, return_counts=True)
for cls, count in zip(unique, counts):
    print(f"Class {cls}: {count} trials ({count/len(y)*100:.1f}%)")
```

If severely imbalanced, use stratified splitting.

### Issue 4: Channel Names Mismatch
**Problem:** Dataset may use slightly different naming
**Solution:** The code handles common variants

If you see "Selected 0 motor cortex channels":
1. Print all channel names from one file
2. Update MOTOR_CHANNELS list in the class
3. Verify against 10-20 system standard

## What You Might Be Missing

### 1. Artifact Rejection (Optional but Recommended)
The current pipeline does NOT remove artifacts. Consider adding:

**Eye blinks/movements:**
- High variance in EOG channels (you have hEOG, vEOG in dataset)
- Could reject trials where EOG variance > threshold

**Muscle artifacts:**
- High frequency components > 40 Hz
- Sudden amplitude spikes

**Implementation:**
```python
def reject_artifacts(self, trials, threshold_std=3):
    """Reject trials with extreme values."""
    trial_stds = np.std(trials, axis=(1,2))
    mean_std = np.mean(trial_stds)
    std_std = np.std(trial_stds)
    
    valid_mask = trial_stds < (mean_std + threshold_std * std_std)
    return trials[valid_mask], valid_mask
```

### 2. Data Normalization (May Help CNNs)
Current pipeline uses raw voltage values. For CNNs, consider:

**Z-score normalization** (per channel, per trial):
```python
def normalize_trials(trials):
    """Z-score normalize each channel in each trial."""
    normalized = np.zeros_like(trials)
    for i in range(len(trials)):
        for ch in range(trials.shape[1]):
            normalized[i, ch] = zscore(trials[i, ch])
    return normalized
```

**When to use:**
- CNN: Often helps convergence
- CSP+LDA: CSP handles scaling, LDA assumes normality
- Random Forest: Doesn't require normalization
- SVM: Important! Should normalize features

### 3. Cross-Validation Strategy
The preprocessing prepares data, but you need to decide:

**Within-subject:**
- Split each subject's data into train/val/test
- Better performance but doesn't generalize to new subjects

**Cross-subject:**
- Train on N-1 subjects, test on held-out subject
- More challenging but realistic for clinical deployment

**Recommended approach** (from your intro):
```python
from sklearn.model_selection import StratifiedKFold

# 70-15-15 split as mentioned
# First: 70% train, 30% temp
# Then: split temp into 15% val, 15% test

from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)
```

### 4. Baseline Removal (Optional)
Some researchers remove pre-movement baseline:

```python
def remove_baseline(trials, baseline_period=(0, 1000)):
    """Remove mean of baseline period from each trial."""
    corrected = trials.copy()
    for i in range(len(trials)):
        baseline = trials[i, :, baseline_period[0]:baseline_period[1]]
        baseline_mean = np.mean(baseline, axis=1, keepdims=True)
        corrected[i] -= baseline_mean
    return corrected
```

### 5. Feature Scaling for SVM
**CRITICAL for SVM:** Must scale features before training

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
```

## Next Steps

1. **Run preprocessing** on a small subset first (1-2 subjects)
2. **Validate outputs** using checklist above
3. **Implement train/val/test split** with stratification
4. **Add normalization** for CNNs and feature scaling for SVM
5. **Consider artifact rejection** if you see noisy trials
6. **Document your choices** for the final report

## Expected Performance (Baseline)

Based on BCI Competition literature for motor tasks:

- **Random guessing:** ~9% (11 classes)
- **CSP+LDA:** 40-60% (typical for motor imagery, higher for real movement)
- **Random Forest:** 45-65%
- **SVM:** 50-70%
- **CNN:** 55-75% (if trained well)

**Your goal:** Establish these baselines, then explore improvements in the 4-day timeframe.

## Time Estimates

- **Preprocessing:** ~30-60 min for full dataset (25 subjects)
- **CSP+LDA training:** ~10-30 min
- **Random Forest training:** ~5-15 min
- **SVM training:** ~1-3 hours (grid search is slow)
- **CNN training:** ~2-6 hours per architecture (GPU recommended)

**Total compute time:** 12-16 hours as you estimated ✓

## Questions to Consider

1. **Within-subject vs cross-subject evaluation?**
   - Affects experimental design and results interpretation

2. **Should you use all 11 classes or group some?**
   - E.g., combine all reaches, all grasps, all twists → 3-class problem
   - Easier but less clinically relevant

3. **Train on all movement types together or separately?**
   - Separate models might perform better but less practical

4. **Use all motor channels or select optimal subset?**
   - CSP will select optimal combinations automatically
   - For ML/CNN, channel selection could reduce overfitting

Good luck with your experiments! The preprocessing is solid based on standard BCI practices from the lectures.
