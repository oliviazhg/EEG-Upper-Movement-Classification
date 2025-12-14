# 1D-CNN EEG Classification Pipeline
**SYDE 522 Final Project - Upper-Limb Movement Classification**

This pipeline implements 1D Convolutional Neural Networks for classifying 11 upper-limb movements from 60-channel EEG signals during real physical movements.

## Overview

**Architecture Variants:**
- 2-layer CNN (64, 128 filters)
- 3-layer CNN (64, 128, 256 filters)  
- 4-layer CNN (64, 128, 256, 512 filters)

**Experimental Protocol:**
- 5 independent trials per architecture (seeds: 0, 1, 2, 3, 4)
- Data split: 70% train, 15% validation, 15% test
- Early stopping (patience=10 on validation loss)
- Adam optimizer (lr=0.001), batch size=32

**Data:**
- Gigascience 2020 Upper Limb Movement Dataset
- 60 EEG channels (motor cortex), 2500 Hz sampling
- 11 movement classes: 6 reaches, 3 grasps, 2 wrist rotations
- Preprocessing: 1-40 Hz bandpass, 2-second trials

## Project Structure

```
.
├── cnn_eeg_classification.py    # Core CNN training code
├── cnn_data_loading.py          # Data loading & preprocessing
├── cnn_analysis.py              # Results analysis & visualization
├── run_cnn_experiments.py       # Master script (run this!)
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── data/                        # Place your EEG data here
│   └── subject_01/
│       └── session_1/
│           └── real_movements.fif
│
├── results/                     # Experimental results
│   └── cnn/
│       ├── all_trials.pkl       # Raw trial data
│       └── aggregated_results.json
│
└── figures/                     # Generated plots
    ├── cnn_loss_curves.png
    ├── cnn_accuracy_curves.png
    ├── cnn_architecture_comparison.png
    ├── cnn_confusion_matrix.png
    └── cnn_per_class_f1.png
```

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Verify TensorFlow installation:**
```bash
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
```

For GPU support (recommended for faster training):
```bash
pip install tensorflow-gpu
```

## Usage

### Quick Start (with synthetic data)

Test the pipeline without real data:

```bash
python run_cnn_experiments.py --use_synthetic --depths 2 3 --seeds 0 1
```

This runs a quick test with:
- 2-layer and 3-layer CNNs
- 2 random seeds (instead of 5)
- Synthetic EEG data (1100 trials)

### Full Experiments (with real data)

```bash
python run_cnn_experiments.py \
    --data_dir /path/to/gigascience/data \
    --subjects 1 2 3 4 5 \
    --session 1 \
    --condition real \
    --depths 2 3 4 \
    --seeds 0 1 2 3 4 \
    --output_dir results/cnn
```

**Parameters:**
- `--data_dir`: Root directory containing EEG data
- `--subjects`: Subject IDs to include (default: all 25)
- `--session`: Session number (1, 2, or 3)
- `--condition`: 'real' or 'imagined' movements
- `--depths`: CNN architectures to test (2, 3, and/or 4 layers)
- `--seeds`: Random seeds for independent trials
- `--output_dir`: Where to save results
- `--verbose`: 0 (quiet), 1 (normal), or 2 (detailed)

### Analyze Existing Results

If you already have trained models and just want to regenerate figures:

```bash
python run_cnn_experiments.py --skip_training --output_dir results/cnn
```

Or directly:

```bash
python cnn_analysis.py
```

## Step-by-Step Workflow

### 1. Prepare Your Data

Organize EEG data following this structure:
```
data/
├── subject_01/
│   ├── session_1/
│   │   ├── real_movements.fif      # MNE-compatible format
│   │   └── imagined_movements.fif
│   ├── session_2/
│   └── session_3/
├── subject_02/
...
```

If your data is in a different format, modify `load_subject_data()` in `cnn_data_loading.py`.

### 2. Run Experiments

```bash
# Full protocol (15 trials total)
python run_cnn_experiments.py --data_dir data/ --subjects 1 2 3

# Faster test (6 trials)
python run_cnn_experiments.py --data_dir data/ --subjects 1 --depths 2 3 --seeds 0 1
```

Training time: ~10 minutes per trial on GPU, ~30 minutes on CPU.

### 3. View Results

Results are automatically analyzed after training. Check:

**Figures** (`figures/`):
- `cnn_loss_curves.png`: Training and validation loss per architecture
- `cnn_accuracy_curves.png`: Training and validation accuracy
- `cnn_architecture_comparison.png`: Bar plot comparing test accuracy
- `cnn_confusion_matrix.png`: Confusion matrix for best architecture
- `cnn_per_class_f1.png`: Per-class F1 scores

**Data** (`results/cnn/`):
- `all_trials.pkl`: Complete trial data (for reanalysis)
- `aggregated_results.json`: Summary statistics with confidence intervals

## Output Metrics

For each architecture, you'll get:

**Performance:**
- Test accuracy (mean ± 95% CI across 5 seeds)
- Test F1 score (macro-averaged)
- Per-class F1 scores (11 classes)
- Confusion matrix

**Training Dynamics:**
- Train/validation loss curves (with 95% CI)
- Train/validation accuracy curves
- Number of epochs to convergence

**Model Characteristics:**
- Total trainable parameters
- Training time
- Inference time per sample

## Expected Results

Based on the project proposal, you should see:

- **Test accuracy:** 70-85% (depending on architecture and data quality)
- **Best architecture:** Likely 3-layer or 4-layer CNN
- **Convergence:** 30-50 epochs typically
- **Overfitting check:** Train accuracy should be within ~5-10% of validation

If train >> val accuracy → increase dropout or reduce model size
If both plateau early → try larger model or more data

## Troubleshooting

### "Data file not found"
- Check `--data_dir` path
- Verify file naming convention in `cnn_data_loading.py`
- Use `--use_synthetic` to test without real data

### "Out of memory" error
- Reduce batch size: edit `cnn_eeg_classification.py` line 212 (batch_size=32 → 16)
- Use fewer subjects
- Close other applications

### "No module named tensorflow"
- Run: `pip install tensorflow`
- For GPU: `pip install tensorflow-gpu`

### Very low accuracy (<20%)
- Check data labels are 0-indexed (not 1-indexed)
- Verify preprocessing didn't corrupt data
- Try with synthetic data first to verify code works

## Customization

### Change CNN architecture

Edit `build_cnn()` in `cnn_eeg_classification.py`:

```python
# Add a 5th layer
if depth == 5:
    model.add(layers.Conv1D(1024, kernel_size=3, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
```

### Change preprocessing

Edit `preprocess_for_cnn()` in `cnn_data_loading.py`:

```python
# Different frequency band
low_freq = 8.0  # mu band only
high_freq = 13.0

# Different trial length
tmin = -0.5  # Include pre-movement
tmax = 2.5
```

### Change hyperparameters

Edit `train_cnn_trial()` in `cnn_eeg_classification.py`:

```python
# Different optimizer
optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

# Different early stopping
patience=20  # More patient
```

## Integration with Other Methods

This pipeline follows the same structure as the CSP+LDA pipeline for easy comparison:

```python
# Compare CNN vs CSP+LDA
from cnn_analysis import analyze_cnn_results
from csp_lda_analysis import analyze_csp_results

cnn_results = analyze_cnn_results('results/cnn/all_trials.pkl')
csp_results = analyze_csp_results('results/csp/all_trials.pkl')

# Compare best accuracies
print(f"Best CNN: {cnn_results[0]['test_acc_mean']:.3f}")
print(f"Best CSP: {csp_results[0]['test_acc_mean']:.3f}")
```

## Citation

If you use this code, please cite:

**Dataset:**
```
Zhang et al. (2020). A dataset of multi-channel EEG and EMG during intended 
and imagined movements of the upper limbs and hands. Scientific Data, 7(1), 1-10.
```

**CNN Architecture:**
```
Schirrmeister et al. (2017). Deep learning with convolutional neural networks 
for EEG decoding and visualization. Human Brain Mapping, 38(11), 5391-5420.
```

## Contact

For questions about this implementation:
- Check GitHub issues
- Refer to SYDE 522 course materials (Lectures 8-9, 12)
- Contact: o2zheng@uwaterloo.ca

## License

This code is provided for educational purposes as part of SYDE 522.
