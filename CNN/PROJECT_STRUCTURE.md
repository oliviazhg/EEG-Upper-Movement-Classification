# 1D-CNN EEG Classification Pipeline - File Structure

## Complete File Listing

```
CNN_EEG_Classification/
│
├── README.md                          # Comprehensive usage guide
├── requirements.txt                   # Python dependencies
│
├── Core Pipeline Scripts              # Main implementation
│   ├── cnn_eeg_classification.py     # CNN model training
│   ├── cnn_data_loading.py           # Data preprocessing
│   ├── cnn_analysis.py               # Results analysis & visualization
│   └── run_cnn_experiments.py        # Master orchestration script
│
├── Utilities
│   ├── test_pipeline.py              # Verify installation
│   └── compare_methods.py            # Cross-method comparison
│
└── Generated Outputs (after running)
    ├── results/
    │   └── cnn/
    │       ├── all_trials.pkl         # Raw experimental data
    │       └── aggregated_results.json # Summary statistics
    │
    └── figures/
        ├── cnn_loss_curves.png        # Training/val loss
        ├── cnn_accuracy_curves.png    # Training/val accuracy
        ├── cnn_architecture_comparison.png
        ├── cnn_confusion_matrix.png
        ├── cnn_per_class_f1.png
        ├── method_comparison.png       # CNN vs CSP/SVM/RF
        └── accuracy_vs_f1.png
```

## File Descriptions

### Core Implementation (4 files)

**1. `cnn_eeg_classification.py`** (480 lines)
- `build_cnn()`: Build 2/3/4-layer architectures
- `train_cnn_trial()`: Train single trial and evaluate
- `run_cnn_experiments()`: Full experimental protocol
- Implements He initialization, early stopping, Adam optimizer
- Saves results after each trial

**2. `cnn_data_loading.py`** (350 lines)
- `bandpass_filter()`: 1-40 Hz Butterworth filter
- `segment_trials()`: Extract 2-second epochs
- `normalize_trials()`: Z-score normalization
- `reject_artifacts()`: Remove noisy trials
- `preprocess_for_cnn()`: Complete pipeline
- `load_subject_data()`: MNE data loading (template)
- `load_multiple_subjects()`: Multi-subject aggregation

**3. `cnn_analysis.py`** (420 lines)
- `compute_bootstrap_ci()`: 95% confidence intervals
- `aggregate_architecture_results()`: Per-architecture summary
- `plot_learning_curves()`: Train/val loss & accuracy
- `plot_architecture_comparison()`: Bar chart
- `plot_confusion_matrix()`: Heatmap for best model
- `plot_per_class_f1()`: Per-class F1 scores
- `analyze_cnn_results()`: Master analysis function

**4. `run_cnn_experiments.py`** (280 lines)
- Command-line interface
- Orchestrates data loading → training → analysis
- Supports synthetic data for testing
- Progress tracking and time estimates

### Utilities (2 files)

**5. `test_pipeline.py`** (220 lines)
- 6 test modules: imports, architecture, preprocessing, training, analysis
- Runs on synthetic data
- Verifies everything works before using real data
- ~2 minutes to run

**6. `compare_methods.py`** (320 lines)
- Load results from multiple methods (CNN, CSP, SVM, RF)
- Generate comparison plots
- Print statistical comparison table
- CI overlap analysis

### Documentation (2 files)

**7. `README.md`**
- Installation instructions
- Usage examples
- Expected results
- Troubleshooting guide
- Customization tips

**8. `requirements.txt`**
- TensorFlow, NumPy, SciPy, scikit-learn
- MNE (EEG processing)
- Matplotlib, Seaborn (visualization)

## Workflow Summary

### Quick Start (Testing)
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Test installation
python test_pipeline.py

# 3. Run with synthetic data
python run_cnn_experiments.py --use_synthetic --depths 2 3 --seeds 0 1
```

### Full Experiments (Real Data)
```bash
# 1. Run all experiments (15 trials)
python run_cnn_experiments.py \
    --data_dir /path/to/data \
    --subjects 1 2 3 4 5 \
    --depths 2 3 4 \
    --seeds 0 1 2 3 4

# 2. Results automatically analyzed and saved to:
#    - results/cnn/all_trials.pkl
#    - results/cnn/aggregated_results.json
#    - figures/cnn_*.png

# 3. Compare with other methods
python compare_methods.py --methods cnn csp svm
```

## Key Design Decisions

**1. Architecture Specification**
- Follows Schirrmeister et al. (2017) Deep4Net
- Progressive kernel size reduction: 50→25→10→5
- Captures temporal features at multiple scales
- Deeper networks have more parameters

**2. Data Pipeline**
- Minimal preprocessing (1-40 Hz) to let CNN learn features
- Broader than CSP (8-30 Hz) which pre-selects features
- Z-score normalization per channel
- 2-second trials capture movement preparation + execution

**3. Training Protocol**
- Early stopping prevents overfitting (patience=10)
- Adam optimizer (lr=0.001) per Lecture 9
- Batch size=32 balances GPU memory and gradient stability
- 5 seeds ensure statistical reliability

**4. Evaluation**
- Bootstrap CIs (10,000 resamples) for test accuracy
- Per-class F1 to detect difficult movements
- Learning curves check for overfitting
- Confusion matrices reveal common misclassifications

**5. Reproducibility**
- All random seeds fixed
- Complete hyperparameter specification
- Saves training history for reanalysis
- Results in standard JSON format

## Expected Timeline

**Testing** (no real data): ~30 minutes
- Install dependencies: 10 min
- Run test_pipeline.py: 2 min  
- Run synthetic experiments (6 trials): 15 min
- Verify outputs: 3 min

**Full Experiments** (with real data): ~3 hours
- Data loading: 10 min
- Training (15 trials × 10 min): 150 min
- Analysis: 5 min
- Total: ~165 min

**Paper Integration**: ~1 hour
- Extract figures for paper
- Copy results table
- Write results section
- Compare with other methods

## Integration with Project

This CNN pipeline directly supports your SYDE 522 final project:

**Introduction & Background**: Already written ✓
- Cites Schirrmeister et al. (2017)
- Explains architectural design
- Justifies hyperparameters

**Methods**: Implemented in code ✓
- Architecture specifications
- Training protocol
- Evaluation metrics

**Results**: Auto-generated ✓
- All required figures
- Summary statistics
- Statistical tests

**Discussion**: Template provided ✓
- Best architecture analysis
- Comparison with baselines
- Limitations and future work

## Output Formats

**For IEEE Paper**:
- All figures saved as PNG (300 dpi) and PDF
- Tables exported as JSON (easy LaTeX conversion)
- Results reported with 95% CIs

**For Presentation**:
- High-resolution figures
- Confusion matrix visualization
- Architecture comparison bar chart

**For Appendix**:
- Complete hyperparameter tables
- Per-class results
- Training curves for all architectures

## Troubleshooting Reference

**Low accuracy (<30%)**
→ Check data labels (should be 0-10, not 1-11)
→ Verify preprocessing didn't corrupt data
→ Try synthetic data to verify code works

**Out of memory**
→ Reduce batch size in `cnn_eeg_classification.py` line 212
→ Use fewer subjects
→ Close other applications

**Slow training**
→ Verify TensorFlow is using GPU: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`
→ Reduce n_timepoints (try 2500 instead of 5000)
→ Use fewer trials per seed

**Import errors**
→ Run `pip install -r requirements.txt`
→ Check Python version (need ≥3.7)

## Citation Information

If using this code for your project, acknowledge:

**Original CNN Architecture**:
Schirrmeister et al. (2017). Deep learning with convolutional neural networks for EEG decoding and visualization. Human Brain Mapping, 38(11), 5391-5420.

**Dataset**:
Zhang et al. (2020). A dataset of multi-channel EEG and EMG during intended and imagined movements of the upper limbs and hands. Scientific Data, 7(1), 1-10.

**Course Reference**:
SYDE 522: Fundamentals of AI, University of Waterloo (2025)
- Lectures 8-9: Backpropagation
- Lecture 12: Modern AI (CNNs)

## Contact

For issues with this implementation:
- Refer to README.md
- Check SYDE 522 course materials
- Email: o2zheng@uwaterloo.ca

---

**Last Updated**: December 2024  
**Version**: 1.0  
**Status**: Ready for experiments ✓
