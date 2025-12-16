"""
Data Validation Script for CNN Experiments
Quickly checks if preprocessed data is in correct format before starting experiments

Usage:
    python validate_cnn_data.py --data-path /path/to/cnn_data.npz
"""

import numpy as np
import argparse
from pathlib import Path


def validate_data_file(data_path: str) -> bool:
    """
    Validate preprocessed CNN data file.
    
    Returns:
    --------
    bool: True if all checks pass, False otherwise
    """
    print(f"\n{'='*70}")
    print(f"CNN DATA VALIDATION")
    print(f"{'='*70}\n")
    
    data_path = Path(data_path)
    
    # Warn if wrong file
    if 'csp_lda' in data_path.name or 'ml_features' in data_path.name:
        print(f"  ⚠️  WARNING: You're validating '{data_path.name}'")
        print(f"     For CNN experiments, you need 'cnn_data.npz'")
        print(f"     (csp_lda_data.npz and ml_features_data.npz are for other models)")
        print()
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return False
        print()
    
    # Check 1: File exists
    print(f"[1/10] Checking file exists...")
    if not data_path.exists():
        print(f"  ❌ FAILED: File not found: {data_path}")
        return False
    print(f"  ✓ File found: {data_path}")
    print(f"  File size: {data_path.stat().st_size / 1e9:.2f} GB")
    
    # Check 2: Can load file
    print(f"\n[2/10] Loading data...")
    try:
        # Use mmap_mode to avoid loading entire file into memory
        data = np.load(data_path, allow_pickle=True, mmap_mode='r')
    except Exception as e:
        print(f"  ❌ FAILED: Cannot load file: {e}")
        return False
    print(f"  ✓ File loaded successfully (memory-mapped)")
    
    # Check 3: Required keys present
    print(f"\n[3/10] Checking required keys...")
    required_keys = ['X', 'y']
    for key in required_keys:
        if key not in data:
            print(f"  ❌ FAILED: Missing key '{key}'")
            return False
    print(f"  ✓ All required keys present: {list(data.keys())}")
    
    # Access data via memory mapping (doesn't load full arrays)
    X = data['X']
    y = data['y']
    
    print(f"  ℹ️  Using memory-mapped mode (arrays not fully loaded)")
    
    # Check 4: X shape
    print(f"\n[4/10] Checking X shape...")
    print(f"  X shape: {X.shape}")
    
    if X.ndim != 3:
        print(f"  ❌ FAILED: X should be 3D (n_trials, n_channels, n_samples), got {X.ndim}D")
        return False
    
    n_trials, n_channels, n_samples = X.shape
    print(f"  ✓ X is 3D array")
    print(f"    - Trials: {n_trials:,}")
    print(f"    - Channels: {n_channels}")
    print(f"    - Samples: {n_samples:,}")
    
    # Check 5: y shape
    print(f"\n[5/10] Checking y shape...")
    print(f"  y shape: {y.shape}")
    
    if y.ndim != 1:
        print(f"  ❌ FAILED: y should be 1D, got {y.ndim}D")
        return False
    
    if len(y) != n_trials:
        print(f"  ❌ FAILED: y length ({len(y)}) doesn't match X trials ({n_trials})")
        return False
    
    print(f"  ✓ y is 1D array with correct length")
    
    # Check 6: Data types
    print(f"\n[6/10] Checking data types...")
    print(f"  X dtype: {X.dtype}")
    print(f"  y dtype: {y.dtype}")
    
    if not np.issubdtype(X.dtype, np.floating):
        print(f"  ⚠️  WARNING: X dtype is {X.dtype}, expected float32 or float64")
        print(f"     (will be converted automatically, but uses more memory)")
    else:
        print(f"  ✓ X has floating point dtype")
    
    if not np.issubdtype(y.dtype, np.integer):
        print(f"  ⚠️  WARNING: y dtype is {y.dtype}, expected integer")
    else:
        print(f"  ✓ y has integer dtype")
    
    # Check 7: Value ranges (sample to avoid loading full array)
    print(f"\n[7/10] Checking value ranges...")
    
    # Sample random subset to check (much faster than loading all 70GB)
    sample_size = min(100, n_trials)  # Check 100 trials max
    sample_indices = np.random.choice(n_trials, size=sample_size, replace=False)
    X_sample = X[sample_indices]  # Only loads sampled trials
    
    print(f"  Sampling {sample_size} trials for validation...")
    
    # Check for NaN or Inf in sample
    if np.any(np.isnan(X_sample)):
        print(f"  ❌ FAILED: X contains NaN values (found in sample)")
        return False
    if np.any(np.isinf(X_sample)):
        print(f"  ❌ FAILED: X contains Inf values (found in sample)")
        return False
    print(f"  ✓ No NaN or Inf in sampled data")
    
    # Check voltage range (typical EEG is ±100 µV, but after filtering could be different)
    x_min, x_max = X_sample.min(), X_sample.max()
    print(f"  X range (sampled): [{x_min:.6f}, {x_max:.6f}]")
    
    if abs(x_max) > 1e6:
        print(f"  ⚠️  WARNING: X values very large, check if scaling is correct")
    
    del X_sample  # Free memory
    
    # Check 8: Class distribution
    print(f"\n[8/10] Checking class distribution...")
    unique_classes = np.unique(y)
    print(f"  Unique classes: {unique_classes}")
    print(f"  Number of classes: {len(unique_classes)}")
    
    if len(unique_classes) != 11:
        print(f"  ⚠️  WARNING: Expected 11 classes (0-10), found {len(unique_classes)}")
    
    print(f"\n  Class distribution:")
    for cls in unique_classes:
        count = np.sum(y == cls)
        percentage = count / len(y) * 100
        print(f"    Class {cls:2d}: {count:5d} trials ({percentage:5.2f}%)")
    
    # Check for severe class imbalance
    class_counts = [np.sum(y == cls) for cls in unique_classes]
    max_count = max(class_counts)
    min_count = min(class_counts)
    imbalance_ratio = max_count / min_count
    
    if imbalance_ratio > 5:
        print(f"  ⚠️  WARNING: Severe class imbalance (ratio: {imbalance_ratio:.1f})")
        print(f"     Consider using stratified splits (already implemented in experiments)")
    else:
        print(f"  ✓ Class balance reasonable (ratio: {imbalance_ratio:.1f})")
    
    # Check 9: Memory requirements
    print(f"\n[9/10] Estimating memory requirements...")
    
    # Calculate memory from shape and dtype (not .nbytes which would load array)
    bytes_per_element = np.dtype(X.dtype).itemsize
    total_elements = np.prod(X.shape)
    x_memory_gb = (total_elements * bytes_per_element) / 1e9
    
    print(f"  X memory: {x_memory_gb:.2f} GB")
    
    # Estimated peak memory during training
    # Rule of thumb: need ~3-4x data size for model, gradients, optimizer states
    estimated_peak_gb = x_memory_gb * 4
    print(f"  Estimated peak GPU memory: ~{estimated_peak_gb:.1f} GB")
    
    if estimated_peak_gb > 200:
        print(f"  ⚠️  WARNING: High memory usage expected")
        print(f"     Consider:")
        print(f"       - Reducing batch size (--batch-size 16 or 8)")
        print(f"       - Running one architecture at a time")
    else:
        print(f"  ✓ Memory requirements reasonable")
    
    # Check 10: Sample split feasibility
    print(f"\n[10/10] Checking data split feasibility...")
    
    train_size = int(0.70 * n_trials)
    val_size = int(0.15 * n_trials)
    test_size = int(0.15 * n_trials)
    
    print(f"  Expected splits (70-15-15):")
    print(f"    Train: {train_size:,} trials")
    print(f"    Val:   {val_size:,} trials")
    print(f"    Test:  {test_size:,} trials")
    
    # Check minimum samples per class in test set
    min_samples_per_class = min([np.sum(y == cls) for cls in unique_classes])
    expected_test_samples_per_class = int(0.15 * min_samples_per_class)
    
    if expected_test_samples_per_class < 5:
        print(f"  ⚠️  WARNING: Small test set for some classes (~{expected_test_samples_per_class} samples)")
        print(f"     Results may have high variance")
    else:
        print(f"  ✓ Adequate samples per class in test set (~{expected_test_samples_per_class})")
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"VALIDATION COMPLETE")
    print(f"{'='*70}\n")
    print(f"✓ All checks passed!")
    print(f"\nData summary:")
    print(f"  - Total trials: {n_trials:,}")
    print(f"  - Channels: {n_channels}")
    print(f"  - Samples per trial: {n_samples:,}")
    print(f"  - Classes: {len(unique_classes)}")
    print(f"  - Data size: {x_memory_gb:.2f} GB")
    print(f"\nReady for CNN experiments!")
    print(f"\nNext step:")
    print(f"  ./run_cnn_experiments.sh --data-path {data_path}")
    print()
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Validate CNN experiment data format'
    )
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to preprocessed cnn_data.npz file')
    
    args = parser.parse_args()
    
    success = validate_data_file(args.data_path)
    
    if not success:
        print(f"\n❌ Validation failed!")
        print(f"\nPlease fix the issues above before running experiments.")
        print(f"If the file is from preprocessing_fixed.py, try re-running:")
        print(f"  python preprocessing_fixed.py --only cnn")
        exit(1)


if __name__ == "__main__":
    main()