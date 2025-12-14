"""
Data Loading Module for SVM Experiments
Loads preprocessed EEG features from NPZ files
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
from sklearn.model_selection import train_test_split


# ============================================================================
# DATA LOADING
# ============================================================================

def load_ml_features_npz(
    npz_path: Path,
    exclude_rest: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load ML features from preprocessed NPZ file
    
    Expected NPZ structure (from preprocessing_fixed.py):
        - 'X': (n_trials, n_features) - extracted features
               Features are: mu_power + beta_power + mean + std per channel
        - 'y': (n_trials,) - class labels (0-11)
        - 'metadata': dict with preprocessing info
        
    Args:
        npz_path: Path to ml_features_data.npz file
        exclude_rest: If True, exclude rest class (label 0)
        
    Returns:
        Tuple of (features, labels)
    """
    print(f"\nLoading data from: {npz_path}")
    
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    
    # Load NPZ file
    data = np.load(npz_path, allow_pickle=True)
    
    # Extract features and labels (using X and y keys from preprocessing)
    if 'X' in data and 'y' in data:
        features = data['X']
        labels = data['y']
    elif 'features' in data and 'labels' in data:
        # Fallback for alternative naming
        features = data['features']
        labels = data['labels']
    else:
        raise KeyError(f"NPZ file must contain 'X' and 'y' (or 'features' and 'labels')")
    
    # Load metadata if available
    if 'metadata' in data:
        metadata = data['metadata'].item()  # Convert 0-d array to dict
        print(f"\n  Metadata:")
        print(f"    Files processed: {metadata.get('n_files', 'N/A')}")
        print(f"    Channel names: {len(metadata.get('channel_names', []))} channels")
        print(f"    Filter: {metadata.get('filter', 'N/A')} Hz")
        print(f"    ICA artifact removal: {metadata.get('ica_artifact_removal', 'N/A')}")
    
    print(f"\n  Loaded data:")
    print(f"    Total samples: {len(features)}")
    print(f"    Feature dimensions: {features.shape[1]}")
    print(f"    Unique classes: {np.unique(labels)}")
    
    # Exclude rest class if requested
    if exclude_rest:
        mask = labels != 0
        features = features[mask]
        labels = labels[mask]
        print(f"\n  After excluding rest (class 0):")
        print(f"    Remaining samples: {len(features)}")
        print(f"    Classes: {np.unique(labels)}")
    
    # Verify class distribution
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\n  Class distribution:")
    for cls, count in zip(unique, counts):
        print(f"    Class {cls:2d}: {count:4d} samples")
    
    return features, labels


def verify_feature_dimensions(features: np.ndarray):
    """
    Verify and report feature dimensions
    
    Args:
        features: Feature array
    """
    actual_dim = features.shape[1]
    
    # Expected: 60 channels × 4 features (mu_power, beta_power, mean, std) = 240
    expected_dim = 240
    
    if actual_dim == expected_dim:
        print(f"\n✓ Feature dimensions verified: {actual_dim} features")
        print(f"  (60 channels × 4 features: mu_power, beta_power, mean, std)")
    else:
        print(f"\n✓ Feature dimensions: {actual_dim} features")
        print(f"  Note: Expected {expected_dim} for standard extraction")
        print(f"  Proceeding with {actual_dim} features...")


# ============================================================================
# DATA SPLITTING
# ============================================================================

def split_data(
    features: np.ndarray,
    labels: np.ndarray,
    train_split: float = 0.70,
    val_split: float = 0.15,
    test_split: float = 0.15,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train, validation, and test sets
    
    Args:
        features: Feature array (n_samples, n_features)
        labels: Label array (n_samples,)
        train_split: Fraction for training (default: 0.70)
        val_split: Fraction for validation (default: 0.15)
        test_split: Fraction for test (default: 0.15)
        random_state: Random seed for reproducibility
        stratify: If True, maintain class proportions in splits
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    # Verify splits sum to 1
    total = train_split + val_split + test_split
    assert abs(total - 1.0) < 1e-6, f"Splits must sum to 1.0, got {total}"
    
    print(f"\nSplitting data:")
    print(f"  Train: {train_split:.0%}")
    print(f"  Val:   {val_split:.0%}")
    print(f"  Test:  {test_split:.0%}")
    print(f"  Stratify: {stratify}")
    print(f"  Random seed: {random_state}")
    
    # First split: separate test set
    test_fraction = test_split / (train_split + val_split + test_split)
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        features, labels,
        test_size=test_fraction,
        random_state=random_state,
        stratify=labels if stratify else None
    )
    
    # Second split: separate train and validation from temp
    val_fraction = val_split / (train_split + val_split)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_fraction,
        random_state=random_state,
        stratify=y_temp if stratify else None
    )
    
    print(f"\n  Result:")
    print(f"    Train: {len(X_train):5d} samples ({len(X_train)/len(features):.1%})")
    print(f"    Val:   {len(X_val):5d} samples ({len(X_val)/len(features):.1%})")
    print(f"    Test:  {len(X_test):5d} samples ({len(X_test)/len(features):.1%})")
    
    # Verify class distribution
    if stratify:
        print(f"\n  Class distribution check:")
        for dataset_name, y_data in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
            unique, counts = np.unique(y_data, return_counts=True)
            total_samples = len(y_data)
            print(f"    {dataset_name}:")
            for cls, count in zip(unique, counts):
                print(f"      Class {cls:2d}: {count:4d} ({count/total_samples:.1%})")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def load_and_split_ml_features(
    data_path: Path,
    train_split: float = 0.70,
    val_split: float = 0.15,
    test_split: float = 0.15,
    random_state: int = 42,
    exclude_rest: bool = True,
    verify_dims: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load ml_features_data.npz and split into train/val/test
    
    Args:
        data_path: Path to ml_features_data.npz file
        train_split: Training fraction
        val_split: Validation fraction
        test_split: Test fraction
        random_state: Random seed
        exclude_rest: Exclude rest class
        verify_dims: Check feature dimensions
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    # Load data
    features, labels = load_ml_features_npz(data_path, exclude_rest)
    
    # Verify dimensions
    if verify_dims:
        verify_feature_dimensions(features)
    
    # Split data
    return split_data(
        features, labels,
        train_split, val_split, test_split,
        random_state, stratify=True
    )


# ============================================================================
# DATASET INFO
# ============================================================================

def print_dataset_info(data_dir: Path):
    """
    Print information about available NPZ files
    
    Args:
        data_dir: Directory to scan for NPZ files
    """
    print("\n" + "="*70)
    print("AVAILABLE DATASETS")
    print("="*70)
    
    # Look for ml_features_data.npz specifically
    ml_features_file = data_dir / "ml_features_data.npz"
    
    if ml_features_file.exists():
        print(f"\n✓ Found: ml_features_data.npz")
        
        try:
            data = np.load(ml_features_file, allow_pickle=True)
            
            X = data['X'] if 'X' in data else data.get('features')
            y = data['y'] if 'y' in data else data.get('labels')
            
            print(f"  Samples: {len(X)}")
            print(f"  Features: {X.shape[1]}")
            print(f"  Classes: {len(np.unique(y))}")
            
            # Class distribution
            unique, counts = np.unique(y, return_counts=True)
            print(f"\n  Class distribution:")
            for cls, count in zip(unique, counts):
                class_name = {
                    0: 'Rest', 1: 'Forward', 2: 'Backward',
                    3: 'Left', 4: 'Right', 5: 'Up', 6: 'Down',
                    7: 'Power Grasp', 8: 'Precision Grasp', 9: 'Lateral Grasp',
                    10: 'Pronation', 11: 'Supination'
                }.get(cls, f'Class {cls}')
                print(f"    {cls:2d} ({class_name:20s}): {count:5d} samples")
            
            # Metadata
            if 'metadata' in data:
                metadata = data['metadata'].item()
                print(f"\n  Metadata:")
                print(f"    Files processed: {metadata.get('n_files', 'N/A')}")
                print(f"    Channels: {len(metadata.get('channel_names', []))}")
                print(f"    Filter: {metadata.get('filter', 'N/A')} Hz")
                print(f"    ICA: {metadata.get('ica_artifact_removal', 'N/A')}")
            
        except Exception as e:
            print(f"  Error loading: {e}")
    
    else:
        print(f"\n✗ ml_features_data.npz not found in {data_dir}")
        print(f"\nLooking for other NPZ files...")
        
        npz_files = sorted(data_dir.glob("*.npz"))
        if npz_files:
            for npz_file in npz_files:
                print(f"\n  Found: {npz_file.name}")
        else:
            print(f"  No NPZ files found")
    
    print("="*70)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Load ML features
    data_dir = Path("/home/ubuntu/multimodal-signal-dataset-for-11-upper-body-movements/PreprocessedData2")
    ml_features_file = data_dir / "ml_features_data.npz"
    
    # Print available datasets
    if data_dir.exists():
        print_dataset_info(data_dir)
    else:
        print(f"Data directory not found: {data_dir}")
    
    # Example: Load and split data
    if ml_features_file.exists():
        try:
            X_train, y_train, X_val, y_val, X_test, y_test = load_and_split_ml_features(
                data_path=ml_features_file,
                random_state=42
            )
            
            print("\n✓ Data loaded successfully!")
            print(f"  Ready for SVM training")
            
        except Exception as e:
            print(f"\n✗ Error: {e}")
    else:
        print(f"\nPlease update data_dir to point to your ml_features_data.npz file")