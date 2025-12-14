"""
SVM Training Module
Implements grid search and final evaluation with checkpointing
"""

import numpy as np
import time
import json
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from typing import Dict, Tuple, Optional
import pickle

from svm_config import (
    GRID_SEARCH_CONFIG,
    FINAL_EVAL_CONFIG,
    SCALING_CONFIG,
    CHECKPOINT_CONFIG,
    N_CLASSES,
)


# ============================================================================
# FEATURE SCALING
# ============================================================================

def create_scaler():
    """Create StandardScaler for SVM preprocessing"""
    return StandardScaler(
        with_mean=SCALING_CONFIG['with_mean'],
        with_std=SCALING_CONFIG['with_std']
    )


def scale_features(X_train, X_val, X_test, scaler=None):
    """
    Scale features using StandardScaler
    
    Args:
        X_train: Training features (n_samples, n_features)
        X_val: Validation features
        X_test: Test features
        scaler: Optional pre-fitted scaler
        
    Returns:
        Tuple of (X_train_scaled, X_val_scaled, X_test_scaled, scaler)
    """
    if scaler is None:
        scaler = create_scaler()
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        X_train_scaled = scaler.transform(X_train)
    
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


# ============================================================================
# PHASE 1: GRID SEARCH
# ============================================================================

def train_svm_single_config(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    X_val_scaled: np.ndarray,
    y_val: np.ndarray,
    C: float,
    gamma: float,
    random_state: int = 0
) -> Dict:
    """
    Train SVM with single configuration
    
    Args:
        X_train_scaled: Scaled training features
        y_train: Training labels
        X_val_scaled: Scaled validation features
        y_val: Validation labels
        C: Regularization parameter
        gamma: Kernel coefficient
        random_state: Random seed
        
    Returns:
        Dictionary with results
    """
    start_time = time.time()
    
    # Create and train SVM
    svm = SVC(
        C=C,
        gamma=gamma,
        kernel=GRID_SEARCH_CONFIG['kernel'],
        random_state=random_state
    )
    
    svm.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time
    
    # Evaluate
    train_acc = svm.score(X_train_scaled, y_train)
    val_acc = svm.score(X_val_scaled, y_val)
    
    # Get predictions for detailed metrics
    y_train_pred = svm.predict(X_train_scaled)
    y_val_pred = svm.predict(X_val_scaled)
    
    train_f1 = f1_score(y_train, y_train_pred, average='macro')
    val_f1 = f1_score(y_val, y_val_pred, average='macro')
    
    # Count support vectors
    n_sv_total = len(svm.support_)
    # Get labels of support vectors
    sv_labels = y_train[svm.support_]
    n_sv_per_class = {
        int(cls): int(count) 
        for cls, count in zip(*np.unique(sv_labels, return_counts=True))
    }
    
    return {
        'C': C,
        'gamma': gamma,
        'train_accuracy': float(train_acc),
        'val_accuracy': float(val_acc),
        'train_f1_macro': float(train_f1),
        'val_f1_macro': float(val_f1),
        'n_support_vectors': int(n_sv_total),
        'n_support_vectors_per_class': n_sv_per_class,
        'training_time_sec': float(training_time),
    }


def grid_search_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    checkpoint_dir: Optional[Path] = None
) -> Dict:
    """
    Phase 1: Grid search over C and gamma
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        checkpoint_dir: Directory for saving checkpoints
        
    Returns:
        Dictionary with grid search results
    """
    print("\n" + "="*70)
    print("PHASE 1: GRID SEARCH")
    print("="*70)
    
    # Scale features
    print("\nScaling features...")
    X_train_scaled, X_val_scaled, _, scaler = scale_features(
        X_train, X_val, X_val  # Use X_val as placeholder for test
    )
    
    C_values = GRID_SEARCH_CONFIG['C_values']
    gamma_values = GRID_SEARCH_CONFIG['gamma_values']
    
    n_C = len(C_values)
    n_gamma = len(gamma_values)
    
    # Initialize result arrays
    val_accuracy_grid = np.zeros((n_C, n_gamma))
    training_times = np.zeros((n_C, n_gamma))
    all_results = []
    
    total_configs = n_C * n_gamma
    config_num = 0
    
    print(f"\nTesting {total_configs} configurations:")
    print(f"  C values: {C_values}")
    print(f"  gamma values: {gamma_values}")
    print()
    
    # Grid search
    for i, C in enumerate(C_values):
        for j, gamma in enumerate(gamma_values):
            config_num += 1
            print(f"[{config_num}/{total_configs}] C={C}, gamma={gamma}... ", end='', flush=True)
            
            result = train_svm_single_config(
                X_train_scaled, y_train,
                X_val_scaled, y_val,
                C, gamma,
                random_state=GRID_SEARCH_CONFIG['random_state']
            )
            
            val_accuracy_grid[i, j] = result['val_accuracy']
            training_times[i, j] = result['training_time_sec']
            all_results.append(result)
            
            print(f"Val Acc: {result['val_accuracy']:.4f} ({result['training_time_sec']:.1f}s)")
            
            # Save checkpoint
            if checkpoint_dir and CHECKPOINT_CONFIG['save_grid_search']:
                checkpoint_file = checkpoint_dir / f"grid_C{C}_gamma{gamma}.json"
                with open(checkpoint_file, 'w') as f:
                    json.dump(convert_to_serializable(result), f, indent=2)
    
    # Find best configuration
    best_idx = np.unravel_index(val_accuracy_grid.argmax(), val_accuracy_grid.shape)
    best_C = C_values[best_idx[0]]
    best_gamma = gamma_values[best_idx[1]]
    best_val_acc = val_accuracy_grid[best_idx]
    
    print(f"\n{'='*70}")
    print("GRID SEARCH COMPLETE")
    print(f"{'='*70}")
    print(f"Best configuration:")
    print(f"  C = {best_C}")
    print(f"  gamma = {best_gamma}")
    print(f"  Validation accuracy = {best_val_acc:.4f}")
    print()
    
    return {
        'C_values': C_values,
        'gamma_values': gamma_values,
        'val_accuracy_grid': val_accuracy_grid,
        'training_times': training_times,
        'best_C': best_C,
        'best_gamma': best_gamma,
        'best_val_accuracy': float(best_val_acc),
        'best_idx': best_idx,
        'all_results': all_results,
        'scaler': scaler,  # Save scaler for Phase 2
    }


# ============================================================================
# PHASE 2: FINAL EVALUATION
# ============================================================================

def train_final_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    C: float,
    gamma: float,
    scaler: StandardScaler,
    seed: int
) -> Dict:
    """
    Train SVM with best configuration and evaluate on test set
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        X_test: Test features
        y_test: Test labels
        C: Best C parameter
        gamma: Best gamma parameter
        scaler: Fitted scaler from grid search
        seed: Random seed for this trial
        
    Returns:
        Dictionary with trial results
    """
    # Scale features using the grid search scaler
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    start_time = time.time()
    svm = SVC(C=C, gamma=gamma, kernel='rbf', random_state=seed)
    svm.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time
    
    # Predictions
    y_train_pred = svm.predict(X_train_scaled)
    y_val_pred = svm.predict(X_val_scaled)
    y_test_pred = svm.predict(X_test_scaled)
    
    # Timing inference
    start_time = time.time()
    _ = svm.predict(X_test_scaled)
    inference_time_total = time.time() - start_time
    inference_time_per_sample = inference_time_total / len(X_test_scaled)
    
    # Metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    train_f1 = f1_score(y_train, y_train_pred, average='macro')
    val_f1 = f1_score(y_val, y_val_pred, average='macro')
    test_f1 = f1_score(y_test, y_test_pred, average='macro')
    
    test_f1_per_class = f1_score(y_test, y_test_pred, average=None)
    
    conf_matrix = confusion_matrix(y_test, y_test_pred)
    
    # Support vectors
    n_sv_total = len(svm.support_)
    sv_labels = y_train[svm.support_]
    n_sv_per_class = {
        int(cls): int(count)
        for cls, count in zip(*np.unique(sv_labels, return_counts=True))
    }
    
    return {
        'C': C,
        'gamma': gamma,
        'seed': seed,
        
        # Accuracies
        'train_accuracy': float(train_acc),
        'val_accuracy': float(val_acc),
        'test_accuracy': float(test_acc),
        
        # F1 scores
        'train_f1_macro': float(train_f1),
        'val_f1_macro': float(val_f1),
        'test_f1_macro': float(test_f1),
        'test_f1_per_class': test_f1_per_class.tolist(),
        
        # Confusion matrix
        'confusion_matrix': conf_matrix.tolist(),
        
        # Model characteristics
        'n_support_vectors': int(n_sv_total),
        'n_support_vectors_per_class': n_sv_per_class,
        
        # Timing
        'training_time_sec': float(training_time),
        'inference_time_per_sample': float(inference_time_per_sample),
    }


def run_final_trials(
    X_full: np.ndarray,      # Changed: full dataset, not pre-split
    y_full: np.ndarray,      # Changed: full labels, not pre-split
    best_C: float,
    best_gamma: float,
    train_split: float = 0.70,  # New parameters
    val_split: float = 0.15,
    test_split: float = 0.15,
    checkpoint_dir: Optional[Path] = None
) -> Dict:
    """
    Phase 2: Run multiple trials with best configuration
    Each trial uses a DIFFERENT random split
    """
    from sklearn.model_selection import train_test_split
    
    print("\n" + "="*70)
    print("PHASE 2: FINAL EVALUATION")
    print("="*70)
    print(f"\nBest configuration: C={best_C}, gamma={best_gamma}")
    print(f"Running {FINAL_EVAL_CONFIG['n_trials']} independent trials")
    print("Each trial uses a different train/val/test split\n")
    
    trials = []
    
    for trial_num, seed in enumerate(FINAL_EVAL_CONFIG['seeds'], 1):
        print(f"Trial {trial_num}/{FINAL_EVAL_CONFIG['n_trials']} (seed={seed}):")
        
        # CREATE NEW SPLIT FOR THIS TRIAL
        print(f"  Creating split with seed {seed}...", end=' ')
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_full, y_full,
            test_size=(1 - train_split),
            stratify=y_full,
            random_state=seed  # Different seed each trial!
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(test_split / (val_split + test_split)),
            stratify=y_temp,
            random_state=seed
        )
        print(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
        
        # FIT NEW SCALER FOR THIS SPLIT
        print(f"  Fitting scaler...", end=' ')
        scaler = create_scaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        print("Done")
        
        # TRAIN AND EVALUATE
        print(f"  Training SVM...", end=' ', flush=True)
        result = train_final_svm_single_trial(
            X_train_scaled, y_train,
            X_val_scaled, y_val,
            X_test_scaled, y_test,
            best_C, best_gamma,
            seed
        )
        
        trials.append(result)
        
        print(f"Test Acc: {result['test_accuracy']:.4f}, "
              f"F1: {result['test_f1_macro']:.4f}, "
              f"SVs: {result['n_support_vectors']}")
        
        # Save checkpoint
        if checkpoint_dir and CHECKPOINT_CONFIG['save_per_trial']:
            checkpoint_file = checkpoint_dir / f"trial_{trial_num}_seed{seed}.json"
            with open(checkpoint_file, 'w') as f:
                json.dump(convert_to_serializable(result), f, indent=2)
    
    return {'trials': trials}

def train_final_svm_single_trial(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    X_val_scaled: np.ndarray,
    y_val: np.ndarray,
    X_test_scaled: np.ndarray,
    y_test: np.ndarray,
    C: float,
    gamma: float,
    seed: int
) -> Dict:
    """
    Train SVM for a single trial (renamed from train_final_svm)
    Data is already split and scaled for this trial
    """
    # Train model
    start_time = time.time()
    svm = SVC(C=C, gamma=gamma, kernel='rbf', random_state=seed)
    svm.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time
    
    # Predictions
    y_train_pred = svm.predict(X_train_scaled)
    y_val_pred = svm.predict(X_val_scaled)
    y_test_pred = svm.predict(X_test_scaled)
    
    # Timing inference
    start_time = time.time()
    _ = svm.predict(X_test_scaled)
    inference_time_total = time.time() - start_time
    inference_time_per_sample = inference_time_total / len(X_test_scaled)
    
    # Metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    train_f1 = f1_score(y_train, y_train_pred, average='macro')
    val_f1 = f1_score(y_val, y_val_pred, average='macro')
    test_f1 = f1_score(y_test, y_test_pred, average='macro')
    
    test_f1_per_class = f1_score(y_test, y_test_pred, average=None)
    
    conf_matrix = confusion_matrix(y_test, y_test_pred)
    
    # Support vectors
    n_sv_total = len(svm.support_)
    sv_labels = y_train[svm.support_]
    n_sv_per_class = {
        int(cls): int(count)
        for cls, count in zip(*np.unique(sv_labels, return_counts=True))
    }
    
    return {
        'C': C,
        'gamma': gamma,
        'seed': seed,
        
        # Accuracies
        'train_accuracy': float(train_acc),
        'val_accuracy': float(val_acc),
        'test_accuracy': float(test_acc),
        
        # F1 scores
        'train_f1_macro': float(train_f1),
        'val_f1_macro': float(val_f1),
        'test_f1_macro': float(test_f1),
        'test_f1_per_class': test_f1_per_class.tolist(),
        
        # Confusion matrix
        'confusion_matrix': conf_matrix.tolist(),
        
        # Model characteristics
        'n_support_vectors': int(n_sv_total),
        'n_support_vectors_per_class': n_sv_per_class,
        
        # Timing
        'training_time_sec': float(training_time),
        'inference_time_per_sample': float(inference_time_per_sample),
    }
    
# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

def convert_to_serializable(obj):
    """Recursively convert numpy types to Python types for JSON serialization"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    else:
        return obj


def save_checkpoint(data: Dict, filepath: Path):
    """Save checkpoint to JSON file"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy arrays and types to JSON-serializable formats
    data_serializable = {}
    for key, value in data.items():
        if isinstance(value, StandardScaler):
            # Skip scaler (save separately if needed)
            continue
        else:
            data_serializable[key] = convert_to_serializable(value)
    
    with open(filepath, 'w') as f:
        json.dump(data_serializable, f, indent=2)


def load_checkpoint(filepath: Path) -> Dict:
    """Load checkpoint from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)