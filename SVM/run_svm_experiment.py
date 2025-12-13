"""
Main SVM Experiment Script
Runs complete pipeline: grid search → final evaluation → analysis → visualization
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime

# Import modules
from svm_config import (
    DATA_CONFIG,
    CHECKPOINT_CONFIG,
    OUTPUT_CONFIG,
    FINAL_EVAL_CONFIG,
    validate_config,
    CLASS_LABELS,
)
from svm_train import (
    grid_search_svm,
    run_final_trials,
    save_checkpoint,
)
from svm_analysis import (
    aggregate_trial_results,
    print_summary,
    analyze_confusion_patterns,
    print_confusion_analysis,
)
from svm_visualize import generate_all_plots


def load_data():
    """
    Load and prepare EEG data
    
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    # TODO: Replace with actual data loading
    # This is a placeholder - adapt to your actual data loading code
    
    # For now, using dummy data
    print("WARNING: Using dummy data. Replace with actual EEG data loading.")
    
    # Dummy data dimensions: (n_samples, n_features)
    # Features should be: mu/beta band power, mean, std from 60 channels = 240 features
    n_features = 240
    
    # Approximate number of samples per class (excluding rest)
    # Assuming ~100 trials per class for subject 1
    n_samples_per_class = 100
    n_classes = len(CLASS_LABELS)  # 11 classes
    total_samples = n_samples_per_class * n_classes
    
    # Generate dummy data
    np.random.seed(42)
    X_all = np.random.randn(total_samples, n_features)
    y_all = np.repeat(list(CLASS_LABELS.keys()), n_samples_per_class)
    
    # Shuffle
    indices = np.random.permutation(total_samples)
    X_all = X_all[indices]
    y_all = y_all[indices]
    
    # Split into train/val/test
    n_train = int(total_samples * DATA_CONFIG['train_split'])
    n_val = int(total_samples * DATA_CONFIG['val_split'])
    
    X_train = X_all[:n_train]
    y_train = y_all[:n_train]
    
    X_val = X_all[n_train:n_train+n_val]
    y_val = y_all[n_train:n_train+n_val]
    
    X_test = X_all[n_train+n_val:]
    y_test = y_all[n_train+n_val:]
    
    print(f"\nData loaded:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Features: {n_features}")
    print(f"  Classes: {n_classes} (excluding rest)")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def setup_directories():
    """Create output and checkpoint directories"""
    dirs = {
        'results': Path(OUTPUT_CONFIG['results_dir']),
        'figures': Path(OUTPUT_CONFIG['figures_dir']),
        'checkpoints': Path(CHECKPOINT_CONFIG['checkpoint_dir']),
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs


def save_results(grid_results, trial_results, aggregated, dirs):
    """Save all results to JSON files"""
    results_dir = dirs['results']
    
    # Grid search results
    grid_results_serializable = {
        k: v.tolist() if isinstance(v, np.ndarray) else v
        for k, v in grid_results.items()
        if k != 'scaler'  # Skip scaler object
    }
    
    with open(results_dir / 'grid_search_results.json', 'w') as f:
        json.dump(grid_results_serializable, f, indent=2)
    
    # Trial results
    with open(results_dir / 'trial_results.json', 'w') as f:
        json.dump(trial_results, f, indent=2)
    
    # Aggregated results
    with open(results_dir / 'aggregated_results.json', 'w') as f:
        json.dump(aggregated, f, indent=2)
    
    print(f"\nResults saved to: {results_dir}")


def main():
    """Main experiment pipeline"""
    
    # Validate configuration
    validate_config()
    
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\nSVM Experiment - {timestamp}")
    
    dirs = setup_directories()
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    
    # ========================================================================
    # PHASE 1: GRID SEARCH
    # ========================================================================
    
    print("\n" + "="*70)
    print("Starting Phase 1: Grid Search")
    print("="*70)
    
    checkpoint_dir = dirs['checkpoints'] if CHECKPOINT_CONFIG['save_grid_search'] else None
    
    grid_results = grid_search_svm(
        X_train, y_train,
        X_val, y_val,
        checkpoint_dir=checkpoint_dir
    )
    
    # Save grid search checkpoint
    if checkpoint_dir:
        checkpoint_file = checkpoint_dir / f'grid_search_complete_{timestamp}.json'
        save_checkpoint(grid_results, checkpoint_file)
        print(f"\nGrid search checkpoint saved: {checkpoint_file}")
    
    # ========================================================================
    # PHASE 2: FINAL EVALUATION
    # ========================================================================
    
    print("\n" + "="*70)
    print("Starting Phase 2: Final Evaluation")
    print("="*70)
    
    trial_results = run_final_trials(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        best_C=grid_results['best_C'],
        best_gamma=grid_results['best_gamma'],
        scaler=grid_results['scaler'],
        checkpoint_dir=checkpoint_dir
    )
    
    # ========================================================================
    # PHASE 3: ANALYSIS
    # ========================================================================
    
    print("\n" + "="*70)
    print("Starting Phase 3: Analysis")
    print("="*70)
    
    aggregated = aggregate_trial_results(trial_results['trials'])
    print_summary(aggregated)
    
    # Confusion analysis
    confusion_analysis = analyze_confusion_patterns(
        np.array(aggregated['confusion_matrix_mean'])
    )
    print_confusion_analysis(confusion_analysis)
    
    # ========================================================================
    # PHASE 4: VISUALIZATION
    # ========================================================================
    
    print("\n" + "="*70)
    print("Starting Phase 4: Visualization")
    print("="*70)
    
    generate_all_plots(grid_results, aggregated, dirs['figures'])
    
    # ========================================================================
    # SAVE FINAL RESULTS
    # ========================================================================
    
    save_results(grid_results, trial_results, aggregated, dirs)
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print(f"\nBest Configuration:")
    print(f"  C = {grid_results['best_C']}")
    print(f"  gamma = {grid_results['best_gamma']}")
    print(f"\nFinal Performance:")
    print(f"  Test Accuracy: {aggregated['test_accuracy_mean']:.4f} "
          f"± {aggregated['test_accuracy_std']:.4f}")
    print(f"  95% CI: [{aggregated['test_accuracy_ci_low']:.4f}, "
          f"{aggregated['test_accuracy_ci_high']:.4f}]")
    print(f"  Test F1: {aggregated['test_f1_macro_mean']:.4f} "
          f"± {aggregated['test_f1_macro_std']:.4f}")
    print(f"\nOutputs:")
    print(f"  Results: {dirs['results']}")
    print(f"  Figures: {dirs['figures']}")
    print(f"  Checkpoints: {dirs['checkpoints']}")
    print("="*70)


if __name__ == "__main__":
    main()
