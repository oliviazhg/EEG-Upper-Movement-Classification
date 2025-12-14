"""
Main SVM Experiment Script
Runs complete pipeline: grid search → final evaluation → analysis → visualization

Usage:
    python run_svm_experiment.py           # Start fresh
    python run_svm_experiment.py --resume  # Resume from checkpoints
    python run_svm_experiment.py --status  # Check checkpoint status
"""

import numpy as np
import json
import sys
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split  # ADD THIS LINE


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
from svm_resume import (
    detect_checkpoints,
    print_checkpoint_status,
    resume_experiment,
    get_resume_instructions,
)


def load_data():
    """
    Load preprocessed EEG data
    Returns FULL dataset and GRID SEARCH split
    """
    from svm_data_loader import load_and_split_ml_features
    
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    data_file = Path(DATA_CONFIG.get('data_file'))
    print(f"\nData file: {data_file}")
    
    # Load FULL dataset (unsplit)
    import numpy as np
    data = np.load(data_file, allow_pickle=True)
    X_full = data['X']
    y_full = data['y']
    
    # Exclude rest class if configured
    if DATA_CONFIG['exclude_rest']:
        mask = y_full != 0
        X_full = X_full[mask]
        y_full = y_full[mask]
        print(f"\nExcluded rest class (0)")
        print(f"Remaining samples: {len(y_full)}")
    
    # Create GRID SEARCH split (fixed seed for reproducibility)
    print(f"\nCreating grid search split (seed=42)...")
    X_train_grid, X_temp, y_train_grid, y_temp = train_test_split(
        X_full, y_full,
        test_size=0.30,
        stratify=y_full,
        random_state=42  # Fixed for grid search
    )
    
    X_val_grid, X_test_grid, y_val_grid, y_test_grid = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=42
    )
    
    print(f"  Grid search split: Train={len(y_train_grid)}, "
          f"Val={len(y_val_grid)}, Test={len(y_test_grid)}")
    
    return (X_full, y_full,  # Full dataset for final trials
            X_train_grid, y_train_grid,  # Grid search split
            X_val_grid, y_val_grid,
            X_test_grid, y_test_grid)

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
    from svm_train import convert_to_serializable
    
    results_dir = dirs['results']
    
    # Grid search results (skip scaler)
    grid_results_copy = {k: v for k, v in grid_results.items() if k != 'scaler'}
    grid_results_serializable = convert_to_serializable(grid_results_copy)
    
    with open(results_dir / 'grid_search_results.json', 'w') as f:
        json.dump(grid_results_serializable, f, indent=2)
    
    # Trial results
    trial_results_serializable = convert_to_serializable(trial_results)
    with open(results_dir / 'trial_results.json', 'w') as f:
        json.dump(trial_results_serializable, f, indent=2)
    
    # Aggregated results
    aggregated_serializable = convert_to_serializable(aggregated)
    with open(results_dir / 'aggregated_results.json', 'w') as f:
        json.dump(aggregated_serializable, f, indent=2)
    
    print(f"\nResults saved to: {results_dir}")


def main():
    """Main experiment pipeline"""
    
    # Parse command line arguments
    resume_mode = '--resume' in sys.argv
    status_only = '--status' in sys.argv
    
    # Validate configuration
    validate_config()
    
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\nSVM Experiment - {timestamp}")
    
    dirs = setup_directories()
    checkpoint_dir = dirs['checkpoints']
    
    # Status check mode
    if status_only:
        status = detect_checkpoints(checkpoint_dir)
        print_checkpoint_status(status)
        print(get_resume_instructions(checkpoint_dir))
        return
    
    # Load data
    (X_full, y_full,
     X_train_grid, y_train_grid,
     X_val_grid, y_val_grid,
     X_test_grid, y_test_grid) = load_data()  

    # ========================================================================
    # RESUME MODE
    # ========================================================================

    if resume_mode:
        print("\n" + "="*70)
        print("RESUME MODE ENABLED")
        print("="*70)
        
        grid_results, trial_results = resume_experiment(
            X_full, y_full,              # Full dataset for final trials
            X_train_grid, y_train_grid,  # Grid search split
            X_val_grid, y_val_grid,
            X_test_grid, y_test_grid,    # Added missing arguments
            checkpoint_dir               # Added missing argument
        )
    
    # ========================================================================
    # FRESH START MODE
    # ========================================================================
    
    else:
        # Check if checkpoints exist
        status = detect_checkpoints(checkpoint_dir)
        if status['n_grid_complete'] > 0 or status['n_trials_complete'] > 0:
            print("\n" + "="*70)
            print("WARNING: Existing checkpoints detected!")
            print("="*70)
            print_checkpoint_status(status)
            print("\nOptions:")
            print("  1. Continue anyway (will overwrite existing checkpoints)")
            print("  2. Resume from checkpoints: python run_svm_experiment.py --resume")
            print("  3. Check status: python run_svm_experiment.py --status")
            
            response = input("\nContinue and overwrite? (yes/no): ").strip().lower()
            if response not in ['yes', 'y']:
                print("Aborted. Use --resume to continue from checkpoints.")
                return
        
        # Phase 1: Grid Search
        print("\n" + "="*70)
        print("Starting Phase 1: Grid Search")
        print("="*70)
        
        grid_results = grid_search_svm(
            X_train_grid, y_train_grid,  # Fixed split
            X_val_grid, y_val_grid,
            checkpoint_dir=checkpoint_dir
        )
        
        # Save grid search checkpoint
        checkpoint_file = checkpoint_dir / f'grid_search_complete_{timestamp}.json'
        save_checkpoint(grid_results, checkpoint_file)
        print(f"\nGrid search checkpoint saved: {checkpoint_file}")
        
        # Phase 2: Final Evaluation
        print("\n" + "="*70)
        print("Starting Phase 2: Final Evaluation")
        print("="*70)
        
        trial_results = run_final_trials(
            X_full, y_full,  # Pass FULL dataset
            best_C=grid_results['best_C'],
            best_gamma=grid_results['best_gamma'],
            train_split=DATA_CONFIG['train_split'],
            val_split=DATA_CONFIG['val_split'],
            test_split=DATA_CONFIG['test_split'],
            checkpoint_dir=checkpoint_dir
        )
    
    # ========================================================================
    # PHASE 3: ANALYSIS (same for both modes)
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
    
    # Suggest cleanup
    print(f"\nTo clean up checkpoints:")
    print(f"  from svm_resume import cleanup_checkpoints")
    print(f"  cleanup_checkpoints(Path('{dirs['checkpoints']}'))")
    print("="*70)


if __name__ == "__main__":
    main()