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
    Load and prepare EEG data from preprocessed NPZ files
    
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    from svm_data_loader import (
        load_and_split_subject_data,
        load_multiple_subjects,
        split_data
    )
    
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    data_dir = Path(DATA_CONFIG.get('data_dir', '/home/ubuntu/multimodal-signal-dataset-for-11-upper-body-movements/PreprocessedData2'))
    
    # Check if using all subjects or single subject
    if DATA_CONFIG.get('use_all_subjects', False):
        subject_ids = DATA_CONFIG.get('subject_ids', list(range(1, 26)))
        
        print(f"\nLoading {len(subject_ids)} subjects: {subject_ids}")
        print(f"Data directory: {data_dir}")
        
        # Load all subjects
        features, labels = load_multiple_subjects(
            subject_ids=subject_ids,
            data_dir=data_dir,
            exclude_rest=DATA_CONFIG['exclude_rest']
        )
        
        # Split data
        X_train, y_train, X_val, y_val, X_test, y_test = split_data(
            features, labels,
            train_split=DATA_CONFIG['train_split'],
            val_split=DATA_CONFIG['val_split'],
            test_split=DATA_CONFIG['test_split'],
            random_state=42,
            stratify=True
        )
    else:
        # Single subject mode (backward compatibility)
        subject_id = DATA_CONFIG.get('subject_id', DATA_CONFIG['subject_ids'][0])
        
        print(f"\nSubject: {subject_id}")
        print(f"Data directory: {data_dir}")
        
        X_train, y_train, X_val, y_val, X_test, y_test = load_and_split_subject_data(
            subject_id=subject_id,
            data_dir=data_dir,
            train_split=DATA_CONFIG['train_split'],
            val_split=DATA_CONFIG['val_split'],
            test_split=DATA_CONFIG['test_split'],
            random_state=42,
            exclude_rest=DATA_CONFIG['exclude_rest'],
            verify_dims=True
        )
    
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
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    
    # ========================================================================
    # RESUME MODE
    # ========================================================================
    
    if resume_mode:
        print("\n" + "="*70)
        print("RESUME MODE ENABLED")
        print("="*70)
        
        grid_results, trial_results = resume_experiment(
            X_train, y_train,
            X_val, y_val,
            X_test, y_test,
            checkpoint_dir
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
            X_train, y_train,
            X_val, y_val,
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
            X_train, y_train,
            X_val, y_val,
            X_test, y_test,
            best_C=grid_results['best_C'],
            best_gamma=grid_results['best_gamma'],
            scaler=grid_results['scaler'],
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