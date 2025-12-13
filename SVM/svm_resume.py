"""
SVM Resume Module
Resume interrupted experiments from checkpoints
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pickle

from svm_config import GRID_SEARCH_CONFIG, FINAL_EVAL_CONFIG


# ============================================================================
# CHECKPOINT DETECTION
# ============================================================================

def detect_checkpoints(checkpoint_dir: Path) -> Dict:
    """
    Scan checkpoint directory and detect completed work
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Dictionary with checkpoint status
    """
    if not checkpoint_dir.exists():
        return {
            'grid_search_complete': False,
            'completed_grid_configs': [],
            'completed_trials': [],
            'grid_search_file': None,
            'trial_files': [],
        }
    
    # Check for grid search checkpoints
    C_values = GRID_SEARCH_CONFIG['C_values']
    gamma_values = GRID_SEARCH_CONFIG['gamma_values']
    
    completed_configs = []
    for C in C_values:
        for gamma in gamma_values:
            checkpoint_file = checkpoint_dir / f"grid_C{C}_gamma{gamma}.json"
            if checkpoint_file.exists():
                completed_configs.append((C, gamma))
    
    # Check for complete grid search file
    grid_search_files = list(checkpoint_dir.glob("grid_search_complete_*.json"))
    grid_search_complete = len(grid_search_files) > 0
    grid_search_file = grid_search_files[0] if grid_search_files else None
    
    # Check for trial checkpoints
    trial_files = sorted(checkpoint_dir.glob("trial_*.json"))
    completed_trials = [int(f.stem.split('_')[1]) for f in trial_files]
    
    status = {
        'grid_search_complete': grid_search_complete,
        'completed_grid_configs': completed_configs,
        'completed_trials': completed_trials,
        'grid_search_file': grid_search_file,
        'trial_files': trial_files,
        'n_grid_complete': len(completed_configs),
        'n_trials_complete': len(completed_trials),
    }
    
    return status


def print_checkpoint_status(status: Dict):
    """Print human-readable checkpoint status"""
    print("\n" + "="*70)
    print("CHECKPOINT STATUS")
    print("="*70)
    
    total_grid_configs = (len(GRID_SEARCH_CONFIG['C_values']) * 
                         len(GRID_SEARCH_CONFIG['gamma_values']))
    
    print(f"\nPhase 1 - Grid Search:")
    print(f"  Completed configs: {status['n_grid_complete']}/{total_grid_configs}")
    
    if status['grid_search_complete']:
        print(f"  ✓ Grid search COMPLETE")
        print(f"    File: {status['grid_search_file'].name}")
    else:
        print(f"  ✗ Grid search INCOMPLETE")
        if status['n_grid_complete'] > 0:
            print(f"    Can resume from checkpoint")
    
    print(f"\nPhase 2 - Final Evaluation:")
    print(f"  Completed trials: {status['n_trials_complete']}/{FINAL_EVAL_CONFIG['n_trials']}")
    
    if status['n_trials_complete'] == FINAL_EVAL_CONFIG['n_trials']:
        print(f"  ✓ All trials COMPLETE")
    elif status['n_trials_complete'] > 0:
        print(f"  ⚠ Trials INCOMPLETE - can resume")
    else:
        print(f"  ✗ No trials started")
    
    print("="*70)


# ============================================================================
# RESUME GRID SEARCH
# ============================================================================

def load_grid_checkpoint(checkpoint_file: Path) -> Dict:
    """Load a single grid configuration checkpoint"""
    with open(checkpoint_file, 'r') as f:
        return json.load(f)


def resume_grid_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    checkpoint_dir: Path,
    status: Dict
) -> Dict:
    """
    Resume grid search from checkpoints
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        checkpoint_dir: Checkpoint directory
        status: Checkpoint status from detect_checkpoints()
        
    Returns:
        Complete grid search results
    """
    from svm_train import scale_features, train_svm_single_config
    
    print("\n" + "="*70)
    print("RESUMING PHASE 1: GRID SEARCH")
    print("="*70)
    
    # If grid search already complete, just load it
    if status['grid_search_complete']:
        print("\nGrid search already complete - loading results...")
        with open(status['grid_search_file'], 'r') as f:
            grid_results = json.load(f)
        
        # Convert lists back to numpy arrays
        grid_results['val_accuracy_grid'] = np.array(grid_results['val_accuracy_grid'])
        grid_results['training_times'] = np.array(grid_results['training_times'])
        
        print(f"✓ Loaded grid search results")
        print(f"  Best: C={grid_results['best_C']}, gamma={grid_results['best_gamma']}")
        print(f"  Best val accuracy: {grid_results['best_val_accuracy']:.4f}")
        
        # Need to recreate scaler
        print("\nRecreating scaler...")
        _, _, _, scaler = scale_features(X_train, X_val, X_val)
        grid_results['scaler'] = scaler
        
        return grid_results
    
    # Otherwise, resume from individual checkpoints
    print(f"\nFound {status['n_grid_complete']} completed configurations")
    print("Loading completed configs and continuing...")
    
    # Scale features
    X_train_scaled, X_val_scaled, _, scaler = scale_features(X_train, X_val, X_val)
    
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
    
    # Process each configuration
    for i, C in enumerate(C_values):
        for j, gamma in enumerate(gamma_values):
            config_num += 1
            
            # Check if already completed
            if (C, gamma) in status['completed_grid_configs']:
                print(f"[{config_num}/{total_configs}] C={C}, gamma={gamma}... ", end='')
                
                # Load from checkpoint
                checkpoint_file = checkpoint_dir / f"grid_C{C}_gamma{gamma}.json"
                result = load_grid_checkpoint(checkpoint_file)
                
                val_accuracy_grid[i, j] = result['val_accuracy']
                training_times[i, j] = result['training_time_sec']
                all_results.append(result)
                
                print(f"✓ LOADED from checkpoint (Val Acc: {result['val_accuracy']:.4f})")
            
            else:
                # Need to run this configuration
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
                checkpoint_file = checkpoint_dir / f"grid_C{C}_gamma{gamma}.json"
                with open(checkpoint_file, 'w') as f:
                    json.dump(result, f, indent=2)
    
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
        'scaler': scaler,
    }


# ============================================================================
# RESUME FINAL TRIALS
# ============================================================================

def resume_final_trials(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    best_C: float,
    best_gamma: float,
    scaler,
    checkpoint_dir: Path,
    status: Dict
) -> Dict:
    """
    Resume final evaluation from checkpoints
    
    Args:
        X_train, y_train, X_val, y_val, X_test, y_test: Data
        best_C, best_gamma: Best hyperparameters
        scaler: Fitted scaler
        checkpoint_dir: Checkpoint directory
        status: Checkpoint status
        
    Returns:
        Dictionary with all trial results
    """
    from svm_train import train_final_svm
    
    print("\n" + "="*70)
    print("RESUMING PHASE 2: FINAL EVALUATION")
    print("="*70)
    print(f"\nBest configuration: C={best_C}, gamma={best_gamma}")
    
    trials = []
    
    # Load completed trials
    if status['n_trials_complete'] > 0:
        print(f"\nLoading {status['n_trials_complete']} completed trials...")
        for trial_file in sorted(status['trial_files']):
            with open(trial_file, 'r') as f:
                trial_result = json.load(f)
            trials.append(trial_result)
            trial_num = int(trial_file.stem.split('_')[1])
            print(f"  ✓ Trial {trial_num} loaded (Test Acc: {trial_result['test_accuracy']:.4f})")
    
    # Run remaining trials
    remaining_trials = FINAL_EVAL_CONFIG['n_trials'] - status['n_trials_complete']
    
    if remaining_trials > 0:
        print(f"\nRunning {remaining_trials} remaining trials...")
        
        for trial_num in range(status['n_trials_complete'] + 1, 
                              FINAL_EVAL_CONFIG['n_trials'] + 1):
            seed = FINAL_EVAL_CONFIG['seeds'][trial_num - 1]
            
            print(f"Trial {trial_num}/{FINAL_EVAL_CONFIG['n_trials']} (seed={seed})... ", 
                  end='', flush=True)
            
            result = train_final_svm(
                X_train, y_train,
                X_val, y_val,
                X_test, y_test,
                best_C, best_gamma,
                scaler, seed
            )
            
            trials.append(result)
            
            print(f"Test Acc: {result['test_accuracy']:.4f}, "
                  f"F1: {result['test_f1_macro']:.4f}, "
                  f"SVs: {result['n_support_vectors']}")
            
            # Save checkpoint
            checkpoint_file = checkpoint_dir / f"trial_{trial_num}_seed{seed}.json"
            with open(checkpoint_file, 'w') as f:
                json.dump(result, f, indent=2)
    else:
        print("\n✓ All trials already complete")
    
    return {'trials': trials}


# ============================================================================
# MAIN RESUME FUNCTION
# ============================================================================

def resume_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    checkpoint_dir: Path
) -> Tuple[Dict, Dict]:
    """
    Resume experiment from any checkpoint state
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        checkpoint_dir: Checkpoint directory
        
    Returns:
        Tuple of (grid_results, trial_results)
    """
    # Detect checkpoint status
    status = detect_checkpoints(checkpoint_dir)
    print_checkpoint_status(status)
    
    # Resume grid search
    grid_results = resume_grid_search(
        X_train, y_train,
        X_val, y_val,
        checkpoint_dir,
        status
    )
    
    # Resume final trials
    trial_results = resume_final_trials(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        grid_results['best_C'],
        grid_results['best_gamma'],
        grid_results['scaler'],
        checkpoint_dir,
        status
    )
    
    return grid_results, trial_results


# ============================================================================
# CHECKPOINT CLEANUP
# ============================================================================

def cleanup_checkpoints(checkpoint_dir: Path, keep_final: bool = True):
    """
    Clean up intermediate checkpoints after successful completion
    
    Args:
        checkpoint_dir: Checkpoint directory
        keep_final: Whether to keep the final complete grid search file
    """
    if not checkpoint_dir.exists():
        return
    
    # Remove individual grid config checkpoints
    for checkpoint_file in checkpoint_dir.glob("grid_C*.json"):
        checkpoint_file.unlink()
        print(f"Removed: {checkpoint_file.name}")
    
    # Optionally remove complete grid search file
    if not keep_final:
        for grid_file in checkpoint_dir.glob("grid_search_complete_*.json"):
            grid_file.unlink()
            print(f"Removed: {grid_file.name}")
    
    # Remove trial checkpoints (they're in the final results anyway)
    for trial_file in checkpoint_dir.glob("trial_*.json"):
        trial_file.unlink()
        print(f"Removed: {trial_file.name}")
    
    print(f"\n✓ Cleanup complete")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_resume_instructions(checkpoint_dir: Path) -> str:
    """
    Generate instructions for resuming from current state
    
    Args:
        checkpoint_dir: Checkpoint directory
        
    Returns:
        String with resume instructions
    """
    status = detect_checkpoints(checkpoint_dir)
    
    instructions = []
    instructions.append("\n" + "="*70)
    instructions.append("HOW TO RESUME")
    instructions.append("="*70)
    
    if status['n_grid_complete'] == 0 and status['n_trials_complete'] == 0:
        instructions.append("\nNo checkpoints found. Start fresh with:")
        instructions.append("  python run_svm_experiment.py")
    
    elif not status['grid_search_complete']:
        instructions.append(f"\nGrid search partially complete ({status['n_grid_complete']}/12)")
        instructions.append("Resume with:")
        instructions.append("  python run_svm_experiment.py --resume")
        instructions.append("  # or")
        instructions.append("  from svm_resume import resume_experiment")
    
    elif status['n_trials_complete'] < FINAL_EVAL_CONFIG['n_trials']:
        instructions.append(f"\nGrid search complete, trials incomplete ({status['n_trials_complete']}/5)")
        instructions.append("Resume with:")
        instructions.append("  python run_svm_experiment.py --resume")
    
    else:
        instructions.append("\n✓ Experiment fully complete!")
        instructions.append("\nClean up checkpoints with:")
        instructions.append("  from svm_resume import cleanup_checkpoints")
        instructions.append("  cleanup_checkpoints(Path('checkpoints/svm'))")
    
    instructions.append("="*70)
    
    return "\n".join(instructions)


if __name__ == "__main__":
    # Example: Check checkpoint status
    from pathlib import Path
    
    checkpoint_dir = Path("checkpoints/svm")
    
    if checkpoint_dir.exists():
        status = detect_checkpoints(checkpoint_dir)
        print_checkpoint_status(status)
        print(get_resume_instructions(checkpoint_dir))
    else:
        print("No checkpoint directory found.")
        print("Run: python run_svm_experiment.py")
