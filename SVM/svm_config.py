"""
SVM Configuration for EEG Movement Classification
Excludes 'rest' class from classification
"""

import numpy as np

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

# Data parameters
DATA_CONFIG = {
    # Path to the ml_features_data.npz file containing ALL 25 subjects
    'data_file': '/home/ubuntu/multimodal-signal-dataset-for-11-upper-body-movements/PreprocessedData2/ml_features_data.npz',    
    'exclude_rest': True,  # Exclude rest class (label 0)
    'train_split': 0.70,
    'val_split': 0.15,
    'test_split': 0.15,
}

# Classes to use (excluding rest=0)
# 1: forward, 2: backward, 3: left, 4: right, 5: up, 6: down
# 7: power grasp, 8: precision grasp, 9: lateral grasp
# 10: pronation, 11: supination
CLASS_LABELS = {
    1: 'forward',
    2: 'backward', 
    3: 'left',
    4: 'right',
    5: 'up',
    6: 'down',
    7: 'power_grasp',
    8: 'precision_grasp',
    9: 'lateral_grasp',
    10: 'pronation',
    11: 'supination'
}

N_CLASSES = len(CLASS_LABELS)  # 11 classes (excluding rest)

# ============================================================================
# PHASE 1: GRID SEARCH CONFIGURATION
# ============================================================================

GRID_SEARCH_CONFIG = {
    'C_values': [0.1, 1, 10, 100],
    'gamma_values': [0.001, 0.01, 0.1],
    'kernel': 'rbf',
    'random_state': 0,  # Fixed seed for grid search
}

# Total configurations: 4 × 3 = 12
N_GRID_CONFIGS = len(GRID_SEARCH_CONFIG['C_values']) * len(GRID_SEARCH_CONFIG['gamma_values'])

# ============================================================================
# PHASE 2: FINAL EVALUATION CONFIGURATION
# ============================================================================

FINAL_EVAL_CONFIG = {
    'n_trials': 5,
    'seeds': [0, 1, 2, 3, 4],
    'confidence_level': 0.95,  # For bootstrap CI
    'n_bootstrap': 1000,
}

# ============================================================================
# FEATURE SCALING
# ============================================================================

SCALING_CONFIG = {
    'method': 'standard',  # StandardScaler (required for SVM)
    'with_mean': True,
    'with_std': True,
}

# ============================================================================
# CHECKPOINT CONFIGURATION
# ============================================================================

CHECKPOINT_CONFIG = {
    'save_grid_search': True,
    'save_per_trial': True,
    'save_frequency': 1,  # Save after every config/trial
    'checkpoint_dir': 'checkpoints/svm',
}

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================

OUTPUT_CONFIG = {
    'results_dir': 'results/svm',
    'figures_dir': 'figures/svm',
    'save_models': False,  # SVM models can be large, optionally save
    'verbose': True,
}

# ============================================================================
# PLOTTING CONFIGURATION
# ============================================================================

PLOT_CONFIG = {
    'dpi': 300,
    'format': 'png',
    'figsize_heatmap': (8, 6),
    'figsize_bar': (8, 6),
    'figsize_confusion': (12, 10),
    'figsize_f1': (10, 6),
    'colormap': 'viridis',
    'show_plots': False,  # Set to True for interactive
}

# ============================================================================
# VALIDATION
# ============================================================================

def validate_config():
    """Validate configuration settings"""
    # Check splits sum to 1
    total_split = (DATA_CONFIG['train_split'] + 
                   DATA_CONFIG['val_split'] + 
                   DATA_CONFIG['test_split'])
    assert abs(total_split - 1.0) < 1e-6, "Data splits must sum to 1.0"
    
    # Check number of classes
    assert N_CLASSES == 11, "Expected 11 movement classes (excluding rest)"
    
    # Check seeds match n_trials
    assert len(FINAL_EVAL_CONFIG['seeds']) == FINAL_EVAL_CONFIG['n_trials']
    
    # Check grid search parameters are valid
    assert all(c > 0 for c in GRID_SEARCH_CONFIG['C_values']), "C must be positive"
    assert all(g > 0 for g in GRID_SEARCH_CONFIG['gamma_values']), "gamma must be positive"
    
    print("✓ Configuration validated successfully")

if __name__ == "__main__":
    validate_config()
    print(f"\nSVM Configuration:")
    print(f"  Classes: {N_CLASSES} (excluding rest)")
    print(f"  Grid search: {N_GRID_CONFIGS} configurations")
    print(f"  Final trials: {FINAL_EVAL_CONFIG['n_trials']}")
    print(f"  Total experiments: {N_GRID_CONFIGS + FINAL_EVAL_CONFIG['n_trials']}")