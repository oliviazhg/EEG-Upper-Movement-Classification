"""
Random Forest Experiment for EEG-Based Upper-Limb Movement Classification
SYDE 522 Final Project

This script implements Random Forest classification with hand-crafted features
extracted from 60-channel EEG data for 11 upper-limb movements.

Author: Olivia Zheng
Dataset: Gigascience 2020 Upper Limb Movement Dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import pickle
from typing import Dict, List, Tuple
from scipy import signal
from scipy.stats import bootstrap, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Experimental configuration parameters"""
    # Data parameters
    FS = 2500  # Sampling frequency (Hz)
    N_CHANNELS = 60  # Number of EEG channels
    TRIAL_DURATION = 2.0  # seconds
    N_SAMPLES = int(FS * TRIAL_DURATION)  # 5000 samples
    
    # Frequency bands
    MU_BAND = (8, 12)  # Hz
    BETA_BAND = (13, 30)  # Hz
    
    # Movement classes (11 movements)
    MOVEMENTS = [
        'forward', 'backward', 'left', 'right', 'up', 'down',  # 6 reaches
        'power_grasp', 'precision_grasp', 'lateral_grasp',      # 3 grasps
        'pronation', 'supination'                                # 2 rotations
    ]
    N_CLASSES = len(MOVEMENTS)
    
    # Data split
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # Experimental parameters
    RANDOM_SEEDS = [0, 1, 2, 3, 4]
    N_TRIALS = len(RANDOM_SEEDS)
    
    # Random Forest configurations to test
    RF_CONFIGS = [
        {'n_estimators': 100, 'max_features': 'sqrt'},
        {'n_estimators': 100, 'max_features': 72},  # 0.3 * 240
        {'n_estimators': 200, 'max_features': 'sqrt'},
        {'n_estimators': 200, 'max_features': 72},
    ]
    
    # Output directories
    OUTPUT_DIR = Path('./results/random_forest')
    FIGURE_DIR = OUTPUT_DIR / 'figures'
    DATA_DIR = OUTPUT_DIR / 'data'
    
    # Feature names for interpretability
    FEATURE_NAMES = None  # Will be generated in feature extraction


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def butter_bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, 
                           fs: float, order: int = 5) -> np.ndarray:
    """
    Apply Butterworth bandpass filter to EEG data.
    
    Args:
        data: EEG data (n_trials, n_channels, n_samples)
        lowcut: Lower frequency bound (Hz)
        highcut: Upper frequency bound (Hz)
        fs: Sampling frequency (Hz)
        order: Filter order
    
    Returns:
        Filtered data with same shape as input
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    
    # Apply filter to each trial and channel
    filtered = np.zeros_like(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            filtered[i, j, :] = signal.filtfilt(b, a, data[i, j, :])
    
    return filtered


def extract_band_power(data: np.ndarray, lowcut: float, highcut: float, 
                       fs: float) -> np.ndarray:
    """
    Extract band power features from EEG data.
    
    Args:
        data: EEG data (n_trials, n_channels, n_samples)
        lowcut: Lower frequency bound (Hz)
        highcut: Upper frequency bound (Hz)
        fs: Sampling frequency (Hz)
    
    Returns:
        Band power for each trial and channel (n_trials, n_channels)
    """
    # Filter data
    filtered = butter_bandpass_filter(data, lowcut, highcut, fs)
    
    # Compute variance (power) for each trial and channel
    power = np.var(filtered, axis=2)
    
    return power


def extract_all_features(eeg_data: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """
    Extract all 240 features from EEG data.
    
    Features per channel (4 features × 60 channels = 240 total):
    - Mu band power (8-12 Hz)
    - Beta band power (13-30 Hz)
    - Mean amplitude
    - Standard deviation
    
    Args:
        eeg_data: Raw EEG data (n_trials, n_channels, n_samples)
    
    Returns:
        features: Feature matrix (n_trials, 240)
        feature_names: List of feature names for interpretability
    """
    n_trials = eeg_data.shape[0]
    features = np.zeros((n_trials, 4 * Config.N_CHANNELS))
    feature_names = []
    
    print("Extracting features...")
    
    # 1. Mu band power (60 features)
    print("  - Mu band power...")
    mu_power = extract_band_power(eeg_data, *Config.MU_BAND, Config.FS)
    features[:, 0:60] = mu_power
    feature_names.extend([f'mu_power_ch{i+1}' for i in range(Config.N_CHANNELS)])
    
    # 2. Beta band power (60 features)
    print("  - Beta band power...")
    beta_power = extract_band_power(eeg_data, *Config.BETA_BAND, Config.FS)
    features[:, 60:120] = beta_power
    feature_names.extend([f'beta_power_ch{i+1}' for i in range(Config.N_CHANNELS)])
    
    # 3. Mean amplitude (60 features)
    print("  - Mean amplitude...")
    mean_amp = np.mean(eeg_data, axis=2)
    features[:, 120:180] = mean_amp
    feature_names.extend([f'mean_ch{i+1}' for i in range(Config.N_CHANNELS)])
    
    # 4. Standard deviation (60 features)
    print("  - Standard deviation...")
    std_amp = np.std(eeg_data, axis=2)
    features[:, 180:240] = std_amp
    feature_names.extend([f'std_ch{i+1}' for i in range(Config.N_CHANNELS)])
    
    print(f"Feature extraction complete. Shape: {features.shape}")
    
    return features, feature_names


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_preprocess_data(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess EEG data from the Gigascience dataset.
    
    IMPORTANT: This function removes the rest class (class 0) from the data,
    as we are only classifying the 11 active movements.
    
    Args:
        data_path: Path to the dataset
    
    Returns:
        eeg_data: EEG trials (n_trials, n_channels, n_samples)
        labels: Movement labels (n_trials,) - classes 0-10 for the 11 movements
    """
    print(f"Loading data from {data_path}...")
    
    # Load data based on format
    data_path = Path(data_path)
    
    if (data_path / 'eeg_data.npy').exists():
        # Load from .npy files
        eeg_data = np.load(data_path / 'eeg_data.npy')
        labels = np.load(data_path / 'labels.npy')
        print(f"Loaded from .npy files")
    
    elif data_path.suffix == '.npz':
        # Load from .npz file (from preprocessing script)
        data = np.load(data_path)
        eeg_data = data['X']
        labels = data['y']
        print(f"Loaded from .npz file")
    
    else:
        # For demonstration, create synthetic data
        print("WARNING: Using synthetic data for demonstration!")
        n_trials = 1100  # ~100 trials per class
        eeg_data = np.random.randn(n_trials, Config.N_CHANNELS, Config.N_SAMPLES)
        labels = np.repeat(np.arange(Config.N_CLASSES), n_trials // Config.N_CLASSES)
    
    print(f"Initial data shape: {eeg_data.shape}, Labels: {labels.shape}")
    print(f"Unique classes before filtering: {np.unique(labels)}")
    
    # CRITICAL: Remove rest class (class 0)
    # This is important because:
    # 1. Rest has ~550 trials vs ~50 for each movement (11:1 imbalance)
    # 2. We're only interested in classifying active movements
    # 3. This matches the experimental design in the paper
    
    print("\nRemoving rest class (class 0)...")
    mask = labels != 0  # Keep only non-rest trials
    eeg_data = eeg_data[mask]
    labels = labels[mask]
    
    # Renumber classes from 1-11 to 0-10
    labels = labels - 1
    
    print(f"After removing rest:")
    print(f"  Data shape: {eeg_data.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Unique classes: {np.unique(labels)}")
    print(f"  Number of classes: {len(np.unique(labels))}")
    
    # Verify we have the correct number of classes
    assert len(np.unique(labels)) == Config.N_CLASSES, \
        f"Expected {Config.N_CLASSES} classes, got {len(np.unique(labels))}"
    
    # Verify class distribution
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\nClass distribution:")
    for cls, count in zip(unique, counts):
        print(f"  Class {cls} ({Config.MOVEMENTS[cls]}): {count} trials")
    
    return eeg_data, labels


def split_data(features: np.ndarray, labels: np.ndarray, 
               seed: int) -> Tuple[np.ndarray, ...]:
    """
    Split data into train/validation/test sets with stratification.
    
    Args:
        features: Feature matrix (n_trials, n_features)
        labels: Class labels (n_trials,)
        seed: Random seed for reproducibility
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        features, labels,
        test_size=Config.TEST_RATIO,
        stratify=labels,
        random_state=seed
    )
    
    # Second split: train vs val
    val_ratio_adjusted = Config.VAL_RATIO / (Config.TRAIN_RATIO + Config.VAL_RATIO)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio_adjusted,
        stratify=y_temp,
        random_state=seed
    )
    
    print(f"Data split (seed={seed}):")
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Val:   {X_val.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================================
# RANDOM FOREST TRAINING AND EVALUATION
# ============================================================================

def train_random_forest(X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray,
                       n_estimators: int = 100,
                       max_features: int = 15,
                       seed: int = 0) -> Dict:
    """
    Train Random Forest and collect comprehensive metrics.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        n_estimators: Number of trees
        max_features: Number of features per split
        seed: Random seed
    
    Returns:
        Dictionary containing all metrics and metadata
    """
    print(f"\nTraining RF: n_trees={n_estimators}, max_feat={max_features}, seed={seed}")
    
    # Initialize Random Forest
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features=max_features,
        oob_score=True,
        random_state=seed,
        n_jobs=-1,  # Use all CPU cores
        class_weight='balanced',  # Handle class imbalance
        verbose=0
    )
    
    # Training
    start_time = time.time()
    rf.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Predictions
    y_train_pred = rf.predict(X_train)
    y_val_pred = rf.predict(X_val)
    y_test_pred = rf.predict(X_test)
    
    # Inference time (average over test set)
    start_time = time.time()
    _ = rf.predict(X_test)
    inference_time = (time.time() - start_time) / len(X_test)
    
    # Collect metrics
    results = {
        # Configuration
        'n_estimators': n_estimators,
        'max_features': max_features,
        'seed': seed,
        
        # Accuracy
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'val_accuracy': accuracy_score(y_val, y_val_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        
        # Macro F1
        'train_f1_macro': f1_score(y_train, y_train_pred, average='macro'),
        'val_f1_macro': f1_score(y_val, y_val_pred, average='macro'),
        'test_f1_macro': f1_score(y_test, y_test_pred, average='macro'),
        
        # Per-class F1
        'test_f1_per_class': f1_score(y_test, y_test_pred, average=None),
        
        # Confusion matrix
        'confusion_matrix': confusion_matrix(y_test, y_test_pred),
        
        # Timing
        'training_time_sec': training_time,
        'inference_time_per_sample': inference_time,
        
        # Feature importance
        'feature_importances': rf.feature_importances_,
        
        # OOB score
        'oob_score': rf.oob_score_,
    }
    
    print(f"  Train acc: {results['train_accuracy']:.3f}")
    print(f"  Val acc:   {results['val_accuracy']:.3f}")
    print(f"  Test acc:  {results['test_accuracy']:.3f}")
    print(f"  OOB score: {results['oob_score']:.3f}")
    print(f"  Training time: {training_time:.2f}s")
    
    return results


# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def compute_bootstrap_ci(data: np.ndarray, confidence_level: float = 0.95,
                        n_resamples: int = 1000) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for the mean.
    
    Args:
        data: Array of measurements (e.g., accuracies from 5 trials)
        confidence_level: CI level (default 95%)
        n_resamples: Number of bootstrap resamples
    
    Returns:
        mean, ci_low, ci_high
    """
    rng = np.random.default_rng(seed=42)
    
    res = bootstrap(
        (data,),
        np.mean,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        random_state=rng
    )
    
    mean = np.mean(data)
    ci_low = res.confidence_interval.low
    ci_high = res.confidence_interval.high
    
    return mean, ci_low, ci_high


def analyze_feature_importance_stability(importance_arrays: List[np.ndarray]) -> Dict:
    """
    Analyze stability of feature importance across random seeds.
    
    Args:
        importance_arrays: List of feature importance arrays from different seeds
    
    Returns:
        Dictionary with stability metrics
    """
    n_trials = len(importance_arrays)
    correlations = []
    
    # Compute pairwise Spearman correlations
    for i in range(n_trials):
        for j in range(i + 1, n_trials):
            corr, p_val = spearmanr(importance_arrays[i], importance_arrays[j])
            correlations.append(corr)
    
    stability = {
        'mean_correlation': np.mean(correlations),
        'std_correlation': np.std(correlations),
        'min_correlation': np.min(correlations),
        'max_correlation': np.max(correlations),
    }
    
    return stability


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_parameter_comparison(config_summaries: List[Dict], save_path: Path):
    """
    Create 2×2 grid bar plot comparing RF configurations.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Organize data
    n_trees_100 = [s for s in config_summaries if s['n_estimators'] == 100]
    n_trees_200 = [s for s in config_summaries if s['n_estimators'] == 200]
    
    x = np.arange(2)  # Two groups (100 trees, 200 trees)
    width = 0.35
    
    # Extract accuracies and CIs
    acc_100_sqrt = [s for s in n_trees_100 if s['max_features'] == 'sqrt'][0]
    acc_100_const = [s for s in n_trees_100 if s['max_features'] == 72][0]
    acc_200_sqrt = [s for s in n_trees_200 if s['max_features'] == 'sqrt'][0]
    acc_200_const = [s for s in n_trees_200 if s['max_features'] == 72][0]
    
    means_sqrt = [acc_100_sqrt['test_acc_mean'], acc_200_sqrt['test_acc_mean']]
    means_const = [acc_100_const['test_acc_mean'], acc_200_const['test_acc_mean']]
    
    # Compute error bar sizes
    err_sqrt_low = [acc_100_sqrt['test_acc_mean'] - acc_100_sqrt['test_acc_ci_low'],
                    acc_200_sqrt['test_acc_mean'] - acc_200_sqrt['test_acc_ci_low']]
    err_sqrt_high = [acc_100_sqrt['test_acc_ci_high'] - acc_100_sqrt['test_acc_mean'],
                     acc_200_sqrt['test_acc_ci_high'] - acc_200_sqrt['test_acc_mean']]
    err_const_low = [acc_100_const['test_acc_mean'] - acc_100_const['test_acc_ci_low'],
                     acc_200_const['test_acc_mean'] - acc_200_const['test_acc_ci_low']]
    err_const_high = [acc_100_const['test_acc_ci_high'] - acc_100_const['test_acc_mean'],
                      acc_200_const['test_acc_ci_high'] - acc_200_const['test_acc_mean']]
    
    # Plot bars
    ax.bar(x - width/2, means_sqrt, width, 
           yerr=[err_sqrt_low, err_sqrt_high],
           label='sqrt features', capsize=5)
    ax.bar(x + width/2, means_const, width,
           yerr=[err_const_low, err_const_high],
           label='0.3×features (72)', capsize=5)
    
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_xlabel('Number of Trees', fontsize=12)
    ax.set_title('Random Forest Parameter Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['100 trees', '200 trees'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved parameter comparison plot to {save_path}")


def plot_feature_importance(feature_names: List[str], 
                           importances_mean: np.ndarray,
                           importances_std: np.ndarray,
                           top_n: int = 20,
                           save_path: Path = None):
    """
    Plot top N most important features as horizontal bar chart.
    """
    # Get top N features
    top_indices = np.argsort(importances_mean)[-top_n:][::-1]
    top_names = [feature_names[i] for i in top_indices]
    top_means = importances_mean[top_indices]
    top_stds = importances_std[top_indices]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_pos = np.arange(len(top_names))
    ax.barh(y_pos, top_means, xerr=top_stds, capsize=3, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Mean Feature Importance', fontsize=12)
    ax.set_title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved feature importance plot to {save_path}")
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], 
                         save_path: Path = None):
    """
    Plot confusion matrix heatmap.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Normalize by row (true labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={'label': 'Proportion'})
    
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    plt.close()


def plot_per_class_f1(f1_scores: np.ndarray, class_names: List[str],
                     save_path: Path = None):
    """
    Plot per-class F1 scores as bar chart.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(class_names))
    ax.bar(x, f1_scores, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_xlabel('Movement Class', fontsize=12)
    ax.set_title('Per-Class F1 Scores', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved per-class F1 plot to {save_path}")
    plt.close()


# ============================================================================
# MAIN EXPERIMENTAL PIPELINE
# ============================================================================

def run_experiment(data_path: str = './data'):
    """
    Main experimental pipeline for Random Forest classification.
    """
    print("="*80)
    print("RANDOM FOREST EXPERIMENT - EEG UPPER-LIMB MOVEMENT CLASSIFICATION")
    print("="*80)
    
    # Create output directories
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    Config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # 1. LOAD AND PREPROCESS DATA
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 1: DATA LOADING AND PREPROCESSING")
    print("="*80)
    
    eeg_data, labels = load_and_preprocess_data(data_path)
    
    # ========================================================================
    # 2. FEATURE EXTRACTION (Done once for all experiments)
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2: FEATURE EXTRACTION")
    print("="*80)
    
    features, feature_names = extract_all_features(eeg_data)
    Config.FEATURE_NAMES = feature_names
    
    # Save features for future use
    np.save(Config.DATA_DIR / 'features.npy', features)
    np.save(Config.DATA_DIR / 'labels.npy', labels)
    with open(Config.DATA_DIR / 'feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    
    # ========================================================================
    # 3. RUN EXPERIMENTS FOR ALL CONFIGURATIONS
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 3: TRAINING AND EVALUATION")
    print("="*80)
    
    all_results = []
    
    for config_idx, rf_config in enumerate(Config.RF_CONFIGS):
        print(f"\n{'='*80}")
        print(f"Configuration {config_idx + 1}/{len(Config.RF_CONFIGS)}: {rf_config}")
        print(f"{'='*80}")
        
        config_results = []
        
        for seed in Config.RANDOM_SEEDS:
            # Split data
            X_train, X_val, X_test, y_train, y_val, y_test = split_data(
                features, labels, seed
            )
            
            # Train and evaluate
            result = train_random_forest(
                X_train, y_train, X_val, y_val, X_test, y_test,
                n_estimators=rf_config['n_estimators'],
                max_features=rf_config['max_features'],
                seed=seed
            )
            
            config_results.append(result)
            all_results.append(result)
        
        # Save intermediate results for this configuration
        config_name = f"n{rf_config['n_estimators']}_mf{rf_config['max_features']}"
        with open(Config.DATA_DIR / f'results_{config_name}.pkl', 'wb') as f:
            pickle.dump(config_results, f)
    
    # ========================================================================
    # 4. AGGREGATE RESULTS AND COMPUTE STATISTICS
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 4: STATISTICAL ANALYSIS")
    print("="*80)
    
    config_summaries = []
    
    for rf_config in Config.RF_CONFIGS:
        # Filter results for this configuration
        config_results = [
            r for r in all_results 
            if r['n_estimators'] == rf_config['n_estimators'] 
            and r['max_features'] == rf_config['max_features']
        ]
        
        # Extract metrics arrays
        test_accs = np.array([r['test_accuracy'] for r in config_results])
        test_f1s = np.array([r['test_f1_macro'] for r in config_results])
        
        # Compute bootstrap CIs
        acc_mean, acc_ci_low, acc_ci_high = compute_bootstrap_ci(test_accs)
        f1_mean, f1_ci_low, f1_ci_high = compute_bootstrap_ci(test_f1s)
        
        # Per-class F1 (average across trials)
        f1_per_class_all = np.array([r['test_f1_per_class'] for r in config_results])
        f1_per_class_mean = np.mean(f1_per_class_all, axis=0)
        
        # Feature importance analysis
        importances_all = np.array([r['feature_importances'] for r in config_results])
        importances_mean = np.mean(importances_all, axis=0)
        importances_std = np.std(importances_all, axis=0)
        
        # Top features
        top_20_indices = np.argsort(importances_mean)[-20:][::-1]
        top_20_importance = importances_mean[top_20_indices]
        
        # Feature importance stability
        stability = analyze_feature_importance_stability(importances_all.tolist())
        
        # Average confusion matrix
        cms = np.array([r['confusion_matrix'] for r in config_results])
        cm_mean = np.mean(cms, axis=0)
        
        summary = {
            'n_estimators': rf_config['n_estimators'],
            'max_features': rf_config['max_features'],
            
            'test_acc_mean': acc_mean,
            'test_acc_ci_low': acc_ci_low,
            'test_acc_ci_high': acc_ci_high,
            
            'test_f1_mean': f1_mean,
            'test_f1_ci_low': f1_ci_low,
            'test_f1_ci_high': f1_ci_high,
            
            'test_f1_per_class_mean': f1_per_class_mean,
            
            'feature_importances_mean': importances_mean,
            'feature_importances_std': importances_std,
            
            'top_20_features_indices': top_20_indices.tolist(),
            'top_20_features_importance': top_20_importance.tolist(),
            
            'feature_importance_stability': stability,
            
            'confusion_matrix_mean': cm_mean,
        }
        
        config_summaries.append(summary)
        
        # Print summary
        print(f"\nConfiguration: n_trees={rf_config['n_estimators']}, "
              f"max_features={rf_config['max_features']}")
        print(f"  Test Accuracy: {acc_mean:.3f} [{acc_ci_low:.3f}, {acc_ci_high:.3f}]")
        print(f"  Test F1 (macro): {f1_mean:.3f} [{f1_ci_low:.3f}, {f1_ci_high:.3f}]")
        print(f"  Feature importance stability: {stability['mean_correlation']:.3f} "
              f"± {stability['std_correlation']:.3f}")
    
    # Save aggregated summaries
    with open(Config.DATA_DIR / 'config_summaries.pkl', 'wb') as f:
        pickle.dump(config_summaries, f)
    
    # ========================================================================
    # 5. GENERATE PLOTS
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 5: GENERATING VISUALIZATIONS")
    print("="*80)
    
    # Find best configuration
    best_idx = np.argmax([s['test_acc_mean'] for s in config_summaries])
    best_config = config_summaries[best_idx]
    
    print(f"\nBest configuration: n_trees={best_config['n_estimators']}, "
          f"max_features={best_config['max_features']}")
    print(f"  Test Accuracy: {best_config['test_acc_mean']:.3f}")
    
    # Plot 1: Parameter comparison
    plot_parameter_comparison(
        config_summaries,
        Config.FIGURE_DIR / 'parameter_comparison.png'
    )
    
    # Plot 2: Feature importance (best config)
    plot_feature_importance(
        Config.FEATURE_NAMES,
        best_config['feature_importances_mean'],
        best_config['feature_importances_std'],
        top_n=20,
        save_path=Config.FIGURE_DIR / 'feature_importance.png'
    )
    
    # Plot 3: Confusion matrix (best config)
    plot_confusion_matrix(
        best_config['confusion_matrix_mean'],
        Config.MOVEMENTS,
        save_path=Config.FIGURE_DIR / 'confusion_matrix.png'
    )
    
    # Plot 4: Per-class F1 (best config)
    plot_per_class_f1(
        best_config['test_f1_per_class_mean'],
        Config.MOVEMENTS,
        save_path=Config.FIGURE_DIR / 'per_class_f1.png'
    )
    
    # ========================================================================
    # 6. GENERATE SUMMARY REPORT
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 6: GENERATING SUMMARY REPORT")
    print("="*80)
    
    report_path = Config.OUTPUT_DIR / 'experiment_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("RANDOM FOREST EXPERIMENT SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write("Dataset Information:\n")
        f.write(f"  Total trials: {len(labels)}\n")
        f.write(f"  Number of classes: {Config.N_CLASSES}\n")
        f.write(f"  Number of features: 240\n")
        f.write(f"  Train/Val/Test split: {Config.TRAIN_RATIO}/{Config.VAL_RATIO}/{Config.TEST_RATIO}\n\n")
        
        f.write("Experimental Setup:\n")
        f.write(f"  Number of configurations: {len(Config.RF_CONFIGS)}\n")
        f.write(f"  Trials per configuration: {Config.N_TRIALS}\n")
        f.write(f"  Random seeds: {Config.RANDOM_SEEDS}\n\n")
        
        f.write("Results by Configuration:\n")
        f.write("-"*80 + "\n")
        
        for summary in config_summaries:
            f.write(f"\nConfiguration: n_estimators={summary['n_estimators']}, "
                   f"max_features={summary['max_features']}\n")
            f.write(f"  Test Accuracy: {summary['test_acc_mean']:.4f} "
                   f"[{summary['test_acc_ci_low']:.4f}, {summary['test_acc_ci_high']:.4f}]\n")
            f.write(f"  Test F1 (macro): {summary['test_f1_mean']:.4f} "
                   f"[{summary['test_f1_ci_low']:.4f}, {summary['test_f1_ci_high']:.4f}]\n")
            f.write(f"  Feature stability: {summary['feature_importance_stability']['mean_correlation']:.4f} "
                   f"± {summary['feature_importance_stability']['std_correlation']:.4f}\n")
        
        f.write("\n" + "-"*80 + "\n")
        f.write(f"\nBest Configuration:\n")
        f.write(f"  n_estimators: {best_config['n_estimators']}\n")
        f.write(f"  max_features: {best_config['max_features']}\n")
        f.write(f"  Test Accuracy: {best_config['test_acc_mean']:.4f}\n")
        
        f.write("\nTop 10 Most Important Features:\n")
        for i in range(10):
            idx = best_config['top_20_features_indices'][i]
            imp = best_config['top_20_features_importance'][i]
            f.write(f"  {i+1}. {Config.FEATURE_NAMES[idx]}: {imp:.4f}\n")
    
    print(f"Saved summary report to {report_path}")
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {Config.OUTPUT_DIR}")
    print(f"Figures saved to: {Config.FIGURE_DIR}")
    print(f"Data saved to: {Config.DATA_DIR}")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Random Forest experiment for EEG movement classification'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default='./data',
        help='Path to dataset directory'
    )
    
    args = parser.parse_args()
    
    run_experiment(data_path=args.data_path)