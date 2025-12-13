"""
SVM Analysis Module
Compute statistics, confidence intervals, and aggregate results
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple


def compute_bootstrap_ci(
    data: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 1000
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval
    
    Args:
        data: Array of values
        confidence: Confidence level (default 0.95)
        n_bootstrap: Number of bootstrap samples
        
    Returns:
        Tuple of (mean, ci_low, ci_high)
    """
    n = len(data)
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    
    mean = np.mean(data)
    alpha = 1 - confidence
    ci_low = np.percentile(bootstrap_means, 100 * alpha / 2)
    ci_high = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    
    return mean, ci_low, ci_high


def aggregate_trial_results(trials: List[Dict]) -> Dict:
    """
    Aggregate results from multiple trials
    
    Args:
        trials: List of trial result dictionaries
        
    Returns:
        Dictionary with aggregated statistics
    """
    n_trials = len(trials)
    
    # Extract arrays
    test_accs = np.array([t['test_accuracy'] for t in trials])
    test_f1s = np.array([t['test_f1_macro'] for t in trials])
    train_accs = np.array([t['train_accuracy'] for t in trials])
    val_accs = np.array([t['val_accuracy'] for t in trials])
    
    n_support_vectors = np.array([t['n_support_vectors'] for t in trials])
    training_times = np.array([t['training_time_sec'] for t in trials])
    
    # Per-class F1 (n_trials, n_classes)
    test_f1_per_class = np.array([t['test_f1_per_class'] for t in trials])
    
    # Confusion matrices (n_trials, n_classes, n_classes)
    confusion_matrices = np.array([t['confusion_matrix'] for t in trials])
    
    # Compute statistics with bootstrap CI
    test_acc_mean, test_acc_ci_low, test_acc_ci_high = compute_bootstrap_ci(test_accs)
    test_f1_mean, test_f1_ci_low, test_f1_ci_high = compute_bootstrap_ci(test_f1s)
    
    # Simple statistics for other metrics
    train_acc_mean = np.mean(train_accs)
    train_acc_std = np.std(train_accs, ddof=1)
    
    val_acc_mean = np.mean(val_accs)
    val_acc_std = np.std(val_accs, ddof=1)
    
    n_sv_mean = np.mean(n_support_vectors)
    n_sv_std = np.std(n_support_vectors, ddof=1)
    
    time_mean = np.mean(training_times)
    time_std = np.std(training_times, ddof=1)
    
    # Per-class F1 statistics
    test_f1_per_class_mean = np.mean(test_f1_per_class, axis=0)
    test_f1_per_class_std = np.std(test_f1_per_class, axis=0, ddof=1)
    
    # Average confusion matrix
    confusion_matrix_mean = np.mean(confusion_matrices, axis=0)
    confusion_matrix_std = np.std(confusion_matrices, axis=0, ddof=1)
    
    return {
        'n_trials': n_trials,
        'C': trials[0]['C'],
        'gamma': trials[0]['gamma'],
        
        # Test performance with CI
        'test_accuracy_mean': float(test_acc_mean),
        'test_accuracy_ci_low': float(test_acc_ci_low),
        'test_accuracy_ci_high': float(test_acc_ci_high),
        'test_accuracy_std': float(np.std(test_accs, ddof=1)),
        
        'test_f1_macro_mean': float(test_f1_mean),
        'test_f1_macro_ci_low': float(test_f1_ci_low),
        'test_f1_macro_ci_high': float(test_f1_ci_high),
        'test_f1_macro_std': float(np.std(test_f1s, ddof=1)),
        
        # Training/validation performance
        'train_accuracy_mean': float(train_acc_mean),
        'train_accuracy_std': float(train_acc_std),
        'val_accuracy_mean': float(val_acc_mean),
        'val_accuracy_std': float(val_acc_std),
        
        # Per-class F1
        'test_f1_per_class_mean': test_f1_per_class_mean.tolist(),
        'test_f1_per_class_std': test_f1_per_class_std.tolist(),
        
        # Confusion matrix
        'confusion_matrix_mean': confusion_matrix_mean.tolist(),
        'confusion_matrix_std': confusion_matrix_std.tolist(),
        
        # Model characteristics
        'n_support_vectors_mean': float(n_sv_mean),
        'n_support_vectors_std': float(n_sv_std),
        
        # Timing
        'training_time_mean': float(time_mean),
        'training_time_std': float(time_std),
        
        # Raw data for additional analysis
        'test_accuracies_all': test_accs.tolist(),
        'test_f1_macros_all': test_f1s.tolist(),
    }


def print_summary(aggregated: Dict):
    """Print formatted summary of results"""
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    
    print(f"\nConfiguration:")
    print(f"  C = {aggregated['C']}")
    print(f"  gamma = {aggregated['gamma']}")
    print(f"  Number of trials = {aggregated['n_trials']}")
    
    print(f"\nTest Performance:")
    print(f"  Accuracy: {aggregated['test_accuracy_mean']:.4f} "
          f"[{aggregated['test_accuracy_ci_low']:.4f}, "
          f"{aggregated['test_accuracy_ci_high']:.4f}]")
    print(f"  F1 Macro: {aggregated['test_f1_macro_mean']:.4f} "
          f"[{aggregated['test_f1_macro_ci_low']:.4f}, "
          f"{aggregated['test_f1_macro_ci_high']:.4f}]")
    
    print(f"\nTraining Performance:")
    print(f"  Train Accuracy: {aggregated['train_accuracy_mean']:.4f} "
          f"± {aggregated['train_accuracy_std']:.4f}")
    print(f"  Val Accuracy: {aggregated['val_accuracy_mean']:.4f} "
          f"± {aggregated['val_accuracy_std']:.4f}")
    
    print(f"\nModel Characteristics:")
    print(f"  Support Vectors: {aggregated['n_support_vectors_mean']:.1f} "
          f"± {aggregated['n_support_vectors_std']:.1f}")
    print(f"  Training Time: {aggregated['training_time_mean']:.2f}s "
          f"± {aggregated['training_time_std']:.2f}s")
    
    print("\nPer-Class F1 Scores:")
    from svm_config import CLASS_LABELS
    f1_means = aggregated['test_f1_per_class_mean']
    f1_stds = aggregated['test_f1_per_class_std']
    
    # Sort by F1 score
    class_indices = sorted(range(len(f1_means)), key=lambda i: f1_means[i], reverse=True)
    
    for idx in class_indices:
        class_num = idx + 1  # Classes are 1-indexed
        class_name = CLASS_LABELS.get(class_num, f"Class {class_num}")
        print(f"  {class_name:20s}: {f1_means[idx]:.4f} ± {f1_stds[idx]:.4f}")
    
    print("="*70)


def compare_to_baseline(svm_results: Dict, baseline_results: Dict) -> Dict:
    """
    Compare SVM results to baseline (e.g., Random Forest)
    
    Args:
        svm_results: Aggregated SVM results
        baseline_results: Aggregated baseline results
        
    Returns:
        Dictionary with comparison statistics
    """
    # Extract test accuracies
    svm_accs = np.array(svm_results['test_accuracies_all'])
    baseline_accs = np.array(baseline_results['test_accuracies_all'])
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(svm_accs, baseline_accs)
    
    # Effect size (Cohen's d for paired samples)
    diff = svm_accs - baseline_accs
    d = np.mean(diff) / np.std(diff, ddof=1)
    
    # Mean difference and CI
    mean_diff = np.mean(diff)
    diff_ci_low = np.percentile(diff, 2.5)
    diff_ci_high = np.percentile(diff, 97.5)
    
    comparison = {
        'svm_mean': float(np.mean(svm_accs)),
        'baseline_mean': float(np.mean(baseline_accs)),
        'mean_difference': float(mean_diff),
        'difference_ci_low': float(diff_ci_low),
        'difference_ci_high': float(diff_ci_high),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'cohens_d': float(d),
        'significant_at_0.05': p_value < 0.05,
    }
    
    print("\n" + "="*70)
    print("COMPARISON TO BASELINE")
    print("="*70)
    print(f"\nSVM Accuracy: {comparison['svm_mean']:.4f}")
    print(f"Baseline Accuracy: {comparison['baseline_mean']:.4f}")
    print(f"Difference: {comparison['mean_difference']:.4f} "
          f"[{comparison['difference_ci_low']:.4f}, {comparison['difference_ci_high']:.4f}]")
    print(f"\nPaired t-test:")
    print(f"  t = {comparison['t_statistic']:.3f}")
    print(f"  p = {comparison['p_value']:.4f}")
    print(f"  Cohen's d = {comparison['cohens_d']:.3f}")
    print(f"  Significant at α=0.05: {comparison['significant_at_0.05']}")
    print("="*70)
    
    return comparison


def analyze_confusion_patterns(confusion_matrix_mean: np.ndarray) -> Dict:
    """
    Analyze common confusion patterns
    
    Args:
        confusion_matrix_mean: Average confusion matrix
        
    Returns:
        Dictionary with confusion analysis
    """
    n_classes = confusion_matrix_mean.shape[0]
    
    # Normalize by row (true labels)
    confusion_normalized = confusion_matrix_mean / confusion_matrix_mean.sum(axis=1, keepdims=True)
    
    # Find most confused pairs
    np.fill_diagonal(confusion_normalized, 0)  # Ignore diagonal
    
    confused_pairs = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j:
                confused_pairs.append({
                    'true_class': i + 1,
                    'predicted_class': j + 1,
                    'confusion_rate': float(confusion_normalized[i, j])
                })
    
    # Sort by confusion rate
    confused_pairs.sort(key=lambda x: x['confusion_rate'], reverse=True)
    
    return {
        'top_10_confusions': confused_pairs[:10],
        'confusion_matrix_normalized': confusion_normalized.tolist(),
    }


def print_confusion_analysis(analysis: Dict):
    """Print confusion analysis"""
    from svm_config import CLASS_LABELS
    
    print("\n" + "="*70)
    print("CONFUSION ANALYSIS")
    print("="*70)
    print("\nTop 10 Most Confused Class Pairs:")
    
    for i, pair in enumerate(analysis['top_10_confusions'], 1):
        true_name = CLASS_LABELS.get(pair['true_class'], f"Class {pair['true_class']}")
        pred_name = CLASS_LABELS.get(pair['predicted_class'], f"Class {pair['predicted_class']}")
        rate = pair['confusion_rate']
        
        print(f"{i:2d}. {true_name:20s} → {pred_name:20s}: {rate:.2%}")
    
    print("="*70)
