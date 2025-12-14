"""
CNN Results Analysis and Visualization
SYDE 522 Final Project

Analyzes results from CNN experiments and generates publication-quality figures.
Includes:
- Learning curves (train/val loss and accuracy)
- Architecture comparison
- Confusion matrices
- Per-class F1 scores
- Statistical analysis with bootstrap confidence intervals
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from pathlib import Path
from typing import Dict, List
from scipy import stats
from scipy.stats import bootstrap


# Movement labels from the dataset
MOVEMENT_LABELS = [
    'Forward', 'Backward', 'Left', 'Right', 'Up', 'Down',
    'Power', 'Precision', 'Lateral', 'Pronation', 'Supination'
]


def compute_bootstrap_ci(data: np.ndarray, 
                         confidence_level: float = 0.95,
                         n_resamples: int = 10000) -> tuple:
    """
    Compute bootstrap confidence interval
    
    Args:
        data: 1D array of values (e.g., test accuracies from 5 seeds)
        confidence_level: CI level (default 95%)
        n_resamples: Number of bootstrap samples
        
    Returns:
        (mean, ci_low, ci_high)
    """
    mean = np.mean(data)
    
    # Bootstrap CI using scipy
    rng = np.random.default_rng(42)
    res = bootstrap(
        (data,),
        np.mean,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        random_state=rng
    )
    
    return mean, res.confidence_interval.low, res.confidence_interval.high


def aggregate_architecture_results(trials: List[Dict], 
                                   architecture: str) -> Dict:
    """
    Aggregate results for a specific architecture across all seeds
    
    Args:
        trials: List of trial dictionaries
        architecture: '2layer', '3layer', or '4layer'
        
    Returns:
        Dictionary with aggregated metrics and confidence intervals
    """
    # Filter trials for this architecture
    arch_trials = [t for t in trials if t['architecture'] == architecture]
    
    if len(arch_trials) == 0:
        raise ValueError(f"No trials found for architecture {architecture}")
    
    print(f"\nAggregating {architecture}: {len(arch_trials)} trials")
    
    # Extract metrics
    test_accs = np.array([t['test_accuracy'] for t in arch_trials])
    test_f1s = np.array([t['test_f1_macro'] for t in arch_trials])
    
    # Per-class F1 (shape: n_trials x 11 classes)
    test_f1_per_class = np.array([t['test_f1_per_class'] for t in arch_trials])
    
    # Confusion matrices
    cms = np.array([t['confusion_matrix'] for t in arch_trials])
    
    # Convergence
    epochs_to_converge = np.array([t['final_epoch'] for t in arch_trials])
    
    # Model size
    n_params = arch_trials[0]['n_parameters']
    
    # Compute CIs for test accuracy and F1
    acc_mean, acc_ci_low, acc_ci_high = compute_bootstrap_ci(test_accs)
    f1_mean, f1_ci_low, f1_ci_high = compute_bootstrap_ci(test_f1s)
    
    print(f"Test Acc: {acc_mean:.4f} [{acc_ci_low:.4f}, {acc_ci_high:.4f}]")
    print(f"Test F1:  {f1_mean:.4f} [{f1_ci_low:.4f}, {f1_ci_high:.4f}]")
    
    # Aggregate training curves
    # Find max epoch length
    max_epochs = max(len(t['training_history']['train_loss']) for t in arch_trials)
    
    # Initialize arrays for curves (pad shorter trials with NaN)
    n_trials = len(arch_trials)
    train_loss_curves = np.full((n_trials, max_epochs), np.nan)
    val_loss_curves = np.full((n_trials, max_epochs), np.nan)
    train_acc_curves = np.full((n_trials, max_epochs), np.nan)
    val_acc_curves = np.full((n_trials, max_epochs), np.nan)
    
    for i, trial in enumerate(arch_trials):
        hist = trial['training_history']
        n_epochs = len(hist['train_loss'])
        
        train_loss_curves[i, :n_epochs] = hist['train_loss']
        val_loss_curves[i, :n_epochs] = hist['val_loss']
        train_acc_curves[i, :n_epochs] = hist['train_accuracy']
        val_acc_curves[i, :n_epochs] = hist['val_accuracy']
    
    # Compute mean and CI per epoch (ignoring NaN)
    def nanmean_ci(arr, axis=0):
        mean = np.nanmean(arr, axis=axis)
        std = np.nanstd(arr, axis=axis)
        n = np.sum(~np.isnan(arr), axis=axis)
        # 95% CI using t-distribution
        ci_width = stats.t.ppf(0.975, n-1) * std / np.sqrt(n)
        return mean, mean - ci_width, mean + ci_width
    
    train_loss_mean, train_loss_ci_low, train_loss_ci_high = nanmean_ci(train_loss_curves)
    val_loss_mean, val_loss_ci_low, val_loss_ci_high = nanmean_ci(val_loss_curves)
    train_acc_mean, train_acc_ci_low, train_acc_ci_high = nanmean_ci(train_acc_curves)
    val_acc_mean, val_acc_ci_low, val_acc_ci_high = nanmean_ci(val_acc_curves)
    
    return {
        'architecture': architecture,
        'n_parameters': int(n_params),
        'n_trials': len(arch_trials),
        
        # Test performance with CI
        'test_acc_mean': float(acc_mean),
        'test_acc_ci_low': float(acc_ci_low),
        'test_acc_ci_high': float(acc_ci_high),
        
        'test_f1_mean': float(f1_mean),
        'test_f1_ci_low': float(f1_ci_low),
        'test_f1_ci_high': float(f1_ci_high),
        
        # Per-class F1
        'test_f1_per_class_mean': np.nanmean(test_f1_per_class, axis=0).tolist(),
        'test_f1_per_class_std': np.nanstd(test_f1_per_class, axis=0).tolist(),
        
        # Training curves (mean ± CI)
        'mean_train_loss_curve': train_loss_mean.tolist(),
        'train_loss_ci_low': train_loss_ci_low.tolist(),
        'train_loss_ci_high': train_loss_ci_high.tolist(),
        
        'mean_val_loss_curve': val_loss_mean.tolist(),
        'val_loss_ci_low': val_loss_ci_low.tolist(),
        'val_loss_ci_high': val_loss_ci_high.tolist(),
        
        'mean_train_acc_curve': train_acc_mean.tolist(),
        'train_acc_ci_low': train_acc_ci_low.tolist(),
        'train_acc_ci_high': train_acc_ci_high.tolist(),
        
        'mean_val_acc_curve': val_acc_mean.tolist(),
        'val_acc_ci_low': val_acc_ci_low.tolist(),
        'val_acc_ci_high': val_acc_ci_high.tolist(),
        
        # Convergence
        'mean_epochs_to_converge': float(np.mean(epochs_to_converge)),
        'std_epochs_to_converge': float(np.std(epochs_to_converge)),
        
        # Average confusion matrix
        'confusion_matrix_mean': np.mean(cms, axis=0).tolist(),
    }


def plot_learning_curves(aggregated_results: List[Dict], 
                         output_dir: str = 'figures'):
    """
    Plot learning curves (loss and accuracy) for all architectures
    
    Figure 1: Train/Val Loss curves
    Figure 2: Train/Val Accuracy curves
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    n_archs = len(aggregated_results)
    
    # FIGURE 1: Loss curves
    fig, axes = plt.subplots(1, n_archs, figsize=(6*n_archs, 4))
    if n_archs == 1:
        axes = [axes]
    
    for i, arch_data in enumerate(aggregated_results):
        ax = axes[i]
        arch = arch_data['architecture']
        
        # Extract curves
        epochs = np.arange(len(arch_data['mean_train_loss_curve']))
        
        train_loss = np.array(arch_data['mean_train_loss_curve'])
        train_loss_low = np.array(arch_data['train_loss_ci_low'])
        train_loss_high = np.array(arch_data['train_loss_ci_high'])
        
        val_loss = np.array(arch_data['mean_val_loss_curve'])
        val_loss_low = np.array(arch_data['val_loss_ci_low'])
        val_loss_high = np.array(arch_data['val_loss_ci_high'])
        
        # Plot
        ax.plot(epochs, train_loss, 'b-', label='Train', linewidth=2)
        ax.fill_between(epochs, train_loss_low, train_loss_high, 
                        color='blue', alpha=0.2)
        
        ax.plot(epochs, val_loss, 'r-', label='Validation', linewidth=2)
        ax.fill_between(epochs, val_loss_low, val_loss_high, 
                        color='red', alpha=0.2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss (Categorical Crossentropy)', fontsize=12)
        ax.set_title(f'{arch.upper()} CNN\n({arch_data["n_parameters"]:,} parameters)', 
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(output_path / 'cnn_loss_curves.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'cnn_loss_curves.pdf', bbox_inches='tight')
    print(f"Saved: {output_path / 'cnn_loss_curves.png'}")
    plt.close()
    
    # FIGURE 2: Accuracy curves
    fig, axes = plt.subplots(1, n_archs, figsize=(6*n_archs, 4))
    if n_archs == 1:
        axes = [axes]
    
    for i, arch_data in enumerate(aggregated_results):
        ax = axes[i]
        arch = arch_data['architecture']
        
        epochs = np.arange(len(arch_data['mean_train_acc_curve']))
        
        train_acc = np.array(arch_data['mean_train_acc_curve']) * 100
        train_acc_low = np.array(arch_data['train_acc_ci_low']) * 100
        train_acc_high = np.array(arch_data['train_acc_ci_high']) * 100
        
        val_acc = np.array(arch_data['mean_val_acc_curve']) * 100
        val_acc_low = np.array(arch_data['val_acc_ci_low']) * 100
        val_acc_high = np.array(arch_data['val_acc_ci_high']) * 100
        
        ax.plot(epochs, train_acc, 'b-', label='Train', linewidth=2)
        ax.fill_between(epochs, train_acc_low, train_acc_high, 
                        color='blue', alpha=0.2)
        
        ax.plot(epochs, val_acc, 'r-', label='Validation', linewidth=2)
        ax.fill_between(epochs, val_acc_low, val_acc_high, 
                        color='red', alpha=0.2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title(f'{arch.upper()} CNN', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 100])
        
    plt.tight_layout()
    plt.savefig(output_path / 'cnn_accuracy_curves.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'cnn_accuracy_curves.pdf', bbox_inches='tight')
    print(f"Saved: {output_path / 'cnn_accuracy_curves.png'}")
    plt.close()


def plot_architecture_comparison(aggregated_results: List[Dict],
                                 output_dir: str = 'figures'):
    """
    Bar plot comparing test accuracy across architectures
    """
    output_path = Path(output_dir)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    architectures = [r['architecture'].upper() for r in aggregated_results]
    test_accs = [r['test_acc_mean'] * 100 for r in aggregated_results]
    
    # Error bars
    err_low = [(r['test_acc_mean'] - r['test_acc_ci_low']) * 100 
               for r in aggregated_results]
    err_high = [(r['test_acc_ci_high'] - r['test_acc_mean']) * 100 
                for r in aggregated_results]
    errors = np.array([err_low, err_high])
    
    # Bar plot
    x = np.arange(len(architectures))
    bars = ax.bar(x, test_accs, yerr=errors, capsize=10,
                   color=['#3498db', '#e74c3c', '#2ecc71'][:len(architectures)],
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, test_accs)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add parameter counts
    for i, r in enumerate(aggregated_results):
        ax.text(i, 5, f'{r["n_parameters"]:,}\nparams',
                ha='center', va='bottom', fontsize=9, style='italic')
    
    ax.set_xlabel('CNN Architecture', fontsize=13, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('CNN Architecture Comparison\n(11-class Upper-Limb Movement Classification)',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(architectures, fontsize=12)
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path / 'cnn_architecture_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'cnn_architecture_comparison.pdf', bbox_inches='tight')
    print(f"Saved: {output_path / 'cnn_architecture_comparison.png'}")
    plt.close()


def plot_confusion_matrix(aggregated_results: List[Dict],
                         output_dir: str = 'figures'):
    """
    Plot confusion matrix for best architecture
    """
    output_path = Path(output_dir)
    
    # Find best architecture
    best_arch = max(aggregated_results, key=lambda x: x['test_acc_mean'])
    cm = np.array(best_arch['confusion_matrix_mean'])
    
    # Normalize to percentages
    cm_norm = cm / cm.sum(axis=1, keepdims=True) * 100
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=MOVEMENT_LABELS,
                yticklabels=MOVEMENT_LABELS,
                cbar_kws={'label': 'Percentage (%)'},
                ax=ax, vmin=0, vmax=100)
    
    ax.set_xlabel('Predicted Movement', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Movement', fontsize=13, fontweight='bold')
    ax.set_title(f'Confusion Matrix - {best_arch["architecture"].upper()} CNN\n'
                f'(Test Acc: {best_arch["test_acc_mean"]*100:.2f}%)',
                fontsize=14, fontweight='bold')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path / 'cnn_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'cnn_confusion_matrix.pdf', bbox_inches='tight')
    print(f"Saved: {output_path / 'cnn_confusion_matrix.png'}")
    plt.close()


def plot_per_class_f1(aggregated_results: List[Dict],
                      output_dir: str = 'figures'):
    """
    Plot per-class F1 scores for best architecture
    """
    output_path = Path(output_dir)
    
    # Find best architecture
    best_arch = max(aggregated_results, key=lambda x: x['test_acc_mean'])
    
    f1_mean = np.array(best_arch['test_f1_per_class_mean']) * 100
    f1_std = np.array(best_arch['test_f1_per_class_std']) * 100
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(MOVEMENT_LABELS))
    bars = ax.bar(x, f1_mean, yerr=f1_std, capsize=5,
                   color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Color code by movement type
    colors = ['#e74c3c']*6 + ['#2ecc71']*3 + ['#f39c12']*2  # Reaches, Grasps, Rotations
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Add value labels
    for i, (mean, std) in enumerate(zip(f1_mean, f1_std)):
        ax.text(i, mean + std + 2, f'{mean:.1f}',
                ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Movement Class', fontsize=13, fontweight='bold')
    ax.set_ylabel('F1 Score (%)', fontsize=13, fontweight='bold')
    ax.set_title(f'Per-Class F1 Scores - {best_arch["architecture"].upper()} CNN',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(MOVEMENT_LABELS, rotation=45, ha='right', fontsize=11)
    ax.set_ylim([0, 100])
    ax.axhline(y=best_arch['test_f1_mean']*100, color='black', 
               linestyle='--', linewidth=2, label=f'Mean F1: {best_arch["test_f1_mean"]*100:.1f}%')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='Reaches (6)'),
        Patch(facecolor='#2ecc71', label='Grasps (3)'),
        Patch(facecolor='#f39c12', label='Rotations (2)')
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path / 'cnn_per_class_f1.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'cnn_per_class_f1.pdf', bbox_inches='tight')
    print(f"Saved: {output_path / 'cnn_per_class_f1.png'}")
    plt.close()


def analyze_cnn_results(results_file: str = 'results/cnn/all_trials.pkl',
                       output_dir: str = 'results/cnn'):
    """
    Complete analysis pipeline for CNN results
    
    Args:
        results_file: Path to pickled trial results
        output_dir: Where to save outputs
    """
    print(f"\n{'='*60}")
    print(f"CNN RESULTS ANALYSIS")
    print(f"{'='*60}")
    
    # Load results
    with open(results_file, 'rb') as f:
        all_trials = pickle.load(f)
    
    print(f"Loaded {len(all_trials)} trials")
    
    # Aggregate by architecture
    architectures = sorted(set(t['architecture'] for t in all_trials))
    print(f"Architectures: {architectures}")
    
    aggregated = []
    for arch in architectures:
        agg = aggregate_architecture_results(all_trials, arch)
        aggregated.append(agg)
    
    # Save aggregated results
    output_path = Path(output_dir)
    with open(output_path / 'aggregated_results.json', 'w') as f:
        json.dump(aggregated, f, indent=2)
    print(f"\nSaved aggregated results to: {output_path / 'aggregated_results.json'}")
    
    # Generate all plots
    print(f"\n{'Generating figures...':-^60}")
    plot_learning_curves(aggregated, output_dir='figures')
    plot_architecture_comparison(aggregated, output_dir='figures')
    plot_confusion_matrix(aggregated, output_dir='figures')
    plot_per_class_f1(aggregated, output_dir='figures')
    
    # Print summary table
    print(f"\n{'SUMMARY TABLE':-^60}")
    print(f"{'Architecture':<15} {'Params':<12} {'Test Acc (%)':<20} {'Test F1 (%)':<20}")
    print(f"{'-'*60}")
    for agg in aggregated:
        acc_str = f"{agg['test_acc_mean']*100:.2f} [{agg['test_acc_ci_low']*100:.2f}, {agg['test_acc_ci_high']*100:.2f}]"
        f1_str = f"{agg['test_f1_mean']*100:.2f} [{agg['test_f1_ci_low']*100:.2f}, {agg['test_f1_ci_high']*100:.2f}]"
        print(f"{agg['architecture'].upper():<15} {agg['n_parameters']:<12,} {acc_str:<20} {f1_str:<20}")
    
    print(f"\n{'='*60}")
    print(f"Analysis complete!")
    print(f"{'='*60}")
    
    return aggregated


if __name__ == '__main__':
    # Run analysis if results exist
    results_path = Path('results/cnn/all_trials.pkl')
    if results_path.exists():
        analyze_cnn_results()
    else:
        print(f"No results found at {results_path}")
        print("Run cnn_eeg_classification.py first to generate results")
