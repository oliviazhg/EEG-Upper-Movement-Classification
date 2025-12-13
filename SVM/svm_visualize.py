"""
SVM Visualization Module
Generate heatmaps, bar plots, confusion matrices, and F1 plots
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional

from svm_config import CLASS_LABELS, PLOT_CONFIG


def setup_plotting_style():
    """Set up consistent plotting style"""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    plt.rcParams['figure.dpi'] = PLOT_CONFIG['dpi']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10


def plot_grid_search_heatmap(
    grid_results: Dict,
    save_path: Optional[Path] = None
):
    """
    Plot heatmap of validation accuracy vs (C, gamma)
    
    Args:
        grid_results: Results from grid_search_svm
        save_path: Path to save figure
    """
    setup_plotting_style()
    
    C_values = grid_results['C_values']
    gamma_values = grid_results['gamma_values']
    val_acc_grid = np.array(grid_results['val_accuracy_grid'])
    best_C = grid_results['best_C']
    best_gamma = grid_results['best_gamma']
    
    fig, ax = plt.subplots(figsize=PLOT_CONFIG['figsize_heatmap'])
    
    # Create heatmap
    im = ax.imshow(val_acc_grid, aspect='auto', cmap=PLOT_CONFIG['colormap'])
    
    # Set ticks
    ax.set_xticks(np.arange(len(gamma_values)))
    ax.set_yticks(np.arange(len(C_values)))
    ax.set_xticklabels(gamma_values)
    ax.set_yticklabels(C_values)
    
    # Labels
    ax.set_xlabel('Gamma (γ)', fontweight='bold')
    ax.set_ylabel('Regularization (C)', fontweight='bold')
    ax.set_title('SVM Grid Search: Validation Accuracy vs Hyperparameters', 
                 fontweight='bold', pad=20)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Validation Accuracy', rotation=270, labelpad=20)
    
    # Annotate cells with accuracy values
    for i in range(len(C_values)):
        for j in range(len(gamma_values)):
            text = ax.text(j, i, f'{val_acc_grid[i, j]:.3f}',
                          ha="center", va="center", color="white", fontweight='bold')
    
    # Mark best configuration
    best_i = C_values.index(best_C)
    best_j = gamma_values.index(best_gamma)
    ax.add_patch(plt.Rectangle((best_j - 0.5, best_i - 0.5), 1, 1,
                               fill=False, edgecolor='red', linewidth=3))
    ax.text(best_j, best_i - 0.35, '★', ha='center', va='center',
            color='red', fontsize=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if PLOT_CONFIG['show_plots']:
        plt.show()
    else:
        plt.close()


def plot_final_performance_bar(
    aggregated: Dict,
    save_path: Optional[Path] = None
):
    """
    Plot bar chart of final test performance with confidence intervals
    
    Args:
        aggregated: Aggregated results from multiple trials
        save_path: Path to save figure
    """
    setup_plotting_style()
    
    fig, ax = plt.subplots(figsize=PLOT_CONFIG['figsize_bar'])
    
    # Data
    test_acc_mean = aggregated['test_accuracy_mean']
    test_acc_ci_low = aggregated['test_accuracy_ci_low']
    test_acc_ci_high = aggregated['test_accuracy_ci_high']
    
    # Error bar
    error = [[test_acc_mean - test_acc_ci_low],
             [test_acc_ci_high - test_acc_mean]]
    
    # Bar plot
    bars = ax.bar(['SVM\n(RBF Kernel)'], [test_acc_mean * 100], 
                  width=0.5, color='steelblue', alpha=0.8, edgecolor='black')
    
    # Error bars
    ax.errorbar(['SVM\n(RBF Kernel)'], [test_acc_mean * 100],
                yerr=np.array(error) * 100, fmt='none', ecolor='black',
                capsize=10, capthick=2, linewidth=2)
    
    # Labels and title
    ax.set_ylabel('Test Accuracy (%)', fontweight='bold')
    ax.set_title(f'SVM Final Performance\n(C={aggregated["C"]}, γ={aggregated["gamma"]})',
                 fontweight='bold', pad=20)
    ax.set_ylim([0, 100])
    
    # Add grid
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    # Annotate with exact values
    ax.text(0, test_acc_mean * 100 + 3,
            f'{test_acc_mean*100:.2f}%\n[{test_acc_ci_low*100:.2f}, {test_acc_ci_high*100:.2f}]',
            ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if PLOT_CONFIG['show_plots']:
        plt.show()
    else:
        plt.close()


def plot_confusion_matrix(
    aggregated: Dict,
    save_path: Optional[Path] = None
):
    """
    Plot averaged confusion matrix
    
    Args:
        aggregated: Aggregated results
        save_path: Path to save figure
    """
    setup_plotting_style()
    
    confusion = np.array(aggregated['confusion_matrix_mean'])
    
    # Normalize by row (true labels)
    confusion_norm = confusion / confusion.sum(axis=1, keepdims=True)
    
    fig, ax = plt.subplots(figsize=PLOT_CONFIG['figsize_confusion'])
    
    # Heatmap
    im = ax.imshow(confusion_norm, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Count', rotation=270, labelpad=20)
    
    # Ticks and labels
    class_names = [CLASS_LABELS[i+1] for i in range(len(confusion))]
    
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    
    ax.set_xlabel('Predicted Label', fontweight='bold')
    ax.set_ylabel('True Label', fontweight='bold')
    ax.set_title('SVM Confusion Matrix (Averaged over 5 trials)', 
                 fontweight='bold', pad=20)
    
    # Annotate cells
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text_color = 'white' if confusion_norm[i, j] > 0.5 else 'black'
            ax.text(j, i, f'{confusion_norm[i, j]:.2f}',
                   ha="center", va="center", color=text_color, fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if PLOT_CONFIG['show_plots']:
        plt.show()
    else:
        plt.close()


def plot_per_class_f1(
    aggregated: Dict,
    save_path: Optional[Path] = None
):
    """
    Plot per-class F1 scores with error bars
    
    Args:
        aggregated: Aggregated results
        save_path: Path to save figure
    """
    setup_plotting_style()
    
    f1_means = np.array(aggregated['test_f1_per_class_mean'])
    f1_stds = np.array(aggregated['test_f1_per_class_std'])
    
    class_names = [CLASS_LABELS[i+1] for i in range(len(f1_means))]
    
    # Sort by F1 score
    sorted_indices = np.argsort(f1_means)[::-1]
    f1_means_sorted = f1_means[sorted_indices]
    f1_stds_sorted = f1_stds[sorted_indices]
    class_names_sorted = [class_names[i] for i in sorted_indices]
    
    fig, ax = plt.subplots(figsize=PLOT_CONFIG['figsize_f1'])
    
    # Bar plot
    x = np.arange(len(class_names_sorted))
    bars = ax.barh(x, f1_means_sorted, xerr=f1_stds_sorted,
                   capsize=5, color='steelblue', alpha=0.8, edgecolor='black')
    
    # Labels
    ax.set_yticks(x)
    ax.set_yticklabels(class_names_sorted)
    ax.set_xlabel('F1 Score', fontweight='bold')
    ax.set_ylabel('Movement Class', fontweight='bold')
    ax.set_title('SVM Per-Class F1 Scores (Test Set)', fontweight='bold', pad=20)
    ax.set_xlim([0, 1])
    
    # Add grid
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    # Add mean line
    overall_mean = aggregated['test_f1_macro_mean']
    ax.axvline(overall_mean, color='red', linestyle='--', linewidth=2, 
               label=f'Overall Mean: {overall_mean:.3f}')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if PLOT_CONFIG['show_plots']:
        plt.show()
    else:
        plt.close()


def plot_training_time_heatmap(
    grid_results: Dict,
    save_path: Optional[Path] = None
):
    """
    Plot heatmap of training time vs (C, gamma)
    
    Args:
        grid_results: Results from grid search
        save_path: Path to save figure
    """
    setup_plotting_style()
    
    C_values = grid_results['C_values']
    gamma_values = grid_results['gamma_values']
    times = np.array(grid_results['training_times'])
    
    fig, ax = plt.subplots(figsize=PLOT_CONFIG['figsize_heatmap'])
    
    # Heatmap
    im = ax.imshow(times, aspect='auto', cmap='YlOrRd')
    
    # Ticks
    ax.set_xticks(np.arange(len(gamma_values)))
    ax.set_yticks(np.arange(len(C_values)))
    ax.set_xticklabels(gamma_values)
    ax.set_yticklabels(C_values)
    
    # Labels
    ax.set_xlabel('Gamma (γ)', fontweight='bold')
    ax.set_ylabel('Regularization (C)', fontweight='bold')
    ax.set_title('SVM Training Time (seconds)', fontweight='bold', pad=20)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Training Time (s)', rotation=270, labelpad=20)
    
    # Annotate
    for i in range(len(C_values)):
        for j in range(len(gamma_values)):
            text = ax.text(j, i, f'{times[i, j]:.1f}s',
                          ha="center", va="center", color="black", fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=PLOT_CONFIG['dpi'], bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if PLOT_CONFIG['show_plots']:
        plt.show()
    else:
        plt.close()


def generate_all_plots(
    grid_results: Dict,
    aggregated: Dict,
    output_dir: Path
):
    """
    Generate all required plots
    
    Args:
        grid_results: Results from grid search
        aggregated: Aggregated final results
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating plots...")
    
    # 1. Grid search heatmap
    plot_grid_search_heatmap(
        grid_results,
        save_path=output_dir / 'grid_search_heatmap.png'
    )
    
    # 2. Final performance bar
    plot_final_performance_bar(
        aggregated,
        save_path=output_dir / 'final_performance.png'
    )
    
    # 3. Confusion matrix
    plot_confusion_matrix(
        aggregated,
        save_path=output_dir / 'confusion_matrix.png'
    )
    
    # 4. Per-class F1
    plot_per_class_f1(
        aggregated,
        save_path=output_dir / 'per_class_f1.png'
    )
    
    # 5. Training time heatmap
    plot_training_time_heatmap(
        grid_results,
        save_path=output_dir / 'training_time_heatmap.png'
    )
    
    print(f"\nAll plots saved to: {output_dir}")
