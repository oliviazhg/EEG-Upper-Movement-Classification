"""
Post-Hoc Analysis Script for Random Forest Results
SYDE 522 Final Project

This script performs detailed analysis of Random Forest experiment results,
including statistical significance testing, effect size calculations, and
additional visualizations.

Author: Olivia Zheng
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from scipy import stats
from scipy.stats import ttest_rel
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size.
    
    Args:
        x, y: Two sets of measurements to compare
    
    Returns:
        Cohen's d value
    """
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof)
    
    return (np.mean(x) - np.mean(y)) / pooled_std


def bonferroni_correction(p_values: np.ndarray) -> np.ndarray:
    """
    Apply Bonferroni correction to p-values.
    
    Args:
        p_values: Array of p-values
    
    Returns:
        Corrected p-values
    """
    n_comparisons = len(p_values)
    return np.minimum(p_values * n_comparisons, 1.0)


def pairwise_statistical_tests(config_summaries: List[Dict], 
                               all_results: List[Dict]) -> pd.DataFrame:
    """
    Perform pairwise statistical comparisons between configurations.
    
    Args:
        config_summaries: List of configuration summaries
        all_results: List of all trial results
    
    Returns:
        DataFrame with statistical test results
    """
    n_configs = len(config_summaries)
    results = []
    
    print("Performing pairwise statistical tests...")
    
    for i in range(n_configs):
        for j in range(i + 1, n_configs):
            config_i = config_summaries[i]
            config_j = config_summaries[j]
            
            # Get accuracies for these configurations
            accs_i = [r['test_accuracy'] for r in all_results 
                     if r['n_estimators'] == config_i['n_estimators']
                     and r['max_features'] == config_i['max_features']]
            accs_j = [r['test_accuracy'] for r in all_results 
                     if r['n_estimators'] == config_j['n_estimators']
                     and r['max_features'] == config_j['max_features']]
            
            # Paired t-test
            t_stat, p_value = ttest_rel(accs_i, accs_j)
            
            # Effect size
            effect_size = cohen_d(np.array(accs_i), np.array(accs_j))
            
            results.append({
                'config_1': f"n{config_i['n_estimators']}_mf{config_i['max_features']}",
                'config_2': f"n{config_j['n_estimators']}_mf{config_j['max_features']}",
                'mean_diff': np.mean(accs_i) - np.mean(accs_j),
                't_statistic': t_stat,
                'p_value': p_value,
                'cohen_d': effect_size,
            })
    
    df = pd.DataFrame(results)
    
    # Apply Bonferroni correction
    df['p_value_corrected'] = bonferroni_correction(df['p_value'].values)
    
    # Determine significance
    df['significant'] = df['p_value_corrected'] < 0.05
    
    # Interpret effect size
    def interpret_effect_size(d):
        abs_d = abs(d)
        if abs_d < 0.2:
            return 'negligible'
        elif abs_d < 0.5:
            return 'small'
        elif abs_d < 0.8:
            return 'medium'
        else:
            return 'large'
    
    df['effect_size_interpretation'] = df['cohen_d'].apply(interpret_effect_size)
    
    return df


# ============================================================================
# FEATURE IMPORTANCE ANALYSIS
# ============================================================================

def analyze_feature_importance(config_summaries: List[Dict],
                               feature_names: List[str]) -> Dict:
    """
    Analyze feature importance patterns across configurations.
    
    Returns:
        Dictionary with analysis results
    """
    print("\nAnalyzing feature importance patterns...")
    
    # Find best configuration
    best_idx = np.argmax([s['test_acc_mean'] for s in config_summaries])
    best_config = config_summaries[best_idx]
    
    # Get top features
    top_indices = best_config['top_20_features_indices']
    top_names = [feature_names[i] for i in top_indices]
    top_importance = best_config['top_20_features_importance']
    
    # Categorize features by type
    feature_categories = {
        'mu_power': [],
        'beta_power': [],
        'mean_amplitude': [],
        'std_deviation': []
    }
    
    for idx, name in zip(top_indices, top_names):
        if 'mu_power' in name:
            feature_categories['mu_power'].append((name, best_config['feature_importances_mean'][idx]))
        elif 'beta_power' in name:
            feature_categories['beta_power'].append((name, best_config['feature_importances_mean'][idx]))
        elif 'mean' in name:
            feature_categories['mean_amplitude'].append((name, best_config['feature_importances_mean'][idx]))
        elif 'std' in name:
            feature_categories['std_deviation'].append((name, best_config['feature_importances_mean'][idx]))
    
    # Calculate category contributions
    category_totals = {
        cat: sum(imp for _, imp in features)
        for cat, features in feature_categories.items()
    }
    
    return {
        'top_features': list(zip(top_names, top_importance)),
        'feature_categories': feature_categories,
        'category_totals': category_totals,
        'best_config': best_config,
    }


# ============================================================================
# ADDITIONAL VISUALIZATIONS
# ============================================================================

def plot_accuracy_distribution(all_results: List[Dict], save_path: Path):
    """
    Plot distribution of accuracies across configurations and seeds.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Organize data by configuration
    configs = {}
    for result in all_results:
        key = f"n{result['n_estimators']}_mf{result['max_features']}"
        if key not in configs:
            configs[key] = []
        configs[key].append(result['test_accuracy'])
    
    # Create box plot
    data_to_plot = [configs[key] for key in sorted(configs.keys())]
    labels = sorted(configs.keys())
    
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
    
    # Color boxes
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_title('Distribution of Test Accuracies Across Random Seeds', 
                fontweight='bold', fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=15, ha='right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved accuracy distribution plot to {save_path}")


def plot_feature_category_contribution(importance_analysis: Dict, save_path: Path):
    """
    Plot contribution of different feature categories.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = list(importance_analysis['category_totals'].keys())
    totals = list(importance_analysis['category_totals'].values())
    
    # Create pie chart
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    explode = (0.05, 0.05, 0.05, 0.05)
    
    ax.pie(totals, labels=categories, autopct='%1.1f%%', startangle=90,
           colors=colors, explode=explode)
    ax.set_title('Feature Category Contribution to Top 20 Features',
                fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved feature category plot to {save_path}")


def plot_training_time_comparison(all_results: List[Dict], save_path: Path):
    """
    Compare training times across configurations.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Organize data
    configs = {}
    for result in all_results:
        key = f"n{result['n_estimators']}_mf{result['max_features']}"
        if key not in configs:
            configs[key] = []
        configs[key].append(result['training_time_sec'])
    
    # Calculate means and stds
    labels = sorted(configs.keys())
    means = [np.mean(configs[key]) for key in labels]
    stds = [np.std(configs[key]) for key in labels]
    
    # Create bar plot
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylabel('Training Time (seconds)', fontsize=12)
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_title('Training Time Comparison', fontweight='bold', fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved training time comparison to {save_path}")


def plot_learning_curves(all_results: List[Dict], save_path: Path):
    """
    Plot learning curves (train vs val vs test accuracy).
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get best configuration
    # First, aggregate by config
    config_accs = {}
    for result in all_results:
        key = f"n{result['n_estimators']}_mf{result['max_features']}"
        if key not in config_accs:
            config_accs[key] = []
        config_accs[key].append(result['test_accuracy'])
    
    # Find best
    best_config = max(config_accs.keys(), 
                     key=lambda k: np.mean(config_accs[k]))
    
    # Get results for best config
    best_results = [r for r in all_results 
                   if f"n{r['n_estimators']}_mf{r['max_features']}" == best_config]
    
    # Extract accuracies
    train_accs = [r['train_accuracy'] for r in best_results]
    val_accs = [r['val_accuracy'] for r in best_results]
    test_accs = [r['test_accuracy'] for r in best_results]
    
    # Plot
    x = range(1, len(train_accs) + 1)
    ax.plot(x, train_accs, 'o-', label='Train', linewidth=2, markersize=8)
    ax.plot(x, val_accs, 's-', label='Validation', linewidth=2, markersize=8)
    ax.plot(x, test_accs, '^-', label='Test', linewidth=2, markersize=8)
    
    ax.set_xlabel('Trial (Random Seed)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'Learning Curves for Best Configuration ({best_config})',
                fontweight='bold', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved learning curves to {save_path}")


def plot_statistical_significance_matrix(stat_df: pd.DataFrame, save_path: Path):
    """
    Create a matrix showing statistical significance between configurations.
    """
    # Get unique configurations
    all_configs = sorted(set(stat_df['config_1'].tolist() + stat_df['config_2'].tolist()))
    n_configs = len(all_configs)
    
    # Create matrix
    p_matrix = np.ones((n_configs, n_configs))
    effect_matrix = np.zeros((n_configs, n_configs))
    
    for _, row in stat_df.iterrows():
        i = all_configs.index(row['config_1'])
        j = all_configs.index(row['config_2'])
        
        p_matrix[i, j] = row['p_value_corrected']
        p_matrix[j, i] = row['p_value_corrected']
        
        effect_matrix[i, j] = row['cohen_d']
        effect_matrix[j, i] = -row['cohen_d']  # Symmetric
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # P-value matrix
    sns.heatmap(-np.log10(p_matrix), annot=p_matrix, fmt='.3f',
                cmap='RdYlGn', center=-np.log10(0.05),
                xticklabels=all_configs, yticklabels=all_configs,
                ax=ax1, cbar_kws={'label': '-log10(p-value)'})
    ax1.set_title('Statistical Significance (Bonferroni-corrected p-values)',
                 fontweight='bold')
    
    # Effect size matrix
    sns.heatmap(effect_matrix, annot=True, fmt='.2f',
                cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                xticklabels=all_configs, yticklabels=all_configs,
                ax=ax2, cbar_kws={'label': "Cohen's d"})
    ax2.set_title("Effect Sizes (Cohen's d)", fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved statistical significance matrix to {save_path}")


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def analyze_results(results_dir: str = './results/random_forest'):
    """
    Perform comprehensive post-hoc analysis of Random Forest results.
    
    Args:
        results_dir: Directory containing experiment results
    """
    results_path = Path(results_dir)
    data_dir = results_path / 'data'
    figure_dir = results_path / 'figures'
    
    if not data_dir.exists():
        print(f"ERROR: Results directory not found: {data_dir}")
        print("Make sure you've run the experiment first!")
        return
    
    print("="*80)
    print("POST-HOC ANALYSIS OF RANDOM FOREST RESULTS")
    print("="*80)
    
    # Create additional figures directory
    posthoc_dir = figure_dir / 'posthoc'
    posthoc_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # LOAD RESULTS
    # ========================================================================
    print("\nLoading results...")
    
    # Load all trial results
    all_results = []
    for result_file in data_dir.glob('results_*.pkl'):
        with open(result_file, 'rb') as f:
            all_results.extend(pickle.load(f))
    
    # Load config summaries
    with open(data_dir / 'config_summaries.pkl', 'rb') as f:
        config_summaries = pickle.load(f)
    
    # Load feature names
    with open(data_dir / 'feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    
    print(f"  Loaded {len(all_results)} trial results")
    print(f"  Loaded {len(config_summaries)} configuration summaries")
    
    # ========================================================================
    # STATISTICAL TESTS
    # ========================================================================
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS")
    print("="*80)
    
    stat_df = pairwise_statistical_tests(config_summaries, all_results)
    
    # Save statistical results
    stat_df.to_csv(results_path / 'statistical_tests.csv', index=False)
    print(f"\nSaved statistical test results to {results_path / 'statistical_tests.csv'}")
    
    # Print summary
    print("\nStatistically Significant Differences (p < 0.05, Bonferroni-corrected):")
    significant = stat_df[stat_df['significant']]
    if len(significant) > 0:
        for _, row in significant.iterrows():
            print(f"  {row['config_1']} vs {row['config_2']}: "
                  f"Δ = {row['mean_diff']:.4f}, "
                  f"p = {row['p_value_corrected']:.4f}, "
                  f"d = {row['cohen_d']:.2f} ({row['effect_size_interpretation']})")
    else:
        print("  No statistically significant differences found.")
    
    # ========================================================================
    # FEATURE IMPORTANCE ANALYSIS
    # ========================================================================
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*80)
    
    importance_analysis = analyze_feature_importance(config_summaries, feature_names)
    
    print("\nTop 10 Most Important Features:")
    for i, (name, imp) in enumerate(importance_analysis['top_features'][:10]):
        print(f"  {i+1}. {name}: {imp:.4f}")
    
    print("\nFeature Category Contributions:")
    for cat, total in importance_analysis['category_totals'].items():
        print(f"  {cat}: {total:.4f}")
    
    # ========================================================================
    # ADDITIONAL VISUALIZATIONS
    # ========================================================================
    print("\n" + "="*80)
    print("GENERATING ADDITIONAL VISUALIZATIONS")
    print("="*80)
    
    plot_accuracy_distribution(all_results, posthoc_dir / 'accuracy_distribution.png')
    plot_feature_category_contribution(importance_analysis, 
                                      posthoc_dir / 'feature_category_contribution.png')
    plot_training_time_comparison(all_results, posthoc_dir / 'training_time_comparison.png')
    plot_learning_curves(all_results, posthoc_dir / 'learning_curves.png')
    plot_statistical_significance_matrix(stat_df, 
                                        posthoc_dir / 'statistical_significance_matrix.png')
    
    # ========================================================================
    # GENERATE ENHANCED REPORT
    # ========================================================================
    print("\n" + "="*80)
    print("GENERATING ENHANCED REPORT")
    print("="*80)
    
    report_path = results_path / 'posthoc_analysis_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("POST-HOC ANALYSIS REPORT - RANDOM FOREST EXPERIMENT\n")
        f.write("="*80 + "\n\n")
        
        # Statistical tests
        f.write("1. STATISTICAL SIGNIFICANCE TESTS\n")
        f.write("-"*80 + "\n\n")
        f.write("Pairwise Comparisons (Bonferroni-corrected):\n\n")
        for _, row in stat_df.iterrows():
            f.write(f"{row['config_1']} vs {row['config_2']}:\n")
            f.write(f"  Mean difference: {row['mean_diff']:.4f}\n")
            f.write(f"  t-statistic: {row['t_statistic']:.4f}\n")
            f.write(f"  p-value (uncorrected): {row['p_value']:.4f}\n")
            f.write(f"  p-value (Bonferroni): {row['p_value_corrected']:.4f}\n")
            f.write(f"  Cohen's d: {row['cohen_d']:.4f} ({row['effect_size_interpretation']})\n")
            f.write(f"  Significant: {'Yes' if row['significant'] else 'No'}\n\n")
        
        # Feature importance
        f.write("\n2. FEATURE IMPORTANCE ANALYSIS\n")
        f.write("-"*80 + "\n\n")
        f.write("Top 20 Most Important Features:\n\n")
        for i, (name, imp) in enumerate(importance_analysis['top_features']):
            f.write(f"{i+1:2d}. {name:30s}: {imp:.6f}\n")
        
        f.write("\nFeature Category Contributions:\n\n")
        for cat, total in sorted(importance_analysis['category_totals'].items(), 
                                key=lambda x: x[1], reverse=True):
            pct = 100 * total / sum(importance_analysis['category_totals'].values())
            f.write(f"  {cat:20s}: {total:.6f} ({pct:.2f}%)\n")
        
        # Best configuration details
        f.write("\n3. BEST CONFIGURATION DETAILS\n")
        f.write("-"*80 + "\n\n")
        best = importance_analysis['best_config']
        f.write(f"Configuration: n_estimators={best['n_estimators']}, "
               f"max_features={best['max_features']}\n\n")
        f.write(f"Test Accuracy: {best['test_acc_mean']:.4f} "
               f"[{best['test_acc_ci_low']:.4f}, {best['test_acc_ci_high']:.4f}]\n")
        f.write(f"Test F1 (macro): {best['test_f1_mean']:.4f} "
               f"[{best['test_f1_ci_low']:.4f}, {best['test_f1_ci_high']:.4f}]\n\n")
        
        f.write("Per-class F1 scores:\n")
        # Assuming class names from Config
        from random_forest_experiment import Config
        for i, (movement, f1) in enumerate(zip(Config.MOVEMENTS, 
                                               best['test_f1_per_class_mean'])):
            f.write(f"  {movement:20s}: {f1:.4f}\n")
    
    print(f"Saved enhanced report to {report_path}")
    
    print("\n" + "="*80)
    print("POST-HOC ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nAdditional figures saved to: {posthoc_dir}")
    print(f"Enhanced report saved to: {report_path}")
    print(f"Statistical tests saved to: {results_path / 'statistical_tests.csv'}")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Post-hoc analysis of Random Forest experiment results'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='./results/random_forest',
        help='Directory containing experiment results'
    )
    
    args = parser.parse_args()
    
    analyze_results(results_dir=args.results_dir)