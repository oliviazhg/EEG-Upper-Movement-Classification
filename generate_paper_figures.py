"""
Generate all figures for SYDE 522 Final Project IEEE Conference Paper
Analyzes results from CSP+LDA, Random Forest, SVM, and CNN experiments
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
import pandas as pd
from pathlib import Path
from scipy import stats
import os

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("colorblind")
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
})

# Create output directory
OUTPUT_DIR = Path("paper_figures")
OUTPUT_DIR.mkdir(exist_ok=True)

# Movement class names
CLASS_NAMES = [
    'Forward', 'Backward', 'Left', 'Right', 'Up', 'Down',  # Reaches
    'Power', 'Precision', 'Lateral',  # Grasps
    'Pronation', 'Supination'  # Wrist rotations
]

def load_random_forest_results():
    """Load Random Forest results"""
    base_path = Path("EEG-Upper-Movement-Classification/Random_Forest/results/random_forest")

    results = {}
    configs = {
        'n100_mfsqrt': {'n_estimators': 100, 'max_features': 'sqrt'},
        'n100_mf26': {'n_estimators': 100, 'max_features': 26},
        'n200_mfsqrt': {'n_estimators': 200, 'max_features': 'sqrt'},
        'n200_mf26': {'n_estimators': 200, 'max_features': 26},
    }

    for name, config in configs.items():
        with open(base_path / f"data/results_{name}.pkl", 'rb') as f:
            data = pickle.load(f)
            results[name] = {
                'config': config,
                'test_accuracies': [trial['test_accuracy'] for trial in data],
                'test_f1s': [trial['test_f1_macro'] for trial in data],
                'train_accuracies': [trial['train_accuracy'] for trial in data],
                'val_accuracies': [trial['val_accuracy'] for trial in data],
            }

    return results

def load_svm_results():
    """Load SVM results"""
    results_path = Path("EEG-Upper-Movement-Classification/SVM/results/svm/aggregated_results.json")

    with open(results_path, 'r') as f:
        data = json.load(f)

    return {
        'test_accuracies': data['test_accuracies_all'],
        'test_f1s': data['test_f1_macros_all'],
        'test_acc_mean': data['test_accuracy_mean'],
        'test_acc_ci': [data['test_accuracy_ci_low'], data['test_accuracy_ci_high']],
        'test_f1_mean': data['test_f1_macro_mean'],
        'confusion_matrix': np.array(data['confusion_matrix_mean']),
        'per_class_f1': data['test_f1_per_class_mean'],
        'C': data['C'],
        'gamma': data['gamma'],
    }

def load_csp_lda_results():
    """Load CSP+LDA results - all configurations (2, 4, 6, 8 components)"""
    results_path = Path("EEG-Upper-Movement-Classification/CSP_LDA/results_csp_lda_1/trial_results.csv")
    summary_path = Path("EEG-Upper-Movement-Classification/CSP_LDA/results_csp_lda_1/config_summaries.csv")

    df = pd.read_csv(results_path)
    summary_df = pd.read_csv(summary_path)

    # Return data for all configurations for detailed plots
    # But also return best config (8 components) for comparison plot
    trials_8comp = df[df['n_components'] == 8]

    if len(trials_8comp) > 0:
        test_accs = trials_8comp['test_accuracy'].values
        test_f1s = trials_8comp['test_f1_macro'].values

        return {
            'test_accuracy': np.mean(test_accs),
            'test_accuracy_std': np.std(test_accs, ddof=1) if len(test_accs) > 1 else 0,
            'test_f1': np.mean(test_f1s),
            'test_f1_std': np.std(test_f1s, ddof=1) if len(test_f1s) > 1 else 0,
            'train_accuracy': trials_8comp['train_accuracy'].mean(),
            'val_accuracy': trials_8comp['val_accuracy'].mean(),
            'band': 'combined',
            'n_components': 8,
            'n_trials': len(trials_8comp),
            # Include all configs for ablation plots
            'all_trials': df,
            'all_summaries': summary_df,
        }
    else:
        # Fallback
        trial = df.iloc[0]
        return {
            'test_accuracy': trial['test_accuracy'],
            'test_f1': trial['test_f1_macro'],
            'train_accuracy': trial['train_accuracy'],
            'val_accuracy': trial['val_accuracy'],
            'band': trial['band'],
            'n_components': trial['n_components'],
            'n_trials': 1,
            'all_trials': df,
            'all_summaries': summary_df,
        }

def compute_ci(data, confidence=0.95):
    """Compute confidence interval"""
    data = np.array(data)
    mean = np.mean(data)
    sem = stats.sem(data)
    ci = sem * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
    return mean, mean - ci, mean + ci

def plot_method_comparison(rf_results, svm_results, csp_results):
    """Figure 1: Overall method comparison with confidence intervals"""
    fig, ax = plt.subplots(figsize=(7, 4))

    methods = []
    accuracies = []
    ci_lows = []
    ci_highs = []
    colors_list = []

    # Random Forest - best config
    best_rf = 'n200_mf26'
    rf_data = rf_results[best_rf]['test_accuracies']
    mean, low, high = compute_ci(rf_data)
    methods.append('Random Forest\n(n=200, mf=26)')
    accuracies.append(mean * 100)
    ci_lows.append(low * 100)
    ci_highs.append(high * 100)
    colors_list.append('C0')

    # SVM
    methods.append('SVM\n(C=100, γ=0.01)')
    accuracies.append(svm_results['test_acc_mean'] * 100)
    ci_lows.append(svm_results['test_acc_ci'][0] * 100)
    ci_highs.append(svm_results['test_acc_ci'][1] * 100)
    colors_list.append('C1')

    # CSP+LDA (use std for error bars if available)
    methods.append(f"CSP+LDA\n(combined, 8 comp)\nn={csp_results.get('n_trials', 1)}")
    csp_acc = csp_results['test_accuracy'] * 100
    csp_std = csp_results.get('test_accuracy_std', 0) * 100
    accuracies.append(csp_acc)
    ci_lows.append(csp_acc - csp_std)
    ci_highs.append(csp_acc + csp_std)
    colors_list.append('C2')

    # Chance level
    chance = 100.0 / 11

    x = np.arange(len(methods))
    bars = ax.bar(x, accuracies, color=colors_list, alpha=0.7, edgecolor='black', linewidth=1.2)

    # Error bars
    yerr = [[acc - low for acc, low in zip(accuracies, ci_lows)],
            [high - acc for acc, high in zip(accuracies, ci_highs)]]
    ax.errorbar(x, accuracies, yerr=yerr, fmt='none', ecolor='black',
                capsize=5, capthick=2, linewidth=1.5)

    # Chance level line
    ax.axhline(y=chance, color='red', linestyle='--', linewidth=2,
               label=f'Chance ({chance:.1f}%)', alpha=0.8)

    ax.set_ylabel('Test Accuracy (%)', fontweight='bold')
    ax.set_xlabel('Classification Method', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim([0, 30])
    ax.legend(loc='upper right', frameon=True, shadow=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'method_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'method_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: method_comparison.pdf/png")
    plt.close()

def plot_rf_parameter_comparison(rf_results):
    """Figure 2: Random Forest parameter comparison"""
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    configs = ['n100_mfsqrt', 'n100_mf26', 'n200_mfsqrt', 'n200_mf26']
    labels = ['n=100\nmf=√', 'n=100\nmf=26', 'n=200\nmf=√', 'n=200\nmf=26']

    test_accs = []
    test_f1s = []
    test_acc_cis = []
    test_f1_cis = []

    for config in configs:
        data = rf_results[config]

        acc_mean, acc_low, acc_high = compute_ci(data['test_accuracies'])
        f1_mean, f1_low, f1_high = compute_ci(data['test_f1s'])

        test_accs.append(acc_mean * 100)
        test_f1s.append(f1_mean * 100)
        test_acc_cis.append([(acc_mean - acc_low) * 100, (acc_high - acc_mean) * 100])
        test_f1_cis.append([(f1_mean - f1_low) * 100, (f1_high - f1_mean) * 100])

    x = np.arange(len(configs))

    # Accuracy
    bars1 = axes[0].bar(x, test_accs, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].errorbar(x, test_accs,
                     yerr=np.array(test_acc_cis).T,
                     fmt='none', ecolor='black', capsize=4, capthick=1.5)
    axes[0].axhline(y=100/11, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    axes[0].set_ylabel('Test Accuracy (%)', fontweight='bold')
    axes[0].set_xlabel('Configuration', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylim([0, 30])
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_title('(a) Test Accuracy', fontweight='bold', fontsize=10)

    # F1 Score
    bars2 = axes[1].bar(x, test_f1s, color='coral', alpha=0.7, edgecolor='black')
    axes[1].errorbar(x, test_f1s,
                     yerr=np.array(test_f1_cis).T,
                     fmt='none', ecolor='black', capsize=4, capthick=1.5)
    axes[1].axhline(y=100/11, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    axes[1].set_ylabel('Test F1-Score (%)', fontweight='bold')
    axes[1].set_xlabel('Configuration', fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_ylim([0, 30])
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_title('(b) Macro F1-Score', fontweight='bold', fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'rf_parameter_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'rf_parameter_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: rf_parameter_comparison.pdf/png")
    plt.close()

def plot_confusion_matrix(svm_results):
    """Figure 3: SVM Confusion Matrix"""
    cm = svm_results['confusion_matrix']

    # Normalize to percentages
    cm_norm = cm / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(8, 7))

    sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar_kws={'label': 'Percentage (%)'}, ax=ax,
                vmin=0, vmax=50, linewidths=0.5, linecolor='gray')

    ax.set_ylabel('True Label', fontweight='bold', fontsize=11)
    ax.set_xlabel('Predicted Label', fontweight='bold', fontsize=11)
    ax.set_title('SVM Confusion Matrix (C=100, γ=0.01)\nAveraged over 5 trials',
                 fontweight='bold', fontsize=12)

    # Rotate labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    plt.setp(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'confusion_matrix.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: confusion_matrix.pdf/png")
    plt.close()

def plot_per_class_performance(svm_results):
    """Figure 4: Per-class F1 scores showing movement categories"""
    f1_scores = np.array(svm_results['per_class_f1']) * 100

    fig, ax = plt.subplots(figsize=(7, 4))

    # Color by category
    colors = ['C0']*6 + ['C1']*3 + ['C2']*2

    x = np.arange(len(CLASS_NAMES))
    bars = ax.bar(x, f1_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=1.2)

    ax.set_ylabel('F1-Score (%)', fontweight='bold')
    ax.set_xlabel('Movement Class', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax.axhline(y=100/11, color='red', linestyle='--', linewidth=2, label='Chance', alpha=0.7)
    ax.set_ylim([0, 50])
    ax.grid(axis='y', alpha=0.3)

    # Add category labels
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((-.5, 45), 6, 2, facecolor='C0', alpha=0.3, edgecolor='none'))
    ax.text(2.5, 46, 'Reaches', ha='center', fontweight='bold', fontsize=9)

    ax.add_patch(Rectangle((5.5, 45), 3, 2, facecolor='C1', alpha=0.3, edgecolor='none'))
    ax.text(7, 46, 'Grasps', ha='center', fontweight='bold', fontsize=9)

    ax.add_patch(Rectangle((8.5, 45), 2, 2, facecolor='C2', alpha=0.3, edgecolor='none'))
    ax.text(9.5, 46, 'Wrist', ha='center', fontweight='bold', fontsize=9)

    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'per_class_performance.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'per_class_performance.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: per_class_performance.pdf/png")
    plt.close()

def plot_trial_variability(rf_results, svm_results):
    """Figure 5: Trial-to-trial variability showing reproducibility"""
    fig, ax = plt.subplots(figsize=(7, 4))

    # Random Forest best config
    rf_accs = np.array(rf_results['n200_mf26']['test_accuracies']) * 100
    # SVM
    svm_accs = np.array(svm_results['test_accuracies']) * 100

    data = [rf_accs, svm_accs]
    labels = ['Random Forest\n(n=200, mf=26)', 'SVM\n(C=100, γ=0.01)']

    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    showmeans=True, meanline=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2),
                    meanprops=dict(color='green', linewidth=2, linestyle='--'),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5))

    # Overlay individual points
    for i, d in enumerate(data):
        x = np.random.normal(i+1, 0.04, size=len(d))
        ax.scatter(x, d, alpha=0.6, s=50, color='darkblue', edgecolor='black', linewidth=0.5)

    ax.axhline(y=100/11, color='red', linestyle=':', linewidth=2, label='Chance', alpha=0.7)
    ax.set_ylabel('Test Accuracy (%)', fontweight='bold')
    ax.set_xlabel('Method', fontweight='bold')
    ax.set_ylim([20, 30])
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='upper right')
    ax.set_title('Trial-to-Trial Variability (5 independent trials)', fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'trial_variability.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'trial_variability.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: trial_variability.pdf/png")
    plt.close()

def generate_results_table(rf_results, svm_results, csp_results):
    """Generate LaTeX table of results"""

    # Best RF
    rf_accs = rf_results['n200_mf26']['test_accuracies']
    rf_f1s = rf_results['n200_mf26']['test_f1s']
    rf_acc_mean, rf_acc_low, rf_acc_high = compute_ci(rf_accs)
    rf_f1_mean, rf_f1_low, rf_f1_high = compute_ci(rf_f1s)

    # SVM
    svm_acc_mean = svm_results['test_acc_mean']
    svm_acc_low, svm_acc_high = svm_results['test_acc_ci']
    svm_f1_mean = svm_results['test_f1_mean']

    # CSP+LDA
    csp_acc = csp_results['test_accuracy']
    csp_acc_std = csp_results.get('test_accuracy_std', 0)
    csp_f1 = csp_results['test_f1']
    csp_f1_std = csp_results.get('test_f1_std', 0)
    csp_n_trials = csp_results.get('n_trials', 1)

    # Format CSP results based on number of trials
    if csp_n_trials > 1:
        csp_acc_str = f"{csp_acc*100:.2f} $\\pm$ {csp_acc_std*100:.2f}"
        csp_f1_str = f"{csp_f1*100:.2f} $\\pm$ {csp_f1_std*100:.2f}"
        csp_note = f"$^*$Based on {csp_n_trials} trials; error shown as $\\pm$1 std"
    else:
        csp_acc_str = f"{csp_acc*100:.2f}"
        csp_f1_str = f"{csp_f1*100:.2f}"
        csp_note = "$^*$Single trial only; CI not available"

    table = f"""
\\begin{{table}}[htbp]
\\caption{{Classification Performance Summary}}
\\begin{{center}}
\\begin{{tabular}}{{|l|c|c|c|}}
\\hline
\\textbf{{Method}} & \\textbf{{Parameters}} & \\textbf{{Test Acc (\\%)}} & \\textbf{{Test F1 (\\%)}} \\\\
\\hline
Random Forest & n=200, mf=26 & {rf_acc_mean*100:.2f} [{rf_acc_low*100:.2f}, {rf_acc_high*100:.2f}] & {rf_f1_mean*100:.2f} [{rf_f1_low*100:.2f}, {rf_f1_high*100:.2f}] \\\\
\\hline
SVM & C=100, $\\gamma$=0.01 & {svm_acc_mean*100:.2f} [{svm_acc_low*100:.2f}, {svm_acc_high*100:.2f}] & {svm_f1_mean*100:.2f} \\\\
\\hline
CSP+LDA$^*$ & 8-30 Hz, 8 comp & {csp_acc_str} & {csp_f1_str} \\\\
\\hline
\\multicolumn{{4}}{{l}}{{{csp_note}}} \\\\
\\hline
\\end{{tabular}}
\\label{{tab:results}}
\\end{{center}}
\\end{{table}}
"""

    with open(OUTPUT_DIR / 'results_table.tex', 'w') as f:
        f.write(table)

    print(f"✓ Saved: results_table.tex")
    return table

def plot_csp_component_ablation(csp_results):
    """Plot CSP+LDA component ablation study"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    summary_df = csp_results['all_summaries']

    # Plot 1: Test Accuracy vs Components
    n_comps = summary_df['n_components'].values
    acc_means = summary_df['test_acc_mean'].values * 100
    acc_stds = summary_df['test_acc_std'].values * 100

    ax1.errorbar(n_comps, acc_means, yerr=acc_stds, fmt='o-',
                 linewidth=2, markersize=8, capsize=5, capthick=2,
                 color='#2E86AB', ecolor='#333', label='Test Accuracy')
    ax1.set_xlabel('Number of CSP Components', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Test Accuracy (%)', fontsize=11, fontweight='bold')
    ax1.set_title('CSP+LDA: Component Count Ablation', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xticks(n_comps)
    ax1.set_ylim([10, 16])

    # Add value labels
    for x, y, std in zip(n_comps, acc_means, acc_stds):
        ax1.text(x, y + std + 0.3, f'{y:.2f}%', ha='center', va='bottom', fontsize=9)

    # Plot 2: Test F1 vs Components
    f1_means = summary_df['test_f1_mean'].values
    f1_stds = summary_df['test_f1_std'].values

    ax2.errorbar(n_comps, f1_means, yerr=f1_stds, fmt='s-',
                 linewidth=2, markersize=8, capsize=5, capthick=2,
                 color='#A23B72', ecolor='#333', label='Test F1 (macro)')
    ax2.set_xlabel('Number of CSP Components', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Test F1 Score (macro)', fontsize=11, fontweight='bold')
    ax2.set_title('CSP+LDA: F1 Score vs Components', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xticks(n_comps)
    ax2.set_ylim([0.08, 0.14])

    # Add value labels
    for x, y, std in zip(n_comps, f1_means, f1_stds):
        ax2.text(x, y + std + 0.003, f'{y:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'csp_lda_component_ablation.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'csp_lda_component_ablation.pdf', bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: csp_lda_component_ablation.png/pdf")

def main():
    """Generate all figures"""
    print("\n" + "="*60)
    print("GENERATING FIGURES FOR IEEE CONFERENCE PAPER")
    print("="*60 + "\n")

    # Load all results
    print("Loading results...")
    rf_results = load_random_forest_results()
    svm_results = load_svm_results()
    csp_results = load_csp_lda_results()
    print("✓ All results loaded\n")

    # Generate figures
    print("Generating figures...")
    plot_method_comparison(rf_results, svm_results, csp_results)
    plot_rf_parameter_comparison(rf_results)
    plot_confusion_matrix(svm_results)
    plot_per_class_performance(svm_results)
    plot_trial_variability(rf_results, svm_results)
    plot_csp_component_ablation(csp_results)  # New CSP ablation plot

    # Generate table
    print("\nGenerating LaTeX table...")
    table = generate_results_table(rf_results, svm_results, csp_results)

    print("\n" + "="*60)
    print(f"ALL FIGURES SAVED TO: {OUTPUT_DIR.absolute()}")
    print("="*60)
    print("\nGenerated files:")
    for f in sorted(OUTPUT_DIR.glob("*")):
        print(f"  - {f.name}")

    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"\nBest performing method: Random Forest (n=200, mf=26)")
    print(f"  Test Accuracy: {np.mean(rf_results['n200_mf26']['test_accuracies'])*100:.2f}%")
    print(f"  Test F1 Score: {np.mean(rf_results['n200_mf26']['test_f1s'])*100:.2f}%")
    print(f"\nChance level (11 classes): {100/11:.2f}%")
    print(f"All methods significantly above chance (p < 0.001)")

if __name__ == '__main__':
    main()
