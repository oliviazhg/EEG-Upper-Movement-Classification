"""
CNN Results Analysis Script
Generates comprehensive analysis and visualizations from completed experiments

Features:
- Statistical significance testing (paired t-tests with Bonferroni correction)
- Cohen's d effect sizes
- Confusion matrix visualization
- Per-class performance analysis
- Training dynamics analysis
- Model comparison tables

Usage:
    python analyze_cnn_results.py --results-dir results_cnn/
"""

import numpy as np
import json
import argparse
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10


class CNNResultsAnalyzer:
    """Comprehensive analysis of CNN experiment results."""
    
    CLASS_NAMES = {
        0: 'Rest', 1: 'Reach Fwd', 2: 'Reach Back', 
        3: 'Reach Left', 4: 'Reach Right', 5: 'Reach Up', 6: 'Reach Down',
        7: 'Grasp Cup', 8: 'Grasp Ball', 9: 'Grasp Card',
        10: 'Twist Pron', 11: 'Twist Sup'
    }
    
    def __init__(self, results_dir: str):
        """
        Initialize analyzer.
        
        Parameters:
        -----------
        results_dir : str
            Directory containing experiment results
        """
        self.results_dir = Path(results_dir)
        self.logs_dir = self.results_dir / 'logs'
        self.plots_dir = self.results_dir / 'analysis_plots'
        self.plots_dir.mkdir(exist_ok=True)
        
        # Load results
        self.aggregate_results = self._load_aggregate_results()
        self.trial_results = self._load_all_trial_results()
        
        print(f"Loaded results from: {results_dir}")
        print(f"Architectures: {list(self.aggregate_results.keys())}")
    
    def _load_aggregate_results(self) -> Dict:
        """Load aggregate results JSON."""
        path = self.results_dir / 'aggregate_results.json'
        with open(path, 'r') as f:
            return json.load(f)
    
    def _load_all_trial_results(self) -> Dict:
        """Load all individual trial results."""
        results = {
            'CNN-2L': [],
            'CNN-3L': [],
            'CNN-4L': []
        }
        
        for arch in results.keys():
            trial_files = sorted(self.logs_dir.glob(f"{arch}_trial*_results.json"))
            for trial_file in trial_files:
                with open(trial_file, 'r') as f:
                    results[arch].append(json.load(f))
        
        return results
    
    def print_summary_table(self):
        """Print formatted summary table."""
        print(f"\n{'='*80}")
        print(f"CNN EXPERIMENT RESULTS SUMMARY")
        print(f"{'='*80}\n")
        
        # Create DataFrame
        data = []
        for arch, results in self.aggregate_results.items():
            data.append({
                'Architecture': arch,
                'Accuracy (%)': f"{results['accuracy']['mean']*100:.2f} ± {results['accuracy']['std']*100:.2f}",
                'F1-Macro (%)': f"{results['f1_macro']['mean']*100:.2f} ± {results['f1_macro']['std']*100:.2f}",
                'Min Acc': f"{results['accuracy']['min']*100:.2f}%",
                'Max Acc': f"{results['accuracy']['max']*100:.2f}%",
                'Trials': results['n_trials']
            })
        
        df = pd.DataFrame(data)
        print(df.to_string(index=False))
        print()
        
        # Save to CSV
        csv_path = self.results_dir / 'summary_table.csv'
        df.to_csv(csv_path, index=False)
        print(f"Summary table saved to: {csv_path}\n")
    
    def statistical_comparison(self):
        """
        Perform statistical significance testing between architectures.
        Uses paired t-tests with Bonferroni correction.
        """
        print(f"\n{'='*80}")
        print(f"STATISTICAL SIGNIFICANCE TESTING")
        print(f"{'='*80}\n")
        
        architectures = ['CNN-2L', 'CNN-3L', 'CNN-4L']
        n_comparisons = 3  # 2L vs 3L, 2L vs 4L, 3L vs 4L
        alpha = 0.05
        bonferroni_alpha = alpha / n_comparisons
        
        print(f"Method: Paired t-tests with Bonferroni correction")
        print(f"Significance level: α = {alpha}")
        print(f"Bonferroni-corrected α = {bonferroni_alpha:.4f}\n")
        
        # Collect accuracies
        accuracies = {}
        for arch in architectures:
            if arch in self.aggregate_results:
                accuracies[arch] = self.aggregate_results[arch]['accuracy']['trials']
        
        # Pairwise comparisons
        comparisons = [
            ('CNN-2L', 'CNN-3L'),
            ('CNN-2L', 'CNN-4L'),
            ('CNN-3L', 'CNN-4L')
        ]
        
        results = []
        for arch1, arch2 in comparisons:
            if arch1 not in accuracies or arch2 not in accuracies:
                continue
            
            acc1 = np.array(accuracies[arch1])
            acc2 = np.array(accuracies[arch2])
            
            # Paired t-test
            t_stat, p_value = stats.ttest_rel(acc1, acc2)
            
            # Cohen's d for paired samples
            diff = acc1 - acc2
            cohens_d = np.mean(diff) / np.std(diff, ddof=1)
            
            # Interpret effect size
            if abs(cohens_d) < 0.2:
                effect_size = "negligible"
            elif abs(cohens_d) < 0.5:
                effect_size = "small"
            elif abs(cohens_d) < 0.8:
                effect_size = "medium"
            else:
                effect_size = "large"
            
            # Determine significance
            is_significant = p_value < bonferroni_alpha
            
            results.append({
                'Comparison': f"{arch1} vs {arch2}",
                't-statistic': t_stat,
                'p-value': p_value,
                'Significant': 'Yes' if is_significant else 'No',
                "Cohen's d": cohens_d,
                'Effect size': effect_size
            })
            
            print(f"{arch1} vs {arch2}:")
            print(f"  t-statistic: {t_stat:.4f}")
            print(f"  p-value: {p_value:.4f} {'***' if is_significant else ''}")
            print(f"  Cohen's d: {cohens_d:.4f} ({effect_size})")
            print(f"  Result: {'Significant difference' if is_significant else 'No significant difference'}")
            print()
        
        # Save results
        df = pd.DataFrame(results)
        csv_path = self.results_dir / 'statistical_tests.csv'
        df.to_csv(csv_path, index=False)
        print(f"Statistical test results saved to: {csv_path}\n")
    
    def plot_detailed_comparison(self):
        """Create comprehensive comparison plots."""
        print(f"\nGenerating detailed comparison plots...")
        
        # 1. Box plot of accuracies
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy box plot
        arch_names = []
        acc_data = []
        for arch, results in self.aggregate_results.items():
            arch_names.append(arch)
            acc_data.append(np.array(results['accuracy']['trials']) * 100)
        
        bp1 = ax1.boxplot(acc_data, labels=arch_names, patch_artist=True)
        for patch in bp1['boxes']:
            patch.set_facecolor('lightblue')
        ax1.set_ylabel('Test Accuracy (%)')
        ax1.set_title('Test Accuracy Distribution Across 5 Trials')
        ax1.grid(axis='y', alpha=0.3)
        
        # F1 box plot
        f1_data = []
        for arch in arch_names:
            f1_data.append(np.array(self.aggregate_results[arch]['f1_macro']['trials']) * 100)
        
        bp2 = ax2.boxplot(f1_data, labels=arch_names, patch_artist=True)
        for patch in bp2['boxes']:
            patch.set_facecolor('lightcoral')
        ax2.set_ylabel('F1-Score (macro, %)')
        ax2.set_title('F1-Score Distribution Across 5 Trials')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'detailed_comparison_boxplots.png', dpi=300)
        plt.close()
        
        # 2. Bar plot with error bars and individual points
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x_pos = np.arange(len(arch_names))
        means = [np.mean(data) for data in acc_data]
        stds = [np.std(data) for data in acc_data]
        
        # Bar plot
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7,
                     color='steelblue', edgecolor='black', linewidth=1.5)
        
        # Overlay individual trial points
        for i, data in enumerate(acc_data):
            # Add jitter for visibility
            jitter = np.random.normal(0, 0.05, len(data))
            ax.scatter(x_pos[i] + jitter, data, alpha=0.6, 
                      color='darkred', s=50, zorder=3)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(arch_names)
        ax.set_ylabel('Test Accuracy (%)')
        ax.set_title('CNN Architecture Comparison\n(Mean ± Std, Individual Trials Shown)')
        ax.grid(axis='y', alpha=0.3)
        
        # Add sample size annotation
        for i, n in enumerate([self.aggregate_results[a]['n_trials'] for a in arch_names]):
            ax.text(i, means[i] + stds[i] + 1, f'n={n}', 
                   ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'comparison_with_trials.png', dpi=300)
        plt.close()
        
        print(f"  ✓ Comparison plots saved")
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for best trial of each architecture."""
        print(f"\nGenerating confusion matrices...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, arch in enumerate(['CNN-2L', 'CNN-3L', 'CNN-4L']):
            if arch not in self.trial_results or not self.trial_results[arch]:
                continue
            
            # Get best trial
            trials = self.trial_results[arch]
            best_trial = max(trials, key=lambda x: x['metrics']['accuracy'])
            
            # Get confusion matrix
            cm = np.array(best_trial['metrics']['confusion_matrix'])
            
            # Normalize by row (true labels)
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Plot
            ax = axes[idx]
            im = ax.imshow(cm_norm, cmap='Blues', aspect='auto')
            
            # Labels
            class_labels = [self.CLASS_NAMES.get(i, f'C{i}') for i in range(len(cm))]
            ax.set_xticks(np.arange(len(class_labels)))
            ax.set_yticks(np.arange(len(class_labels)))
            ax.set_xticklabels(class_labels, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(class_labels, fontsize=8)
            
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title(f'{arch} (Best Trial)\nAcc: {best_trial["metrics"]["accuracy"]:.3f}')
            
            # Colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'confusion_matrices.png', dpi=300)
        plt.close()
        
        print(f"  ✓ Confusion matrices saved")
    
    def plot_per_class_performance(self):
        """Analyze and plot per-class performance."""
        print(f"\nAnalyzing per-class performance...")
        
        # Collect per-class F1 scores for best trial of each architecture
        per_class_data = {}
        
        for arch in ['CNN-2L', 'CNN-3L', 'CNN-4L']:
            if arch not in self.trial_results or not self.trial_results[arch]:
                continue
            
            trials = self.trial_results[arch]
            best_trial = max(trials, key=lambda x: x['metrics']['accuracy'])
            
            # Extract per-class F1 scores
            class_report = best_trial['metrics']['classification_report']
            
            f1_scores = []
            for class_id in range(len(self.CLASS_NAMES)):
                class_key = str(class_id)
                if class_key in class_report:
                    f1_scores.append(class_report[class_key]['f1-score'])
                else:
                    f1_scores.append(0.0)
            
            per_class_data[arch] = f1_scores
        
        # Create plot
        fig, ax = plt.subplots(figsize=(14, 6))
        
        class_labels = [self.CLASS_NAMES[i] for i in range(len(self.CLASS_NAMES))]
        x = np.arange(len(class_labels))
        width = 0.25
        
        for idx, (arch, f1_scores) in enumerate(per_class_data.items()):
            offset = (idx - 1) * width
            ax.bar(x + offset, np.array(f1_scores) * 100, width, 
                  label=arch, alpha=0.8)
        
        ax.set_xlabel('Movement Class')
        ax.set_ylabel('F1-Score (%)')
        ax.set_title('Per-Class F1-Scores (Best Trial of Each Architecture)')
        ax.set_xticks(x)
        ax.set_xticklabels(class_labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'per_class_performance.png', dpi=300)
        plt.close()
        
        # Save numerical data
        df = pd.DataFrame(per_class_data, index=class_labels)
        csv_path = self.results_dir / 'per_class_f1_scores.csv'
        df.to_csv(csv_path)
        
        print(f"  ✓ Per-class performance analysis saved")
    
    def plot_training_dynamics(self):
        """Plot training curves for all trials."""
        print(f"\nGenerating training dynamics plots...")
        
        for arch in ['CNN-2L', 'CNN-3L', 'CNN-4L']:
            if arch not in self.trial_results or not self.trial_results[arch]:
                continue
            
            trials = self.trial_results[arch]
            n_trials = len(trials)
            
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            
            # Plot all trials
            for trial in trials:
                history = trial['history']
                epochs = range(1, len(history['loss']) + 1)
                
                # Loss
                axes[0].plot(epochs, history['loss'], alpha=0.3, color='blue')
                axes[0].plot(epochs, history['val_loss'], alpha=0.3, color='red')
                
                # Accuracy
                axes[1].plot(epochs, np.array(history['accuracy']) * 100, 
                           alpha=0.3, color='blue')
                axes[1].plot(epochs, np.array(history['val_accuracy']) * 100, 
                           alpha=0.3, color='red')
            
            # Labels and styling
            axes[0].set_ylabel('Loss')
            axes[0].set_title(f'{arch} - Training Dynamics (All {n_trials} Trials)')
            axes[0].grid(alpha=0.3)
            axes[0].legend(['Train Loss', 'Val Loss'], loc='upper right')
            
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy (%)')
            axes[1].grid(alpha=0.3)
            axes[1].legend(['Train Acc', 'Val Acc'], loc='lower right')
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / f'{arch}_training_dynamics.png', dpi=300)
            plt.close()
        
        print(f"  ✓ Training dynamics plots saved")
    
    def generate_latex_table(self):
        """Generate LaTeX table for paper."""
        print(f"\nGenerating LaTeX table...")
        
        latex = "\\begin{table}[h]\n"
        latex += "\\centering\n"
        latex += "\\caption{CNN Architecture Comparison Results}\n"
        latex += "\\label{tab:cnn_results}\n"
        latex += "\\begin{tabular}{lcccc}\n"
        latex += "\\hline\n"
        latex += "Architecture & Accuracy (\\%) & F1-Macro (\\%) & Min/Max Acc (\\%) & Trials \\\\\n"
        latex += "\\hline\n"
        
        for arch in ['CNN-2L', 'CNN-3L', 'CNN-4L']:
            if arch not in self.aggregate_results:
                continue
            
            results = self.aggregate_results[arch]
            
            acc_mean = results['accuracy']['mean'] * 100
            acc_std = results['accuracy']['std'] * 100
            f1_mean = results['f1_macro']['mean'] * 100
            f1_std = results['f1_macro']['std'] * 100
            acc_min = results['accuracy']['min'] * 100
            acc_max = results['accuracy']['max'] * 100
            n_trials = results['n_trials']
            
            latex += f"{arch} & "
            latex += f"${acc_mean:.2f} \\pm {acc_std:.2f}$ & "
            latex += f"${f1_mean:.2f} \\pm {f1_std:.2f}$ & "
            latex += f"{acc_min:.1f}/{acc_max:.1f} & "
            latex += f"{n_trials} \\\\\n"
        
        latex += "\\hline\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"
        
        # Save
        tex_path = self.results_dir / 'results_table.tex'
        with open(tex_path, 'w') as f:
            f.write(latex)
        
        print(f"  ✓ LaTeX table saved to: {tex_path}")
        print(f"\n{latex}")
    
    def run_full_analysis(self):
        """Run complete analysis pipeline."""
        print(f"\n{'='*80}")
        print(f"CNN RESULTS COMPREHENSIVE ANALYSIS")
        print(f"{'='*80}")
        
        self.print_summary_table()
        self.statistical_comparison()
        self.plot_detailed_comparison()
        self.plot_confusion_matrices()
        self.plot_per_class_performance()
        self.plot_training_dynamics()
        self.generate_latex_table()
        
        print(f"\n{'='*80}")
        print(f"ANALYSIS COMPLETE")
        print(f"{'='*80}")
        print(f"\nAll outputs saved to:")
        print(f"  Results: {self.results_dir}")
        print(f"  Plots: {self.plots_dir}")
        print(f"\nKey files:")
        print(f"  - summary_table.csv: Overall performance summary")
        print(f"  - statistical_tests.csv: Significance test results")
        print(f"  - results_table.tex: LaTeX table for paper")
        print(f"  - analysis_plots/: All visualization plots")
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive CNN Results Analysis'
    )
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Directory containing experiment results')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = CNNResultsAnalyzer(args.results_dir)
    
    # Run full analysis
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()