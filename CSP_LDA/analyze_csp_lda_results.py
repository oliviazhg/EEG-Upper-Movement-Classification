"""
Utility script for analyzing and visualizing saved CSP+LDA results.
Use this to generate additional plots or perform custom analysis on saved results.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


class CSPLDAAnalyzer:
    """Analyze saved CSP+LDA experimental results."""
    
    def __init__(self, results_dir='./results_csp_lda'):
        """
        Initialize analyzer.
        
        Parameters:
        -----------
        results_dir : str
            Directory containing saved results
        """
        self.results_dir = Path(results_dir)
        
        if not self.results_dir.exists():
            raise ValueError(f"Results directory not found: {results_dir}")
        
        # Load results
        self.load_results()
        
        # Movement class names
        self.class_names = [
            'Forward', 'Backward', 'Left', 'Right', 'Up', 'Down',
            'Power Grasp', 'Precision Grasp', 'Lateral Grasp',
            'Pronation', 'Supination'
        ]
    
    def load_results(self):
        """Load saved experimental results."""
        print("Loading results...")
        
        # Load trial results
        with open(self.results_dir / 'trial_results.pkl', 'rb') as f:
            self.trial_results = pickle.load(f)
        
        # Load config summaries
        with open(self.results_dir / 'config_summaries.pkl', 'rb') as f:
            self.config_summaries = pickle.load(f)
        
        # Load CSV versions for easier manipulation
        self.trial_df = pd.read_csv(self.results_dir / 'trial_results.csv')
        self.summary_df = pd.read_csv(self.results_dir / 'config_summaries.csv')
        
        if (self.results_dir / 'statistical_comparisons.csv').exists():
            self.comparisons_df = pd.read_csv(
                self.results_dir / 'statistical_comparisons.csv'
            )
        
        print(f"Loaded {len(self.trial_results)} trials across "
              f"{len(self.config_summaries)} configurations")
    
    def plot_learning_curves(self, save=True):
        """Plot train vs validation accuracy for each configuration."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        configs = [(s['band'], s['n_components']) 
                  for s in self.config_summaries]
        
        for idx, (band, n_comp) in enumerate(configs):
            ax = axes[idx]
            
            # Get trials for this config
            config_trials = self.trial_df[
                (self.trial_df['band'] == band) & 
                (self.trial_df['n_components'] == n_comp)
            ]
            
            # Plot train and val accuracies
            seeds = config_trials['seed'].values
            train_accs = config_trials['train_accuracy'].values * 100
            val_accs = config_trials['val_accuracy'].values * 100
            test_accs = config_trials['test_accuracy'].values * 100
            
            ax.plot(seeds, train_accs, 'o-', label='Train', linewidth=2, 
                   markersize=8, alpha=0.7)
            ax.plot(seeds, val_accs, 's-', label='Validation', linewidth=2,
                   markersize=8, alpha=0.7)
            ax.plot(seeds, test_accs, '^-', label='Test', linewidth=2,
                   markersize=8, alpha=0.7)
            
            ax.set_xlabel('Trial (Seed)', fontsize=10, fontweight='bold')
            ax.set_ylabel('Accuracy (%)', fontsize=10, fontweight='bold')
            ax.set_title(f'{band.capitalize()} Band, {n_comp} Components',
                        fontsize=11, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 100])
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.results_dir / 'analysis_learning_curves.png',
                       dpi=300, bbox_inches='tight')
            print("✓ Saved learning curves plot")
        
        plt.show()
    
    def plot_accuracy_distribution(self, save=True):
        """Plot distribution of test accuracies across configurations."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Prepare data
        data_to_plot = []
        labels = []
        
        for summary in self.config_summaries:
            band = summary['band']
            n_comp = summary['n_components']
            
            # Get test accuracies for this config
            config_trials = self.trial_df[
                (self.trial_df['band'] == band) & 
                (self.trial_df['n_components'] == n_comp)
            ]
            
            accs = config_trials['test_accuracy'].values * 100
            data_to_plot.append(accs)
            labels.append(f"{band}\n{n_comp}c")
        
        # Create violin plot
        parts = ax.violinplot(data_to_plot, positions=range(len(labels)),
                             showmeans=True, showmedians=True)
        
        # Customize colors
        for pc in parts['bodies']:
            pc.set_facecolor('steelblue')
            pc.set_alpha(0.7)
        
        # Overlay individual points
        for i, data in enumerate(data_to_plot):
            x = np.random.normal(i, 0.04, size=len(data))
            ax.plot(x, data, 'ro', alpha=0.6, markersize=8)
        
        ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
        ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Distribution of Test Accuracies Across Configurations',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim([0, 100])
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.results_dir / 'analysis_accuracy_distribution.png',
                       dpi=300, bbox_inches='tight')
            print("✓ Saved accuracy distribution plot")
        
        plt.show()
    
    def plot_confusion_matrices_grid(self, save=True):
        """Plot confusion matrices for all configurations."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        configs = [(s['band'], s['n_components']) 
                  for s in self.config_summaries]
        
        for idx, (band, n_comp) in enumerate(configs):
            ax = axes[idx]
            
            # Get confusion matrix for this config
            summary = next(s for s in self.config_summaries
                         if s['band'] == band and s['n_components'] == n_comp)
            
            cm = summary['confusion_matrix_mean']
            
            # Normalize by row (true labels)
            cm_norm = cm / cm.sum(axis=1, keepdims=True)
            
            # Plot heatmap
            sns.heatmap(cm_norm, annot=False, cmap='Blues', vmin=0, vmax=1,
                       cbar_kws={'label': 'Proportion'}, ax=ax,
                       xticklabels=False, yticklabels=False)
            
            acc = summary['test_acc_mean'] * 100
            ax.set_title(f"{band.capitalize()}, {n_comp}c (Acc={acc:.1f}%)",
                        fontsize=11, fontweight='bold')
        
        # Add shared labels
        fig.text(0.5, 0.02, 'Predicted Label', ha='center', fontsize=12,
                fontweight='bold')
        fig.text(0.02, 0.5, 'True Label', va='center', rotation='vertical',
                fontsize=12, fontweight='bold')
        
        plt.suptitle('Normalized Confusion Matrices Across Configurations',
                    fontsize=14, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0.03, 0.03, 1, 0.96])
        
        if save:
            plt.savefig(self.results_dir / 'analysis_confusion_matrices_grid.png',
                       dpi=300, bbox_inches='tight')
            print("✓ Saved confusion matrices grid")
        
        plt.show()
    
    def plot_timing_comparison(self, save=True):
        """Compare training and inference times."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Prepare data
        configs = []
        train_times = []
        inference_times = []
        
        for summary in self.config_summaries:
            band = summary['band']
            n_comp = summary['n_components']
            
            # Get times for this config
            config_trials = [t for t in self.trial_results
                           if t['band'] == band and t['n_components'] == n_comp]
            
            train_t = [t['training_time_sec'] for t in config_trials]
            inf_t = [t['inference_time_per_sample'] * 1000 for t in config_trials]  # Convert to ms
            
            configs.append(f"{band}\n{n_comp}c")
            train_times.append(train_t)
            inference_times.append(inf_t)
        
        # Plot 1: Training times
        bp1 = ax1.boxplot(train_times, labels=configs, patch_artist=True)
        for patch in bp1['boxes']:
            patch.set_facecolor('lightblue')
        ax1.set_ylabel('Training Time (seconds)', fontsize=11, fontweight='bold')
        ax1.set_xlabel('Configuration', fontsize=11, fontweight='bold')
        ax1.set_title('Training Time Comparison', fontsize=12, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Inference times
        bp2 = ax2.boxplot(inference_times, labels=configs, patch_artist=True)
        for patch in bp2['boxes']:
            patch.set_facecolor('lightcoral')
        ax2.set_ylabel('Inference Time (ms)', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Configuration', fontsize=11, fontweight='bold')
        ax2.set_title('Inference Time Comparison', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.results_dir / 'analysis_timing_comparison.png',
                       dpi=300, bbox_inches='tight')
            print("✓ Saved timing comparison plot")
        
        plt.show()
    
    def plot_f1_score_heatmap(self, save=True):
        """Create heatmap of F1 scores across classes and configurations."""
        # Prepare data matrix
        configs = [(s['band'], s['n_components']) 
                  for s in self.config_summaries]
        
        f1_matrix = np.zeros((len(configs), 11))
        config_labels = []
        
        for i, (band, n_comp) in enumerate(configs):
            summary = next(s for s in self.config_summaries
                         if s['band'] == band and s['n_components'] == n_comp)
            
            f1_matrix[i, :] = summary['test_f1_per_class_mean']
            config_labels.append(f"{band}_{n_comp}c")
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 6))
        
        sns.heatmap(f1_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                   xticklabels=self.class_names,
                   yticklabels=config_labels,
                   cbar_kws={'label': 'F1 Score'},
                   linewidths=0.5, linecolor='gray', ax=ax)
        
        ax.set_xlabel('Movement Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('Configuration', fontsize=12, fontweight='bold')
        ax.set_title('F1 Scores: Configurations vs Movement Classes',
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=10)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.results_dir / 'analysis_f1_heatmap.png',
                       dpi=300, bbox_inches='tight')
            print("✓ Saved F1 score heatmap")
        
        plt.show()
    
    def identify_difficult_class_pairs(self, top_n=10):
        """Identify most commonly confused class pairs."""
        print("\n" + "="*60)
        print("Most Commonly Confused Class Pairs")
        print("="*60)
        
        # Use best configuration
        best_config = max(self.config_summaries, 
                         key=lambda x: x['test_acc_mean'])
        
        cm = best_config['confusion_matrix_mean']
        
        # Find off-diagonal elements (confusion)
        confusion_pairs = []
        for i in range(11):
            for j in range(11):
                if i != j:
                    confusion_pairs.append({
                        'true_class': self.class_names[i],
                        'pred_class': self.class_names[j],
                        'confusion_count': cm[i, j]
                    })
        
        # Sort by confusion count
        confusion_pairs.sort(key=lambda x: x['confusion_count'], reverse=True)
        
        print(f"\nTop {top_n} most confused pairs "
              f"(from best config: {best_config['band']}, "
              f"{best_config['n_components']} components):\n")
        
        for i, pair in enumerate(confusion_pairs[:top_n], 1):
            print(f"{i:2d}. {pair['true_class']:20s} → {pair['pred_class']:20s} "
                  f"({pair['confusion_count']:.1f} times)")
    
    def generate_summary_report(self):
        """Generate a text summary report."""
        report_path = self.results_dir / 'analysis_summary_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("CSP+LDA EXPERIMENT SUMMARY REPORT\n")
            f.write("="*60 + "\n\n")
            
            # Best configuration
            best_config = max(self.config_summaries, 
                            key=lambda x: x['test_acc_mean'])
            
            f.write("BEST CONFIGURATION:\n")
            f.write(f"  Frequency band: {best_config['band'].capitalize()}\n")
            f.write(f"  CSP components: {best_config['n_components']}\n")
            f.write(f"  Test accuracy:  {best_config['test_acc_mean']*100:.2f}% "
                   f"± {best_config['test_acc_std']*100:.2f}%\n")
            f.write(f"  Test F1 score:  {best_config['test_f1_mean']:.4f} "
                   f"± {best_config['test_f1_std']:.4f}\n")
            f.write(f"  95% CI:         [{best_config['test_acc_ci_low']*100:.2f}%, "
                   f"{best_config['test_acc_ci_high']*100:.2f}%]\n\n")
            
            # All configurations ranked
            f.write("\nALL CONFIGURATIONS (RANKED BY TEST ACCURACY):\n")
            f.write("-"*60 + "\n")
            
            sorted_configs = sorted(self.config_summaries,
                                  key=lambda x: x['test_acc_mean'],
                                  reverse=True)
            
            for i, config in enumerate(sorted_configs, 1):
                f.write(f"{i}. {config['band']:8s} {config['n_components']}comp: "
                       f"{config['test_acc_mean']*100:6.2f}% "
                       f"(± {config['test_acc_std']*100:4.2f}%) "
                       f"F1={config['test_f1_mean']:.4f}\n")
            
            # Statistical significance
            if hasattr(self, 'comparisons_df'):
                f.write("\n\nSTATISTICALLY SIGNIFICANT DIFFERENCES:\n")
                f.write("-"*60 + "\n")
                
                sig_comparisons = self.comparisons_df[
                    self.comparisons_df['significant'] == True
                ]
                
                if len(sig_comparisons) > 0:
                    for _, comp in sig_comparisons.iterrows():
                        f.write(f"{comp['config1']:15s} vs {comp['config2']:15s}: "
                               f"p={comp['p_value']:.6f}, d={comp['cohens_d']:.3f}\n")
                else:
                    f.write("No statistically significant differences found.\n")
        
        print(f"\n✓ Summary report saved to: {report_path}")
    
    def run_all_analyses(self):
        """Run all analysis and generate all plots."""
        print("\n" + "="*60)
        print("Running Complete Analysis")
        print("="*60)
        
        self.plot_learning_curves()
        self.plot_accuracy_distribution()
        self.plot_confusion_matrices_grid()
        self.plot_timing_comparison()
        self.plot_f1_score_heatmap()
        self.identify_difficult_class_pairs()
        self.generate_summary_report()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        print(f"All outputs saved to: {self.results_dir}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze saved CSP+LDA experimental results'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='./results_csp_lda',
        help='Directory containing saved results'
    )
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = CSPLDAAnalyzer(results_dir=args.results_dir)
    
    # Run all analyses
    analyzer.run_all_analyses()


if __name__ == '__main__':
    main()