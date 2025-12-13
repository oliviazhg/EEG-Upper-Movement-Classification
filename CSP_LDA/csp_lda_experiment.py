"""
CSP+LDA Experiment for EEG-Based Upper-Limb Movement Classification
SYDE 522 - Foundations of Artificial Intelligence
Author: Olivia Zheng

This script implements the Common Spatial Patterns + Linear Discriminant Analysis
pipeline for classifying 11 upper-limb movements from EEG signals.
"""

import numpy as np
import pandas as pd
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Scientific computing
from scipy.stats import bootstrap, ttest_rel
from scipy.signal import butter, filtfilt

# Machine learning
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, 
    classification_report
)
from sklearn.preprocessing import LabelEncoder

# MNE for CSP
from mne.decoding import CSP

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Configure matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class CSPLDAExperiment:
    """Manages CSP+LDA experiments with multiple configurations."""
    
    def __init__(self, data_path: str, output_dir: str = './results_csp_lda'):
        """
        Initialize experiment.
        
        Parameters:
        -----------
        data_path : str
            Path to preprocessed data file (.npz or .pkl)
        output_dir : str
            Directory to save results
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Experimental parameters
        self.frequency_bands = {
            'mu': (8, 12),
            'beta': (13, 30),
            'combined': (8, 30)
        }
        self.n_components_list = [4, 6]
        self.random_seeds = [0, 1, 2, 3, 4]
        self.n_trials = len(self.random_seeds)
        
        # Data split ratios
        self.train_ratio = 0.70
        self.val_ratio = 0.15
        self.test_ratio = 0.15
        
        # Results storage
        self.trial_results = []
        self.config_summaries = []
        
        # Movement class names
        self.class_names = [
            'Forward', 'Backward', 'Left', 'Right', 'Up', 'Down',
            'Power Grasp', 'Precision Grasp', 'Lateral Grasp',
            'Pronation', 'Supination'
        ]
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load preprocessed EEG data.
        
        Returns:
        --------
        X : np.ndarray
            EEG data (n_trials, n_channels, n_timepoints)
        y : np.ndarray
            Labels (n_trials,)
        """
        print("Loading data...")
        
        if self.data_path.suffix == '.npz':
            data = np.load(self.data_path)
            X = data['X']  # Shape: (n_trials, n_channels, n_timepoints)
            y = data['y']  # Shape: (n_trials,)
        elif self.data_path.suffix == '.pkl':
            with open(self.data_path, 'rb') as f:
                data = pickle.load(f)
                X = data['X']
                y = data['y']
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
        
        # Remove rest class
        mask = y != 0 # Assuming '0' is rest class
        X = X[mask]
        y = y[mask]
        y = y - 1  # Renumber classes 1-11 → 0-10

        print(f"Data loaded: X shape = {X.shape}, y shape = {y.shape}")
        print(f"Number of classes: {len(np.unique(y))}")
        print(f"Class distribution: {np.bincount(y)}")
        print(f"Rest class removed. Classes now: 0-10")        
        
        return X, y
    
    def bandpass_filter(self, X: np.ndarray, low_freq: float, 
                       high_freq: float, fs: float = 2500) -> np.ndarray:
        """
        Apply bandpass filter to EEG data.
        
        Parameters:
        -----------
        X : np.ndarray
            EEG data (n_trials, n_channels, n_timepoints)
        low_freq : float
            Low cutoff frequency (Hz)
        high_freq : float
            High cutoff frequency (Hz)
        fs : float
            Sampling frequency (Hz)
            
        Returns:
        --------
        X_filtered : np.ndarray
            Filtered EEG data
        """
        nyq = fs / 2
        low = low_freq / nyq
        high = high_freq / nyq
        
        b, a = butter(4, [low, high], btype='band')
        
        X_filtered = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                X_filtered[i, j, :] = filtfilt(b, a, X[i, j, :])
        
        return X_filtered
    
    def stratified_split(self, X: np.ndarray, y: np.ndarray, 
                        seed: int) -> Tuple:
        """
        Split data into train/val/test sets with stratification.
        
        Parameters:
        -----------
        X : np.ndarray
            Features
        y : np.ndarray
            Labels
        seed : int
            Random seed
            
        Returns:
        --------
        X_train, X_val, X_test, y_train, y_val, y_test
        """
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=self.test_ratio,
            stratify=y,
            random_state=seed
        )
        
        # Second split: separate train and validation
        val_size_adjusted = self.val_ratio / (1 - self.test_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            stratify=y_temp,
            random_state=seed
        )
        
        # Verify stratification
        self._verify_stratification(y_train, y_val, y_test)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _verify_stratification(self, y_train: np.ndarray, 
                               y_val: np.ndarray, 
                               y_test: np.ndarray):
        """Verify that class distributions are similar across splits."""
        n_classes = len(np.unique(y_train))
        
        train_dist = np.bincount(y_train, minlength=n_classes) / len(y_train)
        val_dist = np.bincount(y_val, minlength=n_classes) / len(y_val)
        test_dist = np.bincount(y_test, minlength=n_classes) / len(y_test)
        
        # Check train vs test
        assert np.allclose(train_dist, test_dist, atol=0.02), \
            "Train/test stratification failed"
        
        # Check val vs test
        assert np.allclose(val_dist, test_dist, atol=0.02), \
            "Val/test stratification failed"
        
        print("✓ Stratification verified (within 2% tolerance)")
    
    def train_csp_lda(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray,
                     X_test: np.ndarray, y_test: np.ndarray,
                     n_components: int, seed: int) -> Dict:
        """
        Train CSP+LDA pipeline.
        
        Parameters:
        -----------
        X_train, X_val, X_test : np.ndarray
            EEG data (n_trials, n_channels, n_timepoints)
        y_train, y_val, y_test : np.ndarray
            Labels (n_trials,)
        n_components : int
            Number of CSP components per binary classifier
        seed : int
            Random seed
            
        Returns:
        --------
        results : dict
            Training results and metrics
        """
        # Initialize CSP
        # For multi-class, CSP will handle one-vs-rest internally
        csp = CSP(
            n_components=n_components,
            reg=None,
            log=True,  # Apply log transform to variance
            norm_trace=False,
            random_state=seed
        )
        
        # Measure training time
        start_time = time.time()
        
        # Fit CSP on training data
        X_train_csp = csp.fit_transform(X_train, y_train)
        
        # Transform validation and test data
        X_val_csp = csp.transform(X_val)
        X_test_csp = csp.transform(X_test)
        
        # Initialize and train LDA
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train_csp, y_train)
        
        training_time = time.time() - start_time
        
        # Make predictions
        y_train_pred = lda.predict(X_train_csp)
        y_val_pred = lda.predict(X_val_csp)
        y_test_pred = lda.predict(X_test_csp)
        
        # Measure inference time
        start_time = time.time()
        _ = lda.predict(X_test_csp[:1])  # Single sample
        inference_time = (time.time() - start_time)
        
        # Calculate metrics
        results = {
            # Performance metrics
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'val_accuracy': accuracy_score(y_val, y_val_pred),
            'test_accuracy': accuracy_score(y_test, y_test_pred),
            
            'train_f1_macro': f1_score(y_train, y_train_pred, average='macro'),
            'val_f1_macro': f1_score(y_val, y_val_pred, average='macro'),
            'test_f1_macro': f1_score(y_test, y_test_pred, average='macro'),
            
            'test_f1_per_class': f1_score(y_test, y_test_pred, average=None),
            
            # Confusion matrix
            'confusion_matrix': confusion_matrix(y_test, y_test_pred),
            
            # Timing
            'training_time_sec': training_time,
            'inference_time_per_sample': inference_time,
            
            # Model artifacts (save for analysis)
            'csp_patterns': csp.patterns_,
            'csp_filters': csp.filters_,
            'lda_coef': lda.coef_,
            'lda_intercept': lda.intercept_,
        }
        
        return results
    
    def run_single_trial(self, X: np.ndarray, y: np.ndarray,
                        band_name: str, n_components: int, 
                        seed: int) -> Dict:
        """
        Run a single experimental trial.
        
        Parameters:
        -----------
        X : np.ndarray
            Raw EEG data
        y : np.ndarray
            Labels
        band_name : str
            Frequency band ('mu', 'beta', 'combined')
        n_components : int
            Number of CSP components
        seed : int
            Random seed
            
        Returns:
        --------
        trial_data : dict
            Complete trial results
        """
        print(f"\n{'='*60}")
        print(f"Trial: {band_name}, {n_components} components, seed {seed}")
        print(f"{'='*60}")
        
        # Apply bandpass filter
        low_freq, high_freq = self.frequency_bands[band_name]
        print(f"Applying {low_freq}-{high_freq} Hz bandpass filter...")
        X_filtered = self.bandpass_filter(X, low_freq, high_freq)
        
        # Split data
        print("Splitting data...")
        X_train, X_val, X_test, y_train, y_val, y_test = \
            self.stratified_split(X_filtered, y, seed)
        
        print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, "
              f"Test: {X_test.shape[0]}")
        
        # Train model
        print("Training CSP+LDA...")
        results = self.train_csp_lda(
            X_train, y_train, X_val, y_val, X_test, y_test,
            n_components, seed
        )
        
        # Add configuration info
        trial_data = {
            'band': band_name,
            'n_components': n_components,
            'seed': seed,
            **results
        }
        
        print(f"\nResults:")
        print(f"  Train Accuracy: {results['train_accuracy']:.4f}")
        print(f"  Val Accuracy:   {results['val_accuracy']:.4f}")
        print(f"  Test Accuracy:  {results['test_accuracy']:.4f}")
        print(f"  Test F1 (macro): {results['test_f1_macro']:.4f}")
        print(f"  Training time:  {results['training_time_sec']:.2f}s")
        
        return trial_data
    
    def run_all_experiments(self, X: np.ndarray, y: np.ndarray):
        """
        Run all experimental configurations.
        
        Parameters:
        -----------
        X : np.ndarray
            Raw EEG data
        y : np.ndarray
            Labels
        """
        total_configs = len(self.frequency_bands) * len(self.n_components_list)
        total_trials = total_configs * self.n_trials
        
        print(f"\n{'='*60}")
        print(f"Starting CSP+LDA Experiments")
        print(f"{'='*60}")
        print(f"Total configurations: {total_configs}")
        print(f"Trials per configuration: {self.n_trials}")
        print(f"Total trials: {total_trials}\n")
        
        trial_count = 0
        
        for band_name in self.frequency_bands.keys():
            for n_components in self.n_components_list:
                for seed in self.random_seeds:
                    trial_count += 1
                    print(f"\n[Trial {trial_count}/{total_trials}]")
                    
                    trial_data = self.run_single_trial(
                        X, y, band_name, n_components, seed
                    )
                    
                    self.trial_results.append(trial_data)
        
        print(f"\n{'='*60}")
        print("All trials completed!")
        print(f"{'='*60}\n")
    
    def compute_confidence_interval(self, data: np.ndarray, 
                                   confidence_level: float = 0.95) -> Tuple:
        """
        Compute bootstrap confidence interval.
        
        Parameters:
        -----------
        data : np.ndarray
            Data array
        confidence_level : float
            Confidence level (default: 0.95)
            
        Returns:
        --------
        mean, ci_low, ci_high : float
            Mean and confidence interval bounds
        """
        if len(data) < 2:
            return np.mean(data), np.mean(data), np.mean(data)
        
        result = bootstrap(
            [data], 
            np.mean, 
            confidence_level=confidence_level,
            random_state=0,
            n_resamples=10000
        )
        
        return np.mean(data), result.confidence_interval.low, \
               result.confidence_interval.high
    
    def aggregate_results(self):
        """Aggregate trial results by configuration."""
        print("Aggregating results across trials...")
        
        configs = []
        for band_name in self.frequency_bands.keys():
            for n_components in self.n_components_list:
                configs.append((band_name, n_components))
        
        for band_name, n_components in configs:
            # Filter trials for this configuration
            config_trials = [
                t for t in self.trial_results
                if t['band'] == band_name and t['n_components'] == n_components
            ]
            
            # Extract metrics
            test_accs = [t['test_accuracy'] for t in config_trials]
            test_f1s = [t['test_f1_macro'] for t in config_trials]
            test_f1_per_class = np.array([
                t['test_f1_per_class'] for t in config_trials
            ])
            confusion_matrices = np.array([
                t['confusion_matrix'] for t in config_trials
            ])
            
            # Compute statistics
            acc_mean, acc_ci_low, acc_ci_high = \
                self.compute_confidence_interval(test_accs)
            f1_mean, f1_ci_low, f1_ci_high = \
                self.compute_confidence_interval(test_f1s)
            
            config_summary = {
                'band': band_name,
                'n_components': n_components,
                
                # Accuracy statistics
                'test_acc_mean': acc_mean,
                'test_acc_ci_low': acc_ci_low,
                'test_acc_ci_high': acc_ci_high,
                'test_acc_std': np.std(test_accs, ddof=1),
                
                # F1 statistics
                'test_f1_mean': f1_mean,
                'test_f1_ci_low': f1_ci_low,
                'test_f1_ci_high': f1_ci_high,
                'test_f1_std': np.std(test_f1s, ddof=1),
                
                # Per-class F1
                'test_f1_per_class_mean': np.mean(test_f1_per_class, axis=0),
                'test_f1_per_class_std': np.std(test_f1_per_class, axis=0, ddof=1),
                
                # Average confusion matrix
                'confusion_matrix_mean': np.mean(confusion_matrices, axis=0),
                'confusion_matrix_std': np.std(confusion_matrices, axis=0, ddof=1),
            }
            
            self.config_summaries.append(config_summary)
            
            print(f"{band_name} ({n_components} comp): "
                  f"Acc = {acc_mean:.4f} "
                  f"[{acc_ci_low:.4f}, {acc_ci_high:.4f}], "
                  f"F1 = {f1_mean:.4f}")
    
    def statistical_testing(self):
        """Perform statistical tests between configurations."""
        print("\n" + "="*60)
        print("Statistical Testing")
        print("="*60)
        
        # Create DataFrame for easier manipulation
        results_df = pd.DataFrame(self.trial_results)
        
        # Get all configurations
        configs = [(s['band'], s['n_components']) 
                   for s in self.config_summaries]
        
        n_comparisons = len(configs) * (len(configs) - 1) // 2
        bonferroni_alpha = 0.05 / n_comparisons
        
        print(f"Number of pairwise comparisons: {n_comparisons}")
        print(f"Bonferroni-corrected alpha: {bonferroni_alpha:.6f}\n")
        
        # Perform pairwise comparisons
        comparisons = []
        for i, (band1, comp1) in enumerate(configs):
            for band2, comp2 in configs[i+1:]:
                # Get test accuracies for both configs
                accs1 = results_df[
                    (results_df['band'] == band1) & 
                    (results_df['n_components'] == comp1)
                ]['test_accuracy'].values
                
                accs2 = results_df[
                    (results_df['band'] == band2) & 
                    (results_df['n_components'] == comp2)
                ]['test_accuracy'].values
                
                # Paired t-test
                t_stat, p_val = ttest_rel(accs1, accs2)
                
                # Cohen's d effect size
                diff_mean = np.mean(accs1 - accs2)
                diff_std = np.std(accs1 - accs2, ddof=1)
                cohens_d = diff_mean / diff_std if diff_std > 0 else 0
                
                # Determine significance
                significant = p_val < bonferroni_alpha
                
                comparison = {
                    'config1': f"{band1}_{comp1}comp",
                    'config2': f"{band2}_{comp2}comp",
                    'mean_diff': diff_mean,
                    't_statistic': t_stat,
                    'p_value': p_val,
                    'cohens_d': cohens_d,
                    'significant': significant
                }
                
                comparisons.append(comparison)
                
                if significant:
                    print(f"✓ {comparison['config1']} vs {comparison['config2']}: "
                          f"p = {p_val:.6f}, d = {cohens_d:.3f} (SIGNIFICANT)")
        
        # Save comparisons
        comparisons_df = pd.DataFrame(comparisons)
        comparisons_df.to_csv(
            self.output_dir / 'statistical_comparisons.csv', 
            index=False
        )
        
        return comparisons_df
    
    def save_results(self):
        """Save all results to disk."""
        print("\nSaving results...")
        
        # Save trial results
        with open(self.output_dir / 'trial_results.pkl', 'wb') as f:
            pickle.dump(self.trial_results, f)
        
        # Save config summaries
        with open(self.output_dir / 'config_summaries.pkl', 'wb') as f:
            pickle.dump(self.config_summaries, f)
        
        # Save as CSV for easy viewing
        trial_df = pd.DataFrame([
            {k: v for k, v in t.items() 
             if not isinstance(v, np.ndarray)}
            for t in self.trial_results
        ])
        trial_df.to_csv(self.output_dir / 'trial_results.csv', index=False)
        
        summary_df = pd.DataFrame(self.config_summaries)
        # Remove array columns for CSV
        summary_df_csv = summary_df.drop(columns=[
            'test_f1_per_class_mean', 'test_f1_per_class_std',
            'confusion_matrix_mean', 'confusion_matrix_std'
        ])
        summary_df_csv.to_csv(
            self.output_dir / 'config_summaries.csv', 
            index=False
        )
        
        print(f"Results saved to {self.output_dir}")
    
    def plot_parameter_comparison(self):
        """Plot 1: Parameter comparison with error bars."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Prepare data
        configs = []
        means = []
        ci_lows = []
        ci_highs = []
        
        for summary in self.config_summaries:
            config_name = f"{summary['band']}\n{summary['n_components']}comp"
            configs.append(config_name)
            means.append(summary['test_acc_mean'] * 100)
            ci_lows.append(summary['test_acc_ci_low'] * 100)
            ci_highs.append(summary['test_acc_ci_high'] * 100)
        
        means = np.array(means)
        ci_lows = np.array(ci_lows)
        ci_highs = np.array(ci_highs)
        
        # Calculate error bar sizes
        yerr_low = means - ci_lows
        yerr_high = ci_highs - means
        
        # Create bar plot
        x_pos = np.arange(len(configs))
        bars = ax.bar(x_pos, means, color='steelblue', alpha=0.7, 
                     edgecolor='black', linewidth=1.2)
        ax.errorbar(x_pos, means, yerr=[yerr_low, yerr_high], 
                   fmt='none', ecolor='black', capsize=5, capthick=2)
        
        # Formatting
        ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
        ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('CSP+LDA: Frequency Band and Component Count Comparison',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(configs, fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim([0, 100])
        
        # Add value labels on bars
        for i, (bar, mean) in enumerate(zip(bars, means)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{mean:.1f}%', ha='center', va='bottom', 
                   fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plot1_parameter_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'plot1_parameter_comparison.pdf', 
                   bbox_inches='tight')
        print("✓ Saved plot 1: Parameter comparison")
        plt.close()
    
    def plot_frequency_band_ablation(self):
        """Plot 2: Frequency band ablation (grouped bars)."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bands = list(self.frequency_bands.keys())
        n_comp_list = self.n_components_list
        
        x = np.arange(len(bands))
        width = 0.35
        
        # Prepare data for each component count
        data_by_comp = {comp: {'means': [], 'ci_lows': [], 'ci_highs': []} 
                       for comp in n_comp_list}
        
        for band in bands:
            for n_comp in n_comp_list:
                summary = next(s for s in self.config_summaries 
                             if s['band'] == band and 
                             s['n_components'] == n_comp)
                data_by_comp[n_comp]['means'].append(
                    summary['test_acc_mean'] * 100
                )
                data_by_comp[n_comp]['ci_lows'].append(
                    summary['test_acc_ci_low'] * 100
                )
                data_by_comp[n_comp]['ci_highs'].append(
                    summary['test_acc_ci_high'] * 100
                )
        
        # Plot grouped bars
        colors = ['#1f77b4', '#ff7f0e']
        for i, (n_comp, color) in enumerate(zip(n_comp_list, colors)):
            means = np.array(data_by_comp[n_comp]['means'])
            ci_lows = np.array(data_by_comp[n_comp]['ci_lows'])
            ci_highs = np.array(data_by_comp[n_comp]['ci_highs'])
            
            yerr_low = means - ci_lows
            yerr_high = ci_highs - means
            
            offset = width * (i - 0.5)
            bars = ax.bar(x + offset, means, width, label=f'{n_comp} components',
                         color=color, alpha=0.7, edgecolor='black', linewidth=1.2)
            ax.errorbar(x + offset, means, yerr=[yerr_low, yerr_high],
                       fmt='none', ecolor='black', capsize=5, capthick=2)
            
            # Add value labels
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{mean:.1f}%', ha='center', va='bottom',
                       fontsize=9, fontweight='bold')
        
        # Formatting
        ax.set_xlabel('Frequency Band', fontsize=12, fontweight='bold')
        ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('CSP+LDA: Frequency Band Comparison',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([b.capitalize() for b in bands], fontsize=11)
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim([0, 100])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plot2_frequency_band_ablation.png',
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'plot2_frequency_band_ablation.pdf',
                   bbox_inches='tight')
        print("✓ Saved plot 2: Frequency band ablation")
        plt.close()
    
    def plot_confusion_matrix(self):
        """Plot 3: Confusion matrix for best configuration."""
        # Find best configuration
        best_config = max(self.config_summaries, 
                         key=lambda x: x['test_acc_mean'])
        
        cm = best_config['confusion_matrix_mean']
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Count'},
                   linewidths=0.5, linecolor='gray',
                   ax=ax)
        
        # Formatting
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        title = (f"Confusion Matrix - Best Configuration\n"
                f"{best_config['band'].capitalize()} Band, "
                f"{best_config['n_components']} Components "
                f"(Acc = {best_config['test_acc_mean']*100:.1f}%)")
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plot3_confusion_matrix.png',
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'plot3_confusion_matrix.pdf',
                   bbox_inches='tight')
        print("✓ Saved plot 3: Confusion matrix")
        plt.close()
    
    def plot_per_class_f1(self):
        """Plot 4: Per-class F1 comparison."""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Find best n_components for each band
        bands = list(self.frequency_bands.keys())
        
        # Determine best component count per band
        best_configs = {}
        for band in bands:
            band_configs = [s for s in self.config_summaries 
                          if s['band'] == band]
            best_config = max(band_configs, key=lambda x: x['test_acc_mean'])
            best_configs[band] = best_config
        
        x = np.arange(len(self.class_names))
        width = 0.25
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        # Plot bars for each band
        for i, (band, color) in enumerate(zip(bands, colors)):
            config = best_configs[band]
            means = config['test_f1_per_class_mean']
            stds = config['test_f1_per_class_std']
            
            offset = width * (i - 1)
            bars = ax.bar(x + offset, means, width, 
                         label=f"{band.capitalize()} ({config['n_components']} comp)",
                         color=color, alpha=0.7, edgecolor='black', 
                         linewidth=1)
            ax.errorbar(x + offset, means, yerr=stds, fmt='none',
                       ecolor='black', capsize=3, capthick=1.5, alpha=0.7)
        
        # Formatting
        ax.set_xlabel('Movement Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
        ax.set_title('Per-Class F1 Score Comparison Across Frequency Bands',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names, rotation=45, ha='right', fontsize=10)
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plot4_per_class_f1.png',
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'plot4_per_class_f1.pdf',
                   bbox_inches='tight')
        print("✓ Saved plot 4: Per-class F1 comparison")
        plt.close()
    
    def generate_all_plots(self):
        """Generate all required plots."""
        print("\n" + "="*60)
        print("Generating Plots")
        print("="*60)
        
        self.plot_parameter_comparison()
        self.plot_frequency_band_ablation()
        self.plot_confusion_matrix()
        self.plot_per_class_f1()
        
        print(f"\nAll plots saved to {self.output_dir}")
    
    def print_summary_table(self):
        """Print a summary table of results."""
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        
        print(f"\n{'Configuration':<20} {'Test Acc (%)':<15} {'Test F1':<15} "
              f"{'95% CI':<25}")
        print("-" * 75)
        
        # Sort by test accuracy
        sorted_summaries = sorted(
            self.config_summaries, 
            key=lambda x: x['test_acc_mean'], 
            reverse=True
        )
        
        for summary in sorted_summaries:
            config_name = f"{summary['band']}_{summary['n_components']}comp"
            acc = summary['test_acc_mean'] * 100
            f1 = summary['test_f1_mean']
            ci_low = summary['test_acc_ci_low'] * 100
            ci_high = summary['test_acc_ci_high'] * 100
            
            print(f"{config_name:<20} {acc:>6.2f} ± {summary['test_acc_std']*100:>4.2f} "
                  f"{f1:>6.4f} ± {summary['test_f1_std']:>5.4f}   "
                  f"[{ci_low:.2f}, {ci_high:.2f}]")
        
        print("\n" + "="*60)


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='CSP+LDA Experiment for EEG Movement Classification'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to preprocessed data file (.npz or .pkl)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results_csp_lda',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Initialize experiment
    experiment = CSPLDAExperiment(
        data_path=args.data_path,
        output_dir=args.output_dir
    )
    
    # Load data
    X, y = experiment.load_data()
    
    # Run all experiments
    experiment.run_all_experiments(X, y)
    
    # Aggregate results
    experiment.aggregate_results()
    
    # Statistical testing
    experiment.statistical_testing()
    
    # Save results
    experiment.save_results()
    
    # Generate plots
    experiment.generate_all_plots()
    
    # Print summary
    experiment.print_summary_table()
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE!")
    print("="*60)
    print(f"\nResults saved to: {experiment.output_dir}")


if __name__ == '__main__':
    main()