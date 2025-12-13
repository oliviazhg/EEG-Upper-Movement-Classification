"""
Validation and Visualization Tools for Preprocessed EEG Data

Run this after preprocessing to verify everything worked correctly.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple
import seaborn as sns

sns.set_style('whitegrid')


class PreprocessingValidator:
    """Tools to validate and visualize preprocessed EEG data."""
    
    def __init__(self, preprocessed_dir: str):
        """
        Parameters:
        -----------
        preprocessed_dir : str
            Directory containing preprocessed .npz files
        """
        self.data_dir = Path(preprocessed_dir)
        
    def load_data(self, config_name: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Load preprocessed data file."""
        filepath = self.data_dir / f'{config_name}_data.npz'
        
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        data = np.load(filepath, allow_pickle=True)
        X = data['X']
        y = data['y']
        metadata = data['metadata'].item() if 'metadata' in data else {}
        
        return X, y, metadata
    
    def check_basic_stats(self, config_name: str):
        """Print basic statistics about preprocessed data."""
        X, y, metadata = self.load_data(config_name)
        
        print(f"\n{'='*70}")
        print(f"VALIDATION REPORT: {config_name.upper()}")
        print(f"{'='*70}")
        
        print(f"\n1. DATA SHAPES")
        print(f"   X shape: {X.shape}")
        print(f"   y shape: {y.shape}")
        
        if X.ndim == 3:
            print(f"   Format: (n_trials={X.shape[0]}, n_channels={X.shape[1]}, n_samples={X.shape[2]})")
            print(f"   Duration per trial: {X.shape[2]/2500:.2f} seconds")
        else:
            print(f"   Format: (n_trials={X.shape[0]}, n_features={X.shape[1]})")
        
        print(f"\n2. CLASS DISTRIBUTION")
        unique_classes, counts = np.unique(y, return_counts=True)
        print(f"   Number of classes: {len(unique_classes)}")
        print(f"   Classes: {unique_classes}")
        
        for cls, count in zip(unique_classes, counts):
            percentage = count / len(y) * 100
            print(f"   Class {int(cls):2d}: {count:5d} trials ({percentage:5.2f}%)")
        
        # Check balance
        min_count = counts.min()
        max_count = counts.max()
        imbalance_ratio = max_count / min_count
        print(f"\n   Imbalance ratio: {imbalance_ratio:.2f}x")
        if imbalance_ratio > 2.0:
            print(f"   ⚠ WARNING: Significant class imbalance detected!")
        else:
            print(f"   ✓ Classes reasonably balanced")
        
        print(f"\n3. DATA QUALITY")
        
        # Check for NaN/Inf
        has_nan = np.any(np.isnan(X))
        has_inf = np.any(np.isinf(X))
        print(f"   NaN values: {'Yes ⚠' if has_nan else 'None ✓'}")
        print(f"   Inf values: {'Yes ⚠' if has_inf else 'None ✓'}")
        
        # Voltage range (for trial data, not features)
        if X.ndim == 3:
            print(f"\n   Voltage statistics (µV):")
            print(f"   Min:  {X.min():8.2f}")
            print(f"   Max:  {X.max():8.2f}")
            print(f"   Mean: {X.mean():8.2f}")
            print(f"   Std:  {X.std():8.2f}")
            
            # Reasonable range check
            if abs(X.min()) > 200 or abs(X.max()) > 200:
                print(f"   ⚠ WARNING: Voltage values seem unusually large")
            else:
                print(f"   ✓ Voltage range appears reasonable for EEG")
        
        if metadata:
            print(f"\n4. METADATA")
            for key, value in metadata.items():
                if key != 'channel_names':
                    print(f"   {key}: {value}")
        
        print(f"\n{'='*70}\n")
    
    def visualize_sample_trials(self, config_name: str, n_samples: int = 3, save_path: str = None):
        """
        Visualize sample trials from each class.
        
        Parameters:
        -----------
        config_name : str
            Which configuration to visualize
        n_samples : int
            Number of sample trials per class
        save_path : str, optional
            Path to save figure
        """
        X, y, metadata = self.load_data(config_name)
        
        if X.ndim != 3:
            print(f"Can only visualize trial data (3D), not features (2D)")
            print(f"Use config 'csp_lda' or 'cnn' instead of '{config_name}'")
            return
        
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)
        n_channels = X.shape[1]
        
        # Movement class names (from dataset description)
        class_names = {
            0: 'Forward', 1: 'Backward', 2: 'Left', 3: 'Right', 4: 'Up', 5: 'Down',
            6: 'Power Grasp', 7: 'Precision Grasp', 8: 'Lateral Grasp',
            9: 'Pronation', 10: 'Supination'
        }
        
        # Create figure
        fig, axes = plt.subplots(n_classes, n_samples, 
                                figsize=(15, 2*n_classes), 
                                squeeze=False)
        
        time_axis = np.linspace(-1, 2, X.shape[2])  # -1s to +2s
        
        for cls_idx, cls in enumerate(unique_classes):
            # Get trials for this class
            class_trials = X[y == cls]
            
            # Sample random trials
            sample_indices = np.random.choice(len(class_trials), 
                                            size=min(n_samples, len(class_trials)), 
                                            replace=False)
            
            for sample_idx, trial_idx in enumerate(sample_indices):
                ax = axes[cls_idx, sample_idx]
                trial = class_trials[trial_idx]
                
                # Plot all channels (with offset for visibility)
                for ch_idx in range(n_channels):
                    offset = ch_idx * 20  # Offset between channels
                    ax.plot(time_axis, trial[ch_idx] + offset, 
                           linewidth=0.5, alpha=0.7)
                
                # Formatting
                ax.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Movement onset')
                ax.set_xlim(-1, 2)
                
                if sample_idx == 0:
                    class_name = class_names.get(int(cls), f'Class {int(cls)}')
                    ax.set_ylabel(f'{class_name}\n(µV)', fontsize=10)
                
                if cls_idx == 0:
                    ax.set_title(f'Sample {sample_idx+1}', fontsize=10)
                
                if cls_idx == n_classes - 1:
                    ax.set_xlabel('Time (s)', fontsize=9)
                
                ax.set_yticks([])
                ax.grid(alpha=0.3)
        
        plt.suptitle(f'Sample EEG Trials - {config_name.upper()}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
    
    def visualize_class_distribution(self, config_name: str, save_path: str = None):
        """Plot class distribution."""
        X, y, metadata = self.load_data(config_name)
        
        class_names = {
            0: 'Forward', 1: 'Backward', 2: 'Left', 3: 'Right', 4: 'Up', 5: 'Down',
            6: 'Power\nGrasp', 7: 'Precision\nGrasp', 8: 'Lateral\nGrasp',
            9: 'Pronation', 10: 'Supination'
        }
        
        unique_classes, counts = np.unique(y, return_counts=True)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))
        bars = ax.bar(range(len(unique_classes)), counts, color=colors, alpha=0.7)
        
        # Add counts on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count)}',
                   ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel('Movement Class', fontsize=12)
        ax.set_ylabel('Number of Trials', fontsize=12)
        ax.set_title(f'Class Distribution - {config_name.upper()}\n'
                    f'Total: {len(y)} trials across {len(unique_classes)} classes',
                    fontsize=14, fontweight='bold')
        
        ax.set_xticks(range(len(unique_classes)))
        ax.set_xticklabels([class_names.get(int(c), f'{int(c)}') for c in unique_classes],
                          rotation=45, ha='right')
        
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
    
    def plot_average_erp(self, config_name: str, channel_idx: int = 0, save_path: str = None):
        """
        Plot event-related potentials (average across trials) for each class.
        
        Parameters:
        -----------
        config_name : str
            Data configuration
        channel_idx : int
            Which channel to plot
        save_path : str, optional
            Save path for figure
        """
        X, y, metadata = self.load_data(config_name)
        
        if X.ndim != 3:
            print("ERP plotting requires trial data (3D), not features")
            return
        
        unique_classes = np.unique(y)
        time_axis = np.linspace(-1, 2, X.shape[2])
        
        class_names = {
            0: 'Forward', 1: 'Backward', 2: 'Left', 3: 'Right', 4: 'Up', 5: 'Down',
            6: 'Power Grasp', 7: 'Precision Grasp', 8: 'Lateral Grasp',
            9: 'Pronation', 10: 'Supination'
        }
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))
        
        for cls, color in zip(unique_classes, colors):
            class_trials = X[y == cls, channel_idx, :]
            mean_erp = np.mean(class_trials, axis=0)
            sem_erp = np.std(class_trials, axis=0) / np.sqrt(len(class_trials))
            
            label = class_names.get(int(cls), f'Class {int(cls)}')
            ax.plot(time_axis, mean_erp, label=label, color=color, linewidth=2)
            ax.fill_between(time_axis, 
                           mean_erp - sem_erp, 
                           mean_erp + sem_erp, 
                           color=color, alpha=0.2)
        
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.5, label='Movement onset')
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
        
        channel_name = metadata.get('channel_names', [f'Ch{channel_idx}'])[channel_idx]
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Amplitude (µV)', fontsize=12)
        ax.set_title(f'Event-Related Potentials - Channel: {channel_name}\n'
                    f'Mean ± SEM across trials',
                    fontsize=14, fontweight='bold')
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
    
    def compare_configurations(self):
        """Compare data across all configurations."""
        configs = ['csp_lda', 'ml_features', 'cnn']
        
        print(f"\n{'='*70}")
        print(f"COMPARING CONFIGURATIONS")
        print(f"{'='*70}\n")
        
        comparison_data = {}
        
        for config in configs:
            try:
                X, y, metadata = self.load_data(config)
                comparison_data[config] = {
                    'shape': X.shape,
                    'n_trials': len(y),
                    'n_classes': len(np.unique(y)),
                    'data_type': 'trials' if X.ndim == 3 else 'features'
                }
            except FileNotFoundError:
                print(f"⚠ {config} not found")
        
        if comparison_data:
            print(f"{'Config':<15} {'Shape':<25} {'Type':<10} {'Trials':<10} {'Classes'}")
            print(f"{'-'*70}")
            for config, info in comparison_data.items():
                print(f"{config:<15} {str(info['shape']):<25} {info['data_type']:<10} "
                     f"{info['n_trials']:<10} {info['n_classes']}")
        
        print(f"\n{'='*70}\n")


def run_full_validation(preprocessed_dir: str, output_dir: str = None):
    """
    Run complete validation suite.
    
    Parameters:
    -----------
    preprocessed_dir : str
        Directory with preprocessed .npz files
    output_dir : str, optional
        Directory to save validation plots
    """
    validator = PreprocessingValidator(preprocessed_dir)
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
    else:
        output_path = Path(preprocessed_dir)
    
    print("\n" + "="*70)
    print("RUNNING FULL VALIDATION SUITE")
    print("="*70)
    
    # 1. Compare all configurations
    validator.compare_configurations()
    
    # 2. Detailed stats for each configuration
    for config in ['csp_lda', 'ml_features', 'cnn']:
        try:
            validator.check_basic_stats(config)
        except FileNotFoundError:
            print(f"\n⚠ Skipping {config} (file not found)\n")
    
    # 3. Visualizations for trial data
    for config in ['csp_lda', 'cnn']:
        try:
            print(f"\nGenerating visualizations for {config}...")
            
            # Class distribution
            save_path = output_path / f'{config}_class_distribution.png'
            validator.visualize_class_distribution(config, save_path=str(save_path))
            
            # Sample trials
            save_path = output_path / f'{config}_sample_trials.png'
            validator.visualize_sample_trials(config, n_samples=3, save_path=str(save_path))
            
            # ERPs for motor channel (typically C3 or Cz)
            save_path = output_path / f'{config}_erp.png'
            validator.plot_average_erp(config, channel_idx=0, save_path=str(save_path))
            
        except FileNotFoundError:
            print(f"⚠ Skipping visualizations for {config} (file not found)")
    
    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Example usage
    preprocessed_dir = '/home/claude/preprocessed_eeg'
    output_dir = '/home/claude/validation_plots'
    
    run_full_validation(preprocessed_dir, output_dir)