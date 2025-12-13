"""
Validation and Visualization Tools for Preprocessed EEG Data

Includes validation of ICA-based artifact removal.

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
    
    # Class names matching preprocessing.py
    CLASS_NAMES = {
        0: 'Rest',
        1: 'Reach Forward',
        2: 'Reach Backward', 
        3: 'Reach Left',
        4: 'Reach Right',
        5: 'Reach Up',
        6: 'Reach Down',
        7: 'Grasp Cup',
        8: 'Grasp Ball',
        9: 'Grasp Card',
        10: 'Twist Pronation',
        11: 'Twist Supination'
    }
    
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
        print(f"   Classes present: {unique_classes.tolist()}")
        print(f"   Expected: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] (12 classes)")
        
        # Expected total for full dataset
        expected_total = 82500
        if len(y) == expected_total:
            print(f"   ✓ Trial count matches expected: {expected_total}")
        else:
            print(f"   ⚠ Trial count: {len(y)} (expected {expected_total} for full dataset)")
        
        print(f"\n   Class breakdown:")
        for cls, count in zip(unique_classes, counts):
            percentage = count / len(y) * 100
            class_name = self.CLASS_NAMES.get(int(cls), f'Unknown-{cls}')
            print(f"   Class {int(cls):2d} ({class_name:20s}): {count:5d} trials ({percentage:5.2f}%)")
        
        # Check balance
        min_count = counts.min()
        max_count = counts.max()
        imbalance_ratio = max_count / min_count
        print(f"\n   Imbalance ratio: {imbalance_ratio:.2f}x")
        if imbalance_ratio > 3.0:
            print(f"   ⚠ WARNING: Significant class imbalance detected!")
            print(f"      (Rest class expected to have ~2-3x more trials)")
        else:
            print(f"   ✓ Class distribution looks reasonable")
        
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
                print(f"   ⚠ WARNING: Voltage values seem unusually large for EEG")
            else:
                print(f"   ✓ Voltage range appears reasonable for EEG")
        
        if metadata:
            print(f"\n4. METADATA")
            for key, value in metadata.items():
                if key == 'channel_names':
                    print(f"   channel_names: {len(value)} channels")
                    print(f"   First 5: {value[:5]}")
                elif key == 'artifact_removal_stats':
                    # Skip here, will display separately
                    continue
                else:
                    print(f"   {key}: {value}")
        
        # ICA Artifact Removal Statistics
        if metadata and 'ica_artifact_removal' in metadata:
            print(f"\n5. ICA ARTIFACT REMOVAL")
            print(f"   ICA applied: {metadata['ica_artifact_removal']}")
            
            if 'artifact_removal_stats' in metadata:
                stats = metadata['artifact_removal_stats']
                if isinstance(stats, dict):
                    print(f"   Files processed: {stats.get('files_processed', 'N/A')}")
                    print(f"   Files with EOG data: {stats.get('files_with_eog', 'N/A')}")
                    print(f"   Total components removed: {stats.get('total_components_removed', 'N/A')}")
                    print(f"   Avg components removed per file: {stats.get('avg_components_removed', 0):.2f}")
                    
                    # Validate artifact removal effectiveness
                    if stats.get('files_with_eog', 0) > 0:
                        coverage = (stats.get('files_with_eog', 0) / 
                                  stats.get('files_processed', 1)) * 100
                        print(f"   EOG coverage: {coverage:.1f}%")
                        
                        if coverage > 80:
                            print(f"   ✓ Good EOG data coverage")
                        else:
                            print(f"   ⚠ Limited EOG data coverage")
                        
                        avg_removed = stats.get('avg_components_removed', 0)
                        if 1 <= avg_removed <= 4:
                            print(f"   ✓ Reasonable number of components removed")
                        elif avg_removed > 4:
                            print(f"   ⚠ Many components removed (check for over-correction)")
                        else:
                            print(f"   ⚠ Few/no components removed (check EOG correlation threshold)")
        
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
                    class_name = self.CLASS_NAMES.get(int(cls), f'Class {int(cls)}')
                    ax.set_ylabel(f'{class_name}\n(µV)', fontsize=10)
                
                if cls_idx == 0:
                    ax.set_title(f'Sample {sample_idx+1}', fontsize=10)
                
                if cls_idx == n_classes - 1:
                    ax.set_xlabel('Time (s)', fontsize=9)
                
                ax.set_yticks([])
                ax.grid(alpha=0.3)
        
        # Add ICA info to title if available
        title = f'Sample EEG Trials - {config_name.upper()}'
        if metadata and metadata.get('ica_artifact_removal', False):
            title += ' (ICA Artifact Removal Applied)'
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
    
    def visualize_class_distribution(self, config_name: str, save_path: str = None):
        """Plot class distribution."""
        X, y, metadata = self.load_data(config_name)
        
        unique_classes, counts = np.unique(y, return_counts=True)
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_classes)))
        bars = ax.bar(range(len(unique_classes)), counts, color=colors, alpha=0.7)
        
        # Add counts on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count)}',
                   ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Movement Class', fontsize=12)
        ax.set_ylabel('Number of Trials', fontsize=12)
        
        # Add ICA info to title if available
        title = f'Class Distribution - {config_name.upper()}\nTotal: {len(y)} trials across {len(unique_classes)} classes'
        if metadata and metadata.get('ica_artifact_removal', False):
            title += '\n(ICA Artifact Removal Applied)'
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Create labels with class numbers and names
        labels = []
        for c in unique_classes:
            name = self.CLASS_NAMES.get(int(c), f'Class {int(c)}')
            # Shorten long names for x-axis
            if len(name) > 15:
                name = name.replace('Reach ', 'R-').replace('Grasp ', 'G-').replace('Twist ', 'T-')
            labels.append(f'{int(c)}\n{name}')
        
        ax.set_xticks(range(len(unique_classes)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        
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
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_classes)))
        
        for cls, color in zip(unique_classes, colors):
            class_trials = X[y == cls, channel_idx, :]
            mean_erp = np.mean(class_trials, axis=0)
            sem_erp = np.std(class_trials, axis=0) / np.sqrt(len(class_trials))
            
            label = self.CLASS_NAMES.get(int(cls), f'Class {int(cls)}')
            ax.plot(time_axis, mean_erp, label=label, color=color, linewidth=2)
            ax.fill_between(time_axis, 
                           mean_erp - sem_erp, 
                           mean_erp + sem_erp, 
                           color=color, alpha=0.2)
        
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.5, linewidth=2, label='Movement onset')
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
        
        channel_name = metadata.get('channel_names', [f'Ch{channel_idx}'])[channel_idx]
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Amplitude (µV)', fontsize=12)
        
        # Add ICA info to title if available
        title = f'Event-Related Potentials - Channel: {channel_name}\nMean ± SEM across trials'
        if metadata and metadata.get('ica_artifact_removal', False):
            title += ' (ICA Applied)'
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
    
    def plot_artifact_removal_summary(self, save_path: str = None):
        """
        Create summary visualization of ICA artifact removal across all configurations.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save figure
        """
        configs = ['csp_lda', 'ml_features', 'cnn']
        
        stats_data = {}
        for config in configs:
            try:
                _, _, metadata = self.load_data(config)
                if metadata and 'artifact_removal_stats' in metadata:
                    stats = metadata['artifact_removal_stats']
                    if isinstance(stats, dict):
                        stats_data[config] = stats
            except FileNotFoundError:
                continue
        
        if not stats_data:
            print("No ICA artifact removal statistics found in metadata")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        configs = list(stats_data.keys())
        
        # Plot 1: Files with EOG coverage
        files_processed = [stats_data[c].get('files_processed', 0) for c in configs]
        files_with_eog = [stats_data[c].get('files_with_eog', 0) for c in configs]
        
        x = np.arange(len(configs))
        width = 0.35
        
        axes[0].bar(x - width/2, files_processed, width, label='Total Files', alpha=0.7)
        axes[0].bar(x + width/2, files_with_eog, width, label='Files with EOG', alpha=0.7)
        axes[0].set_ylabel('Number of Files')
        axes[0].set_title('EOG Data Availability')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(configs)
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        
        # Plot 2: Total components removed
        total_removed = [stats_data[c].get('total_components_removed', 0) for c in configs]
        
        axes[1].bar(configs, total_removed, alpha=0.7, color='coral')
        axes[1].set_ylabel('Total ICA Components Removed')
        axes[1].set_title('Total Artifact Components Removed')
        axes[1].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(total_removed):
            axes[1].text(i, v, str(int(v)), ha='center', va='bottom')
        
        # Plot 3: Average components removed per file
        avg_removed = [stats_data[c].get('avg_components_removed', 0) for c in configs]
        
        axes[2].bar(configs, avg_removed, alpha=0.7, color='lightgreen')
        axes[2].set_ylabel('Avg Components Removed per File')
        axes[2].set_title('Average Artifact Components per File')
        axes[2].axhline(y=2, color='r', linestyle='--', alpha=0.5, label='Expected range')
        axes[2].axhline(y=3, color='r', linestyle='--', alpha=0.5)
        axes[2].legend()
        axes[2].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(avg_removed):
            axes[2].text(i, v, f'{v:.2f}', ha='center', va='bottom')
        
        plt.suptitle('ICA Artifact Removal Summary', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
    
    def compare_signal_quality(self, config_name: str, n_samples: int = 100, save_path: str = None):
        """
        Compare signal quality metrics to assess artifact removal effectiveness.
        
        Parameters:
        -----------
        config_name : str
            Configuration to analyze
        n_samples : int
            Number of random trials to sample for analysis
        save_path : str, optional
            Path to save figure
        """
        X, y, metadata = self.load_data(config_name)
        
        if X.ndim != 3:
            print("Signal quality analysis requires trial data (3D)")
            return
        
        # Sample random trials
        n_trials = min(n_samples, X.shape[0])
        sample_idx = np.random.choice(X.shape[0], n_trials, replace=False)
        X_sample = X[sample_idx]
        
        # Compute signal quality metrics
        # 1. Signal-to-noise ratio (SNR) estimate
        signal_power = np.mean(X_sample ** 2, axis=2)  # (n_trials, n_channels)
        noise_estimate = np.std(np.diff(X_sample, axis=2), axis=2)  # High-frequency noise
        snr = 10 * np.log10(signal_power / (noise_estimate ** 2 + 1e-10))
        
        # 2. Kurtosis (measure of outliers/artifacts)
        from scipy.stats import kurtosis
        kurt = kurtosis(X_sample, axis=2)  # (n_trials, n_channels)
        
        # 3. Variance
        variance = np.var(X_sample, axis=2)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # SNR distribution
        axes[0, 0].hist(snr.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_xlabel('SNR (dB)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Signal-to-Noise Ratio Distribution')
        axes[0, 0].axvline(x=np.median(snr), color='r', linestyle='--', 
                          label=f'Median: {np.median(snr):.2f} dB')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Kurtosis distribution
        axes[0, 1].hist(kurt.flatten(), bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].set_xlabel('Kurtosis')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Kurtosis Distribution (artifact indicator)')
        axes[0, 1].axvline(x=3, color='r', linestyle='--', label='Normal (3.0)')
        axes[0, 1].axvline(x=np.median(kurt), color='orange', linestyle='--',
                          label=f'Median: {np.median(kurt):.2f}')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # Variance distribution
        axes[1, 0].hist(variance.flatten(), bins=50, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 0].set_xlabel('Variance (µV²)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Signal Variance Distribution')
        axes[1, 0].axvline(x=np.median(variance), color='r', linestyle='--',
                          label=f'Median: {np.median(variance):.2f}')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # Channel-wise SNR
        mean_snr_per_channel = np.mean(snr, axis=0)
        axes[1, 1].bar(range(len(mean_snr_per_channel)), mean_snr_per_channel, alpha=0.7)
        axes[1, 1].set_xlabel('Channel Index')
        axes[1, 1].set_ylabel('Mean SNR (dB)')
        axes[1, 1].set_title('Average SNR per Channel')
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        # Add ICA info to suptitle
        title = f'Signal Quality Analysis - {config_name.upper()}\n'
        if metadata and metadata.get('ica_artifact_removal', False):
            stats = metadata.get('artifact_removal_stats', {})
            avg_removed = stats.get('avg_components_removed', 0)
            title += f'ICA Applied (Avg {avg_removed:.2f} components removed per file)'
        else:
            title += 'No ICA Artifact Removal'
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
        
        # Print summary statistics
        print(f"\n{'='*70}")
        print(f"SIGNAL QUALITY METRICS - {config_name.upper()}")
        print(f"{'='*70}")
        print(f"SNR (dB):")
        print(f"  Mean: {np.mean(snr):.2f}")
        print(f"  Median: {np.median(snr):.2f}")
        print(f"  Std: {np.std(snr):.2f}")
        print(f"\nKurtosis:")
        print(f"  Mean: {np.mean(kurt):.2f}")
        print(f"  Median: {np.median(kurt):.2f}")
        print(f"  % High kurtosis (>5): {100 * np.sum(kurt.flatten() > 5) / kurt.size:.2f}%")
        print(f"\nVariance (µV²):")
        print(f"  Mean: {np.mean(variance):.2f}")
        print(f"  Median: {np.median(variance):.2f}")
        print(f"{'='*70}\n")
    
    def plot_frequency_spectrum(self, config_name: str, n_trials: int = 10, 
                                raw_data_dir: str = None, save_path: str = None):
        """
        Plot frequency spectrum of preprocessed data and optionally compare with raw data.
        
        Parameters:
        -----------
        config_name : str
            Configuration to analyze ('csp_lda' or 'cnn')
        n_trials : int
            Number of random trials to analyze
        raw_data_dir : str, optional
            Directory containing raw .mat files for before/after comparison
        save_path : str, optional
            Path to save figure
        """
        from scipy.fft import fft, fftfreq
        
        X, y, metadata = self.load_data(config_name)
        
        if X.ndim != 3:
            print("Frequency spectrum analysis requires trial data (3D)")
            return
        
        fs = metadata.get('fs', 2500)
        filter_range = metadata.get('filter', (None, None))
        
        # Sample random trials
        n_trials = min(n_trials, X.shape[0])
        sample_idx = np.random.choice(X.shape[0], n_trials, replace=False)
        X_sample = X[sample_idx]
        
        # Compute average spectrum across trials and channels
        all_spectra = []
        for trial in X_sample:
            for channel in trial:
                N = len(channel)
                yf = fft(channel)
                xf = fftfreq(N, 1/fs)[:N//2]
                power = 2.0/N * np.abs(yf[0:N//2])
                all_spectra.append(power)
        
        avg_spectrum = np.mean(all_spectra, axis=0)
        std_spectrum = np.std(all_spectra, axis=0)
        
        # Create figure
        if raw_data_dir:
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            axes = axes.flatten()
        else:
            fig, axes = plt.subplots(1, 2, figsize=(16, 5))
            axes = [axes[0], axes[1], None, None]
        
        # Plot 1: Linear scale (0-60 Hz)
        axes[0].plot(xf, avg_spectrum, linewidth=1.5, label='Mean spectrum')
        axes[0].fill_between(xf, 
                             avg_spectrum - std_spectrum, 
                             avg_spectrum + std_spectrum, 
                             alpha=0.3, label='±1 SD')
        axes[0].set_xlim(0, 60)
        axes[0].set_xlabel('Frequency (Hz)', fontsize=11)
        axes[0].set_ylabel('Power', fontsize=11)
        axes[0].set_title('Frequency Spectrum - Linear Scale (0-60 Hz)', fontsize=12, fontweight='bold')
        
        if filter_range[0] and filter_range[1]:
            axes[0].axvspan(filter_range[0], filter_range[1], alpha=0.2, color='green', 
                           label=f'Filter band ({filter_range[0]}-{filter_range[1]} Hz)')
        
        axes[0].legend(fontsize=9)
        axes[0].grid(alpha=0.3)
        
        # Plot 2: Log scale (0-100 Hz)
        axes[1].semilogy(xf, avg_spectrum + 1e-10, linewidth=1.5)
        axes[1].set_xlim(0, 100)
        axes[1].set_xlabel('Frequency (Hz)', fontsize=11)
        axes[1].set_ylabel('Power (log scale)', fontsize=11)
        axes[1].set_title('Frequency Spectrum - Log Scale (0-100 Hz)', fontsize=12, fontweight='bold')
        
        if filter_range[0] and filter_range[1]:
            axes[1].axvspan(filter_range[0], filter_range[1], alpha=0.2, color='green',
                           label=f'Filter band ({filter_range[0]}-{filter_range[1]} Hz)')
        
        # Mark common frequency bands
        axes[1].axvline(x=8, color='orange', linestyle='--', alpha=0.5, linewidth=1)
        axes[1].axvline(x=12, color='orange', linestyle='--', alpha=0.5, linewidth=1)
        axes[1].axvline(x=30, color='red', linestyle='--', alpha=0.5, linewidth=1)
        axes[1].text(10, axes[1].get_ylim()[1]*0.5, 'μ', fontsize=10, color='orange')
        axes[1].text(20, axes[1].get_ylim()[1]*0.5, 'β', fontsize=10, color='red')
        
        axes[1].legend(fontsize=9)
        axes[1].grid(alpha=0.3)
        
        # If raw data directory provided, load and compare
        if raw_data_dir and axes[2] is not None:
            try:
                import scipy.io as sio
                from pathlib import Path
                
                print(f"  Loading raw data for comparison...")
                raw_dir = Path(raw_data_dir)
                
                # Find a sample raw file
                raw_files = list(raw_dir.glob('EEG_session1_sub1_*.mat'))
                if len(raw_files) == 0:
                    print(f"  No raw files found in {raw_data_dir}")
                    return
                
                # Load first file
                raw_mat = sio.loadmat(raw_files[0], simplify_cells=False)
                
                # Load first channel as example
                raw_channel = raw_mat['ch1'].flatten()
                
                # Compute spectrum
                N_raw = len(raw_channel)
                yf_raw = fft(raw_channel)
                xf_raw = fftfreq(N_raw, 1/fs)[:N_raw//2]
                power_raw = 2.0/N_raw * np.abs(yf_raw[0:N_raw//2])
                
                # Plot 3: Before ICA (raw data)
                axes[2].semilogy(xf_raw, power_raw + 1e-10, linewidth=1.5, color='red', alpha=0.7, label='Before ICA')
                axes[2].set_xlim(0, 100)
                axes[2].set_xlabel('Frequency (Hz)', fontsize=11)
                axes[2].set_ylabel('Power (log scale)', fontsize=11)
                axes[2].set_title('BEFORE ICA - Raw Continuous Data', fontsize=12, fontweight='bold')
                axes[2].axvline(x=50, color='purple', linestyle='--', alpha=0.5, label='Line noise (50 Hz)')
                axes[2].axvline(x=60, color='purple', linestyle='--', alpha=0.5, label='Line noise (60 Hz)')
                axes[2].legend(fontsize=9)
                axes[2].grid(alpha=0.3)
                
                # Plot 4: Before vs After comparison
                axes[3].semilogy(xf_raw, power_raw + 1e-10, linewidth=1.5, color='red', alpha=0.7, label='Before ICA (raw)')
                axes[3].semilogy(xf, avg_spectrum + 1e-10, linewidth=1.5, color='blue', alpha=0.7, label='After ICA (processed)')
                axes[3].set_xlim(0, 60)
                axes[3].set_xlabel('Frequency (Hz)', fontsize=11)
                axes[3].set_ylabel('Power (log scale)', fontsize=11)
                axes[3].set_title('Before vs After ICA Comparison (0-60 Hz)', fontsize=12, fontweight='bold')
                
                if filter_range[0] and filter_range[1]:
                    axes[3].axvspan(filter_range[0], filter_range[1], alpha=0.2, color='green',
                                   label=f'Analysis band ({filter_range[0]}-{filter_range[1]} Hz)')
                
                axes[3].legend(fontsize=9)
                axes[3].grid(alpha=0.3)
                
                print(f"  ✓ Before/after comparison added")
                
            except Exception as e:
                print(f"  Error loading raw data: {e}")
        
        # Analyze power distribution
        total_power = np.sum(avg_spectrum)
        if filter_range[0] and filter_range[1]:
            in_band_mask = (xf >= filter_range[0]) & (xf <= filter_range[1])
            power_in_band = np.sum(avg_spectrum[in_band_mask])
            percent_in_band = (power_in_band / total_power) * 100
            
            # Add text annotation
            annotation = f"Power in {filter_range[0]}-{filter_range[1]} Hz: {percent_in_band:.1f}%"
            if axes[1]:
                axes[1].text(0.98, 0.02, annotation, transform=axes[1].transAxes,
                           fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Overall title
        title = f'Frequency Spectrum Analysis - {config_name.upper()}\n'
        title += f'Averaged over {n_trials} trials × {X_sample.shape[1]} channels'
        if metadata.get('ica_artifact_removal', False):
            stats = metadata.get('artifact_removal_stats', {})
            avg_removed = stats.get('avg_components_removed', 0)
            title += f' | ICA Applied (avg {avg_removed:.2f} components removed)'
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
        
        # Print analysis
        print(f"\n{'='*70}")
        print(f"FREQUENCY SPECTRUM ANALYSIS - {config_name.upper()}")
        print(f"{'='*70}")
        print(f"Total power: {total_power:.2e}")
        
        if filter_range[0] and filter_range[1]:
            print(f"Power in filter band ({filter_range[0]}-{filter_range[1]} Hz): {power_in_band:.2e} ({percent_in_band:.1f}%)")
            print(f"Power outside filter band: {total_power - power_in_band:.2e} ({100-percent_in_band:.1f}%)")
            
            if percent_in_band > 90:
                print(f"\n✓ Most power concentrated in filter band")
                print(f"  Data appears properly filtered")
            elif percent_in_band > 70:
                print(f"\n✓ Good power concentration in filter band")
            else:
                print(f"\n⚠ Significant power outside filter band")
                print(f"  May indicate filtering issues or strong artifacts")
        
        # Check for line noise
        line_noise_50 = np.mean(avg_spectrum[(xf >= 49) & (xf <= 51)])
        line_noise_60 = np.mean(avg_spectrum[(xf >= 59) & (xf <= 61)])
        baseline_power = np.median(avg_spectrum[(xf >= 40) & (xf <= 45)])
        
        if line_noise_50 > baseline_power * 2 or line_noise_60 > baseline_power * 2:
            print(f"\n⚠ Line noise detected at 50/60 Hz")
        else:
            print(f"\n✓ No significant line noise")
        
        print(f"{'='*70}\n")
    
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
                    'data_type': 'trials' if X.ndim == 3 else 'features',
                    'file_size_mb': (self.data_dir / f'{config}_data.npz').stat().st_size / 1e6,
                    'ica_applied': metadata.get('ica_artifact_removal', False)
                }
            except FileNotFoundError:
                print(f"⚠ {config} not found")
        
        if comparison_data:
            print(f"{'Config':<15} {'Shape':<25} {'Type':<10} {'Trials':<10} {'Classes':<10} {'ICA':<8} {'Size (MB)'}")
            print(f"{'-'*100}")
            for config, info in comparison_data.items():
                ica_status = '✓' if info['ica_applied'] else '✗'
                print(f"{config:<15} {str(info['shape']):<25} {info['data_type']:<10} "
                     f"{info['n_trials']:<10} {info['n_classes']:<10} {ica_status:<8} {info['file_size_mb']:.1f}")
        
        print(f"\n{'='*70}\n")


def run_full_validation(preprocessed_dir: str, output_dir: str = None, raw_data_dir: str = None):
    """
    Run complete validation suite.
    
    Parameters:
    -----------
    preprocessed_dir : str
        Directory with preprocessed .npz files
    output_dir : str, optional
        Directory to save validation plots
    raw_data_dir : str, optional
        Directory with raw .mat files for before/after comparison
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
    
    # 3. ICA artifact removal summary (if applicable)
    print("\nGenerating ICA artifact removal summary...")
    try:
        save_path = output_path / 'ica_artifact_removal_summary.png'
        validator.plot_artifact_removal_summary(save_path=str(save_path))
    except Exception as e:
        print(f"  Could not generate ICA summary: {e}")
    
    # 4. Frequency spectrum analysis
    print("\nGenerating frequency spectrum analysis...")
    for config in ['csp_lda', 'cnn']:
        try:
            print(f"\n  Analyzing {config} frequency spectrum...")
            save_path = output_path / f'{config}_frequency_spectrum.png'
            validator.plot_frequency_spectrum(config, n_trials=10, 
                                             raw_data_dir=raw_data_dir,
                                             save_path=str(save_path))
        except FileNotFoundError:
            print(f"  ⚠ Skipping frequency analysis for {config} (file not found)")
        except Exception as e:
            print(f"  ⚠ Error in frequency analysis for {config}: {e}")
    
    # 5. Visualizations for trial data
    print("\nGenerating visualizations...")
    print("(This may take 2-3 minutes for full dataset)")
    
    for config in ['csp_lda', 'cnn']:
        try:
            print(f"\n  Processing {config}...")
            
            # Class distribution (fast)
            save_path = output_path / f'{config}_class_distribution.png'
            validator.visualize_class_distribution(config, save_path=str(save_path))
            
            # Sample trials (moderate speed)
            save_path = output_path / f'{config}_sample_trials.png'
            validator.visualize_sample_trials(config, n_samples=3, save_path=str(save_path))
            
            # ERPs for motor channel (slower - computing averages)
            save_path = output_path / f'{config}_erp.png'
            validator.plot_average_erp(config, channel_idx=7, save_path=str(save_path))  # Channel 7 is Cz
            
            # Signal quality analysis (new - checks artifact removal effectiveness)
            save_path = output_path / f'{config}_signal_quality.png'
            validator.compare_signal_quality(config, n_samples=100, save_path=str(save_path))
            
        except FileNotFoundError:
            print(f"  ⚠ Skipping visualizations for {config} (file not found)")
        except Exception as e:
            print(f"  ⚠ Error in {config}: {e}")
    
    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print(f"Plots saved to: {output_path}")
    print("="*70 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate preprocessed EEG data')
    parser.add_argument('--preprocessed-dir', type=str,
                       default='/home/ubuntu/EEG-Upper-Movement-Classification/TEST/preprocessed',
                       help='Directory containing preprocessed .npz files')
    parser.add_argument('--output-dir', type=str,
                       default='/home/ubuntu/EEG-Upper-Movement-Classification/TEST/preprocessed/validation_reports',
                       help='Directory to save validation plots')
    parser.add_argument('--raw-data-dir', type=str, default='/home/ubuntu/EEG-Upper-Movement-Classification/TEST',
                       help='Directory containing raw EEG .mat files for before/after comparison')
    
    args = parser.parse_args()
    
    run_full_validation(args.preprocessed_dir, args.output_dir, args.raw_data_dir)