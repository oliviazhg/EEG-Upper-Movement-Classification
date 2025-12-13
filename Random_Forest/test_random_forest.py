"""
Testing Script for Random Forest Experiment
SYDE 522 Final Project

This script tests the Random Forest experiment with synthetic data to verify
the pipeline works correctly before running on real data.

Author: Olivia Zheng
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# ============================================================================
# SYNTHETIC DATA GENERATION
# ============================================================================

def generate_synthetic_eeg_data(n_trials_per_class=50, n_channels=60, 
                               n_samples=5000, n_classes=11,
                               snr_db=10, seed=42):
    """
    Generate synthetic EEG-like data with realistic properties.
    
    Creates data where different classes have slightly different patterns
    in specific frequency bands, mimicking real motor-related EEG.
    
    Args:
        n_trials_per_class: Number of trials per movement class
        n_channels: Number of EEG channels
        n_samples: Samples per trial (at 2500 Hz, 5000 = 2 seconds)
        n_classes: Number of movement classes (excluding rest)
        snr_db: Signal-to-noise ratio in dB
        seed: Random seed for reproducibility
    
    Returns:
        eeg_data: Synthetic EEG (n_trials, n_channels, n_samples)
        labels: Class labels (n_trials,) ranging from 0 to n_classes-1
    """
    np.random.seed(seed)
    
    total_trials = n_trials_per_class * n_classes
    eeg_data = np.zeros((total_trials, n_channels, n_samples))
    labels = np.repeat(np.arange(n_classes), n_trials_per_class)
    
    # Sampling frequency
    fs = 2500  # Hz
    t = np.arange(n_samples) / fs
    
    print("Generating synthetic EEG data...")
    print(f"  Classes: {n_classes}")
    print(f"  Trials per class: {n_trials_per_class}")
    print(f"  Total trials: {total_trials}")
    print(f"  Channels: {n_channels}")
    print(f"  Samples per trial: {n_samples}")
    
    # Define frequency bands for different classes
    mu_freq = 10  # Hz (8-12 Hz band)
    beta_freq = 20  # Hz (13-30 Hz band)
    
    for trial_idx in range(total_trials):
        class_label = labels[trial_idx]
        
        # Base noise for all channels
        noise = np.random.randn(n_channels, n_samples)
        
        # Add class-specific patterns
        # Different classes have different combinations of mu and beta activity
        for ch in range(n_channels):
            # Class-specific mu and beta power
            mu_amplitude = 0.5 + 0.3 * np.sin(2 * np.pi * class_label / n_classes)
            beta_amplitude = 0.5 + 0.3 * np.cos(2 * np.pi * class_label / n_classes)
            
            # Add oscillations
            mu_signal = mu_amplitude * np.sin(2 * np.pi * mu_freq * t)
            beta_signal = beta_amplitude * np.sin(2 * np.pi * beta_freq * t)
            
            # Channel-specific variations (some channels more sensitive)
            channel_weight = 0.5 + 0.5 * np.sin(2 * np.pi * ch / n_channels)
            
            # Combine signal and noise
            signal = channel_weight * (mu_signal + beta_signal)
            
            # Apply SNR
            signal_power = np.var(signal)
            noise_power = signal_power / (10 ** (snr_db / 10))
            noise_scaled = np.sqrt(noise_power / np.var(noise[ch])) * noise[ch]
            
            eeg_data[trial_idx, ch, :] = signal + noise_scaled
        
        if (trial_idx + 1) % 100 == 0:
            print(f"  Generated {trial_idx + 1}/{total_trials} trials")
    
    print("Synthetic data generation complete!")
    
    return eeg_data, labels


def save_synthetic_data(output_dir='./synthetic_data'):
    """
    Generate and save synthetic EEG data for testing.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate data
    eeg_data, labels = generate_synthetic_eeg_data(
        n_trials_per_class=50,
        n_channels=60,
        n_samples=5000,
        n_classes=11,
        snr_db=10,
        seed=42
    )
    
    # Save as numpy arrays
    np.save(output_path / 'eeg_data.npy', eeg_data)
    np.save(output_path / 'labels.npy', labels)
    
    print(f"\nSaved synthetic data to {output_path}")
    print(f"  eeg_data.npy: {eeg_data.shape}")
    print(f"  labels.npy: {labels.shape}")
    
    # Save metadata
    metadata = {
        'n_trials': len(labels),
        'n_channels': eeg_data.shape[1],
        'n_samples': eeg_data.shape[2],
        'n_classes': len(np.unique(labels)),
        'sampling_rate': 2500,
        'duration_sec': 2.0,
        'class_distribution': dict(zip(*np.unique(labels, return_counts=True))),
    }
    
    import json
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  metadata.json saved")
    
    return output_path


def visualize_synthetic_data(eeg_data, labels, save_dir='./synthetic_data'):
    """
    Create diagnostic plots for synthetic data.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating diagnostic plots...")
    
    # 1. Time series example
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    for i, class_idx in enumerate([0, 5, 10]):
        trial_idx = np.where(labels == class_idx)[0][0]
        channel = 0  # First channel
        
        t = np.arange(eeg_data.shape[2]) / 2500 * 1000  # ms
        axes[i].plot(t[:1000], eeg_data[trial_idx, channel, :1000])
        axes[i].set_ylabel(f'Class {class_idx}\nAmplitude (μV)')
        axes[i].grid(True, alpha=0.3)
    
    axes[2].set_xlabel('Time (ms)')
    axes[0].set_title('Example Time Series (Channel 1, First 400ms)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path / 'time_series.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved time_series.png")
    
    # 2. Frequency spectrum
    from scipy import signal as scipy_signal
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for class_idx in [0, 5, 10]:
        trial_idx = np.where(labels == class_idx)[0][0]
        channel = 0
        
        freqs, psd = scipy_signal.welch(
            eeg_data[trial_idx, channel, :],
            fs=2500,
            nperseg=1024
        )
        
        ax.semilogy(freqs, psd, label=f'Class {class_idx}', alpha=0.7)
    
    ax.set_xlim([0, 50])
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectral Density (μV²/Hz)')
    ax.set_title('Frequency Spectrum (Channel 1)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path / 'frequency_spectrum.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved frequency_spectrum.png")
    
    # 3. Class distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    
    unique, counts = np.unique(labels, return_counts=True)
    ax.bar(unique, counts, alpha=0.7)
    ax.set_xlabel('Class Label')
    ax.set_ylabel('Number of Trials')
    ax.set_title('Class Distribution', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path / 'class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved class_distribution.png")
    
    print("Diagnostic plots complete!")


# ============================================================================
# TEST RANDOM FOREST EXPERIMENT
# ============================================================================

def test_random_forest_experiment():
    """
    Run the Random Forest experiment on synthetic data to verify pipeline.
    """
    print("="*80)
    print("TESTING RANDOM FOREST EXPERIMENT WITH SYNTHETIC DATA")
    print("="*80)
    
    # Generate synthetic data
    data_path = save_synthetic_data('./test_data')
    
    # Load data for visualization
    eeg_data = np.load(data_path / 'eeg_data.npy')
    labels = np.load(data_path / 'labels.npy')
    
    # Visualize
    visualize_synthetic_data(eeg_data, labels, data_path)
    
    # Import the experiment module
    try:
        # Assuming random_forest_experiment.py is in the same directory
        import random_forest_experiment as rf_exp
        
        print("\n" + "="*80)
        print("RUNNING RANDOM FOREST EXPERIMENT")
        print("="*80)
        
        # Run experiment with reduced scope for testing
        # Temporarily modify config for faster testing
        original_configs = rf_exp.Config.RF_CONFIGS
        original_seeds = rf_exp.Config.RANDOM_SEEDS
        
        # Use only 2 configurations and 2 seeds for quick test
        rf_exp.Config.RF_CONFIGS = [
            {'n_estimators': 100, 'max_features': 'sqrt'},
            {'n_estimators': 100, 'max_features': 72},
        ]
        rf_exp.Config.RANDOM_SEEDS = [0, 1]
        rf_exp.Config.OUTPUT_DIR = Path('./test_results/random_forest')
        rf_exp.Config.FIGURE_DIR = rf_exp.Config.OUTPUT_DIR / 'figures'
        rf_exp.Config.DATA_DIR = rf_exp.Config.OUTPUT_DIR / 'data'
        
        # Run experiment
        rf_exp.run_experiment(data_path=str(data_path))
        
        # Restore original config
        rf_exp.Config.RF_CONFIGS = original_configs
        rf_exp.Config.RANDOM_SEEDS = original_seeds
        
        print("\n" + "="*80)
        print("TEST COMPLETE!")
        print("="*80)
        print(f"\nTest results saved to: ./test_results/random_forest")
        print("\nIf the test ran successfully, you can now run on real data:")
        print("  python random_forest_experiment.py --data_path /path/to/real/data")
        
    except ImportError as e:
        print(f"\nERROR: Could not import random_forest_experiment module")
        print(f"  {e}")
        print("\nMake sure random_forest_experiment.py is in the same directory")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR during experiment execution:")
        print(f"  {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    test_random_forest_experiment()