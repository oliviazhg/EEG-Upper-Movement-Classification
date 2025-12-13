"""
Test script for CSP+LDA experiment using synthetic data.
This allows you to verify the pipeline works before running on real data.
"""

import numpy as np
from pathlib import Path

def generate_synthetic_eeg_data(n_trials=500, n_channels=60, n_timepoints=5000, 
                               n_classes=11, noise_level=1.0):
    """
    Generate synthetic EEG-like data for testing.
    
    Parameters:
    -----------
    n_trials : int
        Number of trials
    n_channels : int
        Number of EEG channels
    n_timepoints : int
        Number of time samples per trial
    n_classes : int
        Number of movement classes
    noise_level : float
        Amount of noise to add
        
    Returns:
    --------
    X : np.ndarray
        Synthetic EEG data (n_trials, n_channels, n_timepoints)
    y : np.ndarray
        Class labels (n_trials,)
    """
    print("Generating synthetic EEG data...")
    
    # Initialize
    X = np.zeros((n_trials, n_channels, n_timepoints))
    y = np.zeros(n_trials, dtype=int)
    
    # Sampling frequency
    fs = 2500  # Hz
    t = np.arange(n_timepoints) / fs
    
    # Generate trials for each class
    trials_per_class = n_trials // n_classes
    
    for class_idx in range(n_classes):
        start_idx = class_idx * trials_per_class
        end_idx = start_idx + trials_per_class
        
        # Class-specific frequency components
        # Different classes have different dominant frequencies
        if class_idx < 6:  # Directional reaches
            # More mu band activity (8-12 Hz)
            dominant_freq = 8 + (class_idx % 6) * 0.5
        elif class_idx < 9:  # Grasps
            # More beta band activity (13-30 Hz)
            dominant_freq = 15 + (class_idx % 3) * 3
        else:  # Wrist rotations
            # Mixed activity
            dominant_freq = 20 + (class_idx % 2) * 5
        
        for trial_idx in range(start_idx, end_idx):
            # Generate oscillatory activity with class-specific frequency
            for ch in range(n_channels):
                # Base oscillation
                signal = np.sin(2 * np.pi * dominant_freq * t + 
                              np.random.rand() * 2 * np.pi)
                
                # Add harmonics
                signal += 0.5 * np.sin(2 * np.pi * dominant_freq * 2 * t + 
                                      np.random.rand() * 2 * np.pi)
                
                # Add slower drift
                signal += 0.3 * np.sin(2 * np.pi * 2 * t + 
                                      np.random.rand() * 2 * np.pi)
                
                # Add noise
                signal += noise_level * np.random.randn(n_timepoints)
                
                # Channel-specific amplitude (simulate spatial patterns)
                spatial_weight = 1.0 + 0.5 * np.sin(ch / n_channels * 2 * np.pi)
                
                X[trial_idx, ch, :] = signal * spatial_weight
            
            y[trial_idx] = class_idx
    
    # Shuffle trials
    shuffle_idx = np.random.permutation(n_trials)
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    print(f"Generated data: X shape = {X.shape}, y shape = {y.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    return X, y


def create_test_data():
    """Create and save synthetic test data."""
    
    # Generate data
    X, y = generate_synthetic_eeg_data(
        n_trials=500,      # Smaller dataset for faster testing
        n_channels=60,     # Match real dataset
        n_timepoints=5000, # 2 seconds at 2500 Hz
        n_classes=11,      # 11 movement classes
        noise_level=1.0
    )
    
    # Create data directory
    data_dir = Path('./test_data')
    data_dir.mkdir(exist_ok=True)
    
    # Save as .npz
    output_path = data_dir / 'synthetic_eeg_data.npz'
    np.savez(output_path, X=X, y=y)
    print(f"\nTest data saved to: {output_path}")
    
    return output_path


def run_quick_test():
    """Run a quick test with reduced parameters."""
    print("="*60)
    print("Running Quick Test with Synthetic Data")
    print("="*60)
    
    # Create synthetic data
    data_path = create_test_data()
    
    # Import the experiment class
    from csp_lda_experiment import CSPLDAExperiment
    
    # Initialize with reduced parameters for quick testing
    experiment = CSPLDAExperiment(
        data_path=str(data_path),
        output_dir='./test_results_csp_lda'
    )
    
    # Reduce to just 1 trial per config for quick test
    print("\nReducing to 1 trial per config for quick testing...")
    experiment.random_seeds = [0]  # Just 1 seed
    experiment.n_trials = 1
    
    # Load data
    X, y = experiment.load_data()
    
    # Run experiments
    experiment.run_all_experiments(X, y)
    
    # Aggregate results
    experiment.aggregate_results()
    
    # Save results
    experiment.save_results()
    
    # Generate plots
    experiment.generate_all_plots()
    
    # Print summary
    experiment.print_summary_table()
    
    print("\n" + "="*60)
    print("QUICK TEST COMPLETE!")
    print("="*60)
    print(f"Results saved to: {experiment.output_dir}")
    print("\nNote: This was a quick test with synthetic data and only 1 trial.")
    print("For real experiments, use 5 trials per configuration.")


def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Test CSP+LDA pipeline with synthetic data'
    )
    parser.add_argument(
        '--generate-only',
        action='store_true',
        help='Only generate synthetic data without running experiment'
    )
    parser.add_argument(
        '--full-test',
        action='store_true',
        help='Run full test with all 5 trials (slower)'
    )
    
    args = parser.parse_args()
    
    if args.generate_only:
        create_test_data()
        print("\nTo run the experiment on this data:")
        print("python csp_lda_experiment.py --data_path ./test_data/synthetic_eeg_data.npz")
    
    elif args.full_test:
        print("Running FULL test (this will take longer)...\n")
        data_path = create_test_data()
        
        from csp_lda_experiment import CSPLDAExperiment
        experiment = CSPLDAExperiment(
            data_path=str(data_path),
            output_dir='./test_results_csp_lda'
        )
        
        X, y = experiment.load_data()
        experiment.run_all_experiments(X, y)
        experiment.aggregate_results()
        experiment.statistical_testing()
        experiment.save_results()
        experiment.generate_all_plots()
        experiment.print_summary_table()
        
    else:
        run_quick_test()


if __name__ == '__main__':
    main()