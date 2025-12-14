"""
Master Script for 1D-CNN EEG Classification
SYDE 522 Final Project

This script orchestrates the complete CNN experimental protocol:
1. Load and preprocess EEG data
2. Train CNN models (2, 3, 4 layers) with 5 random seeds each
3. Analyze results and generate figures
4. Export results for paper

Usage:
    python run_cnn_experiments.py --data_dir /path/to/data --subjects 1 2 3
"""

import argparse
import numpy as np
from pathlib import Path
import sys
import time

# Import custom modules
from cnn_data_loading import load_multiple_subjects, preprocess_for_cnn
from cnn_eeg_classification import run_cnn_experiments
from cnn_analysis import analyze_cnn_results


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Run 1D-CNN experiments for EEG-based movement classification'
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data',
        help='Directory containing EEG data'
    )
    
    parser.add_argument(
        '--subjects',
        type=int,
        nargs='+',
        default=list(range(1, 26)),  # All 25 subjects
        help='Subject IDs to include (default: all 25)'
    )
    
    parser.add_argument(
        '--session',
        type=int,
        default=1,
        choices=[1, 2, 3],
        help='Session number (default: 1)'
    )
    
    parser.add_argument(
        '--condition',
        type=str,
        default='real',
        choices=['real', 'imagined'],
        help='Movement condition (default: real)'
    )
    
    parser.add_argument(
        '--depths',
        type=int,
        nargs='+',
        default=[2, 3, 4],
        help='CNN depths to test (default: 2 3 4)'
    )
    
    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=[0, 1, 2, 3, 4],
        help='Random seeds for trials (default: 0 1 2 3 4)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/cnn',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--skip_training',
        action='store_true',
        help='Skip training and only run analysis on existing results'
    )
    
    parser.add_argument(
        '--verbose',
        type=int,
        default=1,
        choices=[0, 1, 2],
        help='Verbosity level (0=quiet, 1=normal, 2=detailed)'
    )
    
    parser.add_argument(
        '--use_synthetic',
        action='store_true',
        help='Use synthetic data for testing (no real data needed)'
    )
    
    return parser.parse_args()


def generate_synthetic_data(n_trials: int = 1100,
                           n_channels: int = 60,
                           n_timepoints: int = 5000,
                           n_classes: int = 11) -> tuple:
    """
    Generate synthetic EEG data for testing
    
    Args:
        n_trials: Number of trials (default: 100 per class)
        n_channels: Number of EEG channels
        n_timepoints: Samples per trial
        n_classes: Number of movement classes
        
    Returns:
        (X, y) where X is (n_trials, n_channels, n_timepoints), y is (n_trials,)
    """
    print(f"\n{'GENERATING SYNTHETIC DATA':-^60}")
    print(f"This is for TESTING ONLY - use real data for actual experiments")
    
    np.random.seed(42)
    
    # Generate random data with some structure
    X = np.random.randn(n_trials, n_channels, n_timepoints) * 10
    
    # Add class-specific patterns (simple simulation)
    for i in range(n_classes):
        class_mask = np.arange(i, n_trials, n_classes)
        # Add a simple sinusoidal pattern specific to each class
        freq = 10 + i  # Different frequency for each class
        t = np.linspace(0, 2, n_timepoints)
        pattern = 5 * np.sin(2 * np.pi * freq * t)
        X[class_mask, :, :] += pattern[np.newaxis, np.newaxis, :]
    
    # Generate balanced labels
    y = np.repeat(np.arange(n_classes), n_trials // n_classes)
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(y))
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    print(f"Generated {n_trials} trials × {n_channels} channels × {n_timepoints} samples")
    print(f"Classes: {n_classes}, balanced with {n_trials//n_classes} trials each")
    print(f"Data range: [{X.min():.2f}, {X.max():.2f}]")
    
    return X, y


def main():
    """Main experimental pipeline"""
    args = parse_arguments()
    
    print(f"\n{'='*70}")
    print(f"{'1D-CNN EEG CLASSIFICATION EXPERIMENTS':^70}")
    print(f"{'SYDE 522 Final Project':^70}")
    print(f"{'='*70}\n")
    
    # Print configuration
    print(f"{'Configuration':-^70}")
    print(f"Data directory: {args.data_dir}")
    print(f"Subjects: {args.subjects} (n={len(args.subjects)})")
    print(f"Session: {args.session}")
    print(f"Condition: {args.condition.upper()}")
    print(f"CNN depths: {args.depths}")
    print(f"Random seeds: {args.seeds}")
    print(f"Output directory: {args.output_dir}")
    print(f"Skip training: {args.skip_training}")
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load and preprocess data
    if not args.skip_training:
        print(f"\n{'STEP 1: DATA LOADING & PREPROCESSING':-^70}")
        
        if args.use_synthetic:
            # Use synthetic data for testing
            X, y = generate_synthetic_data()
        else:
            # Load real data
            try:
                X, y = load_multiple_subjects(
                    subject_ids=args.subjects,
                    session=args.session,
                    condition=args.condition,
                    data_dir=args.data_dir,
                    preprocess=True
                )
            except Exception as e:
                print(f"\n{'ERROR':-^70}")
                print(f"Failed to load data: {e}")
                print(f"\nTry using --use_synthetic flag to test with synthetic data")
                sys.exit(1)
        
        # Verify data shape
        expected_channels = 60
        expected_timepoints = 5000  # 2 seconds at 2500 Hz
        
        if X.shape[1] != expected_channels or X.shape[2] != expected_timepoints:
            print(f"\n{'WARNING':-^70}")
            print(f"Unexpected data shape: {X.shape}")
            print(f"Expected: (n_trials, {expected_channels}, {expected_timepoints})")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                sys.exit(1)
        
        # Step 2: Run CNN experiments
        print(f"\n{'STEP 2: CNN TRAINING':-^70}")
        print(f"Total experiments: {len(args.depths)} architectures × {len(args.seeds)} seeds = {len(args.depths)*len(args.seeds)} trials")
        print(f"Estimated time: ~{len(args.depths)*len(args.seeds)*10} minutes (10 min/trial average)")
        print(f"\nStarting experiments...")
        
        start_time = time.time()
        
        results = run_cnn_experiments(
            X=X,
            y=y,
            depths=args.depths,
            seeds=args.seeds,
            test_size=0.15,
            val_size=0.15,
            output_dir=args.output_dir,
            verbose=args.verbose
        )
        
        elapsed = time.time() - start_time
        print(f"\n{'Training completed in':-^70}")
        print(f"{elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
    
    # Step 3: Analyze results
    print(f"\n{'STEP 3: RESULTS ANALYSIS':-^70}")
    
    results_file = output_path / 'all_trials.pkl'
    if not results_file.exists():
        print(f"ERROR: No results found at {results_file}")
        print(f"Run training first (without --skip_training flag)")
        sys.exit(1)
    
    aggregated = analyze_cnn_results(
        results_file=str(results_file),
        output_dir=args.output_dir
    )
    
    # Print final summary
    print(f"\n{'='*70}")
    print(f"{'EXPERIMENTS COMPLETE!':^70}")
    print(f"{'='*70}")
    print(f"\nResults saved to: {output_path}")
    print(f"Figures saved to: figures/")
    print(f"\nKey files:")
    print(f"  - all_trials.pkl: Raw trial data")
    print(f"  - aggregated_results.json: Summary statistics")
    print(f"  - figures/cnn_*.png: Visualization figures")
    
    # Best architecture
    best_arch = max(aggregated, key=lambda x: x['test_acc_mean'])
    print(f"\nBest architecture: {best_arch['architecture'].upper()}")
    print(f"  Test accuracy: {best_arch['test_acc_mean']*100:.2f}% "
          f"[{best_arch['test_acc_ci_low']*100:.2f}%, {best_arch['test_acc_ci_high']*100:.2f}%]")
    print(f"  Test F1 score: {best_arch['test_f1_mean']*100:.2f}%")
    print(f"  Parameters: {best_arch['n_parameters']:,}")
    
    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()
