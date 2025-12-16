"""
CSP+LDA Experiment for EEG-Based Upper-Limb Movement Classification
SYDE 522 - Foundations of Artificial Intelligence
Author: Olivia Zheng

GPU-ACCELERATED VERSION with CHECKPOINT SUPPORT for Lambda Labs A100

This script implements the Common Spatial Patterns + Linear Discriminant Analysis
pipeline with GPU acceleration for filtering operations and checkpoint functionality
to resume from interruptions.
"""

import numpy as np
import pandas as pd
import pickle
import time
import gc
import psutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    from cupyx.scipy import signal as cp_signal
    GPU_AVAILABLE = cp.cuda.is_available()
    if GPU_AVAILABLE:
        print("✓ GPU detected - CuPy enabled")
        print(f"  Device: {cp.cuda.Device().compute_capability}")
        mempool = cp.get_default_memory_pool()
        print(f"  VRAM: {mempool.total_bytes() / 1e9:.1f} GB used")
except ImportError:
    GPU_AVAILABLE = False
    print("⚠ CuPy not found - falling back to CPU")
    print("  Install with: pip install cupy-cuda12x")

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

# MNE for CSP
from mne.decoding import CSP

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)

# Configure matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class CSPLDAExperiment:
    """GPU-accelerated CSP+LDA experiments with multiple configurations and checkpoint support."""
    
    def __init__(self, data_path: str, output_dir: str = './results_csp_lda', 
                 use_gpu: bool = True):
        """
        Initialize experiment.
        
        Parameters:
        -----------
        data_path : str
            Path to preprocessed data file (.npz or .pkl)
        output_dir : str
            Directory to save results
        use_gpu : bool
            Whether to use GPU acceleration (if available)
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Checkpoint file
        self.checkpoint_file = self.output_dir / 'experiment_checkpoint.pkl'
        
        # GPU settings
        self.use_gpu = use_gpu and GPU_AVAILABLE
        if self.use_gpu:
            print(f"\n{'='*70}")
            print("GPU ACCELERATION ENABLED")
            print(f"{'='*70}")
            # Set CuPy to use device 0 (A100)
            cp.cuda.Device(0).use()
            print(f"Active GPU: {cp.cuda.Device()}")
            print(f"{'='*70}\n")
        else:
            print("\nRunning on CPU\n")
        
        # Experimental parameters
        # SIMPLIFIED: Focus on combined band (8-30 Hz) with component exploration
        self.frequency_bands = {
            'combined': (8, 30)  # Standard motor BCI band (mu + beta)
        }
        self.n_components_list = [4]  # Run only 4 components
        self.random_seeds = [0, 1, 2]  # 3 trials per config (statistically sufficient)
        self.n_trials = len(self.random_seeds)
        
        # Data split ratios
        self.train_ratio = 0.70
        self.val_ratio = 0.15
        self.test_ratio = 0.15
        
        # Results storage
        self.trial_results = []
        self.config_summaries = []
        
        # Movement class names (0-10 after rest removal)
        self.class_names = [
            'Forward', 'Backward', 'Left', 'Right', 'Up', 'Down',
            'Power Grasp', 'Precision Grasp', 'Lateral Grasp',
            'Pronation', 'Supination'
        ]
    
    def check_ram_usage(self, context: str = ""):
        """Monitor RAM usage and warn if getting too high."""
        mem = psutil.virtual_memory()
        used_gb = mem.used / 1e9
        total_gb = mem.total / 1e9
        percent = mem.percent
        
        status = "✓" if percent < 80 else "⚠" if percent < 90 else "❌"
        
        print(f"  {status} RAM: {used_gb:.1f}/{total_gb:.1f} GB ({percent:.1f}%) {context}")
        
        if percent > 90:
            print(f"  ❌ CRITICAL: RAM usage very high!")
            print(f"     Consider reducing batch size or using swap space")
        elif percent > 80:
            print(f"  ⚠ WARNING: RAM usage high")
        
        return percent
        
    def save_checkpoint(self, trial_counter: int, completed_trials: Dict):
        """
        Save experiment checkpoint.
        
        Parameters:
        -----------
        trial_counter : int
            Current trial number
        completed_trials : dict
            Dictionary mapping trial keys to results
        """
        try:
            checkpoint_data = {
                'trial_counter': trial_counter,
                'completed_trials': completed_trials,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'n_completed': len(completed_trials)
            }
            
            with open(self.checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            print(f"  ✓ Checkpoint saved: {len(completed_trials)} trials completed")
            
        except Exception as e:
            print(f"  ⚠ Warning: Could not save checkpoint: {e}")
    
    def load_checkpoint(self) -> Tuple[int, Dict, bool]:
        """
        Load experiment checkpoint.
        
        Returns:
        --------
        trial_counter : int
            Last completed trial number
        completed_trials : dict
            Dictionary of completed trials
        success : bool
            Whether checkpoint was loaded successfully
        """
        if not self.checkpoint_file.exists():
            return 0, {}, False
        
        try:
            with open(self.checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            trial_counter = checkpoint_data['trial_counter']
            completed_trials = checkpoint_data['completed_trials']
            timestamp = checkpoint_data.get('timestamp', 'unknown')
            
            print(f"\n{'='*60}")
            print(f"CHECKPOINT FOUND")
            print(f"{'='*60}")
            print(f"  Timestamp: {timestamp}")
            print(f"  Completed trials: {len(completed_trials)}")
            print(f"  Last trial: {trial_counter}")
            print(f"{'='*60}\n")
            
            return trial_counter, completed_trials, True
            
        except Exception as e:
            print(f"  ⚠ Warning: Could not load checkpoint: {e}")
            print(f"  Starting from scratch...")
            return 0, {}, False
    
    def clear_checkpoint(self):
        """Remove checkpoint file to start fresh."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            print("  ✓ Checkpoint cleared")
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load preprocessed EEG data.
        
        Returns:
        --------
        X : np.ndarray
            EEG data (n_trials, n_channels, n_timepoints)
        y : np.ndarray
            Labels (n_trials,) with values 0-10 (rest class removed)
        """
        print("="*70)
        print("LOADING DATA")
        print("="*70)
        
        # Auto-detect .npz file if given a directory
        if self.data_path.is_dir():
            npz_file = self.data_path / 'csp_lda_data.npz'
            if not npz_file.exists():
                raise FileNotFoundError(
                    f"Directory provided but csp_lda_data.npz not found in {self.data_path}\n"
                    f"Please provide the full path to the .npz file."
                )
            print(f"Auto-detected file: {npz_file}")
            self.data_path = npz_file
        
        if self.data_path.suffix == '.npz':
            data = np.load(self.data_path, allow_pickle=True)
            X = data['X']  # Shape: (n_trials, n_channels, n_timepoints)
            y = data['y']  # Shape: (n_trials,)
            
            # Load metadata if available
            if 'metadata' in data:
                metadata = data['metadata'].item()
                print(f"\nMetadata found:")
                print(f"  Filter: {metadata.get('filter', 'unknown')}")
                print(f"  Sampling rate: {metadata.get('fs', 'unknown')} Hz")
                print(f"  Channel names: {len(metadata.get('channel_names', []))} channels")
                
        elif self.data_path.suffix == '.pkl':
            with open(self.data_path, 'rb') as f:
                data = pickle.load(f)
                X = data['X']
                y = data['y']
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
        
        print(f"\nRaw data loaded:")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  Data size: {X.nbytes / 1e9:.2f} GB")
        print(f"  Unique classes in y: {np.unique(y)}")
        
        # Warn about very large datasets
        if X.nbytes > 50e9:  # 50GB
            print(f"\n⚠ WARNING: Large dataset ({X.nbytes / 1e9:.1f} GB)")
            print(f"  This will take longer to process")
            print(f"  GPU filtering will use batching to manage memory")
            if self.use_gpu:
                print(f"  Estimated GPU batch size: ~100-200 trials at a time")
            else:
                print(f"  Consider using --no-gpu flag if GPU OOM errors occur")
        
        # Remove rest class (class 0)
        mask = y != 0
        X_filtered = X[mask]
        y_filtered = y[mask]
        
        # Renumber classes from 1-11 to 0-10
        y_filtered = y_filtered - 1
        
        print(f"\nAfter rest class removal:")
        print(f"  X shape: {X_filtered.shape}")
        print(f"  y shape: {y_filtered.shape}")
        print(f"  Number of classes: {len(np.unique(y_filtered))}")
        print(f"  Class range: {np.min(y_filtered)} to {np.max(y_filtered)}")
        
        # Class distribution check
        print(f"\n  Class distribution:")
        unique, counts = np.unique(y_filtered, return_counts=True)
        for cls, count in zip(unique, counts):
            pct = 100 * count / len(y_filtered)
            print(f"    Class {cls}: {count:5d} trials ({pct:5.2f}%)")
        
        # Sample check - print first 20 labels
        print(f"\n  First 20 labels: {y_filtered[:20]}")
        print(f"  Last 20 labels:  {y_filtered[-20:]}")
        
        # Verify we have 11 classes (0-10)
        assert len(np.unique(y_filtered)) == 11, \
            f"Expected 11 classes, got {len(np.unique(y_filtered))}"
        assert np.min(y_filtered) == 0 and np.max(y_filtered) == 10, \
            f"Expected classes 0-10, got {np.min(y_filtered)}-{np.max(y_filtered)}"
        
        # Check for class imbalance
        class_imbalance = np.max(counts) / np.min(counts)
        if class_imbalance > 3:
            print(f"\n  ⚠ WARNING: Significant class imbalance detected!")
            print(f"     Max/min ratio: {class_imbalance:.1f}x")
            print(f"     This may affect classification performance")
        
        print("\n✓ Data validation passed")
        print("="*70 + "\n")
        
        return X_filtered, y_filtered
    
    def bandpass_filter_gpu(self, X: np.ndarray, low_freq: float, 
                           high_freq: float, fs: float = 2500) -> np.ndarray:
        """
        GPU-accelerated bandpass filter with smart memory management.
        
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
        
        # Design filter on CPU (lightweight operation)
        b, a = butter(4, [low, high], btype='band')
        
        # Convert filter to GPU (small arrays)
        b_gpu = cp.asarray(b)
        a_gpu = cp.asarray(a)
        
        # Calculate optimal batch size based on GPU memory and data size
        # Each trial: n_channels × n_timepoints × 4 bytes (float32)
        bytes_per_trial = X.shape[1] * X.shape[2] * 4
        
        # Use ~10GB of GPU memory for batches (leave plenty of headroom)
        available_memory = 10 * 1024**3  # 10GB in bytes
        batch_size = max(1, int(available_memory // bytes_per_trial))
        batch_size = min(batch_size, 200)  # Cap at 200 for efficiency
        
        n_trials = X.shape[0]
        
        print(f"  Filtering {n_trials} trials on GPU")
        print(f"  Batch size: {batch_size} trials ({batch_size * bytes_per_trial / 1e9:.2f} GB per batch)")
        
        # Pre-allocate output on CPU
        X_filtered = np.zeros_like(X)
        
        # Process in batches - only load one batch to GPU at a time
        for batch_start in tqdm(range(0, n_trials, batch_size), 
                               desc="  GPU filtering", leave=False):
            batch_end = min(batch_start + batch_size, n_trials)
            
            # Load ONLY this batch to GPU
            batch_data = X[batch_start:batch_end]
            batch_gpu = cp.asarray(batch_data)
            batch_filtered_gpu = cp.zeros_like(batch_gpu)
            
            # Filter each trial in the batch
            for trial_idx in range(batch_gpu.shape[0]):
                for ch_idx in range(batch_gpu.shape[1]):
                    batch_filtered_gpu[trial_idx, ch_idx, :] = \
                        cp_signal.filtfilt(b_gpu, a_gpu, batch_gpu[trial_idx, ch_idx, :])
            
            # Transfer filtered batch back to CPU
            X_filtered[batch_start:batch_end] = cp.asnumpy(batch_filtered_gpu)
            
            # Clean up batch memory
            del batch_gpu, batch_filtered_gpu
            
            # Clear GPU cache every few batches
            if (batch_start // batch_size) % 5 == 0:
                cp.get_default_memory_pool().free_all_blocks()
        
        # Final cleanup
        del b_gpu, a_gpu
        cp.get_default_memory_pool().free_all_blocks()
        
        return X_filtered
    
    def bandpass_filter_cpu(self, X: np.ndarray, low_freq: float, 
                           high_freq: float, fs: float = 2500) -> np.ndarray:
        """
        CPU-based bandpass filter (fallback).
        
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
        
        print(f"  Filtering {X.shape[0]} trials on CPU...")
        
        for i in tqdm(range(X.shape[0]), desc="  CPU filtering", leave=False):
            for j in range(X.shape[1]):
                X_filtered[i, j, :] = filtfilt(b, a, X[i, j, :])
        
        return X_filtered
    
    def bandpass_filter(self, X: np.ndarray, low_freq: float, 
                       high_freq: float, fs: float = 2500) -> np.ndarray:
        """
        Apply bandpass filter (GPU or CPU based on availability).
        
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
        start_time = time.time()
        
        if self.use_gpu:
            X_filtered = self.bandpass_filter_gpu(X, low_freq, high_freq, fs)
        else:
            X_filtered = self.bandpass_filter_cpu(X, low_freq, high_freq, fs)
        
        elapsed = time.time() - start_time
        print(f"  Filtering completed in {elapsed:.2f}s "
              f"({'GPU' if self.use_gpu else 'CPU'})")
        
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
            Labels (0-10)
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
        n_classes = 11  # We have 11 classes (0-10)
        
        train_dist = np.bincount(y_train.astype(int), minlength=n_classes) / len(y_train)
        val_dist = np.bincount(y_val.astype(int), minlength=n_classes) / len(y_val)
        test_dist = np.bincount(y_test.astype(int), minlength=n_classes) / len(y_test)
        
        # Check train vs test
        max_diff_train_test = np.max(np.abs(train_dist - test_dist))
        max_diff_val_test = np.max(np.abs(val_dist - test_dist))
        
        print(f"  Stratification check:")
        print(f"    Max diff (train vs test): {max_diff_train_test:.4f}")
        print(f"    Max diff (val vs test):   {max_diff_val_test:.4f}")
        
        if max_diff_train_test > 0.02 or max_diff_val_test > 0.02:
            print(f"    ⚠ Warning: Stratification may not be perfect (tolerance: 0.02)")
        else:
            print(f"    ✓ Stratification verified (within 2% tolerance)")
    
    def train_csp_lda(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray,
                     X_test: np.ndarray, y_test: np.ndarray,
                     n_components: int, seed: int) -> Dict:
        """
        Train CSP+LDA pipeline with AGGRESSIVE MEMORY MANAGEMENT.
        
        Parameters:
        -----------
        X_train, X_val, X_test : np.ndarray
            EEG data (n_trials, n_channels, n_timepoints)
        y_train, y_val, y_test : np.ndarray
            Labels (n_trials,) with values 0-10
        n_components : int
            Number of CSP components per binary classifier
        seed : int
            Random seed
            
        Returns:
        --------
        results : dict
            Training results and metrics
        """
        # Set numpy random seed for reproducibility
        np.random.seed(seed)
        
        # Initialize CSP
        csp = CSP(
            n_components=n_components,
            reg=None,
            log=True,  # Apply log transform to variance
            norm_trace=False
        )
        
        # Measure training time
        start_time = time.time()
        
        # Check RAM before fitting
        self.check_ram_usage("before CSP fit")
        
        # Fit CSP on training data
        print("  Fitting CSP...")
        X_train_csp = csp.fit_transform(X_train, y_train)
        
        # CRITICAL: Delete original training data immediately
        del X_train
        gc.collect()
        self.check_ram_usage("after CSP fit")
        
        # Transform validation data IN BATCHES to avoid RAM spike
        print("  Transforming validation data...")
        batch_size = 1000  # Process 1000 trials at a time
        n_val = X_val.shape[0]
        n_features = X_train_csp.shape[1]
        
        X_val_csp = np.zeros((n_val, n_features), dtype=np.float32)
        
        for i in range(0, n_val, batch_size):
            end_idx = min(i + batch_size, n_val)
            X_val_csp[i:end_idx] = csp.transform(X_val[i:end_idx])
        
        # Delete original val data
        del X_val
        gc.collect()
        
        # Transform test data IN BATCHES
        print("  Transforming test data...")
        n_test = X_test.shape[0]
        X_test_csp = np.zeros((n_test, n_features), dtype=np.float32)
        
        for i in range(0, n_test, batch_size):
            end_idx = min(i + batch_size, n_test)
            X_test_csp[i:end_idx] = csp.transform(X_test[i:end_idx])
        
        # Delete original test data
        del X_test
        gc.collect()
        self.check_ram_usage("after CSP transform")
        
        # Initialize and train LDA
        print("  Training LDA...")
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train_csp, y_train)
        
        training_time = time.time() - start_time
        
        # Make predictions
        print("  Making predictions...")
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
            
            # Model artifacts (store as float32 to save memory)
            'csp_patterns': csp.patterns_.astype(np.float32),
            'csp_filters': csp.filters_.astype(np.float32),
            'lda_coef': lda.coef_.astype(np.float32),
            'lda_intercept': lda.intercept_.astype(np.float32),
        }
        
        # Clean up CSP features
        del X_train_csp, X_val_csp, X_test_csp
        del y_train_pred, y_val_pred, y_test_pred
        gc.collect()
        self.check_ram_usage("after LDA training")
        
        return results
    
    def run_single_trial(self, X: np.ndarray, y: np.ndarray,
                        band_name: str, n_components: int, 
                        seed: int) -> Dict:
        """
        Run a single experimental trial with MEMORY-EFFICIENT processing.
        
        Parameters:
        -----------
        X : np.ndarray
            ALREADY FILTERED EEG data (8-30 Hz from preprocessing)
        y : np.ndarray
            Labels (0-10)
        band_name : str
            Frequency band (ignored - data already filtered)
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
        
        # Check initial RAM
        self.check_ram_usage("at trial start")
        
        # DATA ALREADY FILTERED to 8-30 Hz during preprocessing!
        # Skip redundant filtering step
        print(f"Using preprocessed data (already filtered to 8-30 Hz)")
        X_filtered = X  # No filtering needed!
        
        # Split data
        print("Splitting data...")
        X_train, X_val, X_test, y_train, y_val, y_test = \
            self.stratified_split(X_filtered, y, seed)
        
        print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, "
              f"Test: {X_test.shape[0]}")
        
        # Check RAM after split
        self.check_ram_usage("after data split")
        
        # Train model (this function now handles its own memory management)
        print("Training CSP+LDA...")
        results = self.train_csp_lda(
            X_train, y_train, X_val, y_val, X_test, y_test,
            n_components, seed
        )
        
        # Clean up split data (train_csp_lda already deleted X_train, X_val, X_test)
        del y_train, y_val, y_test
        gc.collect()
        
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
        
        # Final RAM check
        self.check_ram_usage("at trial end")
        
        # GPU memory status
        if self.use_gpu:
            mempool = cp.get_default_memory_pool()
            print(f"  GPU memory: {mempool.used_bytes() / 1e9:.2f} GB used, "
                  f"{mempool.total_bytes() / 1e9:.2f} GB total")
        
        # Force garbage collection before next trial
        gc.collect()
        
        return trial_data
    
    def run_all_experiments(self, X: np.ndarray, y: np.ndarray, resume: bool = True):
        """
        Run all experimental configurations with checkpoint support.
        
        Parameters:
        -----------
        X : np.ndarray
            Raw EEG data
        y : np.ndarray
            Labels (0-10)
        resume : bool
            Whether to resume from checkpoint if available
        """
        # Try to load checkpoint
        trial_counter, completed_trials, checkpoint_loaded = self.load_checkpoint()
        
        if not resume:
            print("  Ignoring checkpoint (resume=False)")
            trial_counter = 0
            completed_trials = {}
            checkpoint_loaded = False
        
        # Set up experiment configurations
        configs = []
        for band_name in self.frequency_bands.keys():
            for n_components in self.n_components_list:
                for seed in self.random_seeds:
                    configs.append({
                        'band': band_name,
                        'n_components': n_components,
                        'seed': seed
                    })
        
        total_trials = len(configs)
        
        print(f"\n{'='*60}")
        print(f"Starting CSP+LDA Experiments")
        print(f"{'='*60}")
        print(f"Total configurations: {len(self.frequency_bands) * len(self.n_components_list)}")
        print(f"Trials per configuration: {self.n_trials}")
        print(f"Total trials: {total_trials}")
        print(f"Already completed: {len(completed_trials)}")
        print(f"Remaining: {total_trials - len(completed_trials)}")
        print(f"GPU acceleration: {'ENABLED' if self.use_gpu else 'DISABLED'}")
        print(f"{'='*60}\n")
        
        if len(completed_trials) >= total_trials:
            print("✓ All trials already completed!")
            # Load results from completed trials
            self.trial_results = list(completed_trials.values())
            return
        
        overall_start = time.time()
        
        for trial_idx, config in enumerate(configs, 1):
            # Create unique trial key
            trial_key = f"{config['band']}_{config['n_components']}_seed{config['seed']}"
            
            # Skip if already completed
            if trial_key in completed_trials:
                print(f"\n[Trial {trial_idx}/{total_trials}] - SKIPPED (already completed)")
                print(f"  Configuration: {trial_key}")
                continue
            
            print(f"\n[Trial {trial_idx}/{total_trials}]")
            
            try:
                # Run trial
                trial_data = self.run_single_trial(
                    X, y, 
                    config['band'], 
                    config['n_components'], 
                    config['seed']
                )
                
                # Store result
                completed_trials[trial_key] = trial_data
                self.trial_results.append(trial_data)
                trial_counter = trial_idx
                
                # Save checkpoint after each successful trial
                self.save_checkpoint(trial_counter, completed_trials)
                
            except Exception as e:
                print(f"\n❌ ERROR in trial {trial_idx}: {e}")
                print(f"  Configuration: {trial_key}")
                print(f"  Checkpoint saved up to trial {trial_counter}")
                print(f"  You can resume from this point")
                raise
        
        overall_time = time.time() - overall_start
        
        print(f"\n{'='*60}")
        print("All trials completed!")
        print(f"Total time: {overall_time:.2f}s ({overall_time/60:.1f} min)")
        print(f"Average per trial: {overall_time/total_trials:.2f}s")
        print(f"{'='*60}\n")
        
        # Clear checkpoint after successful completion
        if len(completed_trials) >= total_trials:
            print("✓ Clearing checkpoint (all trials complete)")
            self.clear_checkpoint()
    
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
        ax.set_xlabel('Number of CSP Components', fontsize=12, fontweight='bold')
        ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('CSP+LDA: Component Count Comparison (8-30 Hz Band)',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'{s["n_components"]}' for s in self.config_summaries], fontsize=10)
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
    
    def plot_component_ablation(self):
        """Plot 2: Component count ablation study."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get data for all component counts
        n_comp_list = self.n_components_list
        
        means = []
        ci_lows = []
        ci_highs = []
        
        for n_comp in n_comp_list:
            summary = next(s for s in self.config_summaries 
                         if s['n_components'] == n_comp)
            means.append(summary['test_acc_mean'] * 100)
            ci_lows.append(summary['test_acc_ci_low'] * 100)
            ci_highs.append(summary['test_acc_ci_high'] * 100)
        
        means = np.array(means)
        ci_lows = np.array(ci_lows)
        ci_highs = np.array(ci_highs)
        
        yerr_low = means - ci_lows
        yerr_high = ci_highs - means
        
        # Create line plot with error bars
        x = np.array(n_comp_list)
        
        ax.errorbar(x, means, yerr=[yerr_low, yerr_high],
                   fmt='o-', linewidth=2.5, markersize=10,
                   color='steelblue', ecolor='black',
                   capsize=8, capthick=2, label='Test Accuracy')
        
        # Add value labels
        for xi, mean in zip(x, means):
            ax.text(xi, mean + 2, f'{mean:.1f}%', 
                   ha='center', va='bottom',
                   fontsize=11, fontweight='bold')
        
        # Formatting
        ax.set_xlabel('Number of CSP Components', fontsize=12, fontweight='bold')
        ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('CSP+LDA: Effect of Component Count on Classification Accuracy',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim([0, 100])
        ax.legend(fontsize=10, loc='lower right')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plot2_component_ablation.png',
                   dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'plot2_component_ablation.pdf',
                   bbox_inches='tight')
        print("✓ Saved plot 2: Component count ablation")
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
        """Plot 4: Per-class F1 comparison across component counts."""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        n_comp_list = self.n_components_list
        
        x = np.arange(len(self.class_names))
        width = 0.8 / len(n_comp_list)  # Dynamic width based on number of components
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(n_comp_list)))
        
        # Plot bars for each component count
        for i, (n_comp, color) in enumerate(zip(n_comp_list, colors)):
            config = next(s for s in self.config_summaries 
                         if s['n_components'] == n_comp)
            means = config['test_f1_per_class_mean']
            stds = config['test_f1_per_class_std']
            
            offset = width * (i - (len(n_comp_list) - 1) / 2)
            bars = ax.bar(x + offset, means, width, 
                         label=f"{n_comp} components",
                         color=color, alpha=0.7, edgecolor='black', 
                         linewidth=1)
            ax.errorbar(x + offset, means, yerr=stds, fmt='none',
                       ecolor='black', capsize=3, capthick=1.5, alpha=0.7)
        
        # Formatting
        ax.set_xlabel('Movement Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
        ax.set_title('Per-Class F1 Score: Effect of CSP Component Count',
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
        self.plot_component_ablation()  # Updated function name
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
        description='GPU-Accelerated CSP+LDA Experiment for EEG Movement Classification'
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
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU acceleration (use CPU only)'
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Start from scratch (ignore checkpoint)'
    )
    parser.add_argument(
        '--clear-checkpoint',
        action='store_true',
        help='Clear checkpoint and exit'
    )
    
    args = parser.parse_args()
    
    # Initialize experiment
    experiment = CSPLDAExperiment(
        data_path=args.data_path,
        output_dir=args.output_dir,
        use_gpu=not args.no_gpu
    )
    
    # Clear checkpoint if requested
    if args.clear_checkpoint:
        experiment.clear_checkpoint()
        print("Checkpoint cleared. Exiting.")
        return
    
    # Load data
    X, y = experiment.load_data()
    
    # Run all experiments (with resume support)
    experiment.run_all_experiments(X, y, resume=not args.no_resume)
    
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
    
    # Final GPU cleanup
    if experiment.use_gpu:
        cp.get_default_memory_pool().free_all_blocks()
        print("\nGPU memory freed")


if __name__ == '__main__':
    main()