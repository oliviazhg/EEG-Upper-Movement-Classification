"""
Memory-Efficient CNN Experiment for EEG Movement Classification
SYDE 522 Final Project

Implements 3 CNN architectures with proper memory management:
- CNN-2L: 2 convolutional layers
- CNN-3L: 3 convolutional layers  
- CNN-4L: 4 convolutional layers

Features:
- Data generators to avoid loading all data at once
- Mixed precision training (float16)
- Automatic checkpointing
- 5 independent trials per architecture
- Proper train/val/test splits (70-15-15)
- Comprehensive metrics and logging

Usage:
    python cnn_experiment.py --data-path /path/to/cnn_data.npz --output-dir results/
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import os
import json
import gc
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Force TensorFlow to use only necessary GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU memory growth setting error: {e}")

# DISABLED: Mixed precision causing NaN loss with EEG data
# from tensorflow.keras import mixed_precision
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_global_policy(policy)
# print(f"Mixed precision enabled: {policy.name}")
print("Mixed precision DISABLED - using float32 for numerical stability")


class MemoryEfficientDataGenerator(keras.utils.Sequence):
    """
    Data generator that loads batches on-the-fly to minimize RAM usage.
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32, 
                 shuffle: bool = True, augment: bool = False):
        """
        Parameters:
        -----------
        X : np.ndarray
            Shape (n_trials, n_channels, n_samples)
        y : np.ndarray
            Shape (n_trials,) - class labels
        batch_size : int
            Batch size for training
        shuffle : bool
            Whether to shuffle data after each epoch
        augment : bool
            Whether to apply data augmentation
        """
        self.X = X.astype(np.float32)
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.n_samples = len(self.X)
        self.indices = np.arange(self.n_samples)
        
        # Compute global statistics for normalization (across all trials)
        # This preserves relative differences between trials
        self.global_mean = np.mean(self.X)
        self.global_std = np.std(self.X)
        
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        """Number of batches per epoch."""
        return int(np.ceil(self.n_samples / self.batch_size))
    
    def __getitem__(self, idx):
        """Generate one batch of data."""
        # Get batch indices
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        # Load batch data
        X_batch = self.X[batch_indices].copy()  # Copy to avoid modifying original
        y_batch = self.y[batch_indices]
        
        # Apply augmentation if enabled
        if self.augment:
            X_batch = self._augment_batch(X_batch)
        
        # Global normalization (preserves differences between trials)
        if self.global_std > 0:
            X_batch = (X_batch - self.global_mean) / self.global_std
        
        return X_batch, y_batch
    
    def _augment_batch(self, X_batch):
        """
        Apply simple augmentation: add small Gaussian noise.
        Helps with generalization.
        """
        noise = np.random.normal(0, 0.01, X_batch.shape).astype(np.float32)
        return X_batch + noise
    
    def on_epoch_end(self):
        """Shuffle indices after each epoch."""
        if self.shuffle:
            np.random.shuffle(self.indices)


class CNNArchitecture:
    """Factory for creating different CNN architectures."""
    
    @staticmethod
    def build_cnn_2layer(input_shape: Tuple[int, int], n_classes: int, 
                         dropout_rate: float = 0.5) -> keras.Model:
        """
        Build 2-layer CNN architecture.
        
        Architecture:
        - Conv1D(64, kernel=50) -> ReLU -> MaxPool(4) -> Dropout(0.5)
        - Conv1D(128, kernel=25) -> ReLU -> MaxPool(4) -> Dropout(0.5)
        - Flatten -> Dense(128) -> ReLU -> Dropout(0.5)
        - Dense(n_classes) -> Softmax
        """
        inputs = layers.Input(shape=input_shape)
        
        # Layer 1
        x = layers.Conv1D(64, kernel_size=50, activation='relu', padding='same')(inputs)
        x = layers.MaxPooling1D(pool_size=4)(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Layer 2
        x = layers.Conv1D(128, kernel_size=25, activation='relu', padding='same')(x)
        x = layers.MaxPooling1D(pool_size=4)(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Dense layers
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Output layer
        outputs = layers.Dense(n_classes, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='CNN-2L')
        return model
    
    @staticmethod
    def build_cnn_3layer(input_shape: Tuple[int, int], n_classes: int,
                         dropout_rate: float = 0.5) -> keras.Model:
        """
        Build 3-layer CNN architecture.
        
        Architecture:
        - Conv1D(64, kernel=50) -> ReLU -> MaxPool(4) -> Dropout(0.5)
        - Conv1D(128, kernel=25) -> ReLU -> MaxPool(4) -> Dropout(0.5)
        - Conv1D(256, kernel=10) -> ReLU -> MaxPool(2) -> Dropout(0.5)
        - Flatten -> Dense(256) -> ReLU -> Dropout(0.5)
        - Dense(n_classes) -> Softmax
        """
        inputs = layers.Input(shape=input_shape)
        
        # Layer 1
        x = layers.Conv1D(64, kernel_size=50, activation='relu', padding='same')(inputs)
        x = layers.MaxPooling1D(pool_size=4)(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Layer 2
        x = layers.Conv1D(128, kernel_size=25, activation='relu', padding='same')(x)
        x = layers.MaxPooling1D(pool_size=4)(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Layer 3
        x = layers.Conv1D(256, kernel_size=10, activation='relu', padding='same')(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Dense layers
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Output layer
        outputs = layers.Dense(n_classes, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='CNN-3L')
        return model
    
    @staticmethod
    def build_cnn_4layer(input_shape: Tuple[int, int], n_classes: int,
                         dropout_rate: float = 0.5) -> keras.Model:
        """
        Build 4-layer CNN architecture.
        
        Architecture:
        - Conv1D(64, kernel=50) -> ReLU -> MaxPool(4) -> Dropout(0.5)
        - Conv1D(128, kernel=25) -> ReLU -> MaxPool(4) -> Dropout(0.5)
        - Conv1D(256, kernel=10) -> ReLU -> MaxPool(2) -> Dropout(0.5)
        - Conv1D(512, kernel=5) -> ReLU -> MaxPool(2) -> Dropout(0.5)
        - Flatten -> Dense(512) -> ReLU -> Dropout(0.5)
        - Dense(n_classes) -> Softmax
        """
        inputs = layers.Input(shape=input_shape)
        
        # Layer 1
        x = layers.Conv1D(64, kernel_size=50, activation='relu', padding='same')(inputs)
        x = layers.MaxPooling1D(pool_size=4)(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Layer 2
        x = layers.Conv1D(128, kernel_size=25, activation='relu', padding='same')(x)
        x = layers.MaxPooling1D(pool_size=4)(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Layer 3
        x = layers.Conv1D(256, kernel_size=10, activation='relu', padding='same')(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Layer 4
        x = layers.Conv1D(512, kernel_size=5, activation='relu', padding='same')(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Dense layers
        x = layers.Flatten()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Output layer
        outputs = layers.Dense(n_classes, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='CNN-4L')
        return model


class CNNExperiment:
    """Main experiment runner for CNN architectures."""
    
    def __init__(self, data_path: str, output_dir: str, batch_size: int = 32,
                 max_epochs: int = 100, patience: int = 15):
        """
        Initialize experiment.
        
        Parameters:
        -----------
        data_path : str
            Path to preprocessed .npz file (should be cnn_data.npz)
        output_dir : str
            Directory to save results and checkpoints
        batch_size : int
            Batch size for training
        max_epochs : int
            Maximum epochs per trial
        patience : int
            Early stopping patience
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        
        # Create subdirectories
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.logs_dir = self.output_dir / 'logs'
        self.logs_dir.mkdir(exist_ok=True)
        self.plots_dir = self.output_dir / 'plots'
        self.plots_dir.mkdir(exist_ok=True)
        
        # Load and prepare data
        print(f"\n{'='*70}")
        print(f"LOADING DATA")
        print(f"{'='*70}")
        self._load_data()
        
        # Results storage
        self.results = {
            'CNN-2L': [],
            'CNN-3L': [],
            'CNN-4L': []
        }
    
    def _load_data(self):
        """Load and prepare data from preprocessed file."""
        print(f"Loading data from: {self.data_path}")
        
        data = np.load(self.data_path, allow_pickle=True)
        X = data['X']  # Shape: (n_trials, n_channels, n_samples)
        y = data['y']  # Shape: (n_trials,)
        
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  Data type: {X.dtype}")
        print(f"  Memory: {X.nbytes / 1e9:.2f} GB")
        
        # Store original for multiple trials
        self.X_full = X.astype(np.float32)  # Convert to float32 to save memory
        self.y_full = y
        self.n_classes = len(np.unique(y))
        self.input_shape = (X.shape[1], X.shape[2])  # (n_channels, n_samples)
        
        # CRITICAL: Check for NaN in input data
        print(f"\n  Checking data quality...")
        if np.any(np.isnan(X)):
            n_nan = np.sum(np.isnan(X))
            print(f"  ⚠️  WARNING: Found {n_nan} NaN values in input data!")
            print(f"     Replacing with zeros...")
            self.X_full = np.nan_to_num(self.X_full, nan=0.0)
        else:
            print(f"  ✓ No NaN values in input data")
        
        if np.any(np.isinf(X)):
            n_inf = np.sum(np.isinf(X))
            print(f"  ⚠️  WARNING: Found {n_inf} Inf values in input data!")
            print(f"     Clipping to finite range...")
            self.X_full = np.nan_to_num(self.X_full, posinf=1e6, neginf=-1e6)
        else:
            print(f"  ✓ No Inf values in input data")
        
        # Check data range
        x_min, x_max = self.X_full.min(), self.X_full.max()
        x_mean, x_std = self.X_full.mean(), self.X_full.std()
        print(f"  Data statistics:")
        print(f"    Range: [{x_min:.2e}, {x_max:.2e}]")
        print(f"    Mean: {x_mean:.2e}, Std: {x_std:.2e}")
        
        print(f"  Number of classes: {self.n_classes}")
        print(f"  Input shape: {self.input_shape}")
        
        # Class distribution
        unique, counts = np.unique(y, return_counts=True)
        print(f"\n  Class distribution:")
        for cls, count in zip(unique, counts):
            print(f"    Class {cls}: {count} trials ({count/len(y)*100:.1f}%)")
        
        # Free memory
        del data
        gc.collect()
    
    def _create_data_split(self, seed: int) -> Tuple:
        """
        Create stratified train/val/test split.
        
        Returns:
        --------
        X_train, X_val, X_test, y_train, y_val, y_test
        """
        # First split: 70% train, 30% temp
        X_train, X_temp, y_train, y_temp = train_test_split(
            self.X_full, self.y_full, 
            test_size=0.3, 
            stratify=self.y_full,
            random_state=seed
        )
        
        # Second split: 15% val, 15% test (50-50 split of temp)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=0.5,
            stratify=y_temp,
            random_state=seed
        )
        
        print(f"\n  Data split (seed={seed}):")
        print(f"    Train: {len(X_train)} trials ({len(X_train)/len(self.X_full)*100:.1f}%)")
        print(f"    Val:   {len(X_val)} trials ({len(X_val)/len(self.X_full)*100:.1f}%)")
        print(f"    Test:  {len(X_test)} trials ({len(X_test)/len(self.X_full)*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _create_generators(self, X_train, X_val, X_test, 
                          y_train, y_val, y_test) -> Tuple:
        """Create data generators for memory-efficient training."""
        train_gen = MemoryEfficientDataGenerator(
            X_train, y_train,
            batch_size=self.batch_size,
            shuffle=True,
            augment=True  # Enable augmentation for training
        )
        
        val_gen = MemoryEfficientDataGenerator(
            X_val, y_val,
            batch_size=self.batch_size,
            shuffle=False,
            augment=False
        )
        
        test_gen = MemoryEfficientDataGenerator(
            X_test, y_test,
            batch_size=self.batch_size,
            shuffle=False,
            augment=False
        )
        
        return train_gen, val_gen, test_gen
    
    def _build_model(self, architecture: str) -> keras.Model:
        """Build specified CNN architecture."""
        if architecture == 'CNN-2L':
            model = CNNArchitecture.build_cnn_2layer(
                self.input_shape, self.n_classes
            )
        elif architecture == 'CNN-3L':
            model = CNNArchitecture.build_cnn_3layer(
                self.input_shape, self.n_classes
            )
        elif architecture == 'CNN-4L':
            model = CNNArchitecture.build_cnn_4layer(
                self.input_shape, self.n_classes
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        # Compile model with gradient clipping for stability
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=0.001,
                clipnorm=1.0  # Clip gradients to prevent explosion
            ),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _create_callbacks(self, architecture: str, trial: int) -> List:
        """Create training callbacks."""
        checkpoint_path = self.checkpoint_dir / f"{architecture}_trial{trial}_best.h5"
        
        callback_list = [
            # Terminate on NaN loss (catch problems early)
            callbacks.TerminateOnNaN(),
            
            # Save best model
            callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            
            # Early stopping
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                mode='max',
                patience=self.patience,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            
            # TensorBoard logging
            callbacks.TensorBoard(
                log_dir=str(self.logs_dir / f"{architecture}_trial{trial}"),
                histogram_freq=0,  # Disable histograms to save memory
                write_graph=False
            ),
            
            # CSV logger
            callbacks.CSVLogger(
                str(self.logs_dir / f"{architecture}_trial{trial}_log.csv")
            )
        ]
        
        return callback_list
    
    def _evaluate_model(self, model: keras.Model, test_gen, 
                       y_test: np.ndarray) -> Dict:
        """Evaluate model and compute metrics."""
        # Get predictions
        y_pred_probs = model.predict(test_gen, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Compute metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Per-class metrics
        class_report = classification_report(
            y_test, y_pred, 
            output_dict=True,
            zero_division=0
        )
        
        metrics = {
            'accuracy': float(accuracy),
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted),
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report
        }
        
        return metrics
    
    def _train_single_trial(self, architecture: str, trial: int, seed: int) -> Dict:
        """Train and evaluate a single trial."""
        print(f"\n{'='*70}")
        print(f"{architecture} - TRIAL {trial+1}/5 (seed={seed})")
        print(f"{'='*70}")
        
        # Set random seeds for reproducibility
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        # Create data split
        X_train, X_val, X_test, y_train, y_val, y_test = self._create_data_split(seed)
        
        # Create generators
        train_gen, val_gen, test_gen = self._create_generators(
            X_train, X_val, X_test, y_train, y_val, y_test
        )
        
        # Build model
        print(f"\n  Building {architecture} model...")
        model = self._build_model(architecture)
        
        # Print model summary (only for first trial)
        if trial == 0:
            print(f"\n  Model architecture:")
            model.summary()
            print(f"\n  Total parameters: {model.count_params():,}")
        
        # Create callbacks
        callback_list = self._create_callbacks(architecture, trial)
        
        # Train model
        print(f"\n  Training...")
        start_time = datetime.now()
        
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=self.max_epochs,
            callbacks=callback_list,
            verbose=1
        )
        
        training_time = (datetime.now() - start_time).total_seconds()
        print(f"\n  Training completed in {training_time:.1f} seconds")
        
        # Evaluate on test set
        print(f"\n  Evaluating on test set...")
        metrics = self._evaluate_model(model, test_gen, y_test)
        
        print(f"\n  Results:")
        print(f"    Accuracy: {metrics['accuracy']:.4f}")
        print(f"    F1 (macro): {metrics['f1_macro']:.4f}")
        print(f"    F1 (weighted): {metrics['f1_weighted']:.4f}")
        
        # Save detailed results
        trial_results = {
            'trial': trial,
            'seed': seed,
            'architecture': architecture,
            'training_time': training_time,
            'epochs_trained': len(history.history['loss']),
            'metrics': metrics,
            'history': {
                'loss': [float(x) for x in history.history['loss']],
                'accuracy': [float(x) for x in history.history['accuracy']],
                'val_loss': [float(x) for x in history.history['val_loss']],
                'val_accuracy': [float(x) for x in history.history['val_accuracy']]
            }
        }
        
        # Save trial results
        results_path = self.logs_dir / f"{architecture}_trial{trial}_results.json"
        with open(results_path, 'w') as f:
            json.dump(trial_results, f, indent=2)
        
        # Clear memory
        del model, train_gen, val_gen, test_gen
        del X_train, X_val, X_test, y_train, y_val, y_test
        keras.backend.clear_session()
        gc.collect()
        
        return trial_results
    
    def run_experiments(self, architectures: Optional[List[str]] = None,
                       n_trials: int = 5, start_trial: int = 0):
        """
        Run experiments for all architectures.
        
        Parameters:
        -----------
        architectures : list, optional
            List of architectures to test. Default: all three
        n_trials : int
            Number of independent trials per architecture
        start_trial : int
            Which trial number to start from (default: 0)
            Use this to skip failed trials or continue after interruption
        """
        if architectures is None:
            architectures = ['CNN-2L', 'CNN-3L', 'CNN-4L']
        
        print(f"\n{'='*70}")
        print(f"CNN EXPERIMENTS - STARTING")
        print(f"{'='*70}")
        print(f"Architectures: {architectures}")
        print(f"Trials per architecture: {n_trials}")
        print(f"Starting from trial: {start_trial}")
        print(f"Total experiments: {len(architectures) * n_trials}")
        print(f"Batch size: {self.batch_size}")
        print(f"Max epochs: {self.max_epochs}")
        print(f"Early stopping patience: {self.patience}")
        print(f"{'='*70}\n")
        
        # Seeds for reproducibility - generate enough for all possible trials
        all_seeds = [42, 123, 456, 789, 1011, 2022, 3033, 4044, 5055, 6066]
        
        # Run experiments
        for arch in architectures:
            print(f"\n{'='*70}")
            print(f"ARCHITECTURE: {arch}")
            print(f"{'='*70}\n")
            
            for trial in range(start_trial, start_trial + n_trials):
                seed = all_seeds[trial % len(all_seeds)]
                trial_results = self._train_single_trial(arch, trial, seed)
                self.results[arch].append(trial_results)
                
                # Save intermediate aggregate results after each trial
                self._save_aggregate_results()
        
        # Final summary
        self._print_summary()
        self._plot_results()
        
        print(f"\n{'='*70}")
        print(f"ALL EXPERIMENTS COMPLETE")
        print(f"Results saved to: {self.output_dir}")
        print(f"{'='*70}\n")
    
    def _save_aggregate_results(self):
        """Save aggregate results across all trials."""
        aggregate_path = self.output_dir / 'aggregate_results.json'
        
        summary = {}
        for arch, trials in self.results.items():
            if not trials:
                continue
            
            # Compute statistics across trials
            accuracies = [t['metrics']['accuracy'] for t in trials]
            f1_macros = [t['metrics']['f1_macro'] for t in trials]
            
            summary[arch] = {
                'n_trials': len(trials),
                'accuracy': {
                    'mean': float(np.mean(accuracies)),
                    'std': float(np.std(accuracies)),
                    'min': float(np.min(accuracies)),
                    'max': float(np.max(accuracies)),
                    'trials': accuracies
                },
                'f1_macro': {
                    'mean': float(np.mean(f1_macros)),
                    'std': float(np.std(f1_macros)),
                    'min': float(np.min(f1_macros)),
                    'max': float(np.max(f1_macros)),
                    'trials': f1_macros
                }
            }
        
        with open(aggregate_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def _print_summary(self):
        """Print summary statistics across all trials."""
        print(f"\n{'='*70}")
        print(f"FINAL SUMMARY")
        print(f"{'='*70}\n")
        
        for arch in ['CNN-2L', 'CNN-3L', 'CNN-4L']:
            trials = self.results[arch]
            if not trials:
                continue
            
            accuracies = [t['metrics']['accuracy'] for t in trials]
            f1_macros = [t['metrics']['f1_macro'] for t in trials]
            
            print(f"{arch}:")
            print(f"  Accuracy:  {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
            print(f"  F1 (macro): {np.mean(f1_macros):.4f} ± {np.std(f1_macros):.4f}")
            print()
    
    def _plot_results(self):
        """Generate result visualizations."""
        # 1. Accuracy comparison across architectures
        fig, ax = plt.subplots(figsize=(10, 6))
        
        architectures = []
        means = []
        stds = []
        
        for arch in ['CNN-2L', 'CNN-3L', 'CNN-4L']:
            trials = self.results[arch]
            if not trials:
                continue
            
            accuracies = [t['metrics']['accuracy'] for t in trials]
            architectures.append(arch)
            means.append(np.mean(accuracies))
            stds.append(np.std(accuracies))
        
        x_pos = np.arange(len(architectures))
        ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
        ax.set_xlabel('Architecture')
        ax.set_ylabel('Test Accuracy')
        ax.set_title('CNN Architecture Comparison (5 trials)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(architectures)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'architecture_comparison.png', dpi=300)
        plt.close()
        
        # 2. Training curves for best trial of each architecture
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, arch in enumerate(['CNN-2L', 'CNN-3L', 'CNN-4L']):
            trials = self.results[arch]
            if not trials:
                continue
            
            # Get best trial
            best_trial = max(trials, key=lambda x: x['metrics']['accuracy'])
            history = best_trial['history']
            
            ax = axes[idx]
            epochs = range(1, len(history['loss']) + 1)
            
            ax.plot(epochs, history['accuracy'], 'b-', label='Train Acc')
            ax.plot(epochs, history['val_accuracy'], 'r-', label='Val Acc')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title(f'{arch} (Best Trial)')
            ax.legend()
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'training_curves.png', dpi=300)
        plt.close()
        
        print(f"  Plots saved to: {self.plots_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Memory-Efficient CNN Experiment for EEG Classification'
    )
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to preprocessed cnn_data.npz file')
    parser.add_argument('--output-dir', type=str, default='results_cnn',
                       help='Output directory for results')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--max-epochs', type=int, default=100,
                       help='Maximum epochs per trial')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    parser.add_argument('--architectures', nargs='+', 
                       choices=['CNN-2L', 'CNN-3L', 'CNN-4L'],
                       default=['CNN-2L', 'CNN-3L', 'CNN-4L'],
                       help='Architectures to test')
    parser.add_argument('--n-trials', type=int, default=5,
                       help='Number of independent trials per architecture')
    parser.add_argument('--start-trial', type=int, default=0,
                       help='Trial number to start from (0-based). Use to skip failed trials or continue.')
    
    args = parser.parse_args()
    
    # Create experiment
    experiment = CNNExperiment(
        data_path=args.data_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience
    )
    
    # Run experiments
    experiment.run_experiments(
        architectures=args.architectures,
        n_trials=args.n_trials,
        start_trial=args.start_trial
    )
    
    print("\n✓ CNN experiments complete!")


if __name__ == "__main__":
    main()