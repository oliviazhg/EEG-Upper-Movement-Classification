"""
1D-CNN EEG Classification Pipeline
SYDE 522 Final Project - Upper-Limb Movement Classification

This script implements 1D Convolutional Neural Networks for classifying
11 upper-limb movements from 60-channel EEG data.

Architecture variants: 2-layer, 3-layer, 4-layer CNNs
Training: 5 independent trials per architecture with different random seeds
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             classification_report)
import time
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


# Set global random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


def build_cnn(n_channels: int, 
              n_timepoints: int, 
              n_classes: int, 
              depth: int = 3, 
              seed: int = 0) -> tf.keras.Model:
    """
    Build 1D CNN architecture for EEG classification
    
    Args:
        n_channels: Number of EEG channels (60 for motor cortex)
        n_timepoints: Length of time series (5000 samples = 2 seconds at 2500 Hz)
        n_classes: Number of movement classes (11)
        depth: Number of convolutional layers (2, 3, or 4)
        seed: Random seed for initialization
        
    Returns:
        Compiled Keras model
        
    Architecture follows Lecture 12 (Modern AI) and Schirrmeister et al. 2017:
    - Progressive kernel size reduction: 50 → 25 → 10 → 5
    - Increasing filters: 64 → 128 → 256 → 512
    - He initialization for ReLU (Lecture 9)
    - Dropout (0.5) for regularization
    """
    tf.random.set_seed(seed)
    
    model = models.Sequential(name=f'CNN_{depth}layer')
    
    # Input shape: (n_channels, n_timepoints)
    # Note: channels-first format matches EEG convention
    model.add(layers.Input(shape=(n_channels, n_timepoints)))
    
    # Layer 1 (all depths)
    # Large kernel (50) captures slow oscillations (mu/beta bands)
    model.add(layers.Conv1D(
        filters=64,
        kernel_size=50,
        activation='relu',
        kernel_initializer='he_normal',
        name='conv1'
    ))
    model.add(layers.MaxPooling1D(pool_size=4, name='pool1'))
    
    # Layer 2 (all depths)
    # Medium kernel (25) captures faster transients
    model.add(layers.Conv1D(
        filters=128,
        kernel_size=25,
        activation='relu',
        kernel_initializer='he_normal',
        name='conv2'
    ))
    model.add(layers.MaxPooling1D(pool_size=4, name='pool2'))
    
    # Layer 3 (depth >= 3)
    # Smaller kernel (10) for fine temporal features
    if depth >= 3:
        model.add(layers.Conv1D(
            filters=256,
            kernel_size=10,
            activation='relu',
            kernel_initializer='he_normal',
            name='conv3'
        ))
        model.add(layers.MaxPooling1D(pool_size=2, name='pool3'))
    
    # Layer 4 (depth == 4)
    # Finest temporal resolution
    if depth == 4:
        model.add(layers.Conv1D(
            filters=512,
            kernel_size=5,
            activation='relu',
            kernel_initializer='he_normal',
            name='conv4'
        ))
        model.add(layers.MaxPooling1D(pool_size=2, name='pool4'))
    
    # Classifier head
    model.add(layers.Flatten(name='flatten'))
    model.add(layers.Dropout(0.5, seed=seed, name='dropout'))
    model.add(layers.Dense(
        n_classes,
        activation='softmax',
        kernel_initializer='glorot_uniform',
        name='output'
    ))
    
    return model


def train_cnn_trial(X_train: np.ndarray,
                    y_train: np.ndarray,
                    X_val: np.ndarray,
                    y_val: np.ndarray,
                    X_test: np.ndarray,
                    y_test: np.ndarray,
                    depth: int = 3,
                    seed: int = 0,
                    verbose: int = 0) -> Dict:
    """
    Train a single CNN trial and evaluate
    
    Args:
        X_train, X_val, X_test: EEG data (n_trials, n_channels, n_timepoints)
        y_train, y_val, y_test: Integer labels (n_trials,)
        depth: CNN architecture depth (2, 3, or 4)
        seed: Random seed
        verbose: Keras verbosity level
        
    Returns:
        Dictionary containing:
            - training_history: Loss and accuracy per epoch
            - final_epoch: When early stopping triggered
            - test_accuracy, test_f1_macro: Final performance
            - confusion_matrix: (11, 11) array
            - n_parameters: Model size
            - timing: Training and inference times
    """
    print(f"\n{'='*60}")
    print(f"Training {depth}-layer CNN (seed={seed})")
    print(f"{'='*60}")
    
    # One-hot encode labels for categorical crossentropy
    n_classes = 11
    y_train_oh = tf.keras.utils.to_categorical(y_train, n_classes)
    y_val_oh = tf.keras.utils.to_categorical(y_val, n_classes)
    y_test_oh = tf.keras.utils.to_categorical(y_test, n_classes)
    
    # Build model
    model = build_cnn(
        n_channels=X_train.shape[1],
        n_timepoints=X_train.shape[2],
        n_classes=n_classes,
        depth=depth,
        seed=seed
    )
    
    # Print architecture summary
    if verbose > 0:
        model.summary()
    
    n_params = model.count_params()
    print(f"Total parameters: {n_params:,}")
    
    # Compile model (Lecture 9: Adam optimizer with lr=0.001)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Early stopping (Lecture 9: prevent overfitting)
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1 if verbose > 0 else 0
    )
    
    # Train model
    print(f"Training (max 100 epochs, early stopping patience=10)...")
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train_oh,
        validation_data=(X_val, y_val_oh),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=verbose
    )
    
    training_time = time.time() - start_time
    final_epoch = len(history.history['loss'])
    time_per_epoch = training_time / final_epoch
    
    print(f"Training completed in {training_time:.1f}s ({final_epoch} epochs)")
    print(f"Average time per epoch: {time_per_epoch:.2f}s")
    
    # Evaluate on all splits
    train_loss, train_acc = model.evaluate(X_train, y_train_oh, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val_oh, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test_oh, verbose=0)
    
    # Predictions for detailed metrics
    print("Computing predictions...")
    
    # Measure inference time
    inference_start = time.time()
    y_test_pred = model.predict(X_test, verbose=0).argmax(axis=1)
    inference_time = time.time() - inference_start
    inference_per_sample = inference_time / len(X_test)
    
    y_train_pred = model.predict(X_train, verbose=0).argmax(axis=1)
    y_val_pred = model.predict(X_val, verbose=0).argmax(axis=1)
    
    # Compute F1 scores
    train_f1 = f1_score(y_train, y_train_pred, average='macro')
    val_f1 = f1_score(y_val, y_val_pred, average='macro')
    test_f1 = f1_score(y_test, y_test_pred, average='macro')
    test_f1_per_class = f1_score(y_test, y_test_pred, average=None)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    
    # Print results
    print(f"\n{'Results':-^60}")
    print(f"Train - Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
    print(f"Val   - Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
    print(f"Test  - Acc: {test_acc:.4f}, F1: {test_f1:.4f}")
    print(f"Inference: {inference_per_sample*1000:.2f} ms/sample")
    
    # Return comprehensive results
    return {
        'architecture': f'{depth}layer',
        'seed': seed,
        
        # Training history (per epoch)
        'training_history': {
            'train_loss': history.history['loss'],
            'val_loss': history.history['val_loss'],
            'train_accuracy': history.history['accuracy'],
            'val_accuracy': history.history['val_accuracy'],
        },
        
        # Final performance
        'final_epoch': final_epoch,
        'train_accuracy': float(train_acc),
        'val_accuracy': float(val_acc),
        'test_accuracy': float(test_acc),
        
        'train_f1_macro': float(train_f1),
        'val_f1_macro': float(val_f1),
        'test_f1_macro': float(test_f1),
        
        'test_f1_per_class': test_f1_per_class.tolist(),
        
        # Confusion matrix
        'confusion_matrix': cm.tolist(),
        
        # Model characteristics
        'n_parameters': int(n_params),
        
        # Timing
        'training_time_sec': float(training_time),
        'time_per_epoch': float(time_per_epoch),
        'inference_time_per_sample': float(inference_per_sample),
    }


def run_cnn_experiments(X: np.ndarray,
                        y: np.ndarray,
                        depths: List[int] = [2, 3, 4],
                        seeds: List[int] = [0, 1, 2, 3, 4],
                        test_size: float = 0.15,
                        val_size: float = 0.15,
                        output_dir: str = 'results/cnn',
                        verbose: int = 0) -> Dict:
    """
    Run complete CNN experimental protocol
    
    Args:
        X: EEG data (n_trials, n_channels, n_timepoints)
        y: Movement labels (n_trials,)
        depths: CNN architectures to test
        seeds: Random seeds for trials
        test_size: Proportion for test set
        val_size: Proportion for validation (from remaining after test)
        output_dir: Where to save results
        verbose: Verbosity level
        
    Returns:
        Dictionary containing all trial results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"1D-CNN EEG CLASSIFICATION EXPERIMENTS")
    print(f"{'='*60}")
    print(f"Data shape: {X.shape}")
    print(f"Labels: {np.unique(y)} (n={len(np.unique(y))})")
    print(f"Architectures: {depths}")
    print(f"Seeds: {seeds}")
    print(f"Split: {1-test_size-val_size:.0%} train, {val_size:.0%} val, {test_size:.0%} test")
    
    all_results = []
    
    # Run experiments
    for depth in depths:
        for seed in seeds:
            print(f"\n{'#'*60}")
            print(f"# Architecture: {depth}-layer, Seed: {seed}")
            print(f"{'#'*60}")
            
            # Split data with stratification
            # First split: train+val vs test
            X_trainval, X_test, y_trainval, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=seed,
                stratify=y
            )
            
            # Second split: train vs val
            val_size_adjusted = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_trainval, y_trainval,
                test_size=val_size_adjusted,
                random_state=seed,
                stratify=y_trainval
            )
            
            print(f"Split sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
            
            # Train and evaluate
            results = train_cnn_trial(
                X_train, y_train,
                X_val, y_val,
                X_test, y_test,
                depth=depth,
                seed=seed,
                verbose=verbose
            )
            
            all_results.append(results)
            
            # Save after each trial (in case of crashes)
            with open(output_path / 'all_trials.pkl', 'wb') as f:
                pickle.dump(all_results, f)
    
    print(f"\n{'='*60}")
    print(f"All experiments completed!")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")
    
    return {'trials': all_results}


if __name__ == '__main__':
    # Example usage
    print("CNN EEG Classification Pipeline")
    print("Load your data and call run_cnn_experiments(X, y)")
    print("\nExpected data format:")
    print("  X: (n_trials, n_channels=60, n_timepoints=5000)")
    print("  y: (n_trials,) with values 0-10")
