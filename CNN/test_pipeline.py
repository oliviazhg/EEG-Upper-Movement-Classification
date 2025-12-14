"""
Test Script for CNN Pipeline
Verifies all components work correctly with synthetic data

Run this before using real data to catch any issues.
"""

import numpy as np
import sys
from pathlib import Path

print("="*70)
print("CNN PIPELINE TEST SUITE")
print("="*70)

# Test 1: Imports
print("\n[1/6] Testing imports...")
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import signal, stats
    print("✓ All imports successful")
    print(f"  TensorFlow version: {tf.__version__}")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    print("  Run: pip install -r requirements.txt")
    sys.exit(1)

# Test 2: Module imports
print("\n[2/6] Testing custom modules...")
try:
    from cnn_eeg_classification import build_cnn, train_cnn_trial
    from cnn_data_loading import bandpass_filter, segment_trials, normalize_trials
    from cnn_analysis import compute_bootstrap_ci, aggregate_architecture_results
    print("✓ All custom modules loaded")
except Exception as e:
    print(f"✗ Module import failed: {e}")
    sys.exit(1)

# Test 3: CNN architecture
print("\n[3/6] Testing CNN architecture...")
try:
    for depth in [2, 3, 4]:
        model = build_cnn(
            n_channels=60,
            n_timepoints=5000,
            n_classes=11,
            depth=depth,
            seed=0
        )
        n_params = model.count_params()
        print(f"  {depth}-layer CNN: {n_params:,} parameters")
    print("✓ All architectures built successfully")
except Exception as e:
    print(f"✗ Architecture test failed: {e}")
    sys.exit(1)

# Test 4: Data preprocessing
print("\n[4/6] Testing preprocessing...")
try:
    # Synthetic data
    np.random.seed(42)
    raw_data = np.random.randn(60, 25000) * 10  # 60 channels, 10 seconds
    
    # Filter
    filtered = bandpass_filter(raw_data, sfreq=2500, low_freq=1.0, high_freq=40.0)
    assert filtered.shape == raw_data.shape
    
    # Segment
    events = np.array([[i*2500, 0, i%11+1] for i in range(1, 10)])  # 9 events
    trials = segment_trials(filtered, events, sfreq=2500, tmin=0.0, tmax=2.0)
    assert trials.shape == (9, 60, 5000)
    
    # Normalize
    normalized = normalize_trials(trials, method='standardize')
    assert abs(normalized.mean()) < 0.1  # Should be close to 0
    assert abs(normalized.std() - 1.0) < 0.1  # Should be close to 1
    
    print(f"  Filtered: {filtered.shape}")
    print(f"  Segmented: {trials.shape}")
    print(f"  Normalized: mean={normalized.mean():.3f}, std={normalized.std():.3f}")
    print("✓ Preprocessing working correctly")
except Exception as e:
    print(f"✗ Preprocessing test failed: {e}")
    sys.exit(1)

# Test 5: Training (minimal)
print("\n[5/6] Testing CNN training (minimal)...")
try:
    # Small synthetic dataset
    np.random.seed(42)
    X = np.random.randn(110, 60, 5000)  # 110 trials
    y = np.repeat(np.arange(11), 10)  # 11 classes, 10 each
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.3, random_state=0, stratify=y_train
    )
    
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Quick training (2-layer, 5 epochs)
    print("  Training 2-layer CNN for 5 epochs...")
    model = build_cnn(60, 5000, 11, depth=2, seed=0)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    y_train_oh = tf.keras.utils.to_categorical(y_train, 11)
    y_val_oh = tf.keras.utils.to_categorical(y_val, 11)
    
    history = model.fit(
        X_train, y_train_oh,
        validation_data=(X_val, y_val_oh),
        epochs=5,
        batch_size=32,
        verbose=0
    )
    
    # Evaluate
    y_test_oh = tf.keras.utils.to_categorical(y_test, 11)
    test_loss, test_acc = model.evaluate(X_test, y_test_oh, verbose=0)
    
    print(f"  Final train acc: {history.history['accuracy'][-1]:.3f}")
    print(f"  Final val acc: {history.history['val_accuracy'][-1]:.3f}")
    print(f"  Test acc: {test_acc:.3f}")
    print("✓ Training completed successfully")
    
except Exception as e:
    print(f"✗ Training test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Analysis functions
print("\n[6/6] Testing analysis functions...")
try:
    # Bootstrap CI
    data = np.array([0.75, 0.78, 0.76, 0.79, 0.77])
    mean, ci_low, ci_high = compute_bootstrap_ci(data)
    assert 0.7 < mean < 0.8
    assert ci_low < mean < ci_high
    print(f"  Bootstrap CI: {mean:.3f} [{ci_low:.3f}, {ci_high:.3f}]")
    
    # Mock trial results
    mock_trials = []
    for seed in range(3):
        mock_trials.append({
            'architecture': '2layer',
            'seed': seed,
            'test_accuracy': 0.75 + np.random.rand()*0.05,
            'test_f1_macro': 0.73 + np.random.rand()*0.05,
            'test_f1_per_class': np.random.rand(11)*0.1 + 0.7,
            'confusion_matrix': np.random.randint(0, 10, (11, 11)),
            'final_epoch': 40,
            'n_parameters': 123456,
            'training_history': {
                'train_loss': list(np.random.rand(40)),
                'val_loss': list(np.random.rand(40)),
                'train_accuracy': list(np.random.rand(40)*0.2 + 0.6),
                'val_accuracy': list(np.random.rand(40)*0.2 + 0.5),
            }
        })
    
    # Aggregate
    agg = aggregate_architecture_results(mock_trials, '2layer')
    assert 'test_acc_mean' in agg
    assert 'confusion_matrix_mean' in agg
    print(f"  Aggregated 3 trials: acc={agg['test_acc_mean']:.3f}")
    print("✓ Analysis functions working")
    
except Exception as e:
    print(f"✗ Analysis test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# All tests passed
print("\n" + "="*70)
print("✓ ALL TESTS PASSED")
print("="*70)
print("\nYou can now run the full pipeline:")
print("  python run_cnn_experiments.py --use_synthetic")
print("\nOr with real data:")
print("  python run_cnn_experiments.py --data_dir /path/to/data")
print("="*70)
