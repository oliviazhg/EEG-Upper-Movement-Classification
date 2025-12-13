# SVM Checkpoint System Guide

## Overview

The checkpoint system automatically saves progress during experiments, allowing you to:
1. **Resume interrupted experiments** without losing work
2. **Monitor progress** in real-time
3. **Audit individual results** after completion

---

## How Checkpoints Work

### Checkpoint Files

During execution, the system creates checkpoint files in `checkpoints/svm/`:

```
checkpoints/svm/
├── grid_C0.1_gamma0.001.json       # After each grid config
├── grid_C0.1_gamma0.01.json
├── grid_C0.1_gamma0.1.json
├── grid_C1_gamma0.001.json
├── ...                              # (12 total grid configs)
├── grid_search_complete_20251213_151234.json  # After grid search completes
├── trial_1_seed0.json              # After each final trial
├── trial_2_seed1.json
├── trial_3_seed2.json
├── trial_4_seed3.json
└── trial_5_seed4.json
```

### What Gets Saved

**Grid Configuration Checkpoint** (`grid_C{C}_gamma{gamma}.json`):
```json
{
  "C": 10,
  "gamma": 0.01,
  "train_accuracy": 0.9234,
  "val_accuracy": 0.8542,
  "train_f1_macro": 0.9156,
  "val_f1_macro": 0.8467,
  "n_support_vectors": 542,
  "n_support_vectors_per_class": {...},
  "training_time_sec": 12.34
}
```

**Complete Grid Search** (`grid_search_complete_{timestamp}.json`):
```json
{
  "C_values": [0.1, 1, 10, 100],
  "gamma_values": [0.001, 0.01, 0.1],
  "val_accuracy_grid": [[...], ...],
  "training_times": [[...], ...],
  "best_C": 10,
  "best_gamma": 0.01,
  "best_val_accuracy": 0.8542,
  "all_results": [...]
}
```

**Trial Checkpoint** (`trial_{n}_seed{seed}.json`):
```json
{
  "C": 10,
  "gamma": 0.01,
  "seed": 0,
  "test_accuracy": 0.8234,
  "test_f1_macro": 0.8156,
  "test_f1_per_class": [...],
  "confusion_matrix": [[...], ...],
  "n_support_vectors": 538,
  "training_time_sec": 11.89
}
```

---

## Usage Examples

### 1. Check Checkpoint Status

```bash
python run_svm_experiment.py --status
```

**Output:**
```
======================================================================
CHECKPOINT STATUS
======================================================================

Phase 1 - Grid Search:
  Completed configs: 7/12
  ✗ Grid search INCOMPLETE
    Can resume from checkpoint

Phase 2 - Final Evaluation:
  Completed trials: 0/5
  ✗ No trials started
======================================================================

======================================================================
HOW TO RESUME
======================================================================

Grid search partially complete (7/12)
Resume with:
  python run_svm_experiment.py --resume
======================================================================
```

### 2. Resume Interrupted Experiment

**Scenario:** Your experiment crashed after 7 grid configurations.

```bash
# Resume from where it left off
python run_svm_experiment.py --resume
```

**What happens:**
1. System detects 7 completed grid configs
2. Loads results from checkpoints (instant)
3. Continues with remaining 5 configs
4. Proceeds to Phase 2 when done

**Output:**
```
======================================================================
CHECKPOINT STATUS
======================================================================

Phase 1 - Grid Search:
  Completed configs: 7/12
  ✗ Grid search INCOMPLETE
    Can resume from checkpoint

Phase 2 - Final Evaluation:
  Completed trials: 0/5
  ✗ No trials started
======================================================================

======================================================================
RESUMING PHASE 1: GRID SEARCH
======================================================================

Found 7 completed configurations
Loading completed configs and continuing...

[1/12] C=0.1, gamma=0.001... ✓ LOADED from checkpoint (Val Acc: 0.7234)
[2/12] C=0.1, gamma=0.01... ✓ LOADED from checkpoint (Val Acc: 0.7456)
[3/12] C=0.1, gamma=0.1... ✓ LOADED from checkpoint (Val Acc: 0.7123)
...
[8/12] C=10, gamma=0.001... Val Acc: 0.8234 (8.5s)
[9/12] C=10, gamma=0.01... Val Acc: 0.8542 (9.2s)
...
```

### 3. Resume After Grid Search Complete

**Scenario:** Grid search finished, but only 2 of 5 trials completed before crash.

```bash
python run_svm_experiment.py --resume
```

**Output:**
```
======================================================================
RESUMING PHASE 1: GRID SEARCH
======================================================================

Grid search already complete - loading results...
✓ Loaded grid search results
  Best: C=10, gamma=0.01
  Best val accuracy: 0.8542

======================================================================
RESUMING PHASE 2: FINAL EVALUATION
======================================================================

Best configuration: C=10, gamma=0.01

Loading 2 completed trials...
  ✓ Trial 1 loaded (Test Acc: 0.8234)
  ✓ Trial 2 loaded (Test Acc: 0.8156)

Running 3 remaining trials...
Trial 3/5 (seed=2)... Test Acc: 0.8345, F1: 0.8267, SVs: 542
Trial 4/5 (seed=3)... Test Acc: 0.8198, F1: 0.8112, SVs: 551
Trial 5/5 (seed=4)... Test Acc: 0.8289, F1: 0.8201, SVs: 538
```

### 4. Start Fresh (With Warning)

**Scenario:** You have checkpoints but want to start over.

```bash
python run_svm_experiment.py  # No --resume flag
```

**Output:**
```
======================================================================
WARNING: Existing checkpoints detected!
======================================================================

CHECKPOINT STATUS
======================================================================

Phase 1 - Grid Search:
  Completed configs: 12/12
  ✓ Grid search COMPLETE

Phase 2 - Final Evaluation:
  Completed trials: 5/5
  ✓ All trials COMPLETE
======================================================================

Options:
  1. Continue anyway (will overwrite existing checkpoints)
  2. Resume from checkpoints: python run_svm_experiment.py --resume
  3. Check status: python run_svm_experiment.py --status

Continue and overwrite? (yes/no): 
```

---

## Programmatic Usage

### In Python Script

```python
from pathlib import Path
from svm_resume import (
    detect_checkpoints,
    print_checkpoint_status,
    resume_experiment,
    cleanup_checkpoints,
)

# Check status
checkpoint_dir = Path('checkpoints/svm')
status = detect_checkpoints(checkpoint_dir)
print_checkpoint_status(status)

# Resume if needed
if not status['grid_search_complete']:
    grid_results, trial_results = resume_experiment(
        X_train, y_train, X_val, y_val, X_test, y_test,
        checkpoint_dir
    )
    
# Clean up after completion
cleanup_checkpoints(checkpoint_dir, keep_final=True)
```

### Access Individual Checkpoints

```python
import json

# Load a specific grid config
with open('checkpoints/svm/grid_C10_gamma0.01.json', 'r') as f:
    config_result = json.load(f)
    
print(f"Validation accuracy: {config_result['val_accuracy']:.4f}")
print(f"Support vectors: {config_result['n_support_vectors']}")

# Load a specific trial
with open('checkpoints/svm/trial_1_seed0.json', 'r') as f:
    trial_result = json.load(f)
    
print(f"Test accuracy: {trial_result['test_accuracy']:.4f}")
```

---

## Behind the Scenes

### When Checkpoints Are Created

**Grid Search:**
- ✓ After **each** configuration completes (12 total)
- ✓ After **entire** grid search completes

**Final Trials:**
- ✓ After **each** trial completes (5 total)

### Resume Logic

```python
# Phase 1: Grid Search
for each (C, gamma) config:
    if checkpoint exists:
        → Load from file (instant)
    else:
        → Train model
        → Save checkpoint
        
# Phase 2: Final Trials
for each trial:
    if checkpoint exists:
        → Load from file (instant)
    else:
        → Train model
        → Save checkpoint
```

### Scaler Handling

**Important:** The `StandardScaler` cannot be saved to JSON, so:

1. During grid search: Scaler is fitted on training data
2. Scaler object is passed to Phase 2 in memory
3. If resuming after grid search: Scaler is **recreated** by fitting training data again
4. This is deterministic - same data → same scaler

```python
# When resuming after grid search complete
scaler = StandardScaler()
scaler.fit(X_train)  # Recreates identical scaler
```

---

## Common Scenarios

### Scenario 1: Power Failure During Grid Search

**Before:**
- 5 of 12 configs complete

**Action:**
```bash
python run_svm_experiment.py --resume
```

**Result:**
- Loads 5 completed configs (instant)
- Runs remaining 7 configs
- Continues to Phase 2

**Time Saved:**
- ~5 × 10s = 50 seconds

---

### Scenario 2: Bug in Analysis Code

**Before:**
- Grid search complete
- All 5 trials complete
- Crash during analysis/visualization

**Action:**
```bash
python run_svm_experiment.py --resume
```

**Result:**
- Loads grid search results (instant)
- Loads all 5 trials (instant)
- Skips to analysis/visualization

**Time Saved:**
- ~12 × 10s + 5 × 10s = 170 seconds

---

### Scenario 3: Need to Modify Plots

**Before:**
- Experiment fully complete
- Want to regenerate plots with different style

**Action:**
```python
from svm_visualize import generate_all_plots
import json
from pathlib import Path

# Load saved results
with open('results/svm/grid_search_results.json', 'r') as f:
    grid_results = json.load(f)
    
with open('results/svm/aggregated_results.json', 'r') as f:
    aggregated = json.load(f)

# Modify plot config
from svm_config import PLOT_CONFIG
PLOT_CONFIG['dpi'] = 600  # Higher resolution
PLOT_CONFIG['colormap'] = 'plasma'  # Different colormap

# Regenerate
generate_all_plots(grid_results, aggregated, Path('figures/svm_v2'))
```

---

## Cleanup

After successful completion, you can remove intermediate checkpoints:

```python
from svm_resume import cleanup_checkpoints
from pathlib import Path

# Remove individual config/trial checkpoints
# Keep grid_search_complete_*.json
cleanup_checkpoints(Path('checkpoints/svm'), keep_final=True)

# Remove everything
cleanup_checkpoints(Path('checkpoints/svm'), keep_final=False)
```

**Before cleanup:**
```
checkpoints/svm/
├── grid_C0.1_gamma0.001.json
├── grid_C0.1_gamma0.01.json
├── ...  (12 files)
├── grid_search_complete_20251213.json
├── trial_1_seed0.json
├── trial_2_seed1.json
├── ... (5 files)
```

**After cleanup (keep_final=True):**
```
checkpoints/svm/
└── grid_search_complete_20251213.json
```

---

## Best Practices

### 1. Monitor Progress

During long experiments, check progress in another terminal:

```bash
# Terminal 1: Running experiment
python run_svm_experiment.py

# Terminal 2: Check progress
watch -n 5 'ls -lh checkpoints/svm/ | tail -n 10'
```

### 2. Safe Interruption

If you need to stop the experiment:
- Press `Ctrl+C` (once!)
- Wait for current iteration to finish
- Checkpoint will be saved automatically

**Don't:**
- Force kill (`kill -9`)
- Shut down computer immediately

### 3. Archiving Results

After completion, archive everything together:

```bash
tar -czf svm_experiment_20251213.tar.gz \
    results/svm/ \
    figures/svm/ \
    checkpoints/svm/
```

---

## Troubleshooting

### Issue: Resume Not Working

**Symptom:** Resume loads some checkpoints but starts over

**Cause:** Checkpoint files are corrupted (incomplete JSON)

**Solution:**
```bash
# Check for corrupt files
for f in checkpoints/svm/*.json; do
    python -m json.tool "$f" > /dev/null || echo "Corrupt: $f"
done

# Remove corrupt files and resume
rm checkpoints/svm/grid_C10_gamma0.1.json  # Example
python run_svm_experiment.py --resume
```

### Issue: Different Results After Resume

**Symptom:** Results change slightly when resuming

**Possible Causes:**
1. Random seed not being set correctly
2. Data order changed
3. Scaler recreation (shouldn't happen, but check)

**Debug:**
```python
# Compare checkpoint to fresh run
import json

with open('checkpoints/svm/grid_C1_gamma0.01.json', 'r') as f:
    checkpoint = json.load(f)
    
# Run same config fresh
result = train_svm_single_config(...)

assert abs(checkpoint['val_accuracy'] - result['val_accuracy']) < 1e-6
```

### Issue: Out of Disk Space

**Symptom:** Checkpoint save fails

**Solution:**
```python
# Disable checkpointing temporarily
from svm_config import CHECKPOINT_CONFIG
CHECKPOINT_CONFIG['save_grid_search'] = False
CHECKPOINT_CONFIG['save_per_trial'] = False
```

---

## Summary

**Checkpoints save:**
- ✓ Grid search configs (12 files)
- ✓ Complete grid search (1 file)
- ✓ Final trials (5 files)

**Resume command:**
```bash
python run_svm_experiment.py --resume
```

**Check status:**
```bash
python run_svm_experiment.py --status
```

**Clean up:**
```python
from svm_resume import cleanup_checkpoints
cleanup_checkpoints(Path('checkpoints/svm'))
```

The checkpoint system makes your experiments **robust** and **resumable** without any performance overhead!
