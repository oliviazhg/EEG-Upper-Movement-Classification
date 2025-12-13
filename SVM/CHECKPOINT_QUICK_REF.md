# SVM Checkpoint System - Quick Reference

## How Checkpoints Work

### Automatic Saving

The system automatically saves after:
1. **Each grid configuration** (12 total)
2. **Complete grid search** (1 file with all results)
3. **Each final trial** (5 total)

### File Structure

```
checkpoints/svm/
├── grid_C0.1_gamma0.001.json       ← After config 1
├── grid_C0.1_gamma0.01.json        ← After config 2
├── ...                              ← (12 configs total)
├── grid_search_complete_*.json     ← After all configs
├── trial_1_seed0.json              ← After trial 1
├── trial_2_seed1.json              ← After trial 2
└── ...                              ← (5 trials total)
```

---

## Usage

### Check Status

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
```

### Resume Interrupted Experiment

```bash
python run_svm_experiment.py --resume
```

**What happens:**
1. Detects completed work
2. Loads existing checkpoints (instant)
3. Continues from where it left off
4. Saves new checkpoints as it proceeds

### Start Fresh

```bash
python run_svm_experiment.py
```

If checkpoints exist, you'll get a warning:
```
WARNING: Existing checkpoints detected!
Continue and overwrite? (yes/no):
```

---

## How Resume Works

### Resume Logic Flow

```python
# 1. Detect what's been completed
status = detect_checkpoints('checkpoints/svm/')

# 2. Grid search
for each config in grid:
    if checkpoint exists:
        load_checkpoint()  # ← Instant!
    else:
        train_model()      # ← Only run if needed
        save_checkpoint()

# 3. Final trials
for each trial:
    if checkpoint exists:
        load_checkpoint()  # ← Instant!
    else:
        train_model()      # ← Only run if needed
        save_checkpoint()
```

### Scaler Handling

**Important:** The `StandardScaler` can't be saved to JSON.

**Solution:**
- Grid search: Fit scaler on training data
- If resuming: Recreate scaler by fitting training data again
- This is deterministic: same data → same scaler

```python
# Recreating scaler is safe
scaler = StandardScaler()
scaler.fit(X_train)  # Always gives same result
```

---

## Example Scenarios

### Scenario 1: Crash After 7/12 Grid Configs

**Before:**
```
Completed: 7 grid configs
Time spent: ~70 seconds
```

**Resume:**
```bash
python run_svm_experiment.py --resume
```

**What happens:**
```
[1/12] C=0.1, gamma=0.001... ✓ LOADED (instant)
[2/12] C=0.1, gamma=0.01...  ✓ LOADED (instant)
...
[7/12] C=1, gamma=0.1...     ✓ LOADED (instant)
[8/12] C=10, gamma=0.001...  Training... (10s)
[9/12] C=10, gamma=0.01...   Training... (10s)
...
```

**Time saved:** 70 seconds!

---

### Scenario 2: Crash After Grid Search

**Before:**
```
Grid search: COMPLETE
Trials: 2/5 complete
```

**Resume:**
```bash
python run_svm_experiment.py --resume
```

**What happens:**
```
Grid search already complete - loading results...
✓ Best: C=10, gamma=0.01

Loading 2 completed trials...
  ✓ Trial 1 loaded
  ✓ Trial 2 loaded

Running 3 remaining trials...
Trial 3/5... Training...
Trial 4/5... Training...
Trial 5/5... Training...
```

**Time saved:** 120 seconds for grid search + 20 seconds for trials!

---

### Scenario 3: Bug in Visualization

**Before:**
```
Grid search: COMPLETE
Trials: COMPLETE (5/5)
Crash: During plotting
```

**Resume:**
```bash
python run_svm_experiment.py --resume
```

**What happens:**
```
Grid search already complete - loading...
All trials already complete - loading...

Starting Phase 3: Analysis
Starting Phase 4: Visualization
```

**Time saved:** All training time (~170 seconds)!

---

## Checkpoint Contents

### Grid Config Checkpoint

```json
{
  "C": 10,
  "gamma": 0.01,
  "train_accuracy": 0.9234,
  "val_accuracy": 0.8542,
  "n_support_vectors": 542,
  "training_time_sec": 12.34
}
```

### Trial Checkpoint

```json
{
  "C": 10,
  "gamma": 0.01,
  "seed": 0,
  "test_accuracy": 0.8234,
  "test_f1_macro": 0.8156,
  "test_f1_per_class": [0.82, 0.81, ...],
  "confusion_matrix": [[...], ...],
  "n_support_vectors": 538
}
```

---

## Programmatic Usage

### In Python

```python
from pathlib import Path
from svm_resume import (
    detect_checkpoints,
    resume_experiment,
    cleanup_checkpoints
)

# Check status
checkpoint_dir = Path('checkpoints/svm')
status = detect_checkpoints(checkpoint_dir)

print(f"Grid configs done: {status['n_grid_complete']}/12")
print(f"Trials done: {status['n_trials_complete']}/5")

# Resume if incomplete
if not status['grid_search_complete']:
    grid_results, trial_results = resume_experiment(
        X_train, y_train, X_val, y_val, X_test, y_test,
        checkpoint_dir
    )
```

### Load Individual Checkpoint

```python
import json

# Load specific result
with open('checkpoints/svm/grid_C10_gamma0.01.json') as f:
    result = json.load(f)
    
print(f"Val accuracy: {result['val_accuracy']:.4f}")
```

---

## Cleanup

After completion, remove intermediate checkpoints:

```python
from svm_resume import cleanup_checkpoints

# Keep grid_search_complete_*.json
cleanup_checkpoints(Path('checkpoints/svm'), keep_final=True)

# Remove everything
cleanup_checkpoints(Path('checkpoints/svm'), keep_final=False)
```

Or manually:
```bash
rm checkpoints/svm/grid_C*.json
rm checkpoints/svm/trial_*.json
```

---

## Key Benefits

✅ **No wasted computation** - Never re-run completed work  
✅ **Safe interruption** - Ctrl+C after current iteration finishes  
✅ **Progress monitoring** - Check status anytime  
✅ **Audit trail** - Individual results saved  
✅ **Zero overhead** - Checkpoints are tiny JSON files  

---

## Commands Summary

| Command | Purpose |
|---------|---------|
| `python run_svm_experiment.py` | Start fresh (warns if checkpoints exist) |
| `python run_svm_experiment.py --resume` | Resume from checkpoints |
| `python run_svm_experiment.py --status` | Check progress |

---

## Troubleshooting

### Resume not loading checkpoints?

Check file exists:
```bash
ls -lh checkpoints/svm/
```

Check JSON is valid:
```bash
python -m json.tool checkpoints/svm/grid_C1_gamma0.01.json
```

### Want to force restart?

```bash
# Option 1: Delete checkpoints
rm -rf checkpoints/svm/

# Option 2: Say "yes" to overwrite warning
python run_svm_experiment.py
# Continue and overwrite? yes
```

### Results differ after resume?

This shouldn't happen with fixed random seeds. Debug:
```python
# Compare checkpoint to fresh run
import json

with open('checkpoints/svm/grid_C1_gamma0.01.json') as f:
    checkpoint = json.load(f)
    
# Should be identical
assert checkpoint['val_accuracy'] == 0.8234
```

---

## Complete Example

```bash
# Start experiment
python run_svm_experiment.py

# ... crash after 7 configs ...

# Check what completed
python run_svm_experiment.py --status
# Output: 7/12 configs done

# Resume
python run_svm_experiment.py --resume
# Loads 7, runs 5 more

# ... crash after 3 trials ...

# Resume again
python run_svm_experiment.py --resume
# Loads grid + 3 trials, runs 2 more

# Clean up
python -c "from svm_resume import cleanup_checkpoints; from pathlib import Path; cleanup_checkpoints(Path('checkpoints/svm'))"
```

---

## Summary

The checkpoint system makes your experiments **robust** and **resumable**:

1. Checkpoints save automatically after each step
2. Resume with `--resume` flag
3. No wasted computation
4. Clean up when done

**See CHECKPOINT_GUIDE.md for detailed documentation!**
