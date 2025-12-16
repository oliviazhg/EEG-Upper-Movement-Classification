"""
Extract CSP+LDA results from log files and regenerate complete results.
"""
import re
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

def parse_log_file(log_path):
    """Parse log file to extract all trial results."""
    with open(log_path, 'r') as f:
        content = f.read()

    # Find all trial blocks
    trial_pattern = r'Trial: (\w+), (\d+) components, seed (\d+).*?Results:(.*?)(?=Trial:|All trials completed!|$)'

    trials = []
    for match in re.finditer(trial_pattern, content, re.DOTALL):
        band = match.group(1)
        n_components = int(match.group(2))
        seed = int(match.group(3))
        results_text = match.group(4)

        # Extract metrics
        train_acc = float(re.search(r'Train Accuracy:\s+([\d.]+)', results_text).group(1))
        val_acc = float(re.search(r'Val Accuracy:\s+([\d.]+)', results_text).group(1))
        test_acc = float(re.search(r'Test Accuracy:\s+([\d.]+)', results_text).group(1))
        test_f1 = float(re.search(r'Test F1 \(macro\):\s+([\d.]+)', results_text).group(1))
        training_time = float(re.search(r'Training time:\s+([\d.]+)s', results_text).group(1))

        trials.append({
            'band': band,
            'n_components': n_components,
            'seed': seed,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'test_accuracy': test_acc,
            'train_f1_macro': np.nan,  # Not in logs
            'val_f1_macro': np.nan,    # Not in logs
            'test_f1_macro': test_f1,
            'training_time_sec': training_time,
            'inference_time_per_sample': np.nan  # Not in logs
        })

    return trials

def compute_summaries(trial_results):
    """Compute summary statistics for each configuration."""
    df = pd.DataFrame(trial_results)

    summaries = []
    for (band, n_comp), group in df.groupby(['band', 'n_components']):
        summary = {
            'band': band,
            'n_components': n_comp,
            'test_acc_mean': group['test_accuracy'].mean(),
            'test_acc_ci_low': group['test_accuracy'].min(),
            'test_acc_ci_high': group['test_accuracy'].max(),
            'test_acc_std': group['test_accuracy'].std(),
            'test_f1_mean': group['test_f1_macro'].mean(),
            'test_f1_ci_low': group['test_f1_macro'].min(),
            'test_f1_ci_high': group['test_f1_macro'].max(),
            'test_f1_std': group['test_f1_macro'].std(),
        }
        summaries.append(summary)

    return summaries

# Parse all log files
log_files = [
    'EEG-Upper-Movement-Classification/CSP_LDA/experiment.log',
    'EEG-Upper-Movement-Classification/CSP_LDA/experiment_cont.log',
    'EEG-Upper-Movement-Classification/CSP_LDA/csp_lda_experiment.log',
]

all_trials = []
for log_file in log_files:
    if Path(log_file).exists():
        print(f"Parsing {log_file}...")
        trials = parse_log_file(log_file)
        print(f"  Found {len(trials)} trials")
        all_trials.extend(trials)

# Remove duplicates (keep last occurrence of each config)
print(f"\nTotal trials found: {len(all_trials)}")

# Create a unique key for each trial and keep only the last occurrence
unique_trials = {}
for trial in all_trials:
    key = (trial['band'], trial['n_components'], trial['seed'])
    unique_trials[key] = trial

all_trials = list(unique_trials.values())
print(f"Unique trials: {len(all_trials)}")

# Sort by band, n_components, seed
all_trials.sort(key=lambda x: (x['band'], x['n_components'], x['seed']))

# Print summary
print("\nTrials by configuration:")
df = pd.DataFrame(all_trials)
for (band, n_comp), group in df.groupby(['band', 'n_components']):
    print(f"  {band} ({n_comp} comp): {len(group)} trials (seeds {sorted(group['seed'].tolist())})")

# Compute summaries
summaries = compute_summaries(all_trials)

# Save results
output_dir = Path('EEG-Upper-Movement-Classification/CSP_LDA/results_csp_lda_1')

# Save trial results
trial_df = pd.DataFrame(all_trials)
trial_df.to_csv(output_dir / 'trial_results.csv', index=False)
print(f"\n✓ Saved {len(all_trials)} trials to trial_results.csv")

# Save config summaries
summary_df = pd.DataFrame(summaries)
summary_df.to_csv(output_dir / 'config_summaries.csv', index=False)
print(f"✓ Saved {len(summaries)} config summaries to config_summaries.csv")

# Save pickle files
with open(output_dir / 'trial_results.pkl', 'wb') as f:
    pickle.dump(all_trials, f)
print(f"✓ Saved trial_results.pkl")

with open(output_dir / 'config_summaries.pkl', 'wb') as f:
    pickle.dump(summaries, f)
print(f"✓ Saved config_summaries.pkl")

print("\nResults summary:")
print(summary_df[['band', 'n_components', 'test_acc_mean', 'test_f1_mean']])
