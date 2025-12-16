"""Extract all CSP+LDA trial results from checkpoint and existing data."""
import json
from pathlib import Path

# New trials from checkpoint (7 trials)
new_trials = [
    # 2 components
    {'band': 'combined', 'n_components': 2, 'seed': 0,
     'train_accuracy': 0.13076682750720311, 'val_accuracy': 0.12344079053944597,
     'test_accuracy': 0.12876579203109814, 'test_f1_macro': 0.10336106337553619,
     'training_time_sec': 266.8415114879608},
    {'band': 'combined', 'n_components': 2, 'seed': 1,
     'train_accuracy': 0.12493491165341757, 'val_accuracy': 0.11955289162481776,
     'test_accuracy': 0.1345966958211856, 'test_f1_macro': 0.09896101212406036,
     'training_time_sec': 183.69036197662354},
    {'band': 'combined', 'n_components': 2, 'seed': 2,
     'train_accuracy': 0.13281494081299683, 'val_accuracy': 0.12311679896322696,
     'test_accuracy': 0.12714609653385164, 'test_f1_macro': 0.10374765239206739,
     'training_time_sec': 180.69366431236267},
    # 4 components
    {'band': 'combined', 'n_components': 4, 'seed': 0,
     'train_accuracy': 0.13541847467629395, 'val_accuracy': 0.1307306010043739,
     'test_accuracy': 0.1323291221250405, 'test_f1_macro': 0.11499511113278765,
     'training_time_sec': 181.88388681411743},
    {'band': 'combined', 'n_components': 4, 'seed': 1,
     'train_accuracy': 0.1343076335612872, 'val_accuracy': 0.123764782115665,
     'test_accuracy': 0.13540654356980888, 'test_f1_macro': 0.11681337793161992,
     'training_time_sec': 182.7811517715454},
    {'band': 'combined', 'n_components': 4, 'seed': 2,
     'train_accuracy': 0.1384385739577186, 'val_accuracy': 0.12846265997084075,
     'test_accuracy': 0.12487852283770651, 'test_f1_macro': 0.10444913027918977,
     'training_time_sec': 183.5815076828003},
    # 6 components (1 trial)
    {'band': 'combined', 'n_components': 6, 'seed': 0,
     'train_accuracy': 0.14243065921477419, 'val_accuracy': 0.13688644095253524,
     'test_accuracy': 0.13816002591512797, 'test_f1_macro': 0.1261445224830247,
     'training_time_sec': 186.34095692634583},
]

# Existing trials from logs (3 trials)
existing_trials = [
    {'band': 'combined', 'n_components': 6, 'seed': 2,
     'train_accuracy': 0.1415, 'val_accuracy': 0.1319,
     'test_accuracy': 0.1354, 'test_f1_macro': 0.119,
     'training_time_sec': 261.25},
    {'band': 'combined', 'n_components': 8, 'seed': 1,
     'train_accuracy': 0.1424, 'val_accuracy': 0.1304,
     'test_accuracy': 0.1372, 'test_f1_macro': 0.1268,
     'training_time_sec': 265.01},
    {'band': 'combined', 'n_components': 8, 'seed': 2,
     'train_accuracy': 0.1443, 'val_accuracy': 0.1285,
     'test_accuracy': 0.1388, 'test_f1_macro': 0.1266,
     'training_time_sec': 262.54},
]

# Combine all trials
all_trials = new_trials + existing_trials
all_trials.sort(key=lambda x: (x['n_components'], x['seed']))

print(f"Total trials: {len(all_trials)}")
print("\nTrials by configuration:")
for n_comp in [2, 4, 6, 8]:
    trials_for_comp = [t for t in all_trials if t['n_components'] == n_comp]
    if trials_for_comp:
        seeds = [t['seed'] for t in trials_for_comp]
        print(f"  {n_comp} components: {len(trials_for_comp)} trials (seeds {seeds})")

# Compute summaries
summaries = []
for n_comp in [2, 4, 6, 8]:
    trials_for_comp = [t for t in all_trials if t['n_components'] == n_comp]
    if not trials_for_comp:
        continue

    test_accs = [t['test_accuracy'] for t in trials_for_comp]
    test_f1s = [t['test_f1_macro'] for t in trials_for_comp]

    n = len(test_accs)
    acc_mean = sum(test_accs) / n
    f1_mean = sum(test_f1s) / n

    if n > 1:
        acc_std = (sum((x - acc_mean)**2 for x in test_accs) / (n - 1))**0.5
        f1_std = (sum((x - f1_mean)**2 for x in test_f1s) / (n - 1))**0.5
    else:
        acc_std = 0
        f1_std = 0

    summary = {
        'band': 'combined',
        'n_components': n_comp,
        'test_acc_mean': acc_mean,
        'test_acc_ci_low': min(test_accs),
        'test_acc_ci_high': max(test_accs),
        'test_acc_std': acc_std,
        'test_f1_mean': f1_mean,
        'test_f1_ci_low': min(test_f1s),
        'test_f1_ci_high': max(test_f1s),
        'test_f1_std': f1_std,
    }
    summaries.append(summary)

# Write CSV files
output_dir = Path('EEG-Upper-Movement-Classification/CSP_LDA/results_csp_lda_1')
output_dir.mkdir(parents=True, exist_ok=True)

# Trial results CSV
trial_headers = ['band', 'n_components', 'seed', 'train_accuracy', 'val_accuracy',
                 'test_accuracy', 'test_f1_macro', 'training_time_sec']

with open(output_dir / 'trial_results.csv', 'w') as f:
    f.write(','.join(trial_headers) + '\n')
    for trial in all_trials:
        values = [str(trial.get(h, '')) for h in trial_headers]
        f.write(','.join(values) + '\n')

print(f"\n✓ Saved {len(all_trials)} trials to trial_results.csv")

# Config summaries CSV
summary_headers = ['band', 'n_components', 'test_acc_mean', 'test_acc_ci_low',
                   'test_acc_ci_high', 'test_acc_std', 'test_f1_mean',
                   'test_f1_ci_low', 'test_f1_ci_high', 'test_f1_std']

with open(output_dir / 'config_summaries.csv', 'w') as f:
    f.write(','.join(summary_headers) + '\n')
    for summary in summaries:
        values = [str(summary.get(h, '')) for h in summary_headers]
        f.write(','.join(values) + '\n')

print(f"✓ Saved {len(summaries)} config summaries to config_summaries.csv")

# Save JSON
with open(output_dir / 'all_results.json', 'w') as f:
    json.dump({'trials': all_trials, 'summaries': summaries}, f, indent=2)
print(f"✓ Saved all_results.json")

# Print summary table
print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)
print(f"{'Config':<15} {'Test Acc (%)':<15} {'Test F1':<15} {'Trials':<10}")
print("-"*70)
for summary in summaries:
    n_comp = summary['n_components']
    acc = summary['test_acc_mean'] * 100
    acc_std = summary['test_acc_std'] * 100
    f1 = summary['test_f1_mean']
    f1_std = summary['test_f1_std']
    n_trials = len([t for t in all_trials if t['n_components'] == n_comp])
    print(f"{n_comp} comp{'':<8} {acc:>5.2f} ± {acc_std:>4.2f}     "
          f"{f1:>5.4f} ± {f1_std:>5.4f}   {n_trials}")
print("="*70)
