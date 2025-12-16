"""
Extract CSP+LDA results from checkpoint file and regenerate complete results
"""
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# Try to load checkpoint with all trials
checkpoint_path = Path("EEG-Upper-Movement-Classification/CSP_LDA/experiment_checkpoint.pkl")

if checkpoint_path.exists():
    print("Found checkpoint file!")
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)

    print(f"\nCheckpoint info:")
    print(f"  Timestamp: {checkpoint.get('timestamp', 'N/A')}")
    print(f"  Trial counter: {checkpoint.get('trial_counter', 'N/A')}")
    print(f"  Completed trials: {checkpoint.get('n_completed', 'N/A')}")

    completed_trials = checkpoint.get('completed_trials', {})

    print(f"\nCompleted trial keys:")
    for key in completed_trials.keys():
        print(f"  - {key}")

    # Convert to list of results
    trial_results = list(completed_trials.values())

    # Create DataFrame
    trial_df = pd.DataFrame([
        {k: v for k, v in t.items() if not isinstance(v, np.ndarray)}
        for t in trial_results
    ])

    print(f"\nTrial results summary:")
    print(trial_df[['band', 'n_components', 'seed', 'test_accuracy', 'test_f1_macro']])

    # Save to proper location
    output_dir = Path("EEG-Upper-Movement-Classification/CSP_LDA/results_csp_lda")
    output_dir.mkdir(exist_ok=True)

    # Save pickle
    with open(output_dir / 'trial_results.pkl', 'wb') as f:
        pickle.dump(trial_results, f)

    # Save CSV
    trial_df.to_csv(output_dir / 'trial_results.csv', index=False)

    print(f"\n✓ Saved complete results to {output_dir}")
    print(f"  - trial_results.pkl")
    print(f"  - trial_results.csv")

    # Aggregate by configuration
    configs = trial_df.groupby(['band', 'n_components']).agg({
        'test_accuracy': ['mean', 'std', 'count'],
        'test_f1_macro': ['mean', 'std']
    }).reset_index()

    print(f"\nConfiguration summary:")
    for _, row in configs.iterrows():
        band = row[('band', '')]
        n_comp = row[('n_components', '')]
        acc_mean = row[('test_accuracy', 'mean')]
        acc_std = row[('test_accuracy', 'std')]
        n_trials = row[('test_accuracy', 'count')]

        print(f"  {band}, {n_comp} components ({int(n_trials)} trials): "
              f"{acc_mean*100:.2f}% ± {acc_std*100:.2f}%")

else:
    print("No checkpoint file found in EEG-Upper-Movement-Classification/CSP_LDA/")
    print("\nLooking for alternative locations...")

    # Search for checkpoint files
    import subprocess
    result = subprocess.run(
        ['find', 'EEG-Upper-Movement-Classification/CSP_LDA', '-name', '*checkpoint*'],
        capture_output=True, text=True
    )

    if result.stdout.strip():
        print(f"Found checkpoints at:")
        print(result.stdout)
    else:
        print("No checkpoint files found anywhere.")
