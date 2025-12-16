"""Read and display checkpoint file contents."""
import pickle
from pathlib import Path

checkpoint_path = Path('experiment_checkpoint.pkl')

try:
    with open(checkpoint_path, 'rb') as f:
        data = pickle.load(f)

    print(f"Checkpoint file loaded successfully!")
    print(f"Type: {type(data)}\n")

    if isinstance(data, dict):
        print("Keys:", list(data.keys()))
        print("\n" + "="*60)

        # Print timestamp if available
        if 'timestamp' in data:
            print(f"Timestamp: {data['timestamp']}")

        # Print trial counter
        if 'trial_counter' in data:
            print(f"Trial counter: {data['trial_counter']}")

        # Print completed trials
        if 'completed_trials' in data:
            completed = data['completed_trials']
            print(f"\nCompleted trials: {len(completed)}")
            print("="*60)

            for trial_key, trial_data in completed.items():
                print(f"\n{trial_key}:")
                for key, value in trial_data.items():
                    if key not in ['confusion_matrix', 'test_f1_per_class', 'y_test_pred', 'y_test_true']:
                        print(f"  {key}: {value}")

        # If data structure is different, print everything
        else:
            print("\nFull data:")
            for key, value in data.items():
                if isinstance(value, dict) and len(value) > 3:
                    print(f"{key}: {type(value)} with {len(value)} items")
                else:
                    print(f"{key}: {value}")

    else:
        print("Data:", data)

except Exception as e:
    print(f"Error reading checkpoint: {e}")
    print(f"Error type: {type(e).__name__}")
