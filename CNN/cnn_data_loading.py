"""
Data Loading and Preprocessing for CNN Pipeline
SYDE 522 Final Project

Handles loading EEG data from the Gigascience 2020 dataset and applies
minimal preprocessing for 1D-CNN:
- Bandpass filtering (1-40 Hz) - broader than CSP to let CNN learn features
- Segmentation into 2-second trials
- Artifact rejection (optional)
- Normalization
"""

import numpy as np
import mne
from scipy import signal
from pathlib import Path
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def bandpass_filter(data: np.ndarray,
                    sfreq: float = 2500,
                    low_freq: float = 1.0,
                    high_freq: float = 40.0,
                    order: int = 5) -> np.ndarray:
    """
    Apply Butterworth bandpass filter to EEG data
    
    Args:
        data: EEG data (n_channels, n_timepoints)
        sfreq: Sampling frequency in Hz
        low_freq: Lower cutoff frequency
        high_freq: Upper cutoff frequency
        order: Filter order
        
    Returns:
        Filtered data (same shape as input)
        
    Note: Uses zero-phase filtering (filtfilt) to avoid phase distortion
    """
    nyquist = sfreq / 2
    low = low_freq / nyquist
    high = high_freq / nyquist
    
    # Design filter
    b, a = signal.butter(order, [low, high], btype='band')
    
    # Apply filter (zero-phase)
    filtered = signal.filtfilt(b, a, data, axis=1)
    
    return filtered


def segment_trials(data: np.ndarray,
                   events: np.ndarray,
                   sfreq: float = 2500,
                   tmin: float = 0.0,
                   tmax: float = 2.0) -> np.ndarray:
    """
    Segment continuous EEG into trials
    
    Args:
        data: Continuous EEG (n_channels, n_timepoints)
        events: Event markers (n_events, 3) - [sample, 0, event_id]
        sfreq: Sampling frequency
        tmin: Start time relative to event (seconds)
        tmax: End time relative to event (seconds)
        
    Returns:
        Segmented trials (n_trials, n_channels, n_samples)
    """
    n_channels = data.shape[0]
    n_samples = int((tmax - tmin) * sfreq)
    n_events = len(events)
    
    trials = np.zeros((n_events, n_channels, n_samples))
    
    for i, event in enumerate(events):
        start_sample = event[0] + int(tmin * sfreq)
        end_sample = start_sample + n_samples
        
        # Check bounds
        if start_sample < 0 or end_sample > data.shape[1]:
            warnings.warn(f"Trial {i} exceeds data bounds, skipping")
            continue
        
        trials[i] = data[:, start_sample:end_sample]
    
    return trials


def normalize_trials(trials: np.ndarray,
                     method: str = 'standardize') -> np.ndarray:
    """
    Normalize trial data
    
    Args:
        trials: (n_trials, n_channels, n_timepoints)
        method: 'standardize' (z-score) or 'minmax'
        
    Returns:
        Normalized trials (same shape)
    """
    if method == 'standardize':
        # Z-score per channel across all trials
        mean = trials.mean(axis=(0, 2), keepdims=True)
        std = trials.std(axis=(0, 2), keepdims=True)
        normalized = (trials - mean) / (std + 1e-8)
        
    elif method == 'minmax':
        # Scale to [0, 1] per channel
        min_val = trials.min(axis=(0, 2), keepdims=True)
        max_val = trials.max(axis=(0, 2), keepdims=True)
        normalized = (trials - min_val) / (max_val - min_val + 1e-8)
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized


def reject_artifacts(trials: np.ndarray,
                    labels: np.ndarray,
                    threshold_uV: float = 100.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reject trials with extreme amplitudes (artifacts)
    
    Args:
        trials: (n_trials, n_channels, n_timepoints)
        labels: (n_trials,)
        threshold_uV: Rejection threshold in microvolts
        
    Returns:
        (clean_trials, clean_labels)
    """
    # Compute max absolute amplitude per trial
    max_amps = np.abs(trials).max(axis=(1, 2))
    
    # Keep trials below threshold
    keep_mask = max_amps < threshold_uV
    
    n_rejected = (~keep_mask).sum()
    print(f"Rejected {n_rejected}/{len(trials)} trials ({n_rejected/len(trials)*100:.1f}%)")
    
    return trials[keep_mask], labels[keep_mask]


def preprocess_for_cnn(raw_data: np.ndarray,
                      events: np.ndarray,
                      labels: np.ndarray,
                      sfreq: float = 2500,
                      low_freq: float = 1.0,
                      high_freq: float = 40.0,
                      tmin: float = 0.0,
                      tmax: float = 2.0,
                      reject_artifacts_flag: bool = True,
                      normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Complete preprocessing pipeline for CNN
    
    Args:
        raw_data: Continuous EEG (n_channels, n_timepoints)
        events: Event markers (n_events, 3)
        labels: Movement labels (n_events,)
        sfreq: Sampling frequency
        low_freq, high_freq: Filter cutoffs
        tmin, tmax: Trial epoch window
        reject_artifacts_flag: Whether to reject artifact trials
        normalize: Whether to normalize
        
    Returns:
        (X, y) where X is (n_trials, n_channels, n_samples), y is (n_trials,)
    """
    print(f"\n{'Preprocessing for CNN':-^60}")
    print(f"Input: {raw_data.shape[0]} channels, {raw_data.shape[1]} samples ({raw_data.shape[1]/sfreq:.1f}s)")
    print(f"Events: {len(events)}")
    
    # 1. Bandpass filter (1-40 Hz for CNN - broader than CSP)
    print(f"\n1. Bandpass filtering: {low_freq}-{high_freq} Hz")
    filtered_data = bandpass_filter(raw_data, sfreq, low_freq, high_freq)
    
    # 2. Segment into trials
    print(f"\n2. Segmenting trials: {tmin} to {tmax}s")
    trials = segment_trials(filtered_data, events, sfreq, tmin, tmax)
    print(f"   Trial shape: {trials.shape}")
    
    # 3. Artifact rejection (optional)
    if reject_artifacts_flag:
        print(f"\n3. Artifact rejection (threshold: 100 µV)")
        trials, labels = reject_artifacts(trials, labels, threshold_uV=100.0)
    else:
        print(f"\n3. Skipping artifact rejection")
    
    # 4. Normalization
    if normalize:
        print(f"\n4. Normalizing (z-score per channel)")
        trials = normalize_trials(trials, method='standardize')
    else:
        print(f"\n4. Skipping normalization")
    
    print(f"\n{'Preprocessing complete':-^60}")
    print(f"Final shape: {trials.shape}")
    print(f"Labels: {len(labels)} (classes: {len(np.unique(labels))})")
    
    return trials, labels


def load_subject_data(subject_id: int,
                     session: int = 1,
                     condition: str = 'real',
                     data_dir: str = 'data',
                     channel_subset: Optional[list] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load data for a single subject
    
    Args:
        subject_id: Subject number (1-25)
        session: Session number (1-3)
        condition: 'real' or 'imagined'
        data_dir: Root data directory
        channel_subset: List of channel names to keep (None = all motor cortex channels)
        
    Returns:
        (raw_data, events, labels)
        
    Note: This is a template - adjust paths based on actual dataset structure
    """
    data_path = Path(data_dir)
    
    # Example file structure (adjust as needed):
    # data/subject_01/session_1/real_movements.fif
    subject_dir = data_path / f'subject_{subject_id:02d}'
    session_dir = subject_dir / f'session_{session}'
    eeg_file = session_dir / f'{condition}_movements.fif'
    
    if not eeg_file.exists():
        raise FileNotFoundError(f"Data file not found: {eeg_file}")
    
    print(f"\nLoading: Subject {subject_id}, Session {session}, {condition.upper()} movements")
    
    # Load with MNE
    raw = mne.io.read_raw_fif(eeg_file, preload=True, verbose=False)
    
    # Select channels
    if channel_subset is None:
        # Default: motor cortex channels (FC, C, CP regions)
        channel_subset = [ch for ch in raw.ch_names 
                         if any(region in ch for region in ['FC', 'C', 'CP'])]
    
    raw.pick_channels(channel_subset)
    
    # Get data
    data = raw.get_data()  # (n_channels, n_timepoints)
    sfreq = raw.info['sfreq']
    
    # Get events
    events = mne.find_events(raw, verbose=False)
    
    # Extract labels from event IDs
    # Assumes events are coded as: 1-11 for the 11 movements
    labels = events[:, 2] - 1  # Convert to 0-indexed
    
    print(f"  Channels: {len(channel_subset)} (motor cortex)")
    print(f"  Sampling rate: {sfreq} Hz")
    print(f"  Duration: {data.shape[1]/sfreq:.1f}s")
    print(f"  Events: {len(events)}")
    print(f"  Label distribution: {np.bincount(labels)}")
    
    return data, events, labels


def load_multiple_subjects(subject_ids: list,
                          session: int = 1,
                          condition: str = 'real',
                          data_dir: str = 'data',
                          preprocess: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and combine data from multiple subjects
    
    Args:
        subject_ids: List of subject IDs to load
        session: Session number
        condition: 'real' or 'imagined'
        data_dir: Data directory
        preprocess: Whether to apply preprocessing
        
    Returns:
        (X, y) combined across subjects
    """
    all_trials = []
    all_labels = []
    
    for subject_id in subject_ids:
        try:
            raw_data, events, labels = load_subject_data(
                subject_id, session, condition, data_dir
            )
            
            if preprocess:
                trials, labels = preprocess_for_cnn(raw_data, events, labels)
            else:
                trials = segment_trials(raw_data, events)
            
            all_trials.append(trials)
            all_labels.append(labels)
            
        except Exception as e:
            print(f"Warning: Failed to load subject {subject_id}: {e}")
            continue
    
    # Concatenate
    X = np.concatenate(all_trials, axis=0)
    y = np.concatenate(all_labels, axis=0)
    
    print(f"\n{'Combined dataset':-^60}")
    print(f"Total trials: {len(X)}")
    print(f"Shape: {X.shape}")
    print(f"Label distribution: {np.bincount(y)}")
    
    return X, y


# Example usage and testing
if __name__ == '__main__':
    # Test preprocessing with synthetic data
    print("Testing preprocessing pipeline with synthetic data...")
    
    # Synthetic EEG: 60 channels, 10 seconds at 2500 Hz
    n_channels = 60
    duration = 10.0  # seconds
    sfreq = 2500
    n_samples = int(duration * sfreq)
    
    # Generate random data (in practice, this would be real EEG)
    np.random.seed(42)
    raw_data = np.random.randn(n_channels, n_samples) * 10  # microvolts
    
    # Synthetic events: 11 movements, 10 trials each
    n_events = 110
    event_spacing = n_samples // (n_events + 1)
    event_samples = np.arange(event_spacing, n_samples, event_spacing)[:n_events]
    event_ids = np.repeat(np.arange(11), 10) + 1  # 1-11
    events = np.column_stack([event_samples, np.zeros(n_events, dtype=int), event_ids])
    labels = event_ids - 1  # 0-10
    
    # Preprocess
    X, y = preprocess_for_cnn(
        raw_data, events, labels,
        sfreq=sfreq,
        low_freq=1.0,
        high_freq=40.0,
        tmin=0.0,
        tmax=2.0,
        reject_artifacts_flag=True,
        normalize=True
    )
    
    print(f"\n{'Test complete!':-^60}")
    print(f"Output shape: {X.shape}")
    print(f"Expected: (110, 60, 5000)")
    print(f"Labels shape: {y.shape}")
    print(f"Data range: [{X.min():.2f}, {X.max():.2f}]")
