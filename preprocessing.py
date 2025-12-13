"""
EEG Preprocessing Pipeline for Upper Limb Movement Classification
Based on Gigascience 2020 Dataset

Prepares data for:
- CSP+LDA (bandpass 8-30 Hz)
- Random Forest (features from 8-30 Hz)
- SVM (features from 8-30 Hz)  
- CNN (broader filter 1-40 Hz)
"""

import numpy as np
import scipy.io as sio
from scipy import signal
from scipy.stats import zscore
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class EEGPreprocessor:
    """
    Preprocessing pipeline for motor cortex EEG signals.
    Handles trial extraction, filtering, and feature engineering.
    """
    
    # Motor cortex channels based on 10-20 system
    # These are crucial for motor control and movement-related signals
    MOTOR_CHANNELS = [
        # Primary motor cortex (M1) - central region
        'C3', 'C1', 'Cz', 'C2', 'C4',
        # Premotor/supplementary motor - frontal-central
        'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6',
        # Somatosensory cortex - central-parietal  
        'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6',
        # Extended motor network
        'C5', 'C6',
        # Temporal regions (motor planning)
        'FT7', 'FT8',
    ]
    
    def __init__(self, data_dir: str, output_dir: Optional[str] = None):
        """
        Initialize preprocessor.
        
        Parameters:
        -----------
        data_dir : str
            Directory containing raw .mat files
        output_dir : str, optional
            Directory for preprocessed outputs (default: data_dir/preprocessed)
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else self.data_dir / 'preprocessed'
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Recording parameters
        self.fs = 2500  # Sampling frequency (Hz)
        self.pre_trigger = 1.0  # seconds before movement trigger
        self.post_trigger = 2.0  # seconds after movement trigger
        
        # Movement types from dataset
        self.movement_types = [
            'multigrasp_realMove',
            'reaching_realMove', 
            'twist_realMove'
        ]
        
        print(f"EEG Preprocessor initialized")
        print(f"Data directory: {self.data_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Sampling rate: {self.fs} Hz")
        print(f"Trial window: -{self.pre_trigger}s to +{self.post_trigger}s")
        
    def _load_mat_file(self, filepath: Path) -> Optional[Dict]:
        """Load and validate .mat file."""
        try:
            data = sio.loadmat(filepath, simplify_cells=False)
            return data
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def _extract_channel_names(self, mat_data: Dict) -> List[str]:
        """
        Extract channel names from .mat structure.
        
        The data structure is: dat.clab contains channel labels
        """
        try:
            # Navigate nested MATLAB structure
            clab = mat_data['dat'][0, 0]['clab'][0]
            
            # Handle different array wrapping levels
            channel_names = []
            for ch in clab:
                if isinstance(ch, np.ndarray):
                    if ch.size > 0:
                        name = ch[0] if isinstance(ch[0], str) else str(ch[0])
                    else:
                        name = str(ch)
                else:
                    name = str(ch)
                channel_names.append(name.strip())
            
            return channel_names
            
        except Exception as e:
            print(f"Error extracting channel names: {e}")
            return []
    
    def _get_motor_channel_indices(self, all_channels: List[str]) -> Tuple[List[int], List[str]]:
        """
        Identify indices of motor cortex channels.
        
        Returns:
        --------
        indices : List[int]
            Channel indices
        names : List[str]
            Selected channel names
        """
        indices = []
        names = []
        
        for i, ch_name in enumerate(all_channels):
            # Clean and check channel name
            ch_clean = ch_name.strip().upper()
            
            # Match against motor channels
            for motor_ch in self.MOTOR_CHANNELS:
                if motor_ch.upper() == ch_clean:
                    indices.append(i)
                    names.append(ch_name)
                    break
        
        print(f"  Selected {len(indices)} motor cortex channels")
        print(f"  Channels: {names}")
        
        return indices, names
    
    def _load_channel_data(self, mat_data: Dict, channel_indices: List[int]) -> np.ndarray:
        """
        Load voltage data for selected channels.
        
        Returns:
        --------
        data : np.ndarray
            Shape (n_channels, n_timepoints)
        """
        channel_data = []
        
        for idx in channel_indices:
            # Channel data stored as ch1, ch2, ..., ch64
            ch_key = f'ch{idx + 1}'  # MATLAB uses 1-indexing
            
            if ch_key in mat_data:
                voltage = mat_data[ch_key]
                # Flatten to 1D array
                voltage = voltage.flatten() if voltage.ndim > 1 else voltage
                channel_data.append(voltage)
            else:
                print(f"  Warning: {ch_key} not found in data")
        
        return np.array(channel_data)
    
    def _extract_trigger_info(self, mat_data: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract trigger positions and class labels from markers.
        
        Returns:
        --------
        positions : np.ndarray
            Trigger sample positions
        labels : np.ndarray  
            Class labels for each trigger
        """
        try:
            mrk = mat_data['mrk'][0, 0]
            
            # Trigger positions (convert from MATLAB 1-indexing to Python 0-indexing)
            positions = mrk['pos'].flatten() - 1
            
            # Class labels
            y = mrk['y']
            
            # Handle one-hot encoding (common in this dataset)
            if y.ndim > 1 and y.shape[0] > 1:
                # One-hot encoded: convert to class indices
                labels = np.argmax(y, axis=0)
            else:
                labels = y.flatten()
            
            print(f"  Found {len(positions)} triggers with {len(np.unique(labels))} unique classes")
            
            return positions, labels
            
        except Exception as e:
            print(f"  Error extracting trigger info: {e}")
            return np.array([]), np.array([])
    
    def _extract_trials(
        self, 
        data: np.ndarray, 
        trigger_positions: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract trial segments from continuous data.
        
        Parameters:
        -----------
        data : np.ndarray
            Continuous EEG data (n_channels, n_samples)
        trigger_positions : np.ndarray
            Sample indices of movement triggers
        labels : np.ndarray
            Class label for each trigger
            
        Returns:
        --------
        trials : np.ndarray
            Shape (n_trials, n_channels, n_timepoints)
        trial_labels : np.ndarray
            Corresponding labels
        """
        n_channels = data.shape[0]
        pre_samples = int(self.pre_trigger * self.fs)
        post_samples = int(self.post_trigger * self.fs)
        trial_length = pre_samples + post_samples
        
        trials = []
        trial_labels = []
        
        for i, pos in enumerate(trigger_positions):
            # Define trial window
            start_idx = int(pos - pre_samples)
            end_idx = int(pos + post_samples)
            
            # Check bounds
            if start_idx >= 0 and end_idx <= data.shape[1]:
                trial = data[:, start_idx:end_idx]
                
                # Verify correct length
                if trial.shape[1] == trial_length:
                    trials.append(trial)
                    trial_labels.append(labels[i])
                else:
                    print(f"  Warning: Trial {i} has incorrect length {trial.shape[1]}, skipping")
            else:
                print(f"  Warning: Trial {i} out of bounds, skipping")
        
        trials = np.array(trials)
        trial_labels = np.array(trial_labels)
        
        print(f"  Extracted {len(trials)} valid trials")
        print(f"  Trial shape: {trials[0].shape if len(trials) > 0 else 'N/A'}")
        
        return trials, trial_labels
    
    def apply_bandpass_filter(
        self,
        data: np.ndarray,
        lowcut: float,
        highcut: float,
        order: int = 4
    ) -> np.ndarray:
        """
        Apply Butterworth bandpass filter.
        
        Parameters:
        -----------
        data : np.ndarray
            Input data (trials × channels × time) or (channels × time)
        lowcut : float
            Low cutoff frequency (Hz)
        highcut : float
            High cutoff frequency (Hz)
        order : int
            Filter order
            
        Returns:
        --------
        filtered : np.ndarray
            Filtered data (same shape as input)
        """
        nyquist = 0.5 * self.fs
        low = lowcut / nyquist
        high = highcut / nyquist
        
        # Design filter
        b, a = signal.butter(order, [low, high], btype='band')
        
        # Apply based on dimensionality
        if data.ndim == 3:  # (trials, channels, time)
            filtered = np.zeros_like(data)
            for trial_idx in range(data.shape[0]):
                for ch_idx in range(data.shape[1]):
                    filtered[trial_idx, ch_idx, :] = signal.filtfilt(
                        b, a, data[trial_idx, ch_idx, :]
                    )
                    
        elif data.ndim == 2:  # (channels, time)
            filtered = np.zeros_like(data)
            for ch_idx in range(data.shape[0]):
                filtered[ch_idx, :] = signal.filtfilt(b, a, data[ch_idx, :])
        else:
            raise ValueError(f"Unsupported data shape: {data.shape}")
        
        return filtered
    
    def extract_band_power_features(
        self,
        trials: np.ndarray,
        bands: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> np.ndarray:
        """
        Extract band power + statistical features for ML models.
        
        Features per channel:
        - Mu band power (8-12 Hz)
        - Beta band power (13-30 Hz)
        - Mean amplitude
        - Standard deviation
        
        Total: 4 features × n_channels = 240 features (for 60 channels)
        
        Parameters:
        -----------
        trials : np.ndarray
            Shape (n_trials, n_channels, n_samples)
        bands : dict, optional
            Frequency bands {'name': (low, high)}
            
        Returns:
        --------
        features : np.ndarray
            Shape (n_trials, n_features)
        """
        if bands is None:
            bands = {
                'mu': (8, 12),
                'beta': (13, 30)
            }
        
        n_trials, n_channels, n_samples = trials.shape
        
        features_list = []
        
        for trial in trials:
            trial_features = []
            
            # 1. Band power features
            for band_name, (low, high) in bands.items():
                # Filter to band
                filtered = self.apply_bandpass_filter(trial, low, high)
                # Compute power (mean of squared amplitudes)
                power = np.mean(filtered**2, axis=1)
                trial_features.extend(power)
            
            # 2. Mean amplitude (DC component removed by prior filtering)
            mean_amp = np.mean(trial, axis=1)
            trial_features.extend(mean_amp)
            
            # 3. Standard deviation (signal variability)
            std_amp = np.std(trial, axis=1)
            trial_features.extend(std_amp)
            
            features_list.append(trial_features)
        
        features = np.array(features_list)
        
        n_features = features.shape[1]
        expected = len(bands) * n_channels + 2 * n_channels
        print(f"  Extracted {n_features} features per trial (expected {expected})")
        
        return features
    
    def process_file(
        self,
        subject: str,
        session: str,
        movement_type: str,
        filter_low: float,
        filter_high: float
    ) -> Optional[Dict]:
        """
        Process single .mat file.
        
        Parameters:
        -----------
        subject : str
            e.g., 'sub1'
        session : str
            e.g., 'session1'
        movement_type : str
            One of: multigrasp_realMove, reaching_realMove, twist_realMove
        filter_low : float
            Bandpass low cutoff (Hz)
        filter_high : float
            Bandpass high cutoff (Hz)
            
        Returns:
        --------
        processed : dict
            {
                'trials': ndarray (n_trials, n_channels, n_samples),
                'labels': ndarray (n_trials,),
                'channel_names': list,
                'metadata': dict
            }
        """
        # Construct filename
        filename = f"{session}_{subject}_{movement_type}.mat"
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            print(f"File not found: {filepath}")
            return None
        
        print(f"\nProcessing: {filename}")
        print(f"  Filter: {filter_low}-{filter_high} Hz")
        
        # Load data
        mat_data = self._load_mat_file(filepath)
        if mat_data is None:
            return None
        
        # Extract channel information
        all_channels = self._extract_channel_names(mat_data)
        if not all_channels:
            return None
        
        # Select motor cortex channels
        channel_indices, channel_names = self._get_motor_channel_indices(all_channels)
        if len(channel_indices) == 0:
            print("  Error: No motor cortex channels found")
            return None
        
        # Load voltage data
        eeg_data = self._load_channel_data(mat_data, channel_indices)
        if eeg_data.size == 0:
            return None
        
        # Extract trigger information
        trigger_pos, labels = self._extract_trigger_info(mat_data)
        if len(trigger_pos) == 0:
            return None
        
        # Extract trials
        trials, trial_labels = self._extract_trials(eeg_data, trigger_pos, labels)
        if len(trials) == 0:
            return None
        
        # Apply bandpass filter
        print(f"  Applying {filter_low}-{filter_high} Hz bandpass filter...")
        filtered_trials = self.apply_bandpass_filter(trials, filter_low, filter_high)
        
        # Package results
        processed = {
            'trials': filtered_trials,
            'labels': trial_labels,
            'channel_names': channel_names,
            'metadata': {
                'subject': subject,
                'session': session,
                'movement_type': movement_type,
                'fs': self.fs,
                'filter': (filter_low, filter_high),
                'n_trials': len(trial_labels),
                'n_channels': len(channel_names),
                'n_samples': filtered_trials.shape[2]
            }
        }
        
        print(f"  ✓ Successfully processed {len(trial_labels)} trials")
        
        return processed
    
    def process_dataset(
        self,
        subjects: Optional[List[str]] = None,
        sessions: Optional[List[str]] = None,
        filter_configs: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Dict[str, List[Dict]]:
        """
        Process entire dataset with different filter configurations.
        
        Parameters:
        -----------
        subjects : list, optional
            Subject IDs (default: sub1 to sub25)
        sessions : list, optional  
            Session IDs (default: session1 to session3)
        filter_configs : dict, optional
            Filter settings for each model type
            
        Returns:
        --------
        all_processed : dict
            {
                'csp_lda': [processed_file1, processed_file2, ...],
                'ml_features': [...],
                'cnn': [...]
            }
        """
        # Default parameters
        if subjects is None:
            subjects = [f'sub{i}' for i in range(1, 26)]
        
        if sessions is None:
            sessions = [f'session{i}' for i in range(1, 4)]
        
        if filter_configs is None:
            filter_configs = {
                'csp_lda': (8, 30),      # Mu + Beta bands
                'ml_features': (8, 30),   # Same for Random Forest/SVM
                'cnn': (1, 40)            # Broader for CNN
            }
        
        # Initialize storage
        all_processed = {config_name: [] for config_name in filter_configs}
        
        total_files = len(subjects) * len(sessions) * len(self.movement_types)
        processed_count = 0
        
        print(f"\n{'='*70}")
        print(f"PROCESSING {total_files} FILES ACROSS {len(filter_configs)} CONFIGURATIONS")
        print(f"{'='*70}")
        
        # Process each configuration
        for config_name, (low, high) in filter_configs.items():
            print(f"\n{'='*70}")
            print(f"CONFIGURATION: {config_name.upper()}")
            print(f"Filter: {low}-{high} Hz")
            print(f"{'='*70}")
            
            for subject in subjects:
                for session in sessions:
                    for movement_type in self.movement_types:
                        processed = self.process_file(
                            subject, session, movement_type, low, high
                        )
                        
                        if processed is not None:
                            all_processed[config_name].append(processed)
                            processed_count += 1
        
        print(f"\n{'='*70}")
        print(f"PROCESSING COMPLETE")
        print(f"Successfully processed {processed_count}/{total_files * len(filter_configs)} file-config combinations")
        print(f"{'='*70}")
        
        # Summary statistics
        for config_name, data_list in all_processed.items():
            if data_list:
                total_trials = sum(d['metadata']['n_trials'] for d in data_list)
                print(f"\n{config_name}:")
                print(f"  Files: {len(data_list)}")
                print(f"  Total trials: {total_trials}")
        
        return all_processed
    
    def prepare_for_csp_lda(self, processed_files: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Combine processed files for CSP+LDA.
        
        CSP operates on raw trials, not features.
        
        Returns:
        --------
        X : ndarray (n_trials, n_channels, n_samples)
        y : ndarray (n_trials,)
        """
        X = np.concatenate([f['trials'] for f in processed_files], axis=0)
        y = np.concatenate([f['labels'] for f in processed_files], axis=0)
        
        print(f"\nCSP+LDA data prepared:")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  Classes: {np.unique(y)}")
        
        return X, y
    
    def prepare_for_ml(self, processed_files: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features for Random Forest and SVM.
        
        Returns:
        --------
        X : ndarray (n_trials, n_features)
        y : ndarray (n_trials,)
        """
        # Concatenate all trials
        all_trials = np.concatenate([f['trials'] for f in processed_files], axis=0)
        y = np.concatenate([f['labels'] for f in processed_files], axis=0)
        
        # Extract features
        print(f"\nExtracting features for ML models...")
        X = self.extract_band_power_features(all_trials)
        
        print(f"\nML features prepared:")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  Classes: {np.unique(y)}")
        
        return X, y
    
    def prepare_for_cnn(self, processed_files: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for CNN (same as CSP but broader filter).
        
        Returns:
        --------
        X : ndarray (n_trials, n_channels, n_samples)
        y : ndarray (n_trials,)
        """
        X = np.concatenate([f['trials'] for f in processed_files], axis=0)
        y = np.concatenate([f['labels'] for f in processed_files], axis=0)
        
        print(f"\nCNN data prepared:")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  Classes: {np.unique(y)}")
        
        return X, y
    
    def save_preprocessed(
        self,
        all_processed: Dict[str, List[Dict]],
        include_metadata: bool = True
    ):
        """
        Save preprocessed data to disk.
        
        Parameters:
        -----------
        all_processed : dict
            Output from process_dataset()
        include_metadata : bool
            Whether to save metadata alongside data
        """
        print(f"\n{'='*70}")
        print(f"SAVING PREPROCESSED DATA")
        print(f"{'='*70}")
        
        for config_name, file_list in all_processed.items():
            if not file_list:
                print(f"\nWarning: No data for {config_name}, skipping")
                continue
            
            print(f"\n{config_name}:")
            
            # Prepare data based on model type
            if config_name == 'csp_lda':
                X, y = self.prepare_for_csp_lda(file_list)
            elif config_name == 'ml_features':
                X, y = self.prepare_for_ml(file_list)
            elif config_name == 'cnn':
                X, y = self.prepare_for_cnn(file_list)
            else:
                print(f"  Unknown config: {config_name}, skipping")
                continue
            
            # Save
            output_path = self.output_dir / f'{config_name}_data.npz'
            
            save_dict = {'X': X, 'y': y}
            
            if include_metadata:
                # Aggregate metadata
                metadata = {
                    'n_files': len(file_list),
                    'n_trials': len(y),
                    'n_classes': len(np.unique(y)),
                    'channel_names': file_list[0]['channel_names'],
                    'fs': self.fs,
                    'filter': file_list[0]['metadata']['filter']
                }
                save_dict['metadata'] = metadata
            
            np.savez_compressed(output_path, **save_dict)
            
            print(f"  ✓ Saved to: {output_path}")
            print(f"  File size: {output_path.stat().st_size / 1e6:.1f} MB")
        
        print(f"\n{'='*70}")
        print(f"ALL DATA SAVED")
        print(f"{'='*70}")


def main():
    """Example usage of preprocessing pipeline."""
    
    # Initialize
    preprocessor = EEGPreprocessor(
        data_dir='/mnt/user-data/uploads',  # Update this path
        output_dir='/home/ubuntu/multimodal-signal-dataset-for-11-upper-body-movements/PreprocessedData'
    )
    
    # Process all data (or specify subset)
    # To test: use subjects=['sub1'], sessions=['session1']
    all_processed = preprocessor.process_dataset(
        subjects=None,  # All 25 subjects
        sessions=None,  # All 3 sessions
        filter_configs={
            'csp_lda': (8, 30),
            'ml_features': (8, 30),
            'cnn': (8, 30)
        }
    )
    
    # Save preprocessed data
    preprocessor.save_preprocessed(all_processed)
    
    print("\n✓ Preprocessing complete!")
    print(f"Preprocessed data saved to: {preprocessor.output_dir}")


if __name__ == "__main__":
    main()