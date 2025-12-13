"""
EEG Preprocessing Pipeline for Upper Limb Movement Classification
Based on Gigascience 2020 Dataset

Includes ICA-based EOG artifact removal as described in the paper.

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
from sklearn.decomposition import FastICA
from scipy.spatial.distance import correlation
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')


class EEGPreprocessor:
    """
    Preprocessing pipeline for motor cortex EEG signals.
    Handles trial extraction, filtering, ICA-based artifact removal, and feature engineering.
    
    EOG Artifact Removal:
    - Loads EOG data from separate files (e.g., EOG_session1_sub1_multigrasp_realMove.mat)
    - Each EOG file contains 4 channels (ch1, ch2, ch3, ch4)
    - Uses ICA to identify and remove EOG-contaminated components from EEG
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
    
    # Unified class mapping
    # Maps (movement_type, raw_class_index) -> unified_class_label
    CLASS_MAPPING = {
        # Reaching movements (6 directions + rest)
        # Raw indices after argmax on one-hot: 0=S11(Forward), 1=S21(Backward), 
        # 2=S31(Left), 3=S41(Right), 4=S51(Up), 5=S61(Down), 6=S8(Rest)
        'reaching_realMove': {
            0: 1,   # Forward
            1: 2,   # Backward
            2: 3,   # Left
            3: 4,   # Right
            4: 5,   # Up
            5: 6,   # Down
            6: 0,   # Rest
        },
        # Hand grasping (3 grips + rest)
        # Raw indices: 0=S11(Cup), 1=S21(Ball), 2=S61(Card), 3=S8(Rest)
        'multigrasp_realMove': {
            0: 7,   # Cup
            1: 8,   # Ball
            2: 9,   # Card
            3: 0,   # Rest
        },
        # Wrist twisting (2 rotations + rest)
        # Raw indices: 0=S91(Pronation), 1=S101(Supination), 2=S8(Rest)
        'twist_realMove': {
            0: 10,  # Pronation
            1: 11,  # Supination
            2: 0,   # Rest
        }
    }
    
    # Class names for reference
    CLASS_NAMES = {
        0: 'Rest',
        1: 'Reach Forward',
        2: 'Reach Backward', 
        3: 'Reach Left',
        4: 'Reach Right',
        5: 'Reach Up',
        6: 'Reach Down',
        7: 'Grasp Cup',
        8: 'Grasp Ball',
        9: 'Grasp Card',
        10: 'Twist Pronation',
        11: 'Twist Supination'
    }
    
    def __init__(self, data_dir: str, output_dir: Optional[str] = None, 
                 eog_dir: Optional[str] = None,
                 use_ica_artifact_removal: bool = True):
        """
        Initialize preprocessor.
        
        Parameters:
        -----------
        data_dir : str
            Directory containing raw EEG .mat files
        output_dir : str, optional
            Directory for preprocessed outputs (default: data_dir/preprocessed)
        eog_dir : str, optional
            Directory containing EOG .mat files (default: same as data_dir)
        use_ica_artifact_removal : bool
            Whether to apply ICA-based EOG artifact removal (default: True)
        """
        self.data_dir = Path(data_dir)
        self.eog_dir = Path(eog_dir) if eog_dir else self.data_dir
        self.output_dir = Path(output_dir) if output_dir else self.data_dir / 'preprocessed'
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Recording parameters
        self.fs = 2500  # Sampling frequency (Hz)
        self.pre_trigger = 1.0  # seconds before movement trigger
        self.post_trigger = 2.0  # seconds after movement trigger
        
        # ICA parameters
        self.use_ica_artifact_removal = use_ica_artifact_removal
        self.ica_correlation_threshold = 0.8  # LOWERED from 0.9 for better detection
        
        # Movement types from dataset
        self.movement_types = [
            'multigrasp_realMove',
            'reaching_realMove', 
            'twist_realMove'
        ]
        
        # Statistics for validation
        self.artifact_removal_stats = {
            'files_processed': 0,
            'files_with_eog': 0,
            'total_components_removed': 0,
            'avg_components_removed': 0
        }
        
        print(f"EEG Preprocessor initialized")
        print(f"EEG data directory: {self.data_dir}")
        print(f"EOG data directory: {self.eog_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Sampling rate: {self.fs} Hz")
        print(f"Trial window: -{self.pre_trigger}s to +{self.post_trigger}s")
        print(f"ICA artifact removal: {'ENABLED' if self.use_ica_artifact_removal else 'DISABLED'}")
        if self.use_ica_artifact_removal:
            print(f"ICA correlation threshold: {self.ica_correlation_threshold}")
        
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
    
    def _get_motor_channel_indices(self, all_channels: List[str], verbose: bool = False) -> Tuple[List[int], List[str]]:
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
        
        if verbose:
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
    
    def _load_eog_from_separate_file(self, subject: str, session: str, movement_type: str, 
                                      verbose: bool = False) -> Optional[np.ndarray]:
        """
        Load EOG data from separate EOG .mat file.
        
        EOG files have structure: ch1, ch2, ch3, ch4 (4 EOG channels)
        Filename format: EOG_session1_sub1_multigrasp_realMove.mat
        
        Parameters:
        -----------
        subject : str
            e.g., 'sub1'
        session : str
            e.g., 'session1'
        movement_type : str
            One of: multigrasp_realMove, reaching_realMove, twist_realMove
            
        Returns:
        --------
        eog_data : np.ndarray or None
            EOG channel data (n_eog_channels, n_samples) or None if not found
        """
        # Construct EOG filename: EOG_session1_sub1_multigrasp_realMove.mat
        filename = f"EOG_{session}_{subject}_{movement_type}.mat"
        filepath = self.eog_dir / filename  # Use self.eog_dir instead of self.data_dir
        
        if not filepath.exists():
            if verbose:
                print(f"  EOG file not found: {filepath}")
            return None
        
        # Load EOG file
        mat_data = self._load_mat_file(filepath)
        if mat_data is None:
            if verbose:
                print(f"  Failed to load EOG file: {filename}")
            return None
        
        # EOG files have structure: ch1, ch2, ch3, ch4
        # Extract all 4 EOG channels
        eog_channels = []
        n_eog_channels = 4  # As shown in the image
        
        for ch_idx in range(1, n_eog_channels + 1):
            ch_key = f'ch{ch_idx}'
            
            if ch_key in mat_data:
                voltage = mat_data[ch_key]
                # Flatten to 1D array
                voltage = voltage.flatten() if voltage.ndim > 1 else voltage
                eog_channels.append(voltage)
            else:
                if verbose:
                    print(f"  Warning: {ch_key} not found in EOG file")
        
        if len(eog_channels) == 0:
            if verbose:
                print(f"  No EOG channels found in {filename}")
            return None
        
        eog_data = np.array(eog_channels)
        
        if verbose:
            print(f"  ✓ Loaded {len(eog_channels)} EOG channels from {filename}")
            print(f"    EOG data shape: {eog_data.shape}")
        
        return eog_data
    
    def _apply_ica_artifact_removal(self, eeg_data: np.ndarray, eog_data: np.ndarray, 
                                    verbose: bool = False) -> Tuple[np.ndarray, int]:
        """
        Apply ICA-based artifact removal using EOG reference.
        
        Implementation based on infomax ICA as described in the paper.
        Uses downsampling for efficiency on long continuous recordings.
        
        Parameters:
        -----------
        eeg_data : np.ndarray
            EEG data (n_channels, n_samples)
        eog_data : np.ndarray
            EOG reference data (n_eog_channels, n_samples)
            
        Returns:
        --------
        cleaned_eeg : np.ndarray
            Artifact-removed EEG data (n_channels, n_samples)
        n_components_removed : int
            Number of ICA components removed
        """
        n_channels, n_samples = eeg_data.shape
        
        if verbose:
            print(f"    EEG shape: {eeg_data.shape}, EOG shape: {eog_data.shape}")
        
        # For very long recordings (>500k samples), downsample for ICA computation
        # This dramatically speeds up ICA while preserving artifact patterns
        downsample_factor = 1
        if n_samples > 500000:
            downsample_factor = max(1, n_samples // 250000)  # Target ~250k samples
            if verbose:
                print(f"    Downsampling by factor {downsample_factor} for ICA computation...")
        
        # Downsample data for ICA
        if downsample_factor > 1:
            eeg_for_ica = eeg_data[:, ::downsample_factor]
            eog_for_ica = eog_data[:, ::downsample_factor]
        else:
            eeg_for_ica = eeg_data
            eog_for_ica = eog_data
        
        if verbose:
            print(f"    Computing ICA on {eeg_for_ica.shape[1]} samples...")
        
        # Step 1: Apply ICA decomposition on EEG data
        # Using FastICA (similar to infomax approach)
        ica = FastICA(n_components=n_channels, random_state=42, max_iter=200, tol=0.01)
        
        # Transpose for sklearn format (samples, features)
        eeg_transposed = eeg_for_ica.T
        
        try:
            # Fit ICA and get independent components
            independent_components = ica.fit_transform(eeg_transposed)  # (n_samples, n_components)
            mixing_matrix = ica.mixing_  # (n_features, n_components)
            
            if verbose:
                print(f"    ICA converged in {ica.n_iter_} iterations")
            
        except Exception as e:
            if verbose:
                print(f"  Warning: ICA failed ({e}), returning original data")
            return eeg_data, 0
        
        # Step 2: Identify EOG-contaminated components
        # Correlate each IC with EOG channels (on downsampled data)
        components_to_remove = []
        
        if verbose:
            print(f"    Correlating {n_channels} ICs with {eog_for_ica.shape[0]} EOG channels...")
        
        for ic_idx in range(independent_components.shape[1]):
            ic_signal = independent_components[:, ic_idx]
            
            # Check correlation with each EOG channel
            max_correlation = 0
            for eog_idx in range(eog_for_ica.shape[0]):
                eog_signal = eog_for_ica[eog_idx, :]
                
                # Ensure same length
                min_len = min(len(ic_signal), len(eog_signal))
                ic_trimmed = ic_signal[:min_len]
                eog_trimmed = eog_signal[:min_len]
                
                # Compute correlation
                corr = np.abs(np.corrcoef(ic_trimmed, eog_trimmed)[0, 1])
                max_correlation = max(max_correlation, corr)
            
            # Mark component for removal if highly correlated with EOG
            if max_correlation > self.ica_correlation_threshold:
                components_to_remove.append(ic_idx)
                if verbose:
                    print(f"      Component {ic_idx}: correlation = {max_correlation:.3f} → REMOVED")
        
        # Step 3: Remove contaminated components and reconstruct on FULL data
        if len(components_to_remove) > 0:
            if verbose:
                print(f"    Reconstructing EEG without {len(components_to_remove)} artifact components...")
            
            # Transform full data with learned mixing matrix
            full_components = ica.transform(eeg_data.T)  # (n_samples_full, n_components)
            
            # Zero out contaminated components
            cleaned_components = full_components.copy()
            cleaned_components[:, components_to_remove] = 0
            
            # Reconstruct EEG using cleaned components
            cleaned_eeg = (cleaned_components @ mixing_matrix.T).T
            
            if verbose:
                print(f"    ✓ Removed {len(components_to_remove)} ICA components")
        else:
            cleaned_eeg = eeg_data
            if verbose:
                print(f"    No EOG-contaminated components detected (correlation < {self.ica_correlation_threshold})")
        
        return cleaned_eeg, len(components_to_remove)
    
    def _extract_trigger_info(self, mat_data: Dict, movement_type: str, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract trigger positions and class labels from markers.
        Applies unified class mapping based on movement type.
        
        Parameters:
        -----------
        mat_data : Dict
            Loaded .mat file data
        movement_type : str
            Type of movement (reaching_realMove, multigrasp_realMove, twist_realMove)
        
        Returns:
        --------
        positions : np.ndarray
            Trigger sample positions
        labels : np.ndarray  
            Unified class labels (0-11)
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
                raw_labels = np.argmax(y, axis=0)
            else:
                raw_labels = y.flatten()
            
            # Apply unified class mapping
            if movement_type not in self.CLASS_MAPPING:
                if verbose:
                    print(f"  Warning: Unknown movement type '{movement_type}'")
                labels = raw_labels
            else:
                mapping = self.CLASS_MAPPING[movement_type]
                labels = np.array([mapping.get(int(label), -1) for label in raw_labels])
                
                # Check for unmapped labels
                if np.any(labels == -1):
                    if verbose:
                        print(f"  Warning: Found unmapped labels in {movement_type}")
                        print(f"  Raw labels: {np.unique(raw_labels)}")
            
            if verbose:
                unique_labels = np.unique(labels)
                print(f"  Found {len(positions)} triggers")
                print(f"  Raw classes: {np.unique(raw_labels)} -> Unified classes: {unique_labels}")
                print(f"  Class names: {[self.CLASS_NAMES[l] for l in unique_labels if l in self.CLASS_NAMES]}")
            
            return positions, labels
            
        except Exception as e:
            print(f"  Error extracting trigger info: {e}")
            return np.array([]), np.array([])
    
    def _extract_trials(
        self, 
        data: np.ndarray, 
        trigger_positions: np.ndarray,
        labels: np.ndarray,
        verbose: bool = False
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
                elif verbose:
                    print(f"  Warning: Trial {i} has incorrect length {trial.shape[1]}, skipping")
            elif verbose:
                print(f"  Warning: Trial {i} out of bounds, skipping")
        
        trials = np.array(trials)
        trial_labels = np.array(trial_labels)
        
        if verbose:
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
        filter_high: float,
        verbose: bool = False
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
                'metadata': dict,
                'artifact_removal_applied': bool,
                'n_components_removed': int
            }
        """
        # Construct filename
        filename = f"EEG_{session}_{subject}_{movement_type}.mat"
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            if verbose:
                print(f"File not found: {filepath}")
            return None
        
        if verbose:
            print(f"\nProcessing: {filename}")
            print(f"  Filter: {filter_low}-{filter_high} Hz")
            print(f"  ICA artifact removal: {'ENABLED' if self.use_ica_artifact_removal else 'DISABLED'}")
        
        # Load data
        mat_data = self._load_mat_file(filepath)
        if mat_data is None:
            return None
        
        # Extract channel information
        all_channels = self._extract_channel_names(mat_data)
        if not all_channels:
            return None
        
        # Select motor cortex channels
        channel_indices, channel_names = self._get_motor_channel_indices(all_channels, verbose=verbose)
        if len(channel_indices) == 0:
            if verbose:
                print("  Error: No motor cortex channels found")
            return None
        
        if verbose:
            print(f"  Selected {len(channel_indices)} motor cortex channels")
        
        # Load voltage data for motor cortex channels
        eeg_data = self._load_channel_data(mat_data, channel_indices)
        if eeg_data.size == 0:
            return None
        
        # ICA-based artifact removal (if enabled)
        artifact_removal_applied = False
        n_components_removed = 0
        
        if self.use_ica_artifact_removal:
            # Load EOG data from separate EOG file
            if verbose:
                print(f"  [Step 1/4] Loading EOG data from separate file...")
            
            eog_data = self._load_eog_from_separate_file(subject, session, movement_type, verbose=verbose)
            
            if eog_data is not None:
                if verbose:
                    print(f"  [Step 2/4] Applying ICA artifact removal with {eog_data.shape[0]} EOG channels...")
                    print(f"    This may take 30-60 seconds for long recordings...")
                
                # Apply ICA-based cleaning
                eeg_data, n_components_removed = self._apply_ica_artifact_removal(
                    eeg_data, eog_data, verbose=verbose
                )
                artifact_removal_applied = True
                
                # Update statistics
                self.artifact_removal_stats['files_with_eog'] += 1
                self.artifact_removal_stats['total_components_removed'] += n_components_removed
                
                if verbose:
                    print(f"  [Step 3/4] ICA complete - continuing with trial extraction...")
            else:
                if verbose:
                    print(f"  ⚠ Skipping ICA (EOG file not found)")
                    print(f"     Looking for: EOG_{session}_{subject}_{movement_type}.mat")
        
        # Extract trigger information
        trigger_pos, labels = self._extract_trigger_info(mat_data, movement_type, verbose=verbose)
        if len(trigger_pos) == 0:
            return None
        
        # Extract trials
        trials, trial_labels = self._extract_trials(eeg_data, trigger_pos, labels, verbose=verbose)
        if len(trials) == 0:
            return None
        
        # Apply bandpass filter
        if verbose:
            print(f"  Applying {filter_low}-{filter_high} Hz bandpass filter...")
        filtered_trials = self.apply_bandpass_filter(trials, filter_low, filter_high)
        
        # Package results
        processed = {
            'trials': filtered_trials,
            'labels': trial_labels,
            'channel_names': channel_names,
            'artifact_removal_applied': artifact_removal_applied,
            'n_components_removed': n_components_removed,
            'metadata': {
                'subject': subject,
                'session': session,
                'movement_type': movement_type,
                'fs': self.fs,
                'filter': (filter_low, filter_high),
                'n_trials': len(trial_labels),
                'n_channels': len(channel_names),
                'n_samples': filtered_trials.shape[2],
                'ica_applied': artifact_removal_applied,
                'ica_components_removed': n_components_removed
            }
        }
        
        if verbose:
            print(f"  Successfully processed {len(trial_labels)} trials")
        
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
                'cnn': (1, 40)            # Broader for CNN (captures more temporal dynamics)
            }
        
        # Initialize storage
        all_processed = {config_name: [] for config_name in filter_configs}
        
        # Reset statistics
        self.artifact_removal_stats = {
            'files_processed': 0,
            'files_with_eog': 0,
            'total_components_removed': 0,
            'avg_components_removed': 0
        }
        
        total_files = len(subjects) * len(sessions) * len(self.movement_types)
        
        print(f"\n{'='*70}")
        print(f"PROCESSING DATASET")
        print(f"{'='*70}")
        print(f"Files per configuration: {total_files}")
        print(f"Configurations: {len(filter_configs)}")
        print(f"Total operations: {total_files * len(filter_configs)}")
        print(f"ICA Artifact Removal: {'ENABLED' if self.use_ica_artifact_removal else 'DISABLED'}")
        print(f"{'='*70}\n")
        
        # Process each configuration with progress bar
        for config_name, (low, high) in tqdm(
            filter_configs.items(),
            desc="Configurations",
            position=0,
            leave=True
        ):
            print(f"\n{'='*70}")
            print(f"CONFIGURATION: {config_name.upper()}")
            print(f"Filter: {low}-{high} Hz")
            print(f"{'='*70}\n")
            
            # Create progress bar for files in this configuration
            pbar = tqdm(
                total=total_files,
                desc=f"Processing {config_name}",
                position=1,
                leave=True,
                unit="file"
            )
            
            processed_count = 0
            
            for subject in subjects:
                for session in sessions:
                    for movement_type in self.movement_types:
                        processed = self.process_file(
                            subject, session, movement_type, low, high
                        )
                        
                        if processed is not None:
                            all_processed[config_name].append(processed)
                            processed_count += 1
                            self.artifact_removal_stats['files_processed'] += 1
                        
                        pbar.update(1)
            
            pbar.close()
            
            print(f"\n{config_name}: Successfully processed {processed_count}/{total_files} files")
        
        print(f"\n{'='*70}")
        print(f"PROCESSING COMPLETE")
        total_successful = sum(len(files) for files in all_processed.values())
        total_attempted = total_files * len(filter_configs)
        print(f"Successfully processed {total_successful}/{total_attempted} file-config combinations")
        print(f"{'='*70}")
        
        # ICA statistics
        if self.use_ica_artifact_removal and self.artifact_removal_stats['files_with_eog'] > 0:
            avg_removed = (self.artifact_removal_stats['total_components_removed'] / 
                          self.artifact_removal_stats['files_with_eog'])
            self.artifact_removal_stats['avg_components_removed'] = avg_removed
            
            print(f"\n{'='*70}")
            print(f"ICA ARTIFACT REMOVAL STATISTICS")
            print(f"{'='*70}")
            print(f"Files processed: {self.artifact_removal_stats['files_processed']}")
            print(f"Files with EOG data: {self.artifact_removal_stats['files_with_eog']}")
            print(f"Total ICA components removed: {self.artifact_removal_stats['total_components_removed']}")
            print(f"Average components removed per file: {avg_removed:.2f}")
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
                    'filter': file_list[0]['metadata']['filter'],
                    'ica_artifact_removal': self.use_ica_artifact_removal,
                    'artifact_removal_stats': self.artifact_removal_stats
                }
                save_dict['metadata'] = metadata
            
            np.savez_compressed(output_path, **save_dict)
            
            print(f"  Saved to: {output_path}")
            print(f"  File size: {output_path.stat().st_size / 1e6:.1f} MB")
        
        print(f"\n{'='*70}")
        print(f"ALL DATA SAVED")
        print(f"{'='*70}")


def main():
    """Example usage of preprocessing pipeline."""
    
    import argparse
    parser = argparse.ArgumentParser(description='EEG Preprocessing Pipeline')
    parser.add_argument('--no-ica', action='store_true', 
                       help='Disable ICA artifact removal (faster for testing)')
    parser.add_argument('--subjects', nargs='+', default=None,
                       help='Specific subjects to process (e.g., sub1 sub2)')
    parser.add_argument('--sessions', nargs='+', default=None,
                       help='Specific sessions to process (e.g., session1)')
    parser.add_argument('--eeg-dir', type=str, 
                       default='/home/ubuntu/multimodal-signal-dataset-for-11-upper-body-movements/EEG_ConvertedData',
                       help='Directory containing EEG .mat files')
    parser.add_argument('--eog-dir', type=str, default='/home/ubuntu/multimodal-signal-dataset-for-11-upper-body-movements/EOG_ConvertedData',
                       help='Directory containing EOG .mat files (default: same as EEG dir)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for preprocessed data')
    parser.add_argument('--correlation-threshold', type=float, default=0.8,
                       help='ICA correlation threshold for artifact detection (default: 0.8)')
    args = parser.parse_args()
    
    # Initialize
    preprocessor = EEGPreprocessor(
        data_dir=args.eeg_dir,
        eog_dir=args.eog_dir,  # Can be different from EEG directory
        output_dir=args.output_dir,
        use_ica_artifact_removal=not args.no_ica
    )
    
    # Override correlation threshold if specified
    if not args.no_ica:
        preprocessor.ica_correlation_threshold = args.correlation_threshold
    
    print(f"\n{'='*70}")
    print(f"PREPROCESSING CONFIGURATION")
    print(f"{'='*70}")
    print(f"EEG Directory: {args.eeg_dir}")
    print(f"EOG Directory: {args.eog_dir if args.eog_dir else args.eeg_dir}")
    print(f"Output Directory: {preprocessor.output_dir}")
    print(f"ICA Artifact Removal: {'DISABLED (--no-ica flag)' if args.no_ica else 'ENABLED'}")
    if not args.no_ica:
        print(f"ICA Correlation Threshold: {preprocessor.ica_correlation_threshold}")
    print(f"Subjects: {args.subjects if args.subjects else 'All (sub1-sub25)'}")
    print(f"Sessions: {args.sessions if args.sessions else 'All (session1-session3)'}")
    print(f"{'='*70}\n")
    
    # Process all data (or specify subset)
    all_processed = preprocessor.process_dataset(
        subjects=args.subjects,
        sessions=args.sessions,
        filter_configs={
            'csp_lda': (8, 30),
            'ml_features': (8, 30),
            'cnn': (1, 40)
        }
    )
    
    # Save preprocessed data
    preprocessor.save_preprocessed(all_processed)
    
    print("\nPreprocessing complete!")
    print(f"Preprocessed data saved to: {preprocessor.output_dir}")
    print(f"\nExpected classes: 0-11 (12 total)")
    print(f"  0: Rest")
    print(f"  1-6: Reaching (Forward, Backward, Left, Right, Up, Down)")  
    print(f"  7-9: Grasping (Cup, Ball, Card)")
    print(f"  10-11: Twisting (Pronation, Supination)")


if __name__ == "__main__":
    main()