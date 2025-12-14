"""
MEMORY-EFFICIENT EEG Preprocessing Pipeline
Fixed version that avoids OOM by processing features in batches

Key improvements:
1. Extracts features file-by-file instead of loading all trials at once
2. Can resume from specific configuration (skip already-completed ones)
3. Better memory management for 200+ GB datasets

Usage:
  # Process all configs
  python preprocessing_fixed.py
  
  # Skip csp_lda, only do ml_features and cnn
  python preprocessing_fixed.py --skip csp_lda
  
  # Only process ml_features
  python preprocessing_fixed.py --only ml_features
  
  # Start from ml_features onwards
  python preprocessing_fixed.py --start-from ml_features
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
import argparse

warnings.filterwarnings('ignore')


class EEGPreprocessor:
    """Memory-efficient preprocessing pipeline for motor cortex EEG signals."""
    
    MOTOR_CHANNELS = [
        'C3', 'C1', 'Cz', 'C2', 'C4',
        'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6',
        'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6',
        'C5', 'C6',
        'FT7', 'FT8',
    ]
    
    CLASS_MAPPING = {
        'reaching_realMove': {
            0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 0,
        },
        'multigrasp_realMove': {
            0: 7, 1: 8, 2: 9, 3: 0,
        },
        'twist_realMove': {
            0: 10, 1: 11, 2: 0,
        }
    }
    
    CLASS_NAMES = {
        0: 'Rest', 1: 'Reach Forward', 2: 'Reach Backward', 
        3: 'Reach Left', 4: 'Reach Right', 5: 'Reach Up', 6: 'Reach Down',
        7: 'Grasp Cup', 8: 'Grasp Ball', 9: 'Grasp Card',
        10: 'Twist Pronation', 11: 'Twist Supination'
    }
    
    def __init__(self, data_dir: str, output_dir: Optional[str] = None, 
                 eog_dir: Optional[str] = None,
                 use_ica_artifact_removal: bool = True):
        self.data_dir = Path(data_dir)
        self.eog_dir = Path(eog_dir) if eog_dir else self.data_dir
        self.output_dir = Path(output_dir) if output_dir else self.data_dir / 'preprocessed'
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.fs = 2500
        self.pre_trigger = 1.0
        self.post_trigger = 2.0
        
        self.use_ica_artifact_removal = use_ica_artifact_removal
        self.ica_correlation_threshold = 0.8
        
        self.movement_types = [
            'multigrasp_realMove',
            'reaching_realMove', 
            'twist_realMove'
        ]
        
        self.artifact_removal_stats = {
            'files_processed': 0,
            'files_with_eog': 0,
            'total_components_removed': 0,
            'avg_components_removed': 0
        }
        
        print(f"Memory-Efficient EEG Preprocessor initialized")
        print(f"EEG data directory: {self.data_dir}")
        print(f"EOG data directory: {self.eog_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"ICA artifact removal: {'ENABLED' if self.use_ica_artifact_removal else 'DISABLED'}")
    
    def _load_mat_file(self, filepath: Path) -> Optional[Dict]:
        try:
            data = sio.loadmat(filepath, simplify_cells=False)
            return data
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def _extract_channel_names(self, mat_data: Dict) -> List[str]:
        try:
            clab = mat_data['dat'][0, 0]['clab'][0]
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
        indices = []
        names = []
        
        for i, ch_name in enumerate(all_channels):
            ch_clean = ch_name.strip().upper()
            for motor_ch in self.MOTOR_CHANNELS:
                if motor_ch.upper() == ch_clean:
                    indices.append(i)
                    names.append(ch_name)
                    break
        
        if verbose:
            print(f"  Selected {len(indices)} motor cortex channels")
        
        return indices, names
    
    def _load_channel_data(self, mat_data: Dict, channel_indices: List[int]) -> np.ndarray:
        channel_data = []
        for idx in channel_indices:
            ch_key = f'ch{idx + 1}'
            if ch_key in mat_data:
                voltage = mat_data[ch_key]
                voltage = voltage.flatten() if voltage.ndim > 1 else voltage
                channel_data.append(voltage)
        return np.array(channel_data)
    
    def _load_eog_from_separate_file(self, subject: str, session: str, movement_type: str, 
                                      verbose: bool = False) -> Optional[np.ndarray]:
        filename = f"EOG_{session}_{subject}_{movement_type}.mat"
        filepath = self.eog_dir / filename
        
        if not filepath.exists():
            return None
        
        mat_data = self._load_mat_file(filepath)
        if mat_data is None:
            return None
        
        eog_channels = []
        for ch_idx in range(1, 5):
            ch_key = f'ch{ch_idx}'
            if ch_key in mat_data:
                voltage = mat_data[ch_key]
                voltage = voltage.flatten() if voltage.ndim > 1 else voltage
                eog_channels.append(voltage)
        
        if len(eog_channels) == 0:
            return None
        
        return np.array(eog_channels)
    
    def _apply_ica_artifact_removal(self, eeg_data: np.ndarray, eog_data: np.ndarray, 
                                    verbose: bool = False) -> Tuple[np.ndarray, int]:
        n_channels, n_samples = eeg_data.shape
        
        downsample_factor = 1
        if n_samples > 500000:
            downsample_factor = max(1, n_samples // 250000)
        
        if downsample_factor > 1:
            eeg_for_ica = eeg_data[:, ::downsample_factor]
            eog_for_ica = eog_data[:, ::downsample_factor]
        else:
            eeg_for_ica = eeg_data
            eog_for_ica = eog_data
        
        ica = FastICA(n_components=n_channels, random_state=42, max_iter=200, tol=0.01)
        eeg_transposed = eeg_for_ica.T
        
        try:
            independent_components = ica.fit_transform(eeg_transposed)
            mixing_matrix = ica.mixing_
        except Exception as e:
            if verbose:
                print(f"  Warning: ICA failed ({e}), returning original data")
            return eeg_data, 0
        
        components_to_remove = []
        
        for ic_idx in range(independent_components.shape[1]):
            ic_signal = independent_components[:, ic_idx]
            max_correlation = 0
            
            for eog_idx in range(eog_for_ica.shape[0]):
                eog_signal = eog_for_ica[eog_idx, :]
                min_len = min(len(ic_signal), len(eog_signal))
                corr = np.abs(np.corrcoef(ic_signal[:min_len], eog_signal[:min_len])[0, 1])
                max_correlation = max(max_correlation, corr)
            
            if max_correlation > self.ica_correlation_threshold:
                components_to_remove.append(ic_idx)
        
        if len(components_to_remove) > 0:
            full_components = ica.transform(eeg_data.T)
            cleaned_components = full_components.copy()
            cleaned_components[:, components_to_remove] = 0
            cleaned_eeg = (cleaned_components @ mixing_matrix.T).T
        else:
            cleaned_eeg = eeg_data
        
        return cleaned_eeg, len(components_to_remove)
    
    def _extract_trigger_info(self, mat_data: Dict, movement_type: str, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        try:
            mrk = mat_data['mrk'][0, 0]
            positions = mrk['pos'].flatten() - 1
            y = mrk['y']
            
            if y.ndim > 1 and y.shape[0] > 1:
                raw_labels = np.argmax(y, axis=0)
            else:
                raw_labels = y.flatten()
            
            if movement_type not in self.CLASS_MAPPING:
                labels = raw_labels
            else:
                mapping = self.CLASS_MAPPING[movement_type]
                labels = np.array([mapping.get(int(label), -1) for label in raw_labels])
            
            return positions, labels
        except Exception as e:
            print(f"  Error extracting trigger info: {e}")
            return np.array([]), np.array([])
    
    def _extract_trials(self, data: np.ndarray, trigger_positions: np.ndarray,
                       labels: np.ndarray, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        n_channels = data.shape[0]
        pre_samples = int(self.pre_trigger * self.fs)
        post_samples = int(self.post_trigger * self.fs)
        trial_length = pre_samples + post_samples
        
        trials = []
        trial_labels = []
        
        for i, pos in enumerate(trigger_positions):
            start_idx = int(pos - pre_samples)
            end_idx = int(pos + post_samples)
            
            if start_idx >= 0 and end_idx <= data.shape[1]:
                trial = data[:, start_idx:end_idx]
                if trial.shape[1] == trial_length:
                    trials.append(trial)
                    trial_labels.append(labels[i])
        
        return np.array(trials), np.array(trial_labels)
    
    def apply_bandpass_filter(self, data: np.ndarray, lowcut: float, highcut: float, order: int = 4) -> np.ndarray:
        nyquist = 0.5 * self.fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(order, [low, high], btype='band')
        
        if data.ndim == 3:
            filtered = np.zeros_like(data)
            for trial_idx in range(data.shape[0]):
                for ch_idx in range(data.shape[1]):
                    filtered[trial_idx, ch_idx, :] = signal.filtfilt(b, a, data[trial_idx, ch_idx, :])
        elif data.ndim == 2:
            filtered = np.zeros_like(data)
            for ch_idx in range(data.shape[0]):
                filtered[ch_idx, :] = signal.filtfilt(b, a, data[ch_idx, :])
        else:
            raise ValueError(f"Unsupported data shape: {data.shape}")
        
        return filtered
    
    def extract_band_power_features(self, trials: np.ndarray,
                                    bands: Optional[Dict[str, Tuple[float, float]]] = None) -> np.ndarray:
        """Extract features from trials."""
        if bands is None:
            bands = {'mu': (8, 12), 'beta': (13, 30)}
        
        n_trials, n_channels, n_samples = trials.shape
        features_list = []
        
        for trial in trials:
            trial_features = []
            
            for band_name, (low, high) in bands.items():
                filtered = self.apply_bandpass_filter(trial, low, high)
                power = np.mean(filtered**2, axis=1)
                trial_features.extend(power)
            
            mean_amp = np.mean(trial, axis=1)
            trial_features.extend(mean_amp)
            
            std_amp = np.std(trial, axis=1)
            trial_features.extend(std_amp)
            
            features_list.append(trial_features)
        
        return np.array(features_list)
    
    def process_file(self, subject: str, session: str, movement_type: str,
                    filter_low: float, filter_high: float, verbose: bool = False) -> Optional[Dict]:
        """Process single .mat file."""
        filename = f"EEG_{session}_{subject}_{movement_type}.mat"
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            return None
        
        mat_data = self._load_mat_file(filepath)
        if mat_data is None:
            return None
        
        all_channels = self._extract_channel_names(mat_data)
        if not all_channels:
            return None
        
        channel_indices, channel_names = self._get_motor_channel_indices(all_channels, verbose=verbose)
        if len(channel_indices) == 0:
            return None
        
        eeg_data = self._load_channel_data(mat_data, channel_indices)
        if eeg_data.size == 0:
            return None
        
        artifact_removal_applied = False
        n_components_removed = 0
        
        if self.use_ica_artifact_removal:
            eog_data = self._load_eog_from_separate_file(subject, session, movement_type, verbose=verbose)
            if eog_data is not None:
                eeg_data, n_components_removed = self._apply_ica_artifact_removal(eeg_data, eog_data, verbose=verbose)
                artifact_removal_applied = True
                self.artifact_removal_stats['files_with_eog'] += 1
                self.artifact_removal_stats['total_components_removed'] += n_components_removed
        
        trigger_pos, labels = self._extract_trigger_info(mat_data, movement_type, verbose=verbose)
        if len(trigger_pos) == 0:
            return None
        
        trials, trial_labels = self._extract_trials(eeg_data, trigger_pos, labels, verbose=verbose)
        if len(trials) == 0:
            return None
        
        filtered_trials = self.apply_bandpass_filter(trials, filter_low, filter_high)
        
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
        
        return processed
    
    def prepare_for_csp_lda(self, processed_files: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Combine processed files for CSP+LDA."""
        print("\nPreparing CSP+LDA data (concatenating trials)...")
        print("  This may take 5-10 minutes for large datasets...")
        
        X = np.concatenate([f['trials'] for f in processed_files], axis=0)
        y = np.concatenate([f['labels'] for f in processed_files], axis=0)
        
        print(f"\nCSP+LDA data prepared:")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  Classes: {np.unique(y)}")
        
        return X, y
    
    def prepare_for_ml_MEMORY_EFFICIENT(self, processed_files: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        MEMORY-EFFICIENT feature extraction.
        Processes files one at a time instead of loading all trials into memory.
        """
        print(f"\nExtracting features for ML models (memory-efficient mode)...")
        print(f"  Processing {len(processed_files)} files...")
        
        all_features = []
        all_labels = []
        
        # Process each file individually to avoid OOM
        for file_data in tqdm(processed_files, desc="  Feature extraction", unit="file"):
            trials = file_data['trials']
            labels = file_data['labels']
            
            # Extract features for THIS file only
            features = self.extract_band_power_features(trials)
            
            all_features.append(features)
            all_labels.append(labels)
            
            # Free memory immediately
            del trials
        
        # Concatenate results (features are tiny compared to raw data!)
        print("\n  Concatenating features...")
        X = np.vstack(all_features)
        y = np.concatenate(all_labels)
        
        print(f"\nML features prepared:")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  Classes: {np.unique(y)}")
        
        return X, y
    
    def prepare_for_cnn(self, processed_files: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for CNN (same as CSP but broader filter)."""
        print("\nPreparing CNN data (concatenating trials)...")
        print("  This may take 5-10 minutes for large datasets...")
        
        X = np.concatenate([f['trials'] for f in processed_files], axis=0)
        y = np.concatenate([f['labels'] for f in processed_files], axis=0)
        
        print(f"\nCNN data prepared:")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  Classes: {np.unique(y)}")
        
        return X, y
    
    def _save_single_config(self, config_name: str, file_list: List[Dict]):
        """Save a single configuration immediately after processing."""
        print(f"\n{'='*70}")
        print(f"SAVING {config_name.upper()} DATA")
        print(f"{'='*70}")
        
        # Prepare data based on model type
        if config_name == 'csp_lda':
            X, y = self.prepare_for_csp_lda(file_list)
        elif config_name == 'ml_features':
            X, y = self.prepare_for_ml_MEMORY_EFFICIENT(file_list)  # MEMORY EFFICIENT!
        elif config_name == 'cnn':
            X, y = self.prepare_for_cnn(file_list)
        else:
            print(f"  Unknown config: {config_name}, skipping")
            return
        
        # Save
        output_path = self.output_dir / f'{config_name}_data.npz'
        
        save_dict = {'X': X, 'y': y}
        
        # Aggregate metadata
        metadata = {
            'n_files': len(file_list),
            'n_trials': len(y),
            'n_classes': len(np.unique(y)),
            'channel_names': file_list[0]['channel_names'],
            'fs': self.fs,
            'filter': file_list[0]['metadata']['filter'],
            'ica_artifact_removal': self.use_ica_artifact_removal,
            'artifact_removal_stats': self.artifact_removal_stats.copy()
        }
        save_dict['metadata'] = metadata
        
        print(f"\n  Compressing and writing to disk...")
        print(f"  (This may take 10-30 minutes for large files)")
        np.savez_compressed(output_path, **save_dict)
        
        print(f"  ✓ Saved to: {output_path}")
        print(f"  File size: {output_path.stat().st_size / 1e9:.1f} GB")
        print(f"{'='*70}\n")
    
    def process_dataset(self, subjects: Optional[List[str]] = None,
                       sessions: Optional[List[str]] = None,
                       filter_configs: Optional[Dict[str, Tuple[float, float]]] = None,
                       configs_to_run: Optional[List[str]] = None) -> Dict[str, List[Dict]]:
        """
        Process entire dataset with different filter configurations.
        
        Parameters:
        -----------
        configs_to_run : list, optional
            List of config names to process. If None, runs all configs.
            Example: ['ml_features', 'cnn'] to skip csp_lda
        """
        if subjects is None:
            subjects = [f'sub{i}' for i in range(1, 26)]
        
        if sessions is None:
            sessions = [f'session{i}' for i in range(1, 4)]
        
        if filter_configs is None:
            filter_configs = {
                'csp_lda': (8, 30),
                'ml_features': (8, 30),
                'cnn': (1, 40)
            }
        
        # Filter configs based on what to run
        if configs_to_run is not None:
            filter_configs = {k: v for k, v in filter_configs.items() if k in configs_to_run}
            print(f"\n{'='*70}")
            print(f"RUNNING ONLY: {', '.join(filter_configs.keys())}")
            print(f"{'='*70}\n")
        
        all_processed = {config_name: [] for config_name in filter_configs}
        
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
        
        for config_name, (low, high) in filter_configs.items():
            print(f"\n{'='*70}")
            print(f"CONFIGURATION: {config_name.upper()}")
            print(f"Filter: {low}-{high} Hz")
            print(f"{'='*70}\n")
            
            pbar = tqdm(
                total=total_files,
                desc=f"Processing {config_name}",
                position=0,
                leave=True,
                unit="file"
            )
            
            processed_count = 0
            
            for subject in subjects:
                for session in sessions:
                    for movement_type in self.movement_types:
                        processed = self.process_file(subject, session, movement_type, low, high)
                        
                        if processed is not None:
                            all_processed[config_name].append(processed)
                            processed_count += 1
                            self.artifact_removal_stats['files_processed'] += 1
                        
                        pbar.update(1)
            
            pbar.close()
            
            print(f"\n{config_name}: Successfully processed {processed_count}/{total_files} files")
            
            # SAVE THIS CONFIGURATION IMMEDIATELY
            if all_processed[config_name]:
                self._save_single_config(config_name, all_processed[config_name])
        
        print(f"\n{'='*70}")
        print(f"PROCESSING COMPLETE")
        print(f"{'='*70}")
        
        return all_processed


def main():
    parser = argparse.ArgumentParser(description='Memory-Efficient EEG Preprocessing Pipeline')
    parser.add_argument('--no-ica', action='store_true', 
                       help='Disable ICA artifact removal')
    parser.add_argument('--subjects', nargs='+', default=None,
                       help='Specific subjects to process')
    parser.add_argument('--sessions', nargs='+', default=None,
                       help='Specific sessions to process')
    parser.add_argument('--eeg-dir', type=str, 
                       default='/home/ubuntu/multimodal-signal-dataset-for-11-upper-body-movements/EEG_ConvertedData',
                       help='Directory containing EEG .mat files')
    parser.add_argument('--eog-dir', type=str, 
                       default='/home/ubuntu/multimodal-signal-dataset-for-11-upper-body-movements/EOG_ConvertedData',
                       help='Directory containing EOG .mat files')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for preprocessed data')
    parser.add_argument('--correlation-threshold', type=float, default=0.8,
                       help='ICA correlation threshold')
    
    # NEW: Config selection options
    parser.add_argument('--skip', nargs='+', choices=['csp_lda', 'ml_features', 'cnn'],
                       help='Skip these configurations (e.g., --skip csp_lda)')
    parser.add_argument('--only', nargs='+', choices=['csp_lda', 'ml_features', 'cnn'],
                       help='Only run these configurations (e.g., --only ml_features cnn)')
    parser.add_argument('--start-from', choices=['csp_lda', 'ml_features', 'cnn'],
                       help='Start from this configuration onwards')
    
    args = parser.parse_args()
    
    # Determine which configs to run
    all_configs = ['csp_lda', 'ml_features', 'cnn']
    configs_to_run = None
    
    if args.only:
        configs_to_run = args.only
        print(f"\n==> Running ONLY: {', '.join(configs_to_run)}")
    elif args.skip:
        configs_to_run = [c for c in all_configs if c not in args.skip]
        print(f"\n==> Skipping: {', '.join(args.skip)}")
        print(f"==> Running: {', '.join(configs_to_run)}")
    elif args.start_from:
        start_idx = all_configs.index(args.start_from)
        configs_to_run = all_configs[start_idx:]
        print(f"\n==> Starting from: {args.start_from}")
        print(f"==> Running: {', '.join(configs_to_run)}")
    
    # Initialize
    preprocessor = EEGPreprocessor(
        data_dir=args.eeg_dir,
        eog_dir=args.eog_dir,
        output_dir=args.output_dir,
        use_ica_artifact_removal=not args.no_ica
    )
    
    if not args.no_ica:
        preprocessor.ica_correlation_threshold = args.correlation_threshold
    
    print(f"\n{'='*70}")
    print(f"PREPROCESSING CONFIGURATION")
    print(f"{'='*70}")
    print(f"EEG Directory: {args.eeg_dir}")
    print(f"EOG Directory: {args.eog_dir}")
    print(f"Output Directory: {preprocessor.output_dir}")
    print(f"ICA: {'DISABLED' if args.no_ica else f'ENABLED (threshold={preprocessor.ica_correlation_threshold})'}")
    print(f"Subjects: {args.subjects if args.subjects else 'All (sub1-sub25)'}")
    print(f"Sessions: {args.sessions if args.sessions else 'All (session1-session3)'}")
    if configs_to_run:
        print(f"Configurations: {', '.join(configs_to_run)}")
    print(f"{'='*70}\n")
    
    # Process dataset
    all_processed = preprocessor.process_dataset(
        subjects=args.subjects,
        sessions=args.sessions,
        filter_configs={
            'csp_lda': (8, 30),
            'ml_features': (8, 30),
            'cnn': (1, 40)
        },
        configs_to_run=configs_to_run
    )
    
    print("\n✓ Preprocessing complete!")
    print(f"Preprocessed data saved to: {preprocessor.output_dir}")


if __name__ == "__main__":
    main()