"""
Test script to verify EOG file loading

This script tests that:
1. EOG files can be found and loaded
2. EOG data has the correct structure (4 channels)
3. EOG and EEG data have matching lengths for ICA
"""

import numpy as np
import scipy.io as sio
from pathlib import Path

def test_eog_loading(data_dir):
    """Test EOG file loading."""
    data_path = Path(data_dir)
    
    print(f"\n{'='*70}")
    print(f"TESTING EOG FILE LOADING")
    print(f"{'='*70}")
    print(f"Data directory: {data_path}\n")
    
    # Find sample files
    eeg_files = list(data_path.glob('EEG_session1_sub1_*.mat'))
    eog_files = list(data_path.glob('EOG_session1_sub1_*.mat'))
    
    print(f"Found {len(eeg_files)} EEG files")
    print(f"Found {len(eog_files)} EOG files")
    
    if len(eog_files) == 0:
        print(f"\n✗ ERROR: No EOG files found!")
        print(f"  Looking for pattern: EOG_session1_sub1_*.mat")
        return False
    
    # Test loading first EOG file
    print(f"\n{'─'*70}")
    print(f"Testing: {eog_files[0].name}")
    print(f"{'─'*70}")
    
    try:
        eog_data = sio.loadmat(eog_files[0], simplify_cells=False)
        
        # Check structure
        print(f"\nEOG file structure:")
        for key in eog_data.keys():
            if not key.startswith('__'):
                value = eog_data[key]
                if hasattr(value, 'shape'):
                    print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
                else:
                    print(f"  {key}: {type(value)}")
        
        # Extract 4 EOG channels
        eog_channels = []
        for i in range(1, 5):
            ch_key = f'ch{i}'
            if ch_key in eog_data:
                voltage = eog_data[ch_key]
                voltage = voltage.flatten() if voltage.ndim > 1 else voltage
                eog_channels.append(voltage)
                print(f"\n  ✓ Loaded {ch_key}: {len(voltage)} samples")
            else:
                print(f"\n  ✗ Missing {ch_key}")
        
        if len(eog_channels) == 4:
            eog_array = np.array(eog_channels)
            print(f"\n✓ Successfully loaded EOG data")
            print(f"  Shape: {eog_array.shape} (4 channels × samples)")
            print(f"  Sample values from ch1: {eog_array[0, :5]}")
        else:
            print(f"\n✗ ERROR: Expected 4 EOG channels, found {len(eog_channels)}")
            return False
        
        # Find matching EEG file
        base_name = eog_files[0].name.replace('EOG_', 'EEG_')
        eeg_file = data_path / base_name
        
        if eeg_file.exists():
            print(f"\n{'─'*70}")
            print(f"Checking EEG/EOG length compatibility")
            print(f"{'─'*70}")
            
            eeg_data = sio.loadmat(eeg_file, simplify_cells=False)
            
            # Load one EEG channel to check length
            if 'ch1' in eeg_data:
                eeg_ch1 = eeg_data['ch1'].flatten()
                eog_ch1 = eog_channels[0]
                
                print(f"\nEEG ch1 length: {len(eeg_ch1)}")
                print(f"EOG ch1 length: {len(eog_ch1)}")
                
                if len(eeg_ch1) == len(eog_ch1):
                    print(f"\n✓ EEG and EOG have matching lengths!")
                    print(f"  Ready for ICA artifact removal")
                else:
                    print(f"\n⚠ WARNING: EEG and EOG lengths don't match")
                    print(f"  Difference: {abs(len(eeg_ch1) - len(eog_ch1))} samples")
                    print(f"  ICA will trim to shorter length")
        
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR loading EOG file: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    # UPDATE THIS PATH
    data_dir = '/home/ubuntu/EEG-Upper-Movement-Classification/TEST'
    
    success = test_eog_loading(data_dir)
    
    print(f"\n{'='*70}")
    if success:
        print(f"✓ EOG LOADING TEST PASSED")
        print(f"\nYou can now run preprocessing with ICA enabled!")
    else:
        print(f"✗ EOG LOADING TEST FAILED")
        print(f"\nPlease check:")
        print(f"1. EOG files exist in your data directory")
        print(f"2. EOG files follow naming: EOG_session*_sub*_*_realMove.mat")
        print(f"3. EOG files contain ch1, ch2, ch3, ch4")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()