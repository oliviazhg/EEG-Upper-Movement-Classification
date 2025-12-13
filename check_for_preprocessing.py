import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# UPDATE THIS PATH
filepath = '/home/ubuntu/EEG-Upper-Movement-Classification/TEST/EEG_session1_sub1_multigrasp_realMove.mat'

# Load one file
print("Loading file...")
data = sio.loadmat(filepath, simplify_cells=False)

# Load first channel
ch1 = data['ch1'].flatten()

print("\n" + "="*70)
print("DATA STATISTICS")
print("="*70)
print(f"Mean: {np.mean(ch1):.2f} µV")
print(f"Std: {np.std(ch1):.2f} µV")
print(f"Min: {np.min(ch1):.2f} µV")
print(f"Max: {np.max(ch1):.2f} µV")
print(f"Length: {len(ch1)} samples ({len(ch1)/2500:.1f} seconds)")

# Compute FFT
print("\nComputing frequency spectrum...")
N = len(ch1)
yf = fft(ch1)
xf = fftfreq(N, 1/2500)[:N//2]
power = 2.0/N * np.abs(yf[0:N//2])

# Plot frequency content
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Full spectrum
ax1.plot(xf, power)
ax1.set_xlim(0, 100)
ax1.set_xlabel('Frequency (Hz)')
ax1.set_ylabel('Power')
ax1.set_title('Full Frequency Spectrum (0-100 Hz)')
ax1.axvspan(8, 30, alpha=0.3, color='green', label='Target band (8-30 Hz)')
ax1.legend()
ax1.grid(alpha=0.3)

# Zoomed to 0-60 Hz
ax2.semilogy(xf, power)
ax2.set_xlim(0, 60)
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Power (log scale)')
ax2.set_title('Frequency Spectrum - Log Scale (0-60 Hz)')
ax2.axvspan(8, 30, alpha=0.3, color='green', label='Target band (8-30 Hz)')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('frequency_spectrum.png', dpi=150)
print(f"\nPlot saved to: frequency_spectrum.png")
plt.show()

# Analysis
print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)

total_power = np.sum(power)
power_in_band = np.sum(power[(xf >= 8) & (xf <= 30)])
power_out_band = total_power - power_in_band
percent_in_band = (power_in_band / total_power) * 100

print(f"Total power: {total_power:.2e}")
print(f"Power in 8-30 Hz: {power_in_band:.2e} ({percent_in_band:.1f}%)")
print(f"Power outside 8-30 Hz: {power_out_band:.2e} ({100-percent_in_band:.1f}%)")

print("\n" + "="*70)
if percent_in_band > 90:
    print("✓ DATA IS ALREADY BANDPASS FILTERED (8-30 Hz)")
    print("  - Skip bandpass filtering in your preprocessing")
    print("  - Only apply ICA and trial extraction")
elif percent_in_band < 50:
    print("✓ DATA IS NOT FILTERED (broad frequency content)")
    print("  - Continue with full preprocessing pipeline")
    print("  - Apply bandpass filtering (8-30 Hz)")
    print("  - Apply ICA artifact removal")
else:
    print("? DATA IS PARTIALLY FILTERED OR HAS NATURAL BAND STRUCTURE")
    print("  - Visual inspection needed")
    print("  - Look at the plot to decide")
print("="*70)