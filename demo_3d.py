#!/usr/bin/env python

from omneeg.io import EEG
import matplotlib.pyplot as plt
import numpy as np

# Load 3D spherical harmonics dataset
dataset = EEG(cohort='cohort1', config_file='config.yaml')
data = dataset[0]

print(f"3D Spherical Harmonics Shape: {data.shape}")

# Calculate l_max from number of coefficients
n_coeffs = data.shape[1]
l_max = int(np.sqrt(n_coeffs) - 1)
print(f"L_max = {l_max}, total coefficients = {n_coeffs}")

# Calculate power spectrum for each spherical harmonic degree l
power_spectrum = []

idx = 0
for l in range(l_max + 1):
    power_l = 0
    for m in range(-l, l + 1):
        # Average power across epochs and time
        coeff_power = np.mean(data[:, idx, :] ** 2)
        power_l += coeff_power
        idx += 1
    power_spectrum.append(power_l)

print(f"Power spectrum by degree l: {power_spectrum}")

# Plot
plt.figure(figsize=(10, 6))
plt.bar(range(l_max + 1), power_spectrum)
plt.xlabel('Spherical Harmonic Degree (l)')
plt.ylabel('Power')
plt.title('Real EEG Spherical Harmonic Power Spectrum')
plt.grid(True, alpha=0.3)
plt.savefig('plots/clean_spectrum.png', dpi=300, bbox_inches='tight')
plt.show()
