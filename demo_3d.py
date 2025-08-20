#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : demo_3d.py
# description     : Demonstration of the OmnEEG PyTorch loader
# author          : Mahta Ramezanian Panahi & Guillaume Dumas
# date            : 2025-08-20
# version         : 1
# usage           : python demo_3d.py
# notes           : you need to populate the data folder with YAML files
# python_version  : 3.12
# ==============================================================================

from omneeg.io import EEG
import matplotlib.pyplot as plt
import numpy as np

# Load 3D spherical harmonics dataset
dataset = EEG(cohort='cohort1', config_file='config_3d.yaml')
samp1 = dataset.__getitem__(0)

print(f"3D Spherical Harmonics Shape: {samp1.shape}")

# Calculate l_max from number of coefficients
n_coeffs = samp1.shape[1]
l_max = int(np.sqrt(n_coeffs) - 1)
print(f"L_max = {l_max}, total coefficients = {n_coeffs}")

# Calculate power spectrum for each spherical harmonic degree l
power_spectrum = []

idx = 0
for l in range(l_max + 1):
    power_l = 0
    for m in range(-l, l + 1):
        # Average power across epochs and time
        coeff_power = np.mean(samp1[:, idx, :] ** 2)
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
plt.show()
