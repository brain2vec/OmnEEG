#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : demo_riemann.py
# description     : Demonstration of the Riemannian tangent space transform
# author          : Guillaume Dumas
# date            : 2026-03-03
# version         : 1
# usage           : python demo_riemann.py
# notes           : you need to populate the data folder with YAML files
# python_version  : 3.12
# ==============================================================================

from omneeg.io import EEG
import matplotlib.pyplot as plt
import numpy as np

# Load Riemannian tangent space dataset
# Output shape: (n_epochs, resolution, n_windows)
dataset = EEG(cohort='cohort1', config_file='config_riemann.yaml')
samp1 = dataset.__getitem__(0)

print(f"Riemannian Tangent Space Shape: {samp1.shape}")
print(f"  n_epochs={samp1.shape[0]}, resolution={samp1.shape[1]}, n_windows={samp1.shape[2]}")

# Visualize spatial features x time for one epoch
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(samp1[0], aspect='auto', cmap='RdBu_r')
plt.xlabel('Time windows')
plt.ylabel('Spatial features')
plt.title('Epoch 0: Tangent space features over time')
plt.colorbar()

# Visualize average power across time for each spatial feature
plt.subplot(1, 2, 2)
mean_power = np.mean(samp1[0] ** 2, axis=1)
plt.bar(range(len(mean_power)), mean_power)
plt.xlabel('Spatial feature index')
plt.ylabel('Mean power')
plt.title('Epoch 0: Power per spatial feature')

plt.tight_layout()
plt.show()
