#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : demo_source.py
# description     : Demonstration of the source reconstruction transform
# author          : Guillaume Dumas
# date            : 2026-03-04
# version         : 1
# usage           : python demo_source.py
# notes           : you need to populate the data folder with YAML files
#                   requires MNE fsaverage dataset (auto-downloaded)
# python_version  : 3.12
# ==============================================================================

from omneeg.io import EEG
import matplotlib.pyplot as plt
import numpy as np

# Load source-space parcellated dataset
# Output shape: (n_epochs, resolution, n_times)
dataset = EEG(cohort='cohort1', config_file='config_source.yaml')
samp1 = dataset.__getitem__(0)

print(f"Source Reconstruction Shape: {samp1.shape}")
print(f"  n_epochs={samp1.shape[0]}, resolution={samp1.shape[1]}, n_times={samp1.shape[2]}")

# Visualize region activations over time for one epoch
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(samp1[0], aspect='auto', cmap='RdBu_r')
plt.xlabel('Time (samples)')
plt.ylabel('Anatomical region')
plt.title('Epoch 0: Source activations over time')
plt.colorbar()

# Average activation per region
plt.subplot(1, 2, 2)
mean_act = np.mean(np.abs(samp1[0]), axis=1)
plt.barh(range(len(mean_act)), mean_act)
plt.ylabel('Region index')
plt.xlabel('Mean |activation|')
plt.title('Epoch 0: Mean activation per region')

plt.tight_layout()
plt.show()
