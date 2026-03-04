#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : demo_tphate.py
# description     : Demonstration of the T-PHATE temporal embedding transform
# author          : Guillaume Dumas
# date            : 2026-03-04
# version         : 1
# usage           : python demo_tphate.py
# notes           : you need to populate the data folder with YAML files
# python_version  : 3.12
# ==============================================================================

from omneeg.io import EEG
import matplotlib.pyplot as plt
import numpy as np

# Load T-PHATE embedded dataset
# Output shape: (n_epochs, resolution, n_times)
dataset = EEG(cohort='cohort1', config_file='config_tphate.yaml')
samp1 = dataset.__getitem__(0)

print(f"T-PHATE Embedding Shape: {samp1.shape}")
print(f"  n_epochs={samp1.shape[0]}, resolution={samp1.shape[1]}, n_times={samp1.shape[2]}")

# Visualize embedding dimensions over time for one epoch
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(samp1[0], aspect='auto', cmap='viridis')
plt.xlabel('Time (samples)')
plt.ylabel('Embedding dimension')
plt.title('Epoch 0: T-PHATE embedding over time')
plt.colorbar()

# 2D scatter of first two embedding dimensions (temporal trajectory)
plt.subplot(1, 2, 2)
t = np.arange(samp1.shape[2])
plt.scatter(samp1[0, 0, :], samp1[0, 1, :], c=t, cmap='viridis', s=10)
plt.xlabel('T-PHATE dim 1')
plt.ylabel('T-PHATE dim 2')
plt.title('Epoch 0: Temporal trajectory')
plt.colorbar(label='Time (samples)')

plt.tight_layout()
plt.show()
