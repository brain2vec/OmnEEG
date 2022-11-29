#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : demo.py
# description     : Demonstration of the OmnEEG PyTorch loader
# author          : Guillaume Dumas
# date            : 2022-11-29
# version         : 1
# usage           : python demo.py
# notes           : you need to populate the data folder with YAML files
# python_version  : 3.9
# ==============================================================================

from omneeg.io import EEG
from mne.viz.topomap import plot_topomap
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import trange

# Load the datasets
dataset1 = EEG(cohort='cohort1')
dataset2 = EEG(cohort='cohort2')
dataset3 = EEG(cohort='cohort3')

# Check the number of subjects and channels for each dataset
epo1 = dataset1.__getitem__(0)
print(f"1: N_participants={dataset1.__len__()} N_channels={epo1.shape[1]} Tensor shape: {epo1.shape}")
epo2 = dataset2.__getitem__(0)
print(f"2: N_participants={dataset2.__len__()} N_channels={epo2.shape[1]} Tensor shape: {epo2.shape}")
epo3 = dataset3.__getitem__(0)
print(f"3: N_participants={dataset3.__len__()} N_channels={epo3.shape[1]} Tensor shape: {epo3.shape}")

# Visualization of the first sample of the first epoch in each dataset
# Electrod positions are shown for illustrative purposes only
plt.figure()
ax1 = plt.subplot(1, 3, 1)
im, _ = plot_topomap(epo1[0, :, 0], dataset1.info, sensors=True, contours=0, outlines=None, res=64, axes=ax1, show=False)
ax2 = plt.subplot(1, 3, 2)
im, _ = plot_topomap(epo2[0, :, 0], dataset2.info, sensors=True, contours=0, outlines=None, res=64, axes=ax2, show=False)
ax3 = plt.subplot(1, 3, 3)
im, _ = plot_topomap(epo3[0, :, 0], dataset3.info, sensors=True, contours=0, outlines=None, res=64, axes=ax3, show=False)
plt.show()
plt.pause(1)

# Simple use with PyTorch DataLoader
dataloader = DataLoader(dataset1, batch_size=4,
                        shuffle=False, drop_last=True, num_workers=0)
for epoch in trange(2):
    for batch_idx, batch in enumerate(dataloader):
        print(batch.shape)
        # optimizer.zero_grad()
        # output = model(data, x)
        # loss = criterion(output, target)
