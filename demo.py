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
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import trange

# Load the datasets
dataset1 = EEG(cohort='cohort1')
dataset2 = EEG(cohort='cohort2')
dataset3 = EEG(cohort='cohort3')

# Check the number of subjects and channels for each dataset
samp1 = dataset1.__getitem__(0)
print(f"1: N_participants={dataset1.__len__()} Tensor shape: {samp1.shape}")
samp2 = dataset2.__getitem__(0)
print(f"2: N_participants={dataset2.__len__()} Tensor shape: {samp2.shape}")
samp3 = dataset3.__getitem__(0)
print(f"3: N_participants={dataset3.__len__()} Tensor shape: {samp3.shape}")

# Visualize the transformed data
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
vlim = np.abs(samp1).max()
plt.imshow(samp1[4, :, :, 64], vmin=-vlim, vmax=+vlim, cmap='RdBu_r')
plt.subplot(1, 3, 2)
vlim = np.abs(samp2).max()
plt.imshow(samp2[4, :, :, 64], vmin=-vlim, vmax=+vlim, cmap='RdBu_r')
plt.subplot(1, 3, 3)
vlim = np.abs(samp3).max()
plt.imshow(samp3[4, :, :, 64], vmin=-vlim, vmax=+vlim, cmap='RdBu_r')
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
