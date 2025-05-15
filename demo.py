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

# Configure matplotlib to force interactive mode for macOS
import matplotlib
matplotlib.use('MacOSX')  # Use the native macOS backend

from omneeg.io import EEG
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from tqdm import trange
import os

# Make sure interactive mode is on
plt.ion()
print(f"Matplotlib backend: {plt.get_backend()}")
print(f"Interactive mode: {plt.isinteractive()}")

# Load only one dataset (cohort1) for the demo
dataset = EEG(cohort='cohort1')

# Check the number of subjects and channels
print(f"Dataset: N_participants={dataset.__len__()}")
if dataset.__len__() > 0:
    sample = dataset.__getitem__(0)
    print(f"Tensor shape: {sample.shape}")
    
    # Print sample statistics for debugging
    print(f"Sample min: {np.min(sample)}, max: {np.max(sample)}, mean: {np.mean(sample)}")
    
    # Visualize the transformed data
    fig = plt.figure(figsize=(10, 8))
    
    # Select a time point in the middle if there are many time points
    time_idx = min(64, sample.shape[3] // 2)
    print(f"Using time index: {time_idx} of {sample.shape[3]}")
    
    # Extract the slice to plot
    plot_data = sample[0, :, :, time_idx]
    print(f"Plot data shape: {plot_data.shape}")
    print(f"Plot data min: {np.min(plot_data)}, max: {np.max(plot_data)}")
    
    # Set colormap limits
    vlim = np.abs(plot_data).max()
    print(f"Using vlim: {vlim}")
    
    # Create the plot 
    img = plt.imshow(plot_data, cmap='RdBu_r', vmin=-vlim, vmax=vlim)
    plt.colorbar(img, label='Amplitude (μV)')
    
    # Add proper axis labels
    plt.xlabel('X (Medial-Lateral)')
    plt.ylabel('Y (Anterior-Posterior)')
    
    plt.title('EEG Topographic Map')
    
    # Add time point information to the title
    time_in_seconds = time_idx / dataset.sfreq
    plt.title(f'EEG Topographic Map (t = {time_in_seconds:.2f} s)')
    
    plt.tight_layout()
    
    # Save the figure to a file
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eeg_topomap.png')
    plt.savefig(output_file)
    print(f"\nPlot saved to: {output_file}")
    print(f"You can open it with: open {output_file}")
    
    # Try to display the plot
    print("\nAttempting to show the plot window...")
    plt.draw()
    plt.pause(5)  # Give it time to render
    print("If you don't see a plot window, check the saved PNG file instead.")
    
    # Keep the plot open until user presses a key
    print("\nPress any key in the plot window or wait 30 seconds to continue...")
    plt.waitforbuttonpress(timeout=30)

    # Simple use with PyTorch DataLoader
    print("\nTesting DataLoader functionality:")
    dataloader = DataLoader(dataset, batch_size=4,
                            shuffle=False, drop_last=True, num_workers=0)
    for epoch in trange(2):
        for batch_idx, batch in enumerate(dataloader):
            print(f"Batch shape: {batch.shape}")
            # For demo purposes, only process a few batches
            if batch_idx >= 2:
                break
else:
    print("No data available in the dataset. Check the 'regexp' in cohort1.yaml")
