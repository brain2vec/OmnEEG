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
from omneeg.spherical_harmonics import SphericalHarmonicAnalyzer

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

def visualize_electrode_positions():
    """Visualize the electrode positions on a 3D sphere"""
    
    # Initialize analyzer
    analyzer = SphericalHarmonicAnalyzer()
    
    # Get electrode positions in Cartesian coordinates
    cartesian_positions = analyzer.get_electrode_cartesian_positions()
    
    print("Standard 10-20 electrode positions:")
    for name, (x, y, z) in cartesian_positions.items():
        print(f"{name.upper()}: ({x:.3f}, {y:.3f}, {z:.3f})")
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot electrodes
    x_coords = [pos[0] for pos in cartesian_positions.values()]
    y_coords = [pos[1] for pos in cartesian_positions.values()]
    z_coords = [pos[2] for pos in cartesian_positions.values()]
    names = list(cartesian_positions.keys())
    
    # Plot electrode points
    ax.scatter(x_coords, y_coords, z_coords, c='red', s=100, alpha=0.8)
    
    # Add electrode labels
    for i, name in enumerate(names):
        ax.text(x_coords[i], y_coords[i], z_coords[i], 
                name.upper(), fontsize=10, ha='center')
    
    # Create sphere surface for reference
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Plot sphere wireframe
    ax.plot_wireframe(x_sphere, y_sphere, z_sphere, alpha=0.1, color='gray')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Standard 10-20 EEG Electrode Positions\n(Corrected Spherical Coordinates)')
    
    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])
    
    plt.tight_layout()
    plt.savefig('eeg_electrode_positions_corrected.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return cartesian_positions

def demo_spherical_harmonics_transform():
    """Demonstrate spherical harmonics transformation with simulated data"""
    analyzer = SphericalHarmonicAnalyzer()
    
    # Get number of electrodes
    electrode_positions = analyzer.get_electrode_cartesian_positions()
    n_electrodes = len(electrode_positions)
    print(f"Number of electrodes: {n_electrodes}")
    
    # Create simulated EEG data with known spatial patterns
    n_samples = 1000
    sampling_rate = 250  # Hz
    
    # Generate synthetic EEG signals with spatial patterns
    time = np.linspace(0, n_samples/sampling_rate, n_samples)
    
    # Create different frequency components with spatial patterns
    eeg_data = np.zeros((n_electrodes, n_samples))
    
    electrode_names = list(electrode_positions.keys())
    
    # Alpha wave (10 Hz) with posterior dominance
    alpha_freq = 10
    alpha_signal = np.sin(2 * np.pi * alpha_freq * time)
    for i, name in enumerate(electrode_names):
        # Higher amplitude in occipital and parietal regions
        if name.upper().startswith(('O', 'P')):
            amplitude = 2.0
        else:
            amplitude = 0.5
        eeg_data[i, :] += amplitude * alpha_signal
    
    # Beta wave (20 Hz) with frontal dominance
    beta_freq = 20
    beta_signal = np.sin(2 * np.pi * beta_freq * time)
    for i, name in enumerate(electrode_names):
        # Higher amplitude in frontal regions
        if name.upper().startswith('F'):
            amplitude = 1.5
        else:
            amplitude = 0.3
        eeg_data[i, :] += amplitude * beta_signal
    
    # Add some noise
    eeg_data += 0.2 * np.random.randn(n_electrodes, n_samples)
    
    # Use the analyzer to perform spherical harmonics analysis
    print("Performing spherical harmonics analysis...")
    try:
        result = analyzer.analyze(eeg_data, subsample_factor=50)
        
        print(f"Analysis complete!")
        print(f"Coefficients shape: {result['coefficients'].shape}")
        
        # Visualize the results
        analyzer.visualize_transformation(
            result, 
            time_point=0, 
            output_path='spherical_harmonics_demo.png'
        )
        
        print("Spherical harmonics visualization saved to 'spherical_harmonics_demo.png'")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        print("This is expected since we're using simulated data with artificial positions")

def demo_topographic_mapping():
    """Demonstrate electrode positions visualization"""
    print("Creating 3D visualization of corrected electrode positions...")
    
    analyzer = SphericalHarmonicAnalyzer()
    electrode_positions = analyzer.get_electrode_cartesian_positions()
    
    # Create 3D plot showing the corrected positions
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract coordinates
    x_coords = [pos[0] for pos in electrode_positions.values()]
    y_coords = [pos[1] for pos in electrode_positions.values()]
    z_coords = [pos[2] for pos in electrode_positions.values()]
    names = list(electrode_positions.keys())
    
    # Create different colors for different electrode types
    colors = []
    for name in names:
        if name.upper().startswith('F'):
            colors.append('red')      # Frontal - red
        elif name.upper().startswith('C'):
            colors.append('green')    # Central - green
        elif name.upper().startswith('P'):
            colors.append('blue')     # Parietal - blue
        elif name.upper().startswith('O'):
            colors.append('purple')   # Occipital - purple
        elif name.upper().startswith('T'):
            colors.append('orange')   # Temporal - orange
        else:
            colors.append('gray')     # Other - gray
    
    # Plot electrode points
    ax.scatter(x_coords, y_coords, z_coords, c=colors, s=100, alpha=0.8)
    
    # Add electrode labels
    for i, name in enumerate(names):
        ax.text(x_coords[i]*1.1, y_coords[i]*1.1, z_coords[i]*1.1, 
                name.upper(), fontsize=8, ha='center')
    
    # Create sphere surface for reference
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Plot sphere wireframe
    ax.plot_wireframe(x_sphere, y_sphere, z_sphere, alpha=0.1, color='lightgray')
    
    # Set labels and title
    ax.set_xlabel('X (Left-Right)')
    ax.set_ylabel('Y (Posterior-Anterior)')
    ax.set_zlabel('Z (Inferior-Superior)')
    ax.set_title('Corrected 10-20 EEG Electrode Positions\n(Standard Spherical Coordinates)')
    
    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Frontal'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Central'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Parietal'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=8, label='Occipital'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=8, label='Temporal')
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    plt.tight_layout()
    plt.savefig('eeg_electrode_positions_corrected.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Electrode positions visualization saved to 'eeg_electrode_positions_corrected.png'")
    return electrode_positions

if __name__ == "__main__":
    print("OmnEEG Spherical Harmonics Demo (with corrected electrode positions)")
    print("=" * 60)
    
    print("\n1. Visualizing corrected electrode positions...")
    visualize_electrode_positions()
    
    print("\n2. Demonstrating spherical harmonics transform...")
    demo_spherical_harmonics_transform()
    
    print("\n3. Demonstrating topographic mapping...")
    demo_topographic_mapping()
    
    print("\nDemo completed! Check the generated images for results.")
