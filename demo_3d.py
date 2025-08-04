#!/usr/bin/env python
"""
3D EEG to Spherical Harmonics Demo
Takes EEG data with montage and generates spherical harmonics
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
from omneeg.transform_3d import SphericalHarmonics3D, analyze_eeg_3d

def eeg_to_harmonics(file_path, time_window=(10, 20)):
    """
    Convert EEG data to spherical harmonics
    
    Parameters:
    -----------
    file_path : str
        Path to EEG file (EDF, FIF, etc.)
    time_window : tuple
        Time window to analyze (start, stop) in seconds
        
    Returns:
    --------
    dict
        Analysis results with coefficients and metadata
    """
    print(f"Loading EEG data from: {file_path}")
    
    # Use the analysis function
    result, analyzer = analyze_eeg_3d(file_path, time_window=time_window, lmax=8)
    
    return result, analyzer

def visualize_results(result, analyzer, output_path='eeg_harmonics.png'):
    """Visualization of the results"""
    analyzer.visualize(result, output_path=output_path)
    print(f"Visualization saved to: {output_path}")

if __name__ == "__main__":
    # Example usage
    print("3D EEG to Spherical Harmonics Demo")
    print("=" * 50)
    
    # Path to your EEG file - MODIFY THIS PATH TO YOUR OWN DATA
    eeg_file = "path/to/your/eeg/file.edf"  # Replace with your actual file path
    
    try:
        # Perform analysis
        result, analyzer = eeg_to_harmonics(eeg_file, time_window=(100, 120))
        
        # Visualize results
        visualize_results(result, analyzer, 'eeg_harmonics_3d.png')
        
        print("\nDemo completed successfully!")
        print(f"Analyzed {len(result['electrodes'])} electrodes")
        print(f"Generated {result['coefficients'].shape[0]} time points of spherical harmonics")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the EEG file path is correct and the file exists.")
        print("\nTo use this demo:")
        print("1. Replace 'path/to/your/eeg/file.edf' with your actual EEG file path")
        print("2. Make sure your EEG file has standard 10-20 electrode names")
        print("3. Run the script again") 