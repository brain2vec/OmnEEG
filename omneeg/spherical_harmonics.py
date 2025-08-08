#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : spherical_harmonics.py
# description     : General spherical harmonic analysis for EEG data
# author          : Mahtaao
# date            : 2025-08
# version         : 1
# usage           : from omneeg.spherical_harmonics import SphericalHarmonicAnalyzer
# notes           : Works with various data formats and channel naming conventions
# python_version  : 3.10.17
# ==============================================================================

import pandas as pd
import numpy as np
import pyshtools
import mne
import re
from typing import Union, List, Tuple, Dict, Optional
import warnings
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from pathlib import Path


class SphericalHarmonicAnalyzer:
    """
    Convert sensor-level EEG data to 3D spherical harmonics and visualize them.
    Supports EDF/FIF files and numpy arrays. Uses MNE's standard_1005 montage for positions.
    """
    def __init__(self, lmax: int = 8):
        self.lmax = lmax
        self.montage = mne.channels.make_standard_montage('standard_1005')
        self.elec_pos = self.montage.get_positions()['ch_pos']
        # Build a case-insensitive mapping for channel lookup
        self.elec_pos_lower = {k.lower(): v for k, v in self.elec_pos.items()}

    def load_data(self, data_source: Union[str, np.ndarray], channels: Optional[List[str]] = None, time_window: Optional[Tuple[float, float]] = None):
        """
        Load EEG data from EDF/FIF file or numpy array. Returns (data, ch_names, times, positions).
        """
        def clean_name(ch):
            ch = ch.upper()
            if ch.startswith('EEG '):
                ch = ch[4:]
            if ch.endswith('-REF'):
                ch = ch[:-4]
            if '-' in ch:
                ch = ch.split('-')[0]
            if '_' in ch:
                ch = ch.split('_')[0]
            ch = ch.strip()
            # Map T1/T2 to TP9/TP10 because of the standard_1005 montage
            if ch == 'T1':
                ch = 'TP9'
            if ch == 'T2':
                ch = 'TP10'
            return ch

        if isinstance(data_source, str):
            if data_source.endswith('.edf'):
                raw = mne.io.read_raw_edf(data_source, preload=True, verbose=False)
            elif data_source.endswith('.fif'):
                raw = mne.io.read_raw_fif(data_source, preload=True, verbose=False)
            else:
                raise ValueError("Only EDF and FIF files are supported.")
            if channels:
                raw.pick_channels(channels)
            else:
                raw.pick_types(eeg=True)
            data, times = raw.get_data(return_times=True)
            ch_names = raw.ch_names
        elif isinstance(data_source, np.ndarray):
            data = data_source
            if data.ndim == 1:
                data = data[None, :]
            # Use first N standard names for artificial data
            std_names = list(self.elec_pos.keys())
            n = data.shape[0]
            ch_names = std_names[:n]
            times = np.arange(data.shape[1])
        else:
            raise ValueError("data_source must be a file path or numpy array.")
        # Get 3D positions for each channel
        positions = []
        cleaned_names = []
        valid_indices = []
        for i, ch in enumerate(ch_names):
            ch_clean = clean_name(ch)
            pos = self.elec_pos_lower.get(ch_clean.lower())
            if pos is None:
                print(f"Warning: Channel {ch} (cleaned: {ch_clean}) not found in standard_1005 montage. Skipping.")
                continue
            positions.append(pos)
            cleaned_names.append(ch_clean)
            valid_indices.append(i)
        if not positions:
            raise ValueError("No valid EEG channels found in standard_1005 montage.")
        data = data[valid_indices, :]
        positions = np.array(list(positions))
        return data, cleaned_names, times, positions

    def analyze(self, data_source: Union[str, np.ndarray], channels: Optional[List[str]] = None, time_window: Optional[Tuple[float, float]] = None):
        """
        Compute spherical harmonics for EEG data. Returns dict with coefficients, positions, etc.
        Returns a dictionary containing:
            - coefficients: Spherical harmonic coefficients for each time point
            - positions: 3D electrode positions for visualization and validation
            - ch_names: Channel names to map coefficients to electrodes
            - theta/phi: Angular coordinates for field reconstruction
            - times: Time points for temporal analysis
            - lmax: Maximum harmonic degree used
        """
        data, ch_names, times, positions = self.load_data(data_source, channels, time_window)
        # Convert positions to spherical (theta: colatitude, phi: longitude)
        xyz = positions / np.linalg.norm(positions, axis=1, keepdims=True)
        theta = np.arccos(xyz[:, 2])
        phi = np.arctan2(xyz[:, 1], xyz[:, 0])
        theta_deg = np.degrees(theta)
        phi_deg = np.degrees(phi)
        # Compute harmonics for each time point
        coeffs = []
        for t in range(data.shape[1]):
            vals = data[:, t]
            c = pyshtools.expand.SHExpandLSQ(vals, theta_deg, phi_deg, self.lmax)[0]
            coeffs.append(c)
        coeffs = np.stack(coeffs)  # (time, 2, lmax+1, lmax+1)

        return dict(coefficients=coeffs, positions=positions, ch_names=ch_names, theta=theta_deg, phi=phi_deg, times=times, lmax=self.lmax)

    def visualize(self, result: dict, time_point: int = 0, show_harmonics_up_to: int = 3):
        """
        Visualize sensor data and spherical harmonics at a given time point.
        """
        coeffs = result['coefficients'][time_point]
        positions = result['positions']
        ch_names = result['ch_names']
        theta = result['theta']
        phi = result['phi']
        times = result['times']
        # Sensor values
        sensor_vals = pyshtools.expand.MakeGridPoint(coeffs, theta, phi)
        # Limit harmonics to available degrees
        max_l = coeffs.shape[1] - 1
        n_harmonics = min(show_harmonics_up_to+1, max_l+1)
        n_subplots = n_harmonics + 2
        fig = plt.figure(figsize=(4*n_subplots, 6))
        ax = fig.add_subplot(1, n_subplots, 1, projection='3d')
        xyz = positions / np.linalg.norm(positions, axis=1, keepdims=True)
        sc = ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], c=sensor_vals, cmap='RdBu_r', s=80)
        for i, name in enumerate(ch_names):
            ax.text(xyz[i,0]*1.1, xyz[i,1]*1.1, xyz[i,2]*1.1, name, fontsize=8)
        ax.set_title(f'Sensor Data\nTime: {times[time_point]:.3f}s')
        plt.colorbar(sc, ax=ax, shrink=0.5)
        # Add axis labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # Add wireframe sphere
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 15)
        sphere_x = np.outer(np.cos(u), np.sin(v))
        sphere_y = np.outer(np.sin(u), np.sin(v))
        sphere_z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_wireframe(sphere_x, sphere_y, sphere_z, color='gray', alpha=0.2, linewidth=0.7)
        # Full field reconstruction
        n_grid = 60
        theta_grid = np.linspace(0, 180, n_grid)
        phi_grid = np.linspace(-180, 180, n_grid)
        THETA, PHI = np.meshgrid(theta_grid, phi_grid)
        X = np.sin(np.radians(THETA)) * np.cos(np.radians(PHI))
        Y = np.sin(np.radians(THETA)) * np.sin(np.radians(PHI))
        Z = np.cos(np.radians(THETA))
        field = np.zeros_like(THETA)
        for i in range(n_grid):
            for j in range(n_grid):
                field[i, j] = pyshtools.expand.MakeGridPoint(coeffs, theta_grid[i], phi_grid[j])
        ax2 = fig.add_subplot(1, n_subplots, 2, projection='3d')
        surf = ax2.plot_surface(X, Y, Z, facecolors=plt.cm.RdBu_r((field-field.min())/(np.ptp(field) or 1)), alpha=0.8, linewidth=0)
        ax2.set_title('Reconstructed Field')
        # Rotate 90 degrees counterclockwise around vertical axis to align with head orientation
        ax2.view_init(elev=20, azim=20)
        # Add sensor positions as black dots on the surface
        ax2.scatter(xyz[:,0], xyz[:,1], xyz[:,2], c='black', s=30, alpha=0.8, edgecolors='white', linewidth=0.5)
        # Individual harmonics
        for l in range(n_harmonics):
            c_l = np.zeros_like(coeffs)
            c_l[:, l, :] = coeffs[:, l, :]
            field_l = np.zeros_like(THETA)
            for i in range(n_grid):
                for j in range(n_grid):
                    field_l[i, j] = pyshtools.expand.MakeGridPoint(c_l, theta_grid[i], phi_grid[j])
            axl = fig.add_subplot(1, n_subplots, 3+l)
            im = axl.imshow(field_l, cmap='RdBu_r', origin='lower', extent=[-180,180,0,180], aspect='auto')
            axl.set_title(f'Harmonic l={l}')
            plt.colorbar(im, ax=axl, shrink=0.7)
            # Add sensor positions as black dots on 2D plots
            axl.scatter(phi, theta, c='black', s=20, alpha=0.8, edgecolors='white', linewidth=0.5)
            axl.set_xlabel('Longitude (°)')
            axl.set_ylabel('Colatitude (°)')
        plt.tight_layout()
        plt.show()
  

def compare_reconstruction(result: dict, original: np.ndarray, sensor_idx: int = None):
    """Compare reconstructed harmonics against original data at a sensor point."""
    if sensor_idx is None:
        sensor_idx = np.argmax(result['positions'][:, 2])  # Top sensor
    
    pos = result['positions'][sensor_idx]
    theta = np.degrees(np.arccos(pos[2] / np.linalg.norm(pos)))
    phi = np.degrees(np.arctan2(pos[1], pos[0]))
    
    reconstructed = [pyshtools.expand.MakeGridPoint(coeffs, theta, phi) 
                    for coeffs in result['coefficients']]
    
    plt.figure(figsize=(10, 4))
    plt.plot(result['times'], original[sensor_idx], 'b-', label='Original', alpha=0.7)
    plt.plot(result['times'], reconstructed, 'r-', label='Reconstructed')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Signal at {result["ch_names"][sensor_idx]}')
    plt.legend()
    plt.show()
    
    return reconstructed

if __name__ == "__main__":
    # Example usage with comprehensive visualization
    print("Testing spherical harmonic analysis with visualization...")
    
    # Test with the EDF file
    edf_path = 'path/to/your/eeg/file.edf'  # Replace with your file path
    
    # Create analyzer
    analyzer = SphericalHarmonicAnalyzer(lmax=8)  # Reduced lmax for faster computation
    
    print("Analyzing EDF file with all available EEG channels...")
    result = analyzer.analyze(
        edf_path, 
        time_window=(100, 120)  # Longer time window for better visualization
    )
    
    print(f"Analysis complete!")
    print(f"Electrodes analyzed: {result['ch_names']}")
    print(f"Coefficients shape: {result['coefficients'].shape}")
    print(f"Time points: {len(result['times'])}")
    
    # Generate comprehensive visualizations
    print("\nGenerating visualizations...")
    
    # 1. Main transformation visualization
    print("Creating transformation visualization...")
    analyzer.visualize(
        result, 
        time_point=0, 
        show_harmonics_up_to=3
    )
    
    # 2. Time evolution plots
    if len(result['times']) > 1:
        print("Creating harmonic evolution plots...")
        analyzer.visualize(
            result,
            time_point=0,
            show_harmonics_up_to=3
        )

    # Get original data for comparison
    data, _, _, _ = analyzer.load_data(edf_path)
    compare_reconstruction(result, data)
    
    print("\nAll visualizations complete! Check the generated plot.")


   