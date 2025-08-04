#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : transform_3d.py
# description     : 3D EEG transformation combining interpolation and spherical harmonics
# author          : Generated for OmnEEG
# date            : 2025-01-27
# version         : 1
# usage           : from omneeg.transform_3d import EEG3DTransform
# notes           : Combines interpolation and spherical harmonics for 3D EEG analysis
# python_version  : 3.9
# ==============================================================================

import mne
from mne.channels.layout import _find_topomap_coords
from mne.viz.topomap import _adjust_meg_sphere, _GridData
import numpy as np
import pyshtools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Union, List, Tuple, Dict, Optional
import warnings
from pathlib import Path
import re

from spherical_harmonics import SphericalHarmonicAnalyzer


class EEG3DTransform:
    """
    3D EEG transformation combining interpolation and spherical harmonics analysis.
    """
    
    def __init__(self, output_size: Union[int, Tuple[int, int]] = (64, 64),
                 lmax: int = 16, normalize_radius: bool = True):
        """
        Initialize the 3D EEG transformer.
        
        Parameters:
        -----------
        output_size : int or tuple
            Size of interpolated output grid (width, height)
        lmax : int
            Maximum spherical harmonic degree
        normalize_radius : bool
            Whether to normalize electrode positions to unit sphere
        """
        # Handle output_size properly
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size
            
        self.lmax = lmax
        self.normalize_radius = normalize_radius
        
        # Initialize the spherical harmonic analyzer
        self.sha = SphericalHarmonicAnalyzer(lmax=lmax, normalize_radius=normalize_radius)
        
        # Initialize interpolation parameters
        self.interpolator = None
        
    def _setup_interpolator(self, eeg_data):
        """Set up the interpolator for 2D grid interpolation."""
        sphere, clip_origin = _adjust_meg_sphere(sphere=None, info=eeg_data.info, ch_type='eeg')
        x, y, _, radius = sphere
        picks = mne.pick_types(eeg_data.info, meg=False, eeg=True, ref_meg=False, exclude='bads')
        pos = _find_topomap_coords(eeg_data.info, picks, sphere=sphere)
        
        mask_scale = max(1.0, np.linalg.norm(pos, axis=1).max() * 1.01 / radius)
        clip_radius = (radius * mask_scale,) * 2
        
        xmin, xmax = clip_origin[0] - clip_radius[0], clip_origin[0] + clip_radius[0]
        ymin, ymax = clip_origin[1] - clip_radius[1], clip_origin[1] + clip_radius[1]
        
        xi = np.linspace(xmin, xmax, self.output_size[0])
        yi = np.linspace(ymin, ymax, self.output_size[1])
        Xi, Yi = np.meshgrid(xi, yi)
        
        image_interp = 'cubic'
        extrapolate = 'box'
        border = 'mean'
        
        interp = _GridData(pos, image_interp, extrapolate, clip_origin, clip_radius, border)
        
        self.interpolator = {
            'interp': interp,
            'Xi': Xi,
            'Yi': Yi,
            'pos': pos,
            'extent': (xmin, xmax, ymin, ymax),
            'sphere': sphere,
            'clip_origin': clip_origin
        }
        
    def interpolate_2d(self, eeg_data: Union['mne.io.Raw', 'mne.Epochs']) -> np.ndarray:
        """
        Interpolate EEG data to a 2D regular grid.
        
        Returns:
        --------
        np.ndarray
            Interpolated data of shape (epochs, height, width, timepoints)
        """
        if self.interpolator is None:
            self._setup_interpolator(eeg_data)
            
        data = eeg_data.get_data()
        
        # Handle both Raw (2D) and Epochs (3D)
        if data.ndim == 2:
            # (n_channels, n_times) -> (1, n_channels, n_times)
            data = data[None, ...]
        n_epochs, n_channels, n_times = data.shape
        Z = np.zeros((n_epochs, self.output_size[0], self.output_size[1], n_times))
        
        for epoch in range(n_epochs):
            for time in range(n_times):
                self.interpolator['interp'].set_values(data[epoch, :, time])
                Zi = self.interpolator['interp'].set_locations(
                    self.interpolator['Xi'], self.interpolator['Yi'])()
                Z[epoch, :, :, time] = Zi
        return Z
    
    def transform_3d(self, eeg_data: Union['mne.io.Raw', 'mne.Epochs'],
                     time_window: Optional[Tuple[float, float]] = None,
                     channel_subset: Optional[List[str]] = None,
                     subsample_factor: int = 1,
                     skip_interpolation: bool = False) -> Dict:
        """
        Perform comprehensive 3D transformation combining interpolation and spherical harmonics.
        
        Returns:
        --------
        Dict
            Dictionary containing both 2D interpolated data and spherical harmonic results
        """
        print("Performing 3D EEG transformation...")
        
        # 1. 2D interpolation (optional)
        interpolated_2d = None
        if not skip_interpolation:
            try:
                print("Step 1: Performing 2D interpolation...")
                interpolated_2d = self.interpolate_2d(eeg_data)
            except Exception as e:
                print(f"Warning: 2D interpolation failed: {e}")
                print("Continuing with spherical harmonics only...")
                skip_interpolation = True
        
        # 2. Spherical harmonics analysis
        print("Step 2: Performing spherical harmonics analysis...")
        sh_results = self.sha.analyze(
            eeg_data, time_window, channel_subset, subsample_factor
        )
        
        # 3. Combine results
        result = {
            'interpolated_2d': interpolated_2d,
            'spherical_harmonics': sh_results,
            'output_size': self.output_size,
            'lmax': self.lmax,
            'interpolator_info': self.interpolator,
            'interpolation_successful': not skip_interpolation
        }
        
        print("3D transformation complete!")
        if interpolated_2d is not None:
            print(f"2D interpolation shape: {interpolated_2d.shape}")
        print(f"Spherical harmonics coefficients shape: {sh_results['coefficients'].shape}")
        
        return result
    
    def reconstruct_3d(self, result: Dict, time_point: int = 0,
                      n_harmonics: Optional[int] = None) -> np.ndarray:
        """
        Reconstruct 3D field from spherical harmonics.
        
        Returns:
        --------
        np.ndarray
            3D reconstructed field
        """
        sh_results = result['spherical_harmonics']
        coeffs = sh_results['coefficients'][time_point]
        
        if n_harmonics is not None:
            # Truncate coefficients
            coeffs_truncated = np.zeros_like(coeffs)
            coeffs_truncated[:, :n_harmonics+1, :n_harmonics+1] = coeffs[:, :n_harmonics+1, :n_harmonics+1]
            coeffs = coeffs_truncated
        
        # Create 3D grid
        n_grid = 50
        theta_grid = np.linspace(0, 180, n_grid)
        phi_grid = np.linspace(-180, 180, n_grid)
        r_grid = np.linspace(0.8, 1.2, 10)  # Multiple radial shells
        
        THETA, PHI, R = np.meshgrid(theta_grid, phi_grid, r_grid, indexing='ij')
        
        # Reconstruct field at each point
        reconstructed_3d = np.zeros_like(THETA)
        
        for i in range(n_grid):
            for j in range(n_grid):
                for k in range(len(r_grid)):
                    reconstructed_3d[i, j, k] = pyshtools.expand.MakeGridPoint(
                        coeffs, theta_grid[i], phi_grid[j]
                    )
        
        return reconstructed_3d, THETA, PHI, R
    
    def visualize_3d(self, result: Dict, time_point: int = 0,
                     output_path: Optional[str] = None) -> None:
        """
        Visualize the 3D transformation results.
        """
        # Determine number of subplots based on available data
        has_interpolation = result.get('interpolated_2d') is not None
        
        if has_interpolation:
            fig = plt.figure(figsize=(15, 10))
            plot_idx = 1
        else:
            fig = plt.figure(figsize=(12, 8))
            plot_idx = 1
        
        # 1. 2D interpolation (if available)
        if has_interpolation:
            ax1 = fig.add_subplot(2, 3, plot_idx)
            interpolated_2d = result['interpolated_2d']
            im = ax1.imshow(interpolated_2d[0, :, :, time_point], 
                           cmap='RdBu_r', aspect='equal', origin='lower')
            ax1.set_title(f'2D Interpolation\nTime: {time_point}')
            plt.colorbar(im, ax=ax1)
            plot_idx += 1
        else:
            # Add a placeholder
            ax1 = fig.add_subplot(2, 3, plot_idx)
            ax1.text(0.5, 0.5, '2D Interpolation\nNot Available', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('2D Interpolation')
            plot_idx += 1
        
        # 2. Original sensor data
        ax2 = fig.add_subplot(2, 3, plot_idx, projection='3d')
        sh_results = result['spherical_harmonics']
        positions = sh_results['positions']
        coeffs = sh_results['coefficients'][time_point]
        
        # Get original values at sensor positions
        theta_deg = sh_results['theta']
        phi_deg = sh_results['phi']
        original_values = []
        for i in range(len(theta_deg)):
            val = pyshtools.expand.MakeGridPoint(coeffs, theta_deg[i], phi_deg[i])
            original_values.append(val)
        original_values = np.array(original_values)
        
        scatter = ax2.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                            c=original_values, s=100, cmap='RdBu_r')
        ax2.set_title('Original Sensor Data')
        plt.colorbar(scatter, ax=ax2)
        plot_idx += 1
        
        # 3. 3D spherical reconstruction
        ax3 = fig.add_subplot(2, 3, plot_idx, projection='3d')
        
        # Create sphere surface
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        sphere_x = np.outer(np.cos(u), np.sin(v))
        sphere_y = np.outer(np.sin(u), np.sin(v))
        sphere_z = np.outer(np.ones(np.size(u)), np.cos(v))
        
        # Reconstruct field on sphere surface
        reconstructed_surface = np.zeros_like(sphere_x)
        for i in range(50):
            for j in range(50):
                theta = np.degrees(np.arccos(sphere_z[i, j]))
                phi = np.degrees(np.arctan2(sphere_y[i, j], sphere_x[i, j]))
                reconstructed_surface[i, j] = pyshtools.expand.MakeGridPoint(
                    coeffs, theta, phi
                )
        
        # Plot surface with color mapping
        surf = ax3.plot_surface(sphere_x, sphere_y, sphere_z, 
                               facecolors=plt.cm.RdBu_r(
                                   (reconstructed_surface - reconstructed_surface.min()) / 
                                   (reconstructed_surface.max() - reconstructed_surface.min())
                               ),
                               alpha=0.8)
        ax3.set_title('3D Spherical Reconstruction')
        # Rotate 90 degrees counterclockwise around vertical axis to align with head orientation
        ax3.view_init(elev=20, azim=90)
        plot_idx += 1
        
        # 4. Harmonic power spectrum
        ax4 = fig.add_subplot(2, 3, plot_idx)
        power_by_degree = []
        degrees = []
        for l in range(sh_results['lmax'] + 1):
            power = 0
            for m in range(-l, l + 1):
                m_idx = abs(m)
                if m >= 0:
                    power += coeffs[0, l, m_idx]**2
                else:
                    power += coeffs[1, l, m_idx]**2
            power_by_degree.append(np.sqrt(power))
            degrees.append(l)
        
        ax4.bar(degrees, power_by_degree, alpha=0.7, color='steelblue')
        ax4.set_xlabel('Spherical Harmonic Degree (l)')
        ax4.set_ylabel('Power')
        ax4.set_title('Harmonic Power Spectrum')
        ax4.grid(True, alpha=0.3)
        plot_idx += 1
        
        # 5. Cross-section comparison (only if interpolation available)
        if has_interpolation:
            ax5 = fig.add_subplot(2, 3, plot_idx)
            center_line_2d = interpolated_2d[0, :, interpolated_2d.shape[2]//2, time_point]
            theta_line = np.linspace(0, 180, len(center_line_2d))
            phi_line = np.full_like(theta_line, 0)
            spherical_line = []
            for theta, phi in zip(theta_line, phi_line):
                val = pyshtools.expand.MakeGridPoint(coeffs, theta, phi)
                spherical_line.append(val)
            spherical_line = np.array(spherical_line)
            
            ax5.plot(center_line_2d, label='2D Interpolation', linewidth=2)
            ax5.plot(spherical_line, label='Spherical Harmonics', linewidth=2)
            ax5.set_xlabel('Position')
            ax5.set_ylabel('Amplitude')
            ax5.set_title('2D vs Spherical Comparison')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        else:
            # Add electrode positions plot instead
            ax5 = fig.add_subplot(2, 3, plot_idx)
            ax5.scatter(positions[:, 0], positions[:, 1], c=original_values, 
                       s=100, cmap='RdBu_r')
            ax5.set_xlabel('X')
            ax5.set_ylabel('Y')
            ax5.set_title('Electrode Positions (2D)')
            plt.colorbar(ax5.scatter(positions[:, 0], positions[:, 1], c=original_values, 
                                   s=100, cmap='RdBu_r'), ax=ax5)
        
        plt.tight_layout()
        
        if output_path:
            # Create directory if it doesn't exist
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"3D visualization saved to {output_path}")
        else:
            plt.show()


# Convenience function
def transform_eeg_3d(eeg_data: Union['mne.io.Raw', 'mne.Epochs'],
                     output_size: Union[int, Tuple[int, int]] = (64, 64),
                     lmax: int = 16,
                     time_window: Optional[Tuple[float, float]] = None,
                     channel_subset: Optional[List[str]] = None,
                     subsample_factor: int = 1,
                     visualize: bool = True,
                     output_path: Optional[str] = None,
                     skip_interpolation: bool = False) -> Dict:
    """
    Convenience function for performing 3D EEG transformation.
    """
    transformer = EEG3DTransform(output_size=output_size, lmax=lmax)
    result = transformer.transform_3d(eeg_data, time_window, channel_subset, subsample_factor, skip_interpolation)
    
    if visualize:
        transformer.visualize_3d(result, output_path=output_path)
    
    return result


if __name__ == "__main__":
    # Example usage with real TUEG data
    print("Testing 3D EEG transformation with TUEG data...")
    
    # Use actual TUEG data
    tueg_file = "../data/TUEG/tuh_eeg/v2.0.0/edf/000/aaaaaaac/s003_2002_12_26/02_tcp_le/aaaaaaac_s003_t000.edf"
    
    try:
        # Load the TUEG EDF file
        print(f"Loading TUEG data from: {tueg_file}")
        raw = mne.io.read_raw_edf(tueg_file, preload=True, verbose=False)
        
        # Pick only EEG channels
        eeg_picks = mne.pick_types(raw.info, meg=False, eeg=True, exclude='bads')
        raw.pick(eeg_picks)
        
        print(f"Loaded {len(raw.ch_names)} EEG channels")
        print(f"Data duration: {raw.times[-1]:.2f} seconds")
        print(f"Sampling rate: {raw.info['sfreq']} Hz")
        
        # Add standard electrode positions if missing
        if raw.info['dig'] is None:
            print("Adding standard electrode positions...")
            montage = mne.channels.make_standard_montage('standard_1020')
            raw.set_montage(montage)
        
        # Perform 3D transformation on a time window
        result = transform_eeg_3d(
            raw,
            output_size=(64, 64),
            lmax=12,
            time_window=(10, 30),  # 20 seconds of data
            subsample_factor=10,   # Sample every 10th time point for speed
            visualize=True,
            output_path="results/tueg_3d_transform.png"
        )
        
        print("3D transformation test complete!")
        
    except Exception as e:
        print(f"Error with TUEG data: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback to synthetic data if TUEG fails
        print("Falling back to synthetic data...")
        
        # Create test data with proper electrode positions
        n_channels = 10
        n_timepoints = 100
        test_data = np.random.randn(n_channels, n_timepoints)
        
        # Create MNE Raw object with standard electrode positions
        ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2']
        ch_types = ['eeg'] * n_channels
        info = mne.create_info(ch_names, sfreq=100, ch_types=ch_types)
        
        # Add standard electrode positions
        montage = mne.channels.make_standard_montage('standard_1020')
        info.set_montage(montage)
        
        raw = mne.io.RawArray(test_data, info)
        
        # Perform 3D transformation
        result = transform_eeg_3d(
            raw,
            output_size=(32, 32),
            lmax=8,
            visualize=True,
            output_path="results/test_3d_transform.png"
        )
        
        print("Synthetic data 3D transformation test complete!") 