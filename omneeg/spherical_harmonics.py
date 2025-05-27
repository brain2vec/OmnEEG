#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : spherical_harmonics.py
# description     : General spherical harmonic analysis for EEG data
# author          : Generated for OmnEEG
# date            : 2025-05-27
# version         : 1
# usage           : from omneeg.spherical_harmonics import SphericalHarmonicAnalyzer
# notes           : Works with various data formats and channel naming conventions
# python_version  : 3.9
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
    General spherical harmonic analysis for EEG data.
    
    Works with various data formats (EDF, FIF, arrays), channel naming conventions,
    and electrode configurations.
    """
    
    def __init__(self, electrode_positions_file: str = 'EEG1005.tsv', 
                 lmax: int = 32, normalize_radius: bool = True):
        """
        Initialize the spherical harmonic analyzer.
        
        Parameters:
        -----------
        electrode_positions_file : str
            Path to electrode positions file (TSV format with x,y,z coordinates)
        lmax : int
            Maximum spherical harmonic degree
        normalize_radius : bool
            Whether to normalize electrode positions to unit sphere
        """
        self.lmax = lmax
        self.normalize_radius = normalize_radius
        
        # Load standard electrode positions instead of file
        self.electrode_positions = self._load_standard_electrode_positions()
        print(f"Loaded {len(self.electrode_positions)} standard 10-20 electrode positions")
    
    def _load_standard_electrode_positions(self):
        """
        Load standard electrode positions from MNE-Python's standard_1005 montage.
        
        Returns a dictionary of electrode names mapping to spherical coordinates (theta, phi)
        theta: azimuthal angle (0 to 2π), phi: polar angle from +z axis (0 to π)
        """
        try:
            # Get the standard 10-05 montage which has 345 electrode positions
            montage = mne.channels.make_standard_montage('standard_1005')
            
            # Get the positions in a dict format with keys as electrode names
            positions_3d = montage.get_positions()
            
            # Convert to spherical coordinates
            electrode_positions = {}
            
            # Get the 3D positions and convert to our spherical coordinate convention
            if 'ch_pos' in positions_3d:
                ch_pos = positions_3d['ch_pos']
                
                for ch_name, pos_3d in ch_pos.items():
                    # Normalize to unit sphere
                    norm = np.linalg.norm(pos_3d)
                    if norm > 0:
                        x, y, z = pos_3d / norm
                        
                        # Convert to spherical coordinates
                        # theta: azimuthal angle in x-y plane
                        # phi: polar angle from +z axis
                        theta = np.arctan2(y, x)  # Range: -π to π
                        phi = np.arccos(z)        # Range: 0 to π
                        
                        # Store the position
                        electrode_positions[ch_name.upper()] = (theta, phi)
            else:
                # Fallback method if 'ch_pos' is not available
                sphere_coords = montage._get_sphere_coords()
                for ch_name, (theta, phi, _) in zip(montage.ch_names, sphere_coords):
                    # In MNE, theta is already azimuth and phi is elevation from horizontal
                    # Convert phi from MNE convention to our convention
                    phi_our_conv = np.pi/2 - phi
                    electrode_positions[ch_name.upper()] = (theta, phi_our_conv)
                    
            print(f"Loaded {len(electrode_positions)} standard electrode positions from MNE standard_1005 montage")
            return electrode_positions
            
        except Exception as e:
            print(f"Warning: Could not load MNE standard montage: {e}")
            print("Using fallback electrode positions")
            
            # Fallback with a minimal set of standard positions
            electrode_positions = {
                # Central line
                'CZ': (0.0, 0.0),                    # Top center
                'FZ': (0.0, np.pi/3),                # Front center
                'PZ': (0.0, 2*np.pi/3),              # Back center
                
                # Frontal poles
                'FP1': (-np.pi/6, np.pi/6),          # Front pole left
                'FP2': (np.pi/6, np.pi/6),           # Front pole right
                
                # Frontal
                'F3': (-np.pi/3, np.pi/3),           # Front left
                'F4': (np.pi/3, np.pi/3),            # Front right
                'F7': (-2*np.pi/3, np.pi/3),         # Front temporal left
                'F8': (2*np.pi/3, np.pi/3),          # Front temporal right
                
                # Central
                'C3': (-np.pi/2, np.pi/2),           # Central left
                'C4': (np.pi/2, np.pi/2),            # Central right
                
                # Parietal
                'P3': (-np.pi/3, 2*np.pi/3),         # Parietal left
                'P4': (np.pi/3, 2*np.pi/3),          # Parietal right
                
                # Occipital
                'O1': (-np.pi/6, 5*np.pi/6),         # Occipital left
                'O2': (np.pi/6, 5*np.pi/6),          # Occipital right
            }
            
            print(f"Loaded {len(electrode_positions)} fallback electrode positions")
            return electrode_positions
    
    def _spherical_to_cartesian(self, theta, phi, r=1.0):
        """
        Convert spherical coordinates to Cartesian coordinates
        
        Args:
            theta: azimuthal angle
            phi: polar angle  
            r: radius (default 1.0 for unit sphere)
        """
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        return x, y, z
    
    def get_electrode_cartesian_positions(self):
        """Get electrode positions in Cartesian coordinates on unit sphere"""
        cartesian_positions = {}
        for name, (theta, phi) in self.electrode_positions.items():
            x, y, z = self._spherical_to_cartesian(theta, phi)
            cartesian_positions[name] = (x, y, z)
        return cartesian_positions

    def _standardize_channel_names(self, channel_names: List[str]) -> Dict[str, str]:
        """
        Create mapping from various channel naming conventions to standard names.
        
        Parameters:
        -----------
        channel_names : List[str]
            List of channel names from the data
            
        Returns:
        --------
        Dict[str, str]
            Mapping from original names to standardized names
        """
        mapping = {}
        
        for ch in channel_names:
            # Remove common prefixes and suffixes
            clean_name = ch.upper()
            
            # Handle common patterns:
            # EEG FP1-REF -> FP1
            # EEG_FP1_REF -> FP1
            # Fp1 -> FP1
            # FP1-A1 -> FP1
            # etc.
            
            patterns = [
                r'^EEG\s+([^-_\s]+)[-_]?.*$',  # EEG FP1-REF, EEG_FP1_REF
                r'^([A-Z0-9]+)[-_].*$',        # FP1-REF, FP1_A1
                r'^.*[-_]([A-Z0-9]+)[-_].*$',  # something_FP1_something
                r'^([A-Z0-9]+)$',              # FP1
                r'^([A-Z]+\d+).*$',            # Fp1, FP1x, etc.
            ]
            
            for pattern in patterns:
                match = re.match(pattern, clean_name)
                if match:
                    electrode_name = match.group(1).upper()
                    # Standardize common variations
                    electrode_name = electrode_name.replace('FP', 'FP')  # Ensure FP is uppercase
                    electrode_name = electrode_name.replace('OZ', 'OZ')  # Ensure OZ is uppercase
                    mapping[ch] = electrode_name
                    break
            
            if ch not in mapping:
                # If no pattern matched, try to extract electrode-like strings
                electrode_match = re.search(r'([A-Z]+\d+)', clean_name)
                if electrode_match:
                    mapping[ch] = electrode_match.group(1)
                else:
                    mapping[ch] = clean_name  # Use as-is if nothing else works
        
        return mapping
    
    def _get_electrode_position(self, electrode_name: str) -> Optional[np.ndarray]:
        """
        Get 3D position for an electrode.
        
        Parameters:
        -----------
        electrode_name : str
            Standardized electrode name
            
        Returns:
        --------
        np.ndarray or None
            3D position [x, y, z] or None if not found
        """
        if self.electrode_positions is None:
            return None
        
        # Try exact match with uppercase (most MNE positions are uppercase)
        electrode_upper = electrode_name.upper()
        if electrode_upper in self.electrode_positions:
            theta, phi = self.electrode_positions[electrode_upper]
            x, y, z = self._spherical_to_cartesian(theta, phi)
            return np.array([x, y, z])
        
        # Try lowercase (some positions might be stored lowercase)
        electrode_lower = electrode_name.lower()
        if electrode_lower in self.electrode_positions:
            theta, phi = self.electrode_positions[electrode_lower]
            x, y, z = self._spherical_to_cartesian(theta, phi)
            return np.array([x, y, z])
            
        # Try to find alternate names using our mapping function
        alternate_names = self._get_alternate_electrode_names(electrode_name)
        for alt_name in alternate_names:
            if alt_name in self.electrode_positions:
                theta, phi = self.electrode_positions[alt_name]
                x, y, z = self._spherical_to_cartesian(theta, phi)
                return np.array([x, y, z])
        
        # No position found for this electrode
        return None
    
    def _create_artificial_positions(self, n_electrodes: int) -> np.ndarray:
        """
        Create artificial electrode positions in a spherical arrangement.
        
        Parameters:
        -----------
        n_electrodes : int
            Number of electrodes
            
        Returns:
        --------
        np.ndarray
            Array of shape (n_electrodes, 3) with artificial positions
        """
        # Create positions on a sphere using spherical coordinates
        positions = []
        
        if n_electrodes <= 1:
            return np.array([[0, 0, 1]])  # Single electrode at top
        
        # Distribute electrodes roughly evenly on sphere
        # Use golden spiral or similar distribution
        golden_ratio = (1 + 5**0.5) / 2
        
        for i in range(n_electrodes):
            # Latitude (theta): from 0 to pi
            theta = np.arccos(1 - 2 * i / (n_electrodes - 1)) if n_electrodes > 1 else 0
            
            # Longitude (phi): golden ratio spiral
            phi = 2 * np.pi * i / golden_ratio
            
            # Convert to Cartesian coordinates
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            
            positions.append([x, y, z])
        
        return np.array(positions)
    
    def _cart2sph(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """
        Convert Cartesian to spherical coordinates.
        
        Parameters:
        -----------
        x, y, z : float
            Cartesian coordinates
            
        Returns:
        --------
        Tuple[float, float, float]
            (r, theta, phi) where theta is colatitude (0 to pi) and phi is longitude (-pi to pi)
        """
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r) if r > 0 else 0  # Colatitude (0 to pi)
        phi = np.arctan2(y, x)    # Longitude (-pi to pi)
        return r, theta, phi
    
    def load_data(self, data_source: Union[str, np.ndarray, 'mne.io.Raw', 'mne.Epochs'], 
                  time_window: Optional[Tuple[float, float]] = None,
                  channel_subset: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str], Optional[np.ndarray]]:
        """
        Load EEG data from various sources.
        
        Parameters:
        -----------
        data_source : str, np.ndarray, mne.io.Raw, or mne.Epochs
            Data source (file path, numpy array, or MNE object)
        time_window : Tuple[float, float], optional
            Time window (start, stop) in seconds for file-based sources
        channel_subset : List[str], optional
            Specific channels to analyze
            
        Returns:
        --------
        Tuple[np.ndarray, List[str], Optional[np.ndarray]]
            (data, electrode_names, times) where data is (n_electrodes, n_timepoints)
        """
        if isinstance(data_source, str):
            # File path
            if data_source.endswith('.edf'):
                raw = mne.io.read_raw_edf(data_source, preload=True, verbose=False)
            elif data_source.endswith('.fif'):
                raw = mne.io.read_raw_fif(data_source, preload=True, verbose=False)
            elif data_source.endswith('.mff'):
                raw = mne.io.read_raw_egi(data_source, preload=True, verbose=False)
            else:
                raise ValueError(f"Unsupported file format: {data_source}")
            
            # Get EEG channels
            eeg_picks = mne.pick_types(raw.info, meg=False, eeg=True, exclude='bads')
            raw.pick(eeg_picks)
            
            # Extract data for time window
            if time_window is not None:
                start_time, stop_time = time_window
                data, times = raw.get_data(start=start_time, stop=stop_time, return_times=True)
            else:
                data, times = raw.get_data(return_times=True)
            
            channel_names = raw.ch_names
            
        elif isinstance(data_source, (mne.io.Raw, mne.Epochs)):
            # MNE object
            if isinstance(data_source, mne.Epochs):
                # For epochs, take the mean across epochs or use first epoch
                data = data_source.get_data().mean(axis=0)  # Average across epochs
                times = data_source.times
            else:
                if time_window is not None:
                    start_time, stop_time = time_window
                    data, times = data_source.get_data(start=start_time, stop=stop_time, return_times=True)
                else:
                    data, times = data_source.get_data(return_times=True)
            
            channel_names = data_source.ch_names
            
        elif isinstance(data_source, np.ndarray):
            # Numpy array
            data = data_source
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)  # Single channel, multiple timepoints
            elif len(data.shape) > 2:
                raise ValueError("Data array should be 1D or 2D (channels x timepoints)")
            
            # Create artificial channel names
            channel_names = [f"CH{i+1}" for i in range(data.shape[0])]
            times = np.arange(data.shape[1])
            
        else:
            raise TypeError(f"Unsupported data source type: {type(data_source)}")
        
        # Filter channels if subset specified
        if channel_subset is not None:
            # Create mapping from original to standard names
            name_mapping = self._standardize_channel_names(channel_names)
            
            # Find indices of requested channels
            channel_indices = []
            available_electrodes = []
            
            for target_ch in channel_subset:
                target_std = target_ch.upper()
                for orig_name, std_name in name_mapping.items():
                    if std_name == target_std:
                        if orig_name in channel_names:
                            idx = channel_names.index(orig_name)
                            channel_indices.append(idx)
                            available_electrodes.append(std_name)
                            break
            
            if len(channel_indices) == 0:
                raise ValueError(f"None of the requested channels found in data. "
                               f"Available: {list(name_mapping.values())}")
            
            data = data[channel_indices, :]
            channel_names = available_electrodes
        else:
            # Use all channels, but standardize names
            name_mapping = self._standardize_channel_names(channel_names)
            channel_names = [name_mapping[ch] for ch in channel_names]
        
        return data, channel_names, times
    
    def analyze(self, data_source: Union[str, np.ndarray, 'mne.io.Raw', 'mne.Epochs'],
                time_window: Optional[Tuple[float, float]] = None,
                channel_subset: Optional[List[str]] = None,
                subsample_factor: int = 1) -> Dict:
        """
        Perform spherical harmonic analysis on EEG data.
        
        Parameters:
        -----------
        data_source : str, np.ndarray, mne.io.Raw, or mne.Epochs
            Data source
        time_window : Tuple[float, float], optional
            Time window (start, stop) in seconds
        channel_subset : List[str], optional
            Specific channels to analyze
        subsample_factor : int
            Factor to subsample time points (1 = no subsampling)
            
        Returns:
        --------
        Dict
            Dictionary containing:
            - 'coefficients': Spherical harmonic coefficients (time_points, 2, lmax+1, lmax+1)
            - 'times': Time points analyzed
            - 'electrodes': Electrode names used
            - 'positions': 3D positions used
            - 'lmax': Maximum harmonic degree
        """
        # Load data
        data, electrode_names, times = self.load_data(data_source, time_window, channel_subset)
        
        print(f"Loaded data for electrodes: {electrode_names}")
        print(f"Data shape: {data.shape}")
        if times is not None:
            print(f"Time range: {times[0]:.3f} to {times[-1]:.3f} seconds")
        
        # Get electrode positions
        positions = []
        valid_electrodes = []
        valid_data_indices = []
        
        print("Available electrode positions in database:", list(self.electrode_positions.keys())[:10], f"...and {len(self.electrode_positions)-10} more")
        print("Looking up positions for:", electrode_names)
        
        for i, electrode in enumerate(electrode_names):
            pos = self._get_electrode_position(electrode)
            if pos is not None:
                positions.append(pos)
                valid_electrodes.append(electrode)
                valid_data_indices.append(i)
                print(f"Found position for {electrode}: {pos}")
            else:
                print(f"No position found for {electrode}")
        
        if len(positions) == 0:
            print("WARNING: No electrode positions found in MNE database.")
            print("Using MNE's standard_1005 montage positions instead of artificial positions...")
            
            # Instead of artificial positions, use MNE's standard montage positions directly
            try:
                montage = mne.channels.make_standard_montage('standard_1005')
                ch_pos = montage.get_positions()['ch_pos']
                
                # Create positional mapping for all electrodes, regardless of match
                # This ensures we get proper anatomically-correct positions
                electrode_to_pos = {}
                for i, electrode in enumerate(electrode_names):
                    # Try direct match first
                    if electrode.upper() in ch_pos:
                        electrode_to_pos[electrode] = ch_pos[electrode.upper()]
                    else:
                        # Use the ith position as fallback (better than random artificial)
                        # This keeps the positions anatomically plausible
                        keys = list(ch_pos.keys())
                        fallback_key = keys[i % len(keys)]
                        electrode_to_pos[electrode] = ch_pos[fallback_key]
                        print(f"Using fallback position from {fallback_key} for {electrode}")
                
                # Extract positions in order of electrode_names
                positions = [electrode_to_pos[electrode] for electrode in electrode_names]
                positions = np.array(positions)
                valid_electrodes = electrode_names
                valid_data_indices = list(range(len(electrode_names)))
                print(f"Using {len(positions)} standard montage positions from MNE")
            except Exception as e:
                print(f"Error using MNE montage: {e}")
                print("Falling back to artificial positions (last resort)...")
                positions = self._create_artificial_positions(len(electrode_names))
                valid_electrodes = electrode_names
                valid_data_indices = list(range(len(electrode_names)))
        else:
            print(f"Found positions for {len(positions)}/{len(electrode_names)} electrodes")
            data = data[valid_data_indices, :]
            positions = np.array(positions)
        
        # Convert to spherical coordinates
        r_vals, theta_vals, phi_vals = [], [], []
        
        for pos in positions:
            r, theta, phi = self._cart2sph(pos[0], pos[1], pos[2])
            r_vals.append(r)
            theta_vals.append(theta)
            phi_vals.append(phi)
        
        r_vals = np.array(r_vals)
        theta_vals = np.array(theta_vals)
        phi_vals = np.array(phi_vals)
        
        # Normalize to unit sphere if requested
        if self.normalize_radius:
            r_mean = np.mean(r_vals)
            if r_mean > 0:
                positions_normalized = positions / r_mean
                for i, pos in enumerate(positions_normalized):
                    r_vals[i], theta_vals[i], phi_vals[i] = self._cart2sph(pos[0], pos[1], pos[2])
        
        # Convert to degrees for PyShTools
        theta_deg = np.degrees(theta_vals)
        phi_deg = np.degrees(phi_vals)
        
        # Subsample time points
        n_timepoints = data.shape[1]
        sample_indices = np.arange(0, n_timepoints, subsample_factor)
        
        print(f"Computing spherical harmonics for {len(sample_indices)} time points (lmax={self.lmax})...")
        
        # Pre-allocate coefficients array with consistent shape
        coefficients = np.zeros((len(sample_indices), 2, self.lmax + 1, self.lmax + 1))
        
        for i, time_idx in enumerate(sample_indices):
            if i % max(1, len(sample_indices) // 10) == 0:
                print(f"Processing time point {i+1}/{len(sample_indices)}")
            
            # Get EEG values at this time point
            values = data[:, time_idx]
            
            # Perform spherical harmonic expansion
            try:
                coeffs = pyshtools.expand.SHExpandLSQ(values, theta_deg, phi_deg, self.lmax)
                
                # Handle both tuple and array returns from PyShTools
                if isinstance(coeffs, tuple):
                    coeffs = coeffs[0]  # Take the first element if it's a tuple
                
                # Ensure coeffs has the right shape and copy to pre-allocated array
                if coeffs.shape[1] <= self.lmax + 1 and coeffs.shape[2] <= self.lmax + 1:
                    coefficients[i, :coeffs.shape[0], :coeffs.shape[1], :coeffs.shape[2]] = coeffs
                else:
                    # Truncate if larger than expected
                    coefficients[i] = coeffs[:2, :self.lmax+1, :self.lmax+1]
                    
            except Exception as e:
                print(f"Warning: Failed to compute spherical harmonics at time {i}: {e}")
                # Zero coefficients are already set by pre-allocation
                pass
        
        result = {
            'coefficients': coefficients,
            'times': times[sample_indices] if times is not None else sample_indices,
            'electrodes': valid_electrodes,
            'positions': positions,
            'lmax': self.lmax,
            'theta': theta_deg,
            'phi': phi_deg
        }
        
        print(f"Spherical harmonic analysis complete!")
        print(f"Coefficients shape: {coefficients.shape}")
        
        return result
    
    def reconstruct_signal(self, coefficients: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """
        Reconstruct signal from spherical harmonic coefficients.
        
        Parameters:
        -----------
        coefficients : np.ndarray
            Spherical harmonic coefficients
        theta : np.ndarray
            Colatitude values in degrees
        phi : np.ndarray
            Longitude values in degrees
            
        Returns:
        --------
        np.ndarray
            Reconstructed signals
        """
        reconstructed = []
        
        for i, coeffs in enumerate(coefficients):
            values = pyshtools.expand.MakeGridPoint(coeffs, theta, phi)
            reconstructed.append(values)
        
        return np.array(reconstructed)
    
    def visualize_transformation(self, result: Dict, time_point: int = 0, 
                               output_path: Optional[str] = None, 
                               show_harmonics_up_to: int = 4) -> None:
        """
        Visualize how sensor-level data transforms into spherical harmonics.
        
        Parameters:
        -----------
        result : Dict
            Result from analyze() method
        time_point : int
            Which time point to visualize
        output_path : str, optional
            Path to save the figure
        show_harmonics_up_to : int
            Show individual harmonics up to this degree
        """
        fig = plt.figure(figsize=(20, 12))
        
        # Extract data
        coeffs = result['coefficients'][time_point]  # Shape: (2, lmax+1, lmax+1)
        positions = result['positions']
        electrodes = result['electrodes']
        times = result['times']
        
        # Normalize positions to unit sphere for visualization
        positions_normalized = np.zeros_like(positions)
        for i, pos in enumerate(positions):
            radius = np.linalg.norm(pos)
            if radius > 0:
                positions_normalized[i] = pos / radius
            else:
                positions_normalized[i] = [0, 0, 1]  # Default to top of sphere
        
        # Get original data at this time point
        theta_deg = result['theta']
        phi_deg = result['phi']
        
        # Create high-resolution grid for visualization
        n_grid = 100
        theta_grid = np.linspace(0, 180, n_grid)
        phi_grid = np.linspace(-180, 180, n_grid)
        THETA, PHI = np.meshgrid(theta_grid, phi_grid)
        
        # Convert to Cartesian for 3D plotting
        theta_rad = np.radians(THETA)
        phi_rad = np.radians(PHI)
        X = np.sin(theta_rad) * np.cos(phi_rad)
        Y = np.sin(theta_rad) * np.sin(phi_rad)
        Z = np.cos(theta_rad)
        
        # 1. Original sensor data visualization (3D scatter)
        ax1 = fig.add_subplot(2, 4, 1, projection='3d')
        
        # Get original values at sensor positions (reconstruct from coefficients)
        original_values = []
        for i in range(len(theta_deg)):
            val = pyshtools.expand.MakeGridPoint(coeffs, theta_deg[i], phi_deg[i])
            original_values.append(val)
        original_values = np.array(original_values)
        
        # Draw unit sphere wireframe for reference
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        sphere_x = np.outer(np.cos(u), np.sin(v))
        sphere_y = np.outer(np.sin(u), np.sin(v))
        sphere_z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax1.plot_wireframe(sphere_x, sphere_y, sphere_z, alpha=0.1, color='gray')
        
        # 3D scatter plot of sensor positions colored by values
        scatter = ax1.scatter(positions_normalized[:, 0], positions_normalized[:, 1], positions_normalized[:, 2], 
                            c=original_values, s=100, cmap='RdBu_r', 
                            vmin=-np.max(np.abs(original_values)), 
                            vmax=np.max(np.abs(original_values)))
        
        # Add electrode labels
        for i, (pos, label) in enumerate(zip(positions_normalized, electrodes)):
            ax1.text(pos[0]*1.1, pos[1]*1.1, pos[2]*1.1, label, fontsize=8)
        
        ax1.set_title(f'Original Sensor Data\nTime: {times[time_point]:.3f}s')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_box_aspect([1,1,1])  # Equal aspect ratio
        plt.colorbar(scatter, ax=ax1, shrink=0.5)
        
        # 2. Full reconstruction on sphere
        ax2 = fig.add_subplot(2, 4, 2, projection='3d')
        
        # Reconstruct full field
        reconstructed_grid = np.zeros_like(THETA)
        for i in range(n_grid):
            for j in range(n_grid):
                reconstructed_grid[i, j] = pyshtools.expand.MakeGridPoint(
                    coeffs, theta_grid[i], phi_grid[j])
        
        # Normalize the reconstructed values for color mapping
        vmin, vmax = reconstructed_grid.min(), reconstructed_grid.max()
        if vmax != vmin:
            colors = plt.cm.RdBu_r((reconstructed_grid - vmin) / (vmax - vmin))
        else:
            colors = plt.cm.RdBu_r(np.ones_like(reconstructed_grid) * 0.5)
        
        # Plot surface
        surf = ax2.plot_surface(X, Y, Z, facecolors=colors, 
                               alpha=0.8, linewidth=0, antialiased=True)
        
        # Add sensor positions as black dots
        ax2.scatter(positions_normalized[:, 0], positions_normalized[:, 1], positions_normalized[:, 2], 
                   c='black', s=30, alpha=1.0)
        
        ax2.set_title('Reconstructed Field\n(All Harmonics)')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_box_aspect([1,1,1])  # Equal aspect ratio
        
        # 3. Coefficient magnitude spectrum
        ax3 = fig.add_subplot(2, 4, 3)
        
        # Calculate power spectrum by degree
        power_by_degree = []
        degrees = []
        for l in range(result['lmax'] + 1):
            power = 0
            for m in range(-l, l + 1):
                m_idx = abs(m) if m >= 0 else abs(m)
                if m >= 0:
                    power += coeffs[0, l, m_idx]**2
                else:
                    power += coeffs[1, l, m_idx]**2
            power_by_degree.append(np.sqrt(power))
            degrees.append(l)
        
        ax3.bar(degrees, power_by_degree, alpha=0.7, color='steelblue')
        ax3.set_xlabel('Spherical Harmonic Degree (l)')
        ax3.set_ylabel('Power')
        ax3.set_title('Harmonic Power Spectrum')
        ax3.grid(True, alpha=0.3)
        
        # 4. Individual harmonic contributions
        ax4 = fig.add_subplot(2, 4, 4)
        
        # Show first few harmonics as 2D projections
        harmonic_contributions = []
        for l in range(min(show_harmonics_up_to + 1, result['lmax'] + 1)):
            # Create coefficients with only this degree
            single_harmonic_coeffs = np.zeros_like(coeffs)
            single_harmonic_coeffs[:, l, :] = coeffs[:, l, :]
            
            # Reconstruct
            harmonic_field = np.zeros_like(THETA)
            for i in range(n_grid):
                for j in range(n_grid):
                    harmonic_field[i, j] = pyshtools.expand.MakeGridPoint(
                        single_harmonic_coeffs, theta_grid[i], phi_grid[j])
            
            harmonic_contributions.append(harmonic_field)
        
        # Plot as subplots in a grid
        for l in range(len(harmonic_contributions)):
            ax_harmonic = fig.add_subplot(2, 4, 5 + l)
            
            # Mollweide projection
            theta_plot = np.radians(THETA - 90)  # Convert to latitude
            phi_plot = np.radians(PHI)
            
            im = ax_harmonic.contourf(phi_plot, theta_plot, harmonic_contributions[l], 
                                    levels=20, cmap='RdBu_r', extend='both')
            ax_harmonic.set_title(f'Harmonic l={l}')
            ax_harmonic.set_xlabel('Longitude (rad)')
            ax_harmonic.set_ylabel('Latitude (rad)')
            plt.colorbar(im, ax=ax_harmonic)
            
            # Add sensor positions
            sensor_theta_plot = np.radians(theta_deg - 90)
            sensor_phi_plot = np.radians(phi_deg)
            ax_harmonic.scatter(sensor_phi_plot, sensor_theta_plot, 
                              c='black', s=20, alpha=0.8)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {output_path}")
        else:
            plt.show()
    
    def plot_harmonic_evolution(self, result: Dict, electrode_subset: Optional[List[str]] = None,
                              output_path: Optional[str] = None) -> None:
        """
        Plot how harmonic coefficients evolve over time.
        
        Parameters:
        -----------
        result : Dict
            Result from analyze() method
        electrode_subset : List[str], optional
            Subset of electrodes to highlight
        output_path : str, optional
            Path to save the figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        coeffs = result['coefficients']  # Shape: (time, 2, lmax+1, lmax+1)
        times = result['times']
        
        # 1. Power by degree over time
        ax = axes[0, 0]
        power_evolution = []
        
        for t in range(len(times)):
            power_by_degree = []
            for l in range(result['lmax'] + 1):
                power = 0
                for m in range(-l, l + 1):
                    m_idx = abs(m)
                    if m >= 0:
                        power += coeffs[t, 0, l, m_idx]**2
                    else:
                        power += coeffs[t, 1, l, m_idx]**2
                power_by_degree.append(np.sqrt(power))
            power_evolution.append(power_by_degree)
        
        power_evolution = np.array(power_evolution)  # Shape: (time, degree)
        
        # Plot as heatmap
        im = ax.imshow(power_evolution.T, aspect='auto', origin='lower', 
                      cmap='viridis', extent=[times[0], times[-1], 0, result['lmax']])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Harmonic Degree (l)')
        ax.set_title('Harmonic Power Evolution')
        plt.colorbar(im, ax=ax)
        
        # 2. Total power over time
        ax = axes[0, 1]
        total_power = np.sum(power_evolution, axis=1)
        ax.plot(times, total_power, 'b-', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Total Power')
        ax.set_title('Total Spectral Power')
        ax.grid(True, alpha=0.3)
        
        # 3. Low vs high frequency harmonics
        ax = axes[1, 0]
        low_freq = np.sum(power_evolution[:, :3], axis=1)  # l=0,1,2
        high_freq = np.sum(power_evolution[:, 3:], axis=1)  # l>=3
        
        ax.plot(times, low_freq, 'r-', label='Low degree (l≤2)', linewidth=2)
        ax.plot(times, high_freq, 'b-', label='High degree (l≥3)', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Power')
        ax.set_title('Low vs High Degree Harmonics')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Specific harmonic coefficients
        ax = axes[1, 1]
        
        # Plot first few harmonic coefficients
        for l in range(min(5, result['lmax'] + 1)):
            coeff_evolution = []
            for t in range(len(times)):
                # Take the (0,0) coefficient for each degree
                coeff_evolution.append(coeffs[t, 0, l, 0])
            ax.plot(times, coeff_evolution, label=f'l={l}, m=0', linewidth=2)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Coefficient Value')
        ax.set_title('Individual Harmonic Coefficients')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Evolution plot saved to {output_path}")
        else:
            plt.show()
    
    def compare_reconstruction_quality(self, result: Dict, max_degrees: List[int] = [1, 2, 4, 8],
                                     time_point: int = 0, output_path: Optional[str] = None) -> None:
        """
        Compare reconstruction quality with different numbers of harmonics.
        
        Parameters:
        -----------
        result : Dict
            Result from analyze() method
        max_degrees : List[int]
            Different maximum degrees to compare
        time_point : int
            Which time point to analyze
        output_path : str, optional
            Path to save the figure
        """
        fig, axes = plt.subplots(2, len(max_degrees), figsize=(4*len(max_degrees), 8))
        
        coeffs_full = result['coefficients'][time_point]
        theta_deg = result['theta']
        phi_deg = result['phi']
        positions = result['positions']
        electrodes = result['electrodes']
        
        # Get original values at sensor positions
        original_values = []
        for i in range(len(theta_deg)):
            val = pyshtools.expand.MakeGridPoint(coeffs_full, theta_deg[i], phi_deg[i])
            original_values.append(val)
        original_values = np.array(original_values)
        
        # Create grid for surface plots
        n_grid = 50
        theta_grid = np.linspace(0, 180, n_grid)
        phi_grid = np.linspace(-180, 180, n_grid)
        THETA, PHI = np.meshgrid(theta_grid, phi_grid)
        
        errors = []
        
        for i, max_l in enumerate(max_degrees):
            # Create truncated coefficients
            coeffs_truncated = np.zeros_like(coeffs_full)
            coeffs_truncated[:, :max_l+1, :max_l+1] = coeffs_full[:, :max_l+1, :max_l+1]
            
            # Reconstruct values at sensor positions
            reconstructed_values = []
            for j in range(len(theta_deg)):
                val = pyshtools.expand.MakeGridPoint(coeffs_truncated, theta_deg[j], phi_deg[j])
                reconstructed_values.append(val)
            reconstructed_values = np.array(reconstructed_values)
            
            # Calculate reconstruction error
            error = np.mean((original_values - reconstructed_values)**2)
            errors.append(error)
            
            # Plot 3D reconstruction
            if len(max_degrees) > 1:
                ax = axes[0, i]
            else:
                ax = axes[0]
            
            # Create 3D scatter plot
            ax = fig.add_subplot(2, len(max_degrees), i+1, projection='3d')
            
            scatter = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                               c=reconstructed_values, s=100, cmap='RdBu_r', 
                               vmin=-np.max(np.abs(original_values)), 
                               vmax=np.max(np.abs(original_values)))
            
            ax.set_title(f'Reconstruction (l≤{max_l})\nMSE: {error:.6f}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            # Plot error at each sensor
            if len(max_degrees) > 1:
                ax2 = axes[1, i]
            else:
                ax2 = axes[1]
            
            sensor_errors = (original_values - reconstructed_values)**2
            bars = ax2.bar(range(len(electrodes)), sensor_errors, alpha=0.7)
            ax2.set_xlabel('Electrode')
            ax2.set_ylabel('Squared Error')
            ax2.set_title(f'Per-Electrode Error (l≤{max_l})')
            ax2.set_xticks(range(len(electrodes)))
            ax2.set_xticklabels(electrodes, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Reconstruction comparison saved to {output_path}")
        else:
            plt.show()
        
        # Print summary
        print("\nReconstruction Quality Summary:")
        print("Max Degree | MSE")
        print("-----------|---------")
        for max_l, error in zip(max_degrees, errors):
            print(f"    {max_l:2d}     | {error:.6f}")
    
    def plot_electrode_positions(self, figsize=(10, 8)):
        """Plot electrode positions on a sphere to verify positioning."""
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Create sphere surface
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        sphere_x = np.outer(np.cos(u), np.sin(v))
        sphere_y = np.outer(np.sin(u), np.sin(v))
        sphere_z = np.outer(np.ones(np.size(u)), np.cos(v))
        
        # Plot transparent sphere
        ax.plot_surface(sphere_x, sphere_y, sphere_z, alpha=0.1, color='lightblue')
        
        # Plot electrode positions
        for name, pos in self.electrode_positions.items():
            x, y, z = pos['cartesian']
            ax.scatter(x, y, z, c='red', s=50)
            ax.text(x*1.1, y*1.1, z*1.1, name, fontsize=8)
        
        # Set equal aspect ratio and labels
        ax.set_xlabel('X (Left-Right)')
        ax.set_ylabel('Y (Anterior-Posterior)')
        ax.set_zlabel('Z (Inferior-Superior)')
        ax.set_title('EEG Electrode Positions (Standard 10-20 System)')
        
        # Set viewing angle to show the head from above and slightly in front
        ax.view_init(elev=20, azim=45)
        
        # Make axes equal
        max_range = 0.1
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
        
        plt.tight_layout()
        plt.show()
        return fig
    
    def _map_edf_channels_to_positions(self, channel_names, electrode_positions):
        """
        Map EDF channel names to electrode positions from standard montage.
        
        This method handles various EEG naming conventions found in EDF files
        and maps them to standard electrode positions.
        """
        mapped_positions = []
        mapped_channels = []
        
        # Common EDF channel name patterns
        # EEG naming formats can vary significantly: "EEG FP1-REF", "EEG FP1", "FP1", "FP1-REF", etc.
        eeg_patterns = [
            r'^EEG (\w+)-REF$',        # "EEG FP1-REF"
            r'^EEG (\w+)-\w+$',        # "EEG FP1-A1", "EEG FP1-LE"
            r'^EEG (\w+)$',            # "EEG FP1"
            r'^EEG (\w+)-$',           # "EEG FP1-"
            r'^(\w+)-REF$',            # "FP1-REF"
            r'^(\w+)-\w+$',            # "FP1-A1", "FP1-LE"
            r'^(\w+)$',                # "FP1" (just the electrode name)
        ]
        
        # Try to get positions directly from MNE's standard montage
        try:
            # This gives us all available standard positions
            montage = mne.channels.make_standard_montage('standard_1005')
            std_positions = montage.get_positions()
        except Exception as e:
            print(f"Warning: Could not access MNE standard positions: {e}")
            std_positions = None
        
        for ch_name in channel_names:
            ch_name = ch_name.strip()
            matched = False
            electrode = None
            
            # First try direct match (some EDF files use standard names directly)
            if ch_name.upper() in electrode_positions:
                electrode = ch_name.upper()
                matched = True
                
            # Try various patterns to extract electrode name
            else:
                for pattern in eeg_patterns:
                    match = re.match(pattern, ch_name)
                    if match:
                        # Extract the electrode name part
                        electrode = match.group(1).upper()
                        matched = True
                        break
            
            # Check if the extracted electrode name exists in our positions
            if matched and electrode:
                if electrode in electrode_positions:
                    # Get the position
                    theta, phi = electrode_positions[electrode]
                    
                    # Convert spherical to Cartesian (unit sphere)
                    x = np.sin(phi) * np.cos(theta)
                    y = np.sin(phi) * np.sin(theta)
                    z = np.cos(phi)
                    
                    mapped_positions.append([x, y, z])
                    mapped_channels.append(electrode)
                    print(f"Mapped {ch_name} -> {electrode} at ({x:.3f}, {y:.3f}, {z:.3f})")
                else:
                    # Try alternate electrode name mappings
                    # Some EDF files use variations like 'T3' instead of 'T7' in the newer 10-10 system
                    alternate_names = self._get_alternate_electrode_names(electrode)
                    found_alternate = False
                    
                    for alt_name in alternate_names:
                        if alt_name in electrode_positions:
                            theta, phi = electrode_positions[alt_name]
                            x = np.sin(phi) * np.cos(theta)
                            y = np.sin(phi) * np.sin(theta)
                            z = np.cos(phi)
                            
                            mapped_positions.append([x, y, z])
                            mapped_channels.append(alt_name)
                            print(f"Mapped {ch_name} -> {electrode} (as {alt_name}) at ({x:.3f}, {y:.3f}, {z:.3f})")
                            found_alternate = True
                            break
                    
                    if not found_alternate:
                        print(f"Warning: No position found for electrode {electrode} (from {ch_name})")
            else:
                print(f"Skipping non-EEG channel: {ch_name}")
        
        if not mapped_positions:
            raise ValueError("No EEG channels could be mapped to electrode positions. Check channel naming format.")
        
        return np.array(mapped_positions), mapped_channels
        
    def _get_alternate_electrode_names(self, electrode_name):
        """
        Get alternative names for electrodes to handle different naming conventions.
        
        This helps map between different EEG naming standards like the older 10-20 
        system and newer 10-10/10-5 systems.
        """
        # Common electrode name mappings between systems
        electrode_name = electrode_name.upper()
        
        # Map between older and newer naming conventions
        name_mappings = {
            # Older to newer
            'T3': ['T7'],
            'T4': ['T8'],
            'T5': ['P7'],
            'T6': ['P8'],
            'M1': ['A1', 'TP9'],
            'M2': ['A2', 'TP10'],
            
            # Newer to older
            'T7': ['T3'],
            'T8': ['T4'],
            'P7': ['T5'],
            'P8': ['T6'],
            'A1': ['M1'],
            'A2': ['M2'],
            'TP9': ['M1', 'A1'],
            'TP10': ['M2', 'A2'],
            
            # Possible typos or variations
            'FP1': ['FP1', 'Fp1'],
            'FP2': ['FP2', 'Fp2'],
            'FPZ': ['FPZ', 'Fpz'],
        }
        
        # Return the original name and any alternates
        if electrode_name in name_mappings:
            return [electrode_name] + name_mappings[electrode_name]
        return [electrode_name]
    
if __name__ == "__main__":
    # Example usage with comprehensive visualization
    print("Testing spherical harmonic analysis with visualization...")
    
    # Test with the EDF file
    edf_path = 'data/TUEG/tuh_eeg/v2.0.0/edf/000/aaaaaaaa/s001_2015_12_30/01_tcp_ar/aaaaaaaa_s001_t000.edf'
    
    try:
        # Create analyzer
        analyzer = SphericalHarmonicAnalyzer(lmax=8)  # Reduced lmax for faster computation
        
        print("Analyzing EDF file with all available EEG channels...")
        result = analyzer.analyze(
            edf_path, 
            time_window=(100, 120),  # Longer time window for better visualization
            subsample_factor=50      # More time points for evolution plots
        )
        
        print(f"Analysis complete!")
        print(f"Electrodes analyzed: {result['electrodes']}")
        print(f"Coefficients shape: {result['coefficients'].shape}")
        print(f"Time points: {len(result['times'])}")
        
        # Generate comprehensive visualizations
        print("\nGenerating visualizations...")
        
        # 1. Main transformation visualization
        print("Creating transformation visualization...")
        analyzer.visualize_transformation(
            result, 
            time_point=0, 
            output_path='eeg_harmonic_transformation.png',
            show_harmonics_up_to=3
        )
        
        # 2. Time evolution plots
        if len(result['times']) > 1:
            print("Creating harmonic evolution plots...")
            analyzer.plot_harmonic_evolution(
                result,
                output_path='eeg_harmonic_evolution.png'
            )
        
        # 3. Reconstruction quality comparison
        print("Creating reconstruction quality comparison...")
        analyzer.compare_reconstruction_quality(
            result,
            max_degrees=[1, 2, 4, 8],
            time_point=0,
            output_path='eeg_reconstruction_quality.png'
        )
        
        print("\nAll visualizations complete! Check the generated PNG files.")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback test with artificial data
        print("\nTesting with artificial data...")
        
        # Create realistic artificial EEG data
        n_electrodes = 10
        n_timepoints = 500
        time_vec = np.linspace(0, 2, n_timepoints)  # 2 seconds
        
        # Simulate EEG with some spatial and temporal structure
        artificial_data = np.zeros((n_electrodes, n_timepoints))
        
        # Add some oscillations with spatial patterns
        for i in range(n_electrodes):
            # Different frequencies for different "electrodes"
            freq1 = 8 + i * 0.5  # Alpha-like frequencies
            freq2 = 15 + i * 0.3  # Beta-like frequencies
            
            # Spatial amplitude modulation
            amp_mod = np.cos(i * np.pi / n_electrodes)
            
            artificial_data[i, :] = (
                amp_mod * np.sin(2 * np.pi * freq1 * time_vec) +
                0.5 * amp_mod * np.sin(2 * np.pi * freq2 * time_vec) +
                0.1 * np.random.randn(n_timepoints)  # Add some noise
            )
        
        # Analyze artificial data
        analyzer = SphericalHarmonicAnalyzer(lmax=6)
        result = analyzer.analyze(artificial_data, subsample_factor=25)
        
        print(f"Artificial data analysis complete!")
        print(f"Coefficients shape: {result['coefficients'].shape}")
        
        # Generate visualizations for artificial data
        analyzer.visualize_transformation(
            result, 
            time_point=0, 
            output_path='artificial_eeg_transformation.png'
        )
        
        if len(result['times']) > 1:
            analyzer.plot_harmonic_evolution(
                result,
                output_path='artificial_eeg_evolution.png'
            )
        
        print("Artificial data visualizations complete!")

# Convenience function for quick analysis and visualization
def analyze_and_visualize(data_source, output_prefix="eeg_analysis", **kwargs):
    """
    Convenience function to perform analysis and generate all visualizations.
    
    Parameters:
    -----------
    data_source : str, np.ndarray, mne.io.Raw, or mne.Epochs
        Data source for analysis
    output_prefix : str
        Prefix for output file names
    **kwargs : dict
        Additional arguments for the analyzer
    
    Returns:
    --------
    Dict
        Analysis results
    """
    analyzer = SphericalHarmonicAnalyzer(**kwargs)
    result = analyzer.analyze(data_source)
    
    # Generate all visualizations
    analyzer.visualize_transformation(
        result, 
        output_path=f"{output_prefix}_transformation.png"
    )
    
    if len(result['times']) > 1:
        analyzer.plot_harmonic_evolution(
            result, 
            output_path=f"{output_prefix}_evolution.png"
        )
    
    analyzer.compare_reconstruction_quality(
        result, 
        output_path=f"{output_prefix}_reconstruction.png"
    )
    
    return result