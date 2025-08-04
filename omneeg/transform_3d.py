#!/usr/bin/env python
"""
3D Spherical Harmonics for EEG
Core functionality: convert EEG data to spherical harmonics
"""

import numpy as np
import mne
import pyshtools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class SphericalHarmonics3D:
    """
    3D spherical harmonic analysis for EEG data.
    Takes EEG data with montage and generates spherical harmonics.
    """
    
    def __init__(self, lmax=8):
        """
        Initialize analyzer.
        
        Parameters:
        -----------
        lmax : int
            Maximum spherical harmonic degree
        """
        self.lmax = lmax
    
    def analyze(self, eeg_data, time_window=None):
        """
        Analyze EEG data and convert to spherical harmonics.
        
        Parameters:
        -----------
        eeg_data : mne.io.Raw or mne.Epochs
            EEG data with montage set
        time_window : tuple, optional
            Time window (start, stop) in seconds
            
        Returns:
        --------
        dict
            Analysis results with coefficients and metadata
        """
        # Get EEG channels
        eeg_picks = mne.pick_types(eeg_data.info, meg=False, eeg=True, exclude='bads')
        eeg_data.pick(eeg_picks)
        
        # Extract data
        if time_window is not None:
            start_time, stop_time = time_window
            data, times = eeg_data.get_data(start=start_time, stop=stop_time, return_times=True)
        else:
            data, times = eeg_data.get_data(return_times=True)
        
        channel_names = eeg_data.ch_names
        print(f"Analyzing {len(channel_names)} channels: {channel_names}")
        
        # Get electrode positions from montage
        positions = self._get_electrode_positions(eeg_data)
        
        # Convert positions to spherical coordinates
        theta_deg, phi_deg = self._cartesian_to_spherical(positions)
        
        # Compute spherical harmonics for each time point
        print(f"Computing spherical harmonics (lmax={self.lmax})...")
        
        # Vectorized computation using np.apply_along_axis for efficiency
        def compute_coeffs(values):
            coeffs = pyshtools.expand.SHExpandLSQ(values, theta_deg, phi_deg, self.lmax)
            if isinstance(coeffs, tuple):
                coeffs = coeffs[0]
            return coeffs
        
        # data shape: (n_channels, n_times), so apply along axis=0 for each time point
        coefficients = np.apply_along_axis(compute_coeffs, 0, data)
        
        # Transpose to get the expected shape: (time_points, 2, lmax+1, lmax+1)
        # The apply_along_axis changes the shape from (2, lmax+1, lmax+1, time_points) 
        # to (time_points, 2, lmax+1, lmax+1)
        n_time_points = data.shape[1]
        expected_shape = (n_time_points, 2, self.lmax + 1, self.lmax + 1)
        
        # Move the last dimension (time_points) to the front: (2, lmax+1, lmax+1, time_points) -> (time_points, 2, lmax+1, lmax+1)
        coefficients = np.transpose(coefficients, (3, 0, 1, 2))
        
        # Verify the shape is correct
        assert coefficients.shape == expected_shape, f"Expected shape {expected_shape}, got {coefficients.shape}"
        
        result = {
            'coefficients': coefficients,
            'times': times,
            'electrodes': channel_names,
            'positions': positions,
            'lmax': self.lmax,
            'theta': theta_deg,
            'phi': phi_deg
        }
        
        print(f"Analysis complete! Coefficients shape: {coefficients.shape}")
        return result
    
    def _get_electrode_positions(self, eeg_data):
        """Get electrode positions from MNE data with montage."""
        # Get positions from the montage
        montage = eeg_data.get_montage()
        if montage is None:
            raise ValueError("No montage found. Please set a montage first.")
        
        positions = []
        valid_channels = []
        
        for ch_name in eeg_data.ch_names:
            # Get position for this channel
            pos = montage.get_positions()
            if 'ch_pos' in pos and ch_name in pos['ch_pos']:
                positions.append(pos['ch_pos'][ch_name])
                valid_channels.append(ch_name)
            else:
                # Skip channels without positions (non-EEG channels)
                print(f"Skipping {ch_name} - no position found")
        
        if len(positions) == 0:
            raise ValueError("No EEG channels with valid positions found")
        
        # Update the data to only include valid channels
        eeg_data.pick(valid_channels)
        
        return np.array(positions)
    
    def _cartesian_to_spherical(self, positions):
        """
        Convert Cartesian positions to spherical coordinates.
        
        Returns:
        --------
        tuple
            (theta_deg, phi_deg) where theta is colatitude (0-180°) and phi is longitude (-180 to 180°)
        """
        theta_deg = []
        phi_deg = []
        
        for pos in positions:
            x, y, z = pos
            
            # Convert to spherical coordinates
            r = np.sqrt(x**2 + y**2 + z**2)
            if r > 0:
                # Colatitude (0 to 180°)
                theta = np.arccos(z / r)
                # Longitude (-180 to 180°)
                phi = np.arctan2(y, x)
            else:
                theta = 0
                phi = 0
            
            theta_deg.append(np.degrees(theta))
            phi_deg.append(np.degrees(phi))
        
        return np.array(theta_deg), np.array(phi_deg)
    
    def visualize(self, result, time_point=0, output_path=None):
        """
        Simple visualization of spherical harmonic results.
        
        Parameters:
        -----------
        result : dict
            Analysis results from analyze() method
        time_point : int
            Which time point to visualize
        output_path : str, optional
            Path to save the figure
        """
        fig = plt.figure(figsize=(15, 5))
        
        # Extract data
        coeffs = result['coefficients'][time_point]
        positions = result['positions']
        electrodes = result['electrodes']
        times = result['times']
        
        # 1. Original sensor data
        ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        
        # Get original values at sensor positions
        theta_deg = result['theta']
        phi_deg = result['phi']
        original_values = []
        for i in range(len(theta_deg)):
            val = pyshtools.expand.MakeGridPoint(coeffs, theta_deg[i], phi_deg[i])
            original_values.append(val)
        original_values = np.array(original_values)
        
        # Plot sensor positions colored by values
        scatter = ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                            c=original_values, s=100, cmap='RdBu_r')
        
        # Add electrode labels
        for i, (pos, label) in enumerate(zip(positions, electrodes)):
            ax1.text(pos[0]*1.1, pos[1]*1.1, pos[2]*1.1, label, fontsize=8)
        
        ax1.set_title(f'Sensor Data (t={times[time_point]:.2f}s)')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        plt.colorbar(scatter, ax=ax1, shrink=0.5)
        
        # 2. Harmonic power spectrum
        ax2 = fig.add_subplot(1, 3, 2)
        
        # Calculate power by degree (with overflow protection)
        power_by_degree = []
        degrees = []
        for l in range(self.lmax + 1):
            power = 0
            for m in range(-l, l + 1):
                m_idx = abs(m)
                if m >= 0:
                    # Use abs() to prevent overflow
                    power += abs(coeffs[0, l, m_idx])
                else:
                    power += abs(coeffs[1, l, m_idx])
            power_by_degree.append(power)
            degrees.append(l)
        
        ax2.bar(degrees, power_by_degree, alpha=0.7, color='steelblue')
        ax2.set_xlabel('Harmonic Degree (l)')
        ax2.set_ylabel('Power')
        ax2.set_title('Harmonic Power Spectrum')
        ax2.grid(True, alpha=0.3)
        
        # 3. Reconstructed field on sphere
        ax3 = fig.add_subplot(1, 3, 3, projection='3d')
        
        # Create high-resolution grid
        n_grid = 50
        theta_grid = np.linspace(0, 180, n_grid)
        phi_grid = np.linspace(-180, 180, n_grid)
        THETA, PHI = np.meshgrid(theta_grid, phi_grid)
        
        # Reconstruct field
        reconstructed = np.zeros_like(THETA)
        for i in range(n_grid):
            for j in range(n_grid):
                try:
                    reconstructed[i, j] = pyshtools.expand.MakeGridPoint(
                        coeffs, theta_grid[i], phi_grid[j])
                except Exception:
                    # Handle any reconstruction errors
                    reconstructed[i, j] = 0
        
        # Convert to Cartesian for 3D plotting
        theta_rad = np.radians(THETA)
        phi_rad = np.radians(PHI)
        X = np.sin(theta_rad) * np.cos(phi_rad)
        Y = np.sin(theta_rad) * np.sin(phi_rad)
        Z = np.cos(theta_rad)
        
        # Plot surface with error handling
        try:
            vmin, vmax = reconstructed.min(), reconstructed.max()
            if vmax != vmin:
                colors = plt.cm.RdBu_r((reconstructed - vmin) / (vmax - vmin))
            else:
                colors = plt.cm.RdBu_r(np.ones_like(reconstructed) * 0.5)
            surf = ax3.plot_surface(X, Y, Z, facecolors=colors, alpha=0.8)
        except Exception:
            # Fallback to wireframe if surface plotting fails
            ax3.plot_wireframe(X, Y, Z, alpha=0.3, color='blue')
        
        # Add sensor positions
        ax3.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                   c='black', s=30, alpha=1.0)
        
        ax3.set_title('Reconstructed Field')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {output_path}")
        else:
            plt.show()


def analyze_eeg_3d(file_path, time_window=(10, 20), lmax=8):
    """
    Analyze EEG file and generate spherical harmonics.
    
    Parameters:
    -----------
    file_path : str
        Path to EEG file
    time_window : tuple
        Time window to analyze
    lmax : int
        Maximum harmonic degree
        
    Returns:
    --------
    dict
        Analysis results
    """
    # Load EEG data
    if file_path.endswith('.edf'):
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    elif file_path.endswith('.fif'):
        raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    # Rename channels to match standard montage
    print("Renaming channels to match standard montage...")
    channel_mapping = {}
    for ch_name in raw.ch_names:
        if ch_name.startswith('EEG ') and ch_name.endswith('-REF'):
            # Extract electrode name from "EEG FP1-REF" -> "FP1"
            electrode_name = ch_name[4:-4]  # Remove "EEG " and "-REF"
            channel_mapping[ch_name] = electrode_name
        elif ch_name.endswith('-REF'):
            # Extract electrode name from "FP1-REF" -> "FP1"
            electrode_name = ch_name[:-4]
            channel_mapping[ch_name] = electrode_name
    
    if channel_mapping:
        raw.rename_channels(channel_mapping)
        print(f"Renamed {len(channel_mapping)} channels")
    
    # Set standard montage
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, on_missing='warn')
    
    # Create analyzer and analyze
    analyzer = SphericalHarmonics3D(lmax=lmax)
    result = analyzer.analyze(raw, time_window=time_window)
    
    return result, analyzer


if __name__ == "__main__":
    # Example usage
    print("3D Spherical Harmonics Test")
    
    # Test with your EEG file - MODIFY THIS PATH TO YOUR OWN DATA
    eeg_file = "path/to/your/eeg/file.edf"  # Replace with your actual file path
    
    try:
        result, analyzer = analyze_eeg_3d(eeg_file, time_window=(100, 120))
        analyzer.visualize(result, output_path='harmonics_3d.png')
        print("Analysis completed successfully!")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the EEG file path is correct and the file exists.")
        print("\nTo use this test:")
        print("1. Replace 'path/to/your/eeg/file.edf' with your actual EEG file path")
        print("2. Make sure your EEG file has standard 10-20 electrode names")
        print("3. Run the script again") 