#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : interactive_viewer.py
# description     : Interactive viewer for EEG spherical harmonics with time and rotation controls
# author          : Generated for OmnEEG
# date            : 2025-05-27
# version         : 1
# usage           : from omneeg.interactive_viewer import InteractiveViewer
# notes           : Provides interactive controls for time and rotation
# python_version  : 3.9
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D
import pyshtools
from omneeg.spherical_harmonics import SphericalHarmonicAnalyzer


class InteractiveViewer:
    """
    Interactive viewer for EEG spherical harmonics with time and rotation controls.
    """
    
    def __init__(self, result, initial_time=0):
        self.result = result
        self.current_time = initial_time
        self.sensor_elev = 20
        self.sensor_azim = 45
        self.field_elev = 20
        self.field_azim = 0
        
        # Create figure with subplots
        self.fig = plt.figure(figsize=(16, 8))
        
        # Sensor data subplot
        self.ax_sensor = self.fig.add_subplot(1, 2, 1, projection='3d')
        self.ax_sensor.set_title('Sensor Data')
        
        # Reconstructed field subplot
        self.ax_field = self.fig.add_subplot(1, 2, 2, projection='3d')
        self.ax_field.set_title('Reconstructed Field')
        
        # Create controls
        self._create_controls()
        
        # Initial plot
        self._update_plot()
        
    def _create_controls(self):
        """Create interactive controls for time and rotation."""
        # Adjust layout to make room for controls
        plt.subplots_adjust(bottom=0.25, left=0.1, right=0.9)
        
        # Time slider
        ax_time = plt.axes([0.1, 0.15, 0.3, 0.03])
        self.time_slider = Slider(
            ax_time, 'Time', 0, len(self.result['times'])-1,
            valinit=self.current_time, valstep=1
        )
        self.time_slider.on_changed(self._on_time_change)
        
        # Sensor rotation sliders
        ax_sensor_elev = plt.axes([0.1, 0.10, 0.3, 0.03])
        self.sensor_elev_slider = Slider(
            ax_sensor_elev, 'Sensor Elev', -90, 90,
            valinit=self.sensor_elev, valstep=5
        )
        self.sensor_elev_slider.on_changed(self._on_sensor_elev_change)
        
        ax_sensor_azim = plt.axes([0.1, 0.05, 0.3, 0.03])
        self.sensor_azim_slider = Slider(
            ax_sensor_azim, 'Sensor Azim', 0, 360,
            valinit=self.sensor_azim, valstep=5
        )
        self.sensor_azim_slider.on_changed(self._on_sensor_azim_change)
        
        # Field rotation sliders
        ax_field_elev = plt.axes([0.6, 0.10, 0.3, 0.03])
        self.field_elev_slider = Slider(
            ax_field_elev, 'Field Elev', -90, 90,
            valinit=self.field_elev, valstep=5
        )
        self.field_elev_slider.on_changed(self._on_field_elev_change)
        
        ax_field_azim = plt.axes([0.6, 0.05, 0.3, 0.03])
        self.field_azim_slider = Slider(
            ax_field_azim, 'Field Azim', 0, 360,
            valinit=self.field_azim, valstep=5
        )
        self.field_azim_slider.on_changed(self._on_field_azim_change)
        
        # Reset button
        ax_reset = plt.axes([0.8, 0.15, 0.1, 0.03])
        self.reset_button = Button(ax_reset, 'Reset Views')
        self.reset_button.on_clicked(self._on_reset)
        
    def _on_time_change(self, val):
        """Handle time slider change."""
        self.current_time = int(val)
        self._update_plot()
        
    def _on_sensor_elev_change(self, val):
        """Handle sensor elevation change."""
        self.sensor_elev = val
        self.ax_sensor.view_init(elev=self.sensor_elev, azim=self.sensor_azim)
        self.fig.canvas.draw_idle()
        
    def _on_sensor_azim_change(self, val):
        """Handle sensor azimuth change."""
        self.sensor_azim = val
        self.ax_sensor.view_init(elev=self.sensor_elev, azim=self.sensor_azim)
        self.fig.canvas.draw_idle()
        
    def _on_field_elev_change(self, val):
        """Handle field elevation change."""
        self.field_elev = val
        self.ax_field.view_init(elev=self.field_elev, azim=self.field_azim)
        self.fig.canvas.draw_idle()
        
    def _on_field_azim_change(self, val):
        """Handle field azimuth change."""
        self.field_azim = val
        self.ax_field.view_init(elev=self.field_elev, azim=self.field_azim)
        self.fig.canvas.draw_idle()
        
    def _on_reset(self, event):
        """Reset all views to default."""
        self.sensor_elev = 20
        self.sensor_azim = 45
        self.field_elev = 20
        self.field_azim = 0
        
        self.sensor_elev_slider.set_val(self.sensor_elev)
        self.sensor_azim_slider.set_val(self.sensor_azim)
        self.field_elev_slider.set_val(self.field_elev)
        self.field_azim_slider.set_val(self.field_azim)
        
    def _update_plot(self):
        """Update the plot with current time point."""
        # Clear previous plots
        self.ax_sensor.clear()
        self.ax_field.clear()
        
        # Get current data
        coeffs = self.result['coefficients'][self.current_time]
        positions = self.result['positions']
        ch_names = self.result['ch_names']
        theta = self.result['theta']
        phi = self.result['phi']
        times = self.result['times']
        
        # Normalize positions to unit sphere
        xyz = positions / np.linalg.norm(positions, axis=1, keepdims=True)
        
        # Plot sensor data
        sensor_vals = pyshtools.expand.MakeGridPoint(coeffs, theta, phi)
        sc = self.ax_sensor.scatter(xyz[:,0], xyz[:,1], xyz[:,2], 
                                   c=sensor_vals, cmap='RdBu_r', s=80)
        
        # Add electrode labels
        for i, name in enumerate(ch_names):
            self.ax_sensor.text(xyz[i,0]*1.1, xyz[i,1]*1.1, xyz[i,2]*1.1, 
                               name, fontsize=8)
        
        # Add wireframe sphere
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 15)
        sphere_x = np.outer(np.cos(u), np.sin(v))
        sphere_y = np.outer(np.sin(u), np.sin(v))
        sphere_z = np.outer(np.ones(np.size(u)), np.cos(v))
        self.ax_sensor.plot_wireframe(sphere_x, sphere_y, sphere_z, 
                                     color='gray', alpha=0.2, linewidth=0.7)
        
        # Set sensor view
        self.ax_sensor.view_init(elev=self.sensor_elev, azim=self.sensor_azim)
        self.ax_sensor.set_xlabel('X')
        self.ax_sensor.set_ylabel('Y')
        self.ax_sensor.set_zlabel('Z')
        self.ax_sensor.set_title(f'Sensor Data\nTime: {times[self.current_time]:.3f}s')
        
        # Add colorbar for sensor data
        plt.colorbar(sc, ax=self.ax_sensor, shrink=0.5)
        
        # Plot reconstructed field
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
        
        # Plot surface with colors
        surf = self.ax_field.plot_surface(X, Y, Z, 
                                         facecolors=plt.cm.RdBu_r((field-field.min())/(np.ptp(field) or 1)), 
                                         alpha=0.8, linewidth=0)
        
        # Add sensor positions as black dots
        self.ax_field.scatter(xyz[:,0], xyz[:,1], xyz[:,2], 
                             c='black', s=30, alpha=0.8, edgecolors='white', linewidth=0.5)
        
        # Set field view
        self.ax_field.view_init(elev=self.field_elev, azim=self.field_azim)
        self.ax_field.set_xlabel('X')
        self.ax_field.set_ylabel('Y')
        self.ax_field.set_zlabel('Z')
        self.ax_field.set_title('Reconstructed Field')
        
        # Add colorbar for field with proper RdBu_r colormap
        from matplotlib.cm import ScalarMappable
        sm = ScalarMappable(cmap=plt.cm.RdBu_r)
        sm.set_array(field)
        plt.colorbar(sm, ax=self.ax_field, shrink=0.5)
        
        plt.tight_layout()
        self.fig.canvas.draw_idle()
    
    def show(self):
        """Show the interactive plot."""
        plt.show()


def create_interactive_viewer(data_source, **kwargs):
    """
    Create an interactive viewer for EEG data.
    
    Parameters:
    -----------
    data_source : str or np.ndarray
        EEG data source (file path or numpy array)
    **kwargs : dict
        Additional arguments for SphericalHarmonicAnalyzer
        
    Returns:
    --------
    InteractiveViewer
        The interactive viewer object
    """
    analyzer = SphericalHarmonicAnalyzer(**kwargs)
    result = analyzer.analyze(data_source)
    return InteractiveViewer(result)


if __name__ == "__main__":
    # Example usage
    print("Creating interactive viewer...")
    
    # Test with the EDF file
    edf_path = 'path/to/your/eeg/file.edf'  # Replace with your file path
    
    try:
        viewer = create_interactive_viewer(edf_path, lmax=8)
        print("Interactive viewer created! Use the sliders to:")
        print("- Change time points")
        print("- Rotate sensor view (elevation and azimuth)")
        print("- Rotate field view (elevation and azimuth)")
        print("- Click 'Reset Views' to return to default orientation")
        viewer.show()
        
    except Exception as e:
        print(f"Error creating interactive viewer: {e}")
        import traceback
        traceback.print_exc() 