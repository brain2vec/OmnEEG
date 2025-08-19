from mne.viz.topomap import _adjust_meg_sphere, _GridData
from mne.channels.layout import _find_topomap_coords
import mne
import numpy as np
import pyshtools as pysh


class Interpolate(object):
    """Interpolate EEG to a matrix of pixels or spherical harmonics.
    
    Args:
        output_size (tuple or int): For 2D: desired output size. For 3D: ignored.
        transform_type (str): '2d' for topomap, '3d' for spherical harmonics
        l_max (int, optional): Maximum harmonic degree for 3D transform
    """

    def __init__(self, output_size, transform_type='2d', l_max=None):
        self.transform_type = transform_type
        self.l_max = l_max
        if transform_type == '2d':
            assert isinstance(output_size, (int, tuple))
            self.output_size = output_size
        elif transform_type == '3d':
            if l_max is None:
                raise ValueError("l_max must be specified for 3D transform")
        else:
            raise ValueError("transform_type must be '2d' or '3d'")

    def __call__(self, eeg):
        if self.transform_type == '2d':
            return self._interpolate_2d(eeg)
        elif self.transform_type == '3d':
            return self._interpolate_3d(eeg)

    def _interpolate_2d(self, eeg):
        sphere, clip_origin = _adjust_meg_sphere(sphere=None, info=eeg.info, ch_type='eeg')
        x, y, _, radius = sphere
        picks = mne.pick_types(eeg.info, meg=False, eeg=True, ref_meg=False, exclude='bads')
        pos = _find_topomap_coords(eeg.info, picks, sphere=sphere)
        mask_scale = max(1.0, np.linalg.norm(pos, axis=1).max() * 1.01 / radius)
        clip_radius = (radius * mask_scale,)*2
        res = self.output_size
        image_interp = 'cubic'
        xlim = np.inf, -np.inf,
        ylim = np.inf, -np.inf,
        xmin, xmax = clip_origin[0] - clip_radius[0], clip_origin[0] + clip_radius[0]
        ymin, ymax = clip_origin[1] - clip_radius[1], clip_origin[1] + clip_radius[1]
        xi = np.linspace(xmin, xmax, res[0])
        yi = np.linspace(ymin, ymax, res[1])
        Xi, Yi = np.meshgrid(xi, yi)
        extrapolate = 'box'
        border = 'mean'
        interp = _GridData(pos, image_interp, extrapolate,
                           clip_origin, clip_radius, border)
        extent = (xmin, xmax, ymin, ymax)
        data = eeg.get_data()
        Z=np.zeros((data.shape[0], res[0], res[1], data.shape[2]))
        for epoch in range(data.shape[0]):
            for time in range(data.shape[2]):
                interp.set_values(data[epoch, :, time])
                Zi = interp.set_locations(Xi, Yi)()
                Z[epoch, :, :, time] = Zi
        return Z
    
    def _interpolate_3d(self, eeg):
        """Transform EEG data to spherical harmonic coefficients."""
        picks = mne.pick_types(eeg.info, meg=False, eeg=True, ref_meg=False, exclude='bads')
        sphere, _ = _adjust_meg_sphere(sphere=None, info=eeg.info, ch_type='eeg')
        pos = _find_topomap_coords(eeg.info, picks, sphere=sphere)
        
        # Convert positions to spherical coordinates
        # pos is in Cartesian coordinates (x, y) on the projected sphere
        x, y = pos[:, 0], pos[:, 1]
        # Normalize positions to ensure they're on unit sphere
        pos_norm = np.sqrt(x**2 + y**2)
        if np.any(pos_norm > 1.0):
            # Scale positions to fit within unit circle
            scale_factor = 1.0 / np.max(pos_norm)
            x *= scale_factor
            y *= scale_factor
        z = np.sqrt(1 - x**2 - y**2)  # Project to unit sphere
        
        # Convert to spherical coordinates
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r)  # Colatitude (0 to pi)
        phi = np.arctan2(y, x)    # Azimuth (-pi to pi)
        
        # Convert phi to [0, 2*pi]
        phi = np.where(phi < 0, phi + 2*np.pi, phi)
        
        data = eeg.get_data()
        n_epochs, n_channels, n_times = data.shape
        
        # Calculate number of coefficients for given l_max
        n_coeffs = (self.l_max + 1) ** 2
        
        # Initialize coefficient array
        coeffs = np.zeros((n_epochs, n_coeffs, n_times))
        
        for epoch in range(n_epochs):
            for time in range(n_times):
                signal = data[epoch, :, time]
                
                # Convert to degrees 
                lat_deg = 90 - theta * 180 / np.pi  # Convert colatitude to latitude in degrees
                lon_deg = phi * 180 / np.pi  # Convert to degrees
                
                cilm, chi2 = pysh.expand.shlsq(signal, lat_deg, lon_deg, self.l_max)
                
                # Flatten coefficients to 1D array
                coeffs_flat = []
                for l in range(self.l_max + 1):
                    for m in range(-l, l + 1):
                        if m >= 0:
                            coeffs_flat.append(cilm[0, l, m])  # Cosine coefficients
                        else:
                            coeffs_flat.append(cilm[1, l, abs(m)])  # Sine coefficients
                
                coeffs[epoch, :, time] = coeffs_flat[:n_coeffs]
        
        return coeffs
