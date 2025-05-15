import mne
from mne.channels.layout import _find_topomap_coords
from mne.viz.topomap import _adjust_meg_sphere, _GridData
import numpy as np


class Interpolate(object):
# TODO - Move on to Python3 and get rid of object inheritence
    """Interpolate a EEG to a matrix of pixels.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, eeg):
        # Check if eeg is an MNE epochs object or array-like data
        if hasattr(eeg, 'info'):
            # It's an MNE object with info attribute
            info = eeg.info
            data = eeg.get_data()
            
            # For Raw objects, reshape to epochs format (trials, channels, times)
            if isinstance(eeg, mne.io.Raw):
                data = data.reshape(1, data.shape[0], data.shape[1])  # add trial dimension
        else:
            # Assume it's a slice of epochs from an MNE object
            # This is a workaround for the error
            raise TypeError("Expected an MNE object with 'info' attribute. Please modify the io.py code.")
            
        try:
            # Try to use the standard approach first
            sphere, clip_origin = _adjust_meg_sphere(sphere=None, info=info, ch_type='eeg')
            x, y, _, radius = sphere
            # Use the public API function directly from mne
            picks = mne.pick_types(info, meg=False, eeg=True, ref_meg=False, exclude='bads')
            pos = _find_topomap_coords(info, picks, sphere=sphere)
        except RuntimeError as e:
            if "No digitization points found" in str(e):
                # Fallback: create artificial positions in a circle formation
                print("No electrode positions found, creating artificial positions")
                
                # Get number of EEG channels
                n_channels = len(picks)
                
                # Create artificial positions in a circular pattern
                radius = 0.85  # slightly smaller than unit circle for better visualization
                angles = np.linspace(0, 2*np.pi, n_channels, endpoint=False)
                x_pos = radius * np.cos(angles)
                y_pos = radius * np.sin(angles)
                
                # Create positions for all channels
                pos = np.column_stack([x_pos, y_pos])
                
                # Set default sphere and clip_origin for visualization
                sphere = (0, 0, 0, 1)  # x, y, z, radius
                clip_origin = (0, 0)
            else:
                # Re-raise other errors
                raise
            
        # Check the dimensions of data
        if not isinstance(data, np.ndarray):
            raise TypeError("Data must be a numpy array")
            
        # Ensure data has the right dimensions (epochs, channels, times)
        if len(data.shape) == 2:  # (channels, times)
            data = data.reshape(1, data.shape[0], data.shape[1])  # add epochs dimension
        elif len(data.shape) != 3:
            raise ValueError(f"Expected 2D or 3D data array, got shape {data.shape}")
            
        mask_scale = max(1.0, np.linalg.norm(pos, axis=1).max() * 1.01 / radius)
        clip_radius = (radius * mask_scale,)*2
        res = self.output_size
        image_interp = 'cubic'
        xmin, xmax = clip_origin[0] - clip_radius[0], clip_origin[0] + clip_radius[0]
        ymin, ymax = clip_origin[1] - clip_radius[1], clip_origin[1] + clip_radius[1]
        xi = np.linspace(xmin, xmax, res[0])
        yi = np.linspace(ymin, ymax, res[1])
        Xi, Yi = np.meshgrid(xi, yi)
        extrapolate = 'box'
        border = 'mean'
        interp = _GridData(pos, image_interp, extrapolate,
                          clip_origin, clip_radius, border)
                          
        print(f"Data shape: {data.shape}, Output dimensions: {res}")
        
        # Now create the output array with proper dimensions
        Z = np.zeros((data.shape[0], res[0], res[1], data.shape[2]))
        
        for epoch in range(data.shape[0]):
            for time in range(data.shape[2]):
                interp.set_values(data[epoch, :, time])
                Zi = interp.set_locations(Xi, Yi)()
                Z[epoch, :, :, time] = Zi
        return Z
