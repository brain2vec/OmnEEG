from mne.viz.topomap import _adjust_meg_sphere, _GridData
from mne.channels.layout import _find_topomap_coords

from mne.io.pick import pick_types
import numpy as np


class Interpolate(object):
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
        sphere, clip_origin = _adjust_meg_sphere(sphere=None, info=eeg.info, ch_type='eeg')
        x, y, _, radius = sphere
        picks = pick_types(eeg.info, meg=False, eeg=True, ref_meg=False, exclude='bads')
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