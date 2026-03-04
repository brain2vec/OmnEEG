#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : transform.py
# description     : EEG transforms with uniform output shape (epochs, resolution, times)
# author          : Mahta Ramezanian Panahi & Guillaume Dumas
# date            : 2025-08-20
# version         : 3
# python_version  : 3.12
# ==============================================================================

import math
import numpy as np
import mne
import pyshtools as pysh
from mne.viz.topomap import _adjust_meg_sphere, _GridData
from mne.channels.layout import _find_topomap_coords


class Transform(object):
    """Universal EEG transform with uniform output shape.

    All transforms produce output of shape (n_epochs, resolution, n_times)
    where `resolution` is the number of spatial features.

    Args:
        resolution (int): Number of spatial features in the output.
            Must be a perfect square for '2d' and '3d' transforms
            (e.g. 64 gives 8x8 grid for 2D, l_max=7 for 3D).
        transform_type (str): '2d' for topomap, '3d' for spherical harmonics,
                              'riemann' for Riemannian tangent space,
                              'source' for source-space parcellation,
                              'tphate' for T-PHATE temporal embedding.
        window (int): Window size in samples for Riemannian transform.
        step (int): Step size in samples for Riemannian transform (default: 1).
        parc (str): Parcellation atlas for source transform (default: None = Ward).
        method (str): Inverse method for source transform (default: 'dSPM').
        snr (float): Assumed SNR for inverse regularization (default: 3.0).
        knn (int): Number of nearest neighbors for T-PHATE (default: 5).
    """

    def __init__(self, resolution, transform_type='2d', window=None, step=None,
                 parc=None, method=None, snr=None, knn=None):
        self.resolution = resolution
        self.transform_type = transform_type
        self.window = window
        self.step = step if step is not None else 1

        if transform_type == '2d':
            grid_side = int(math.isqrt(resolution))
            if grid_side * grid_side != resolution:
                raise ValueError(
                    f"For 2D transform, resolution must be a perfect square, got {resolution}")
            self.grid_side = grid_side
        elif transform_type == '3d':
            l_max_plus_1 = int(math.isqrt(resolution))
            if l_max_plus_1 * l_max_plus_1 != resolution:
                raise ValueError(
                    f"For 3D transform, resolution must be a perfect square, got {resolution}")
            self.l_max = l_max_plus_1 - 1
        elif transform_type == 'riemann':
            if window is None:
                raise ValueError("window parameter is required for Riemannian transform")
        elif transform_type == 'source':
            self.parc = parc  # None = dynamic Ward clustering, or atlas name
            self.method = method if method is not None else 'dSPM'
            self.snr = snr if snr is not None else 3.0
        elif transform_type == 'tphate':
            self.knn = knn if knn is not None else 5
        else:
            raise ValueError(
                "transform_type must be '2d', '3d', 'riemann', 'source', or 'tphate'")

    def __call__(self, eeg):
        if self.transform_type == '2d':
            return self._interpolate_2d(eeg)
        elif self.transform_type == '3d':
            return self._interpolate_3d(eeg)
        elif self.transform_type == 'riemann':
            return self._riemann(eeg)
        elif self.transform_type == 'source':
            return self._source(eeg)
        elif self.transform_type == 'tphate':
            return self._tphate(eeg)

    def _interpolate_2d(self, eeg):
        """Transform EEG to flattened topomap features.

        Output shape: (n_epochs, resolution, n_times)
        where resolution = grid_side * grid_side.
        """
        sphere, clip_origin = _adjust_meg_sphere(sphere=None, info=eeg.info, ch_type='eeg')
        x, y, _, radius = sphere
        picks = mne.pick_types(eeg.info, meg=False, eeg=True, ref_meg=False, exclude='bads')
        pos = _find_topomap_coords(eeg.info, picks, sphere=sphere)
        mask_scale = max(1.0, np.linalg.norm(pos, axis=1).max() * 1.01 / radius)
        clip_radius = (radius * mask_scale,) * 2
        res = (self.grid_side, self.grid_side)
        image_interp = 'cubic'
        xmin = clip_origin[0] - clip_radius[0]
        xmax = clip_origin[0] + clip_radius[0]
        ymin = clip_origin[1] - clip_radius[1]
        ymax = clip_origin[1] + clip_radius[1]
        xi = np.linspace(xmin, xmax, res[0])
        yi = np.linspace(ymin, ymax, res[1])
        Xi, Yi = np.meshgrid(xi, yi)
        extrapolate = 'box'
        border = 'mean'
        interp = _GridData(pos, image_interp, extrapolate,
                           clip_origin, clip_radius, border)
        data = eeg.get_data()
        n_epochs, n_channels, n_times = data.shape
        Z = np.zeros((n_epochs, self.resolution, n_times))
        for epoch in range(n_epochs):
            for time in range(n_times):
                interp.set_values(data[epoch, :, time])
                Zi = interp.set_locations(Xi, Yi)()
                Z[epoch, :, time] = Zi.ravel()
        return Z

    def _interpolate_3d(self, eeg):
        """Transform EEG to spherical harmonic coefficients.

        Output shape: (n_epochs, resolution, n_times)
        where resolution = (l_max + 1)^2.
        """
        picks = mne.pick_types(eeg.info, meg=False, eeg=True, ref_meg=False, exclude='bads')
        sphere, _ = _adjust_meg_sphere(sphere=None, info=eeg.info, ch_type='eeg')
        pos = _find_topomap_coords(eeg.info, picks, sphere=sphere)

        x, y = pos[:, 0], pos[:, 1]
        pos_norm = np.sqrt(x**2 + y**2)
        if np.any(pos_norm > 1.0):
            scale_factor = 1.0 / np.max(pos_norm)
            x = x * scale_factor
            y = y * scale_factor
        z = np.sqrt(np.maximum(1 - x**2 - y**2, 0))

        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        phi = np.where(phi < 0, phi + 2 * np.pi, phi)

        lat_deg = 90 - theta * 180 / np.pi
        lon_deg = phi * 180 / np.pi

        data = eeg.get_data()
        n_epochs, n_channels, n_times = data.shape

        coeffs = np.zeros((n_epochs, self.resolution, n_times))

        for epoch in range(n_epochs):
            for time in range(n_times):
                signal = data[epoch, :, time]
                cilm, chi2 = pysh.expand.shlsq(signal, lat_deg, lon_deg, self.l_max)

                coeffs_flat = []
                for l in range(self.l_max + 1):
                    for m in range(-l, l + 1):
                        if m >= 0:
                            coeffs_flat.append(cilm[0, l, m])
                        else:
                            coeffs_flat.append(cilm[1, l, abs(m)])

                coeffs[epoch, :, time] = coeffs_flat[:self.resolution]

        return coeffs

    def _riemann(self, eeg):
        """Transform EEG to Riemannian tangent space features.

        Uses sliding-window covariance matrices projected to tangent space
        via the log-Euclidean framework (matrix logarithm + vectorization).

        The signal is reflect-padded by window//2 on each side so that
        n_windows = n_times, preserving the original time dimension.

        Output shape: (n_epochs, resolution, n_times)
        """
        data = eeg.get_data()
        n_epochs, n_channels, n_times = data.shape

        window = self.window
        step = self.step

        # Reflect-pad so that n_windows = n_times (with step=1)
        pad_left = window // 2
        pad_right = window - 1 - pad_left
        data = np.pad(data, ((0, 0), (0, 0), (pad_left, pad_right)), mode='reflect')
        n_times_padded = data.shape[2]
        n_windows = (n_times_padded - window) // step + 1

        n_tangent = n_channels * (n_channels + 1) // 2

        # Precompute upper triangle indices
        tri_idx = np.triu_indices(n_channels)
        diag_mask = tri_idx[0] == tri_idx[1]

        result = np.zeros((n_epochs, n_tangent, n_windows))

        for epoch in range(n_epochs):
            for w in range(n_windows):
                start = w * step
                end = start + window
                segment = data[epoch, :, start:end]

                # Covariance with Ledoit-Wolf-style regularization
                cov = np.cov(segment)
                cov += 1e-6 * np.eye(n_channels)

                # Log-Euclidean: matrix logarithm via eigendecomposition
                eigvals, eigvecs = np.linalg.eigh(cov)
                eigvals = np.maximum(eigvals, 1e-10)
                log_cov = eigvecs @ np.diag(np.log(eigvals)) @ eigvecs.T

                # Vectorize upper triangle with sqrt(2) scaling on off-diagonal
                features = log_cov[tri_idx]
                features[~diag_mask] *= np.sqrt(2)

                result[epoch, :, w] = features

        # Adaptive pool from n_tangent to target resolution
        return self._adaptive_pool(result, self.resolution)

    def _source(self, eeg):
        """Transform EEG to source-space parcellated time courses.

        Uses MNE's template-based source reconstruction with fsaverage:
        1. Build forward model from fsaverage template (cached after first call)
        2. Parcellate source space into exactly `resolution` ROIs using
           Ward clustering on cortical adjacency (or atlas if `parc` is set)
        3. Estimate noise covariance from epochs
        4. Compute inverse operator and apply to epochs
        5. Average source activity within each ROI

        Output shape: (n_epochs, resolution, n_times)
        """
        import os

        # Fetch fsaverage template (MNE caches the download)
        fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
        subjects_dir = os.path.dirname(fs_dir)

        # Cache forward model and parcellation (same for all subjects)
        if not hasattr(self, '_fwd_cache'):
            src = mne.setup_source_space(
                'fsaverage', spacing='oct6',
                subjects_dir=subjects_dir, add_dist=False)
            bem_fname = os.path.join(
                fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
            bem = mne.read_bem_solution(bem_fname)
            self._fwd_cache = mne.make_forward_solution(
                eeg.info, trans='fsaverage', src=src,
                bem=bem, eeg=True, mindist=5.0)
            self._src_cache = src
            self._subjects_dir = subjects_dir

            if self.parc is None:
                # Dynamic Ward clustering on cortical surface
                from sklearn.cluster import AgglomerativeClustering
                adjacency = mne.spatial_src_adjacency(src)
                coords = np.vstack([s['rr'][s['vertno']] for s in src])
                clustering = AgglomerativeClustering(
                    n_clusters=self.resolution,
                    connectivity=adjacency,
                    linkage='ward')
                self._cluster_labels = clustering.fit_predict(coords)
            else:
                # Atlas-based parcellation
                labels = mne.read_labels_from_annot(
                    'fsaverage', parc=self.parc, subjects_dir=subjects_dir)
                self._atlas_labels = [
                    lb for lb in labels if 'unknown' not in lb.name.lower()]

        fwd = self._fwd_cache
        src = self._src_cache

        # Noise covariance from data (per subject)
        cov = mne.compute_covariance(eeg, method='empirical')

        # Inverse operator
        inv = mne.minimum_norm.make_inverse_operator(eeg.info, fwd, cov)

        # Apply inverse to all epochs
        lambda2 = 1.0 / self.snr ** 2
        stcs = mne.minimum_norm.apply_inverse_epochs(
            eeg, inv, lambda2, method=self.method, verbose=False)

        if self.parc is None:
            # Average source vertices within each Ward cluster
            n_times = stcs[0].data.shape[1]
            result = np.zeros((len(stcs), self.resolution, n_times))
            for i, stc in enumerate(stcs):
                for k in range(self.resolution):
                    mask = self._cluster_labels == k
                    result[i, k, :] = stc.data[mask].mean(axis=0)
            return result
        else:
            # Atlas parcellation + adaptive pool
            label_ts = mne.extract_label_time_course(
                stcs, self._atlas_labels, src, mode='mean_flip')
            result = np.array(label_ts)
            return self._adaptive_pool(result, self.resolution)

    def _tphate(self, eeg):
        """Transform EEG to T-PHATE temporal embedding.

        T-PHATE learns a low-dimensional representation of the temporal
        dynamics by building a time-point affinity graph and applying
        diffusion-based dimensionality reduction.

        Output shape: (n_epochs, resolution, n_times)
        where resolution = n_components of the T-PHATE embedding.
        """
        import tphate

        data = eeg.get_data()
        n_epochs, n_channels, n_times = data.shape

        result = np.zeros((n_epochs, self.resolution, n_times))

        for epoch in range(n_epochs):
            # T-PHATE expects (n_times, n_features)
            op = tphate.TPHATE(n_components=self.resolution, knn=self.knn)
            emb = op.fit_transform(data[epoch].T)  # (n_times, resolution)
            result[epoch] = emb.T  # (resolution, n_times)

        return result

    @staticmethod
    def _adaptive_pool(data, target_features):
        """Pool spatial dimension to target_features via bin averaging.

        Args:
            data: array of shape (n_epochs, n_features, n_times)
            target_features: desired number of spatial features

        Returns:
            array of shape (n_epochs, target_features, n_times)
        """
        n_spatial = data.shape[1]
        if n_spatial == target_features:
            return data
        elif n_spatial > target_features:
            indices = np.array_split(np.arange(n_spatial), target_features)
            return np.stack([data[:, idx].mean(axis=1) for idx in indices], axis=1)
        else:
            pad_width = ((0, 0), (0, target_features - n_spatial), (0, 0))
            return np.pad(data, pad_width, mode='constant', constant_values=0)


# Backward compatibility alias
Interpolate = Transform
