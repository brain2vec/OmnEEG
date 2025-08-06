import numpy as np

# 1. Compute energy distribution of components

def compute_energy_distribution(coeffs, axis=1):
    """
    Compute the energy (sum of squares) of each component over time.
    coeffs: np.ndarray, shape (n_components, n_times) or (n_harmonics, n_times)
    axis: int, axis over which to sum (default: 1 for time)
    Returns: energy (np.ndarray), sorted_indices (np.ndarray)
    """
    energy = np.sum(coeffs**2, axis=axis)
    sorted_indices = np.argsort(energy)[::-1]
    return energy, sorted_indices

# 2. Cumulative energy analysis

def cumulative_energy(energy):
    """
    Compute cumulative energy fraction for sorted energy array.
    energy: np.ndarray, sorted in descending order
    Returns: cumulative_energy (np.ndarray)
    """
    sorted_energy = np.sort(energy)[::-1]
    cumulative = np.cumsum(sorted_energy) / np.sum(energy)
    return cumulative

# 3. Reconstruction error (MSE)

def reconstruction_error(original, reconstructed):
    """
    Compute mean squared error between original and reconstructed data.
    original, reconstructed: np.ndarray, same shape
    Returns: mse (float)
    """
    return np.mean((original - reconstructed)**2)

# 4. Reconstruct EEG from top-k harmonics

def reconstruct_from_harmonics(coeffs, Y, top_k):
    """
    Reconstruct EEG from top-k harmonics (by energy).
    coeffs: (n_harmonics, n_times)
    Y: Spherical harmonics basis matrix, shape (n_channels, n_harmonics)
    top_k: int, number of harmonics to keep
    Returns: reconstructed_data (n_channels, n_times)
    """
    # Compute energy and get indices of top-k harmonics
    energy, sorted_indices = compute_energy_distribution(coeffs, axis=1)
    idx = sorted_indices[:top_k]
    coeffs_subset = np.zeros_like(coeffs)
    coeffs_subset[idx, :] = coeffs[idx, :]
    reconstructed = Y @ coeffs_subset
    return reconstructed

# 5. (Optional) Reconstruct from top-k ICA components

def reconstruct_from_ica(mixing, sources, top_k):
    """
    Reconstruct data from top-k ICA components.
    mixing: (n_channels, n_components)
    sources: (n_components, n_times)
    top_k: int, number of components to keep
    Returns: reconstructed_data (n_channels, n_times)
    """
    idx = np.argsort(np.sum(sources**2, axis=1))[::-1][:top_k]
    mixing_subset = mixing[:, idx]
    sources_subset = sources[idx, :]
    reconstructed = mixing_subset @ sources_subset
    return reconstructed 