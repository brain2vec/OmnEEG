import pandas as pd
import numpy as np
import pyshtools
from scipy.spatial import distance

sensors = pd.read_csv('EEG1005.tsv', sep='\t', header=0)
sensors.name = sensors.name.str.lower()
electrodes = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2']

def get3d(e):
    return sensors.query(f"name == '{e}'").loc[:, ['x', 'y', 'z']].values.flatten()

values = np.random.randn(len(electrodes))
pos = np.vstack([get3d(e.lower()) for e in electrodes])

# Convert 3D Cartesian coordinates to spherical coordinates (r, theta, phi)
def cart2sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)  # Colatitude (0 to pi)
    phi = np.arctan2(y, x)    # Longitude (-pi to pi)
    return r, theta, phi

# Extract spherical coordinates from sensor positions
r_vals = []
theta_vals = []
phi_vals = []
for i in range(pos.shape[0]):
    r, theta, phi = cart2sph(pos[i, 0], pos[i, 1], pos[i, 2])
    r_vals.append(r)
    theta_vals.append(theta)
    phi_vals.append(phi)

# Convert to numpy arrays
r_vals = np.array(r_vals)
theta_vals = np.array(theta_vals)
phi_vals = np.array(phi_vals)

# Normalize radius to unit sphere (if needed)
r_mean = np.mean(r_vals)
pos_normalized = pos / r_mean
for i in range(pos_normalized.shape[0]):
    r_vals[i], theta_vals[i], phi_vals[i] = cart2sph(
        pos_normalized[i, 0], pos_normalized[i, 1], pos_normalized[i, 2])

# Define parameters for spherical harmonic expansion
lmax = 256  # Maximum spherical harmonic degree

# Ensure theta, phi are in degrees
theta_deg = np.degrees(theta_vals)
phi_deg = np.degrees(phi_vals)

# Perform spherical harmonic expansion using least squares
coeffs = pyshtools.expand.SHExpandLSQ(values, theta_deg, phi_deg, lmax)