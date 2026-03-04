# OmnEEG

OmnEEG (pronounce OmnI-I-G) allows you to feed seamlessly multiple large and heterogeneous EEG datasets into your PyTorch models.

## Uniform output

All tokenization schemes produce the same output shape:

```
(n_epochs, resolution, n_times)
```

where `resolution` is the number of spatial features (must be a perfect square for 2D and 3D transforms). This makes downstream models agnostic to the tokenization scheme.

| Transform | `resolution=256` | Internal representation |
|---|---|---|
| 2D Topomap | 16x16 pixel grid, flattened | Interpolated scalp topography |
| 3D Spherical Harmonics | l_max=15, 256 coefficients | Frequency-domain on the sphere |
| Source | 256 Ward-clustered cortical ROIs | Parcellated inverse solution |
| Riemannian | Tangent space pooled to 256 | Log-Euclidean covariance features |
| T-PHATE | 256-dim temporal embedding | Diffusion-based manifold geometry |

## Installation

```bash
pip install -r requirements.txt
```

## Quick start

```python
from omneeg.io import EEG

# All five produce shape (n_epochs, 256, 128)
dataset_2d = EEG(cohort='cohort1', config_file='config_2d.yaml')
dataset_3d = EEG(cohort='cohort1', config_file='config_3d.yaml')
dataset_sr = EEG(cohort='cohort1', config_file='config_source.yaml')
dataset_ri = EEG(cohort='cohort1', config_file='config_riemann.yaml')
dataset_tp = EEG(cohort='cohort1', config_file='config_tphate.yaml')

sample = dataset_2d[0]
print(sample.shape)  # (10, 256, 128)
```

## Configuration

Global settings are in YAML config files. One per transform type:

**`config_2d.yaml`** — Topomap interpolation
```yaml
resolution: 256          # spatial features = grid_side^2 (256 = 16x16)
transform_type: "2d"
```

**`config_3d.yaml`** — Spherical harmonics
```yaml
resolution: 256          # spatial features = (l_max+1)^2 (256 = l_max 15)
transform_type: "3d"
```

**`config_source.yaml`** — Source reconstruction
```yaml
resolution: 256          # number of cortical ROIs (Ward clustering)
transform_type: "source"
# parc: "aparc"         # optional: use atlas instead ('aparc' or 'aparc.a2009s')
method: "dSPM"          # inverse method: 'MNE', 'dSPM', 'sLORETA', 'eLORETA'
snr: 3.0                # assumed SNR for regularization
```

**`config_riemann.yaml`** — Riemannian tangent space
```yaml
resolution: 256          # spatial features (pooled from tangent space)
transform_type: "riemann"
window: 32              # covariance window in samples
step: 1                 # stride in samples
```

**`config_tphate.yaml`** — T-PHATE temporal embedding
```yaml
resolution: 256          # embedding dimensions
transform_type: "tphate"
knn: 5                  # nearest neighbors for affinity graph
```

Cohort-specific settings (file patterns, montage, channel renaming) go in `data/<cohort>.yaml`.

## Transforms

### 2D Topomap

Interpolates EEG onto a flat grid using MNE's topomap machinery ([Bashivan et al. 2015](https://arxiv.org/abs/1511.06448)). The grid is `sqrt(resolution) x sqrt(resolution)` pixels, then flattened.

### 3D Spherical Harmonics

Projects EEG sensor data onto spherical harmonic basis functions via least-squares ([SHTOOLS](https://shtools.github.io/SHTOOLS/pyshexpandlsq.html)). The maximum degree `l_max = sqrt(resolution) - 1`.

### Source Reconstruction

Template-based source reconstruction using MNE's fsaverage ([Gramfort et al. 2013](https://mne.tools/stable/auto_tutorials/inverse/index.html)). Builds a forward model from the fsaverage BEM, estimates noise covariance from epochs, and applies an inverse operator (dSPM, sLORETA, eLORETA, or MNE). By default, the cortical surface is dynamically parcellated into exactly `resolution` ROIs using Ward hierarchical clustering with cortical adjacency constraints — no fixed atlas needed. Optionally, set `parc` to use a standard atlas (`aparc` for Desikan-Killiany 68 regions, `aparc.a2009s` for Destrieux 148 regions) with adaptive pooling. The forward model and parcellation are cached across calls.

### Riemannian Tangent Space

Sliding-window covariance matrices projected to the tangent space at the identity via the log-Euclidean framework ([Sabbagh et al. 2020](https://www.sciencedirect.com/science/article/pii/S1053811920303797)). The signal is reflect-padded so the time dimension is preserved. Tangent space features are adaptively pooled to match `resolution`.

### T-PHATE

Temporal PHATE ([Tong et al. 2022](https://www.nature.com/articles/s43588-023-00419-0)) learns a low-dimensional embedding of the temporal dynamics by building a time-point affinity graph and applying diffusion-based dimensionality reduction. Each epoch is independently embedded into `resolution` dimensions, preserving the temporal structure. The `knn` parameter controls the locality of the affinity graph.

## Citation

If you use this software, please cite:

```bibtex
@software{RamezanianPanahi_Dumas_OmnEEG_2025,
  author = {Ramezanian-Panahi, Mahta and Dumas, Guillaume},
  title = {OmnEEG: Simple EEG tokenizer with PyTorch datasets},
  year = {2025},
  publisher = {GitHub},
  version = {main},
  url = {https://github.com/brain2vec/OmnEEG},
  note = {last updated: 2025-08-23; accessed: 2025-10-27}
}
```
