# OmnEEG

OmnEEG (pronounce OmnI-I-G) allows you to feed seamlessly multiple large and heterogeneous EEG datasets into your PyTorch models.

## Uniform output

All tokenization schemes produce the same output shape:

```
(n_epochs, resolution, n_times)
```

where `resolution` is the number of spatial features (must be a perfect square for 2D and 3D transforms). This makes downstream models agnostic to the tokenization scheme.

| Transform | `resolution=64` | Internal representation |
|---|---|---|
| 2D Topomap | 8x8 pixel grid, flattened | Interpolated scalp topography |
| 3D Spherical Harmonics | l_max=7, 64 coefficients | Frequency-domain on the sphere |
| Riemannian | Tangent space pooled to 64 | Log-Euclidean covariance features |

## Installation

```bash
pip install -r requirements.txt
```

## Quick start

```python
from omneeg.io import EEG

# All three produce shape (n_epochs, 64, 128)
dataset_2d = EEG(cohort='cohort1', config_file='config_2d.yaml')
dataset_3d = EEG(cohort='cohort1', config_file='config_3d.yaml')
dataset_ri = EEG(cohort='cohort1', config_file='config_riemann.yaml')

sample = dataset_2d[0]
print(sample.shape)  # (10, 64, 128)
```

## Configuration

Global settings are in YAML config files. One per transform type:

**`config_2d.yaml`** — Topomap interpolation
```yaml
resolution: 64          # spatial features = grid_side^2 (64 = 8x8)
transform_type: "2d"
```

**`config_3d.yaml`** — Spherical harmonics
```yaml
resolution: 64          # spatial features = (l_max+1)^2 (64 = l_max 7)
transform_type: "3d"
```

**`config_riemann.yaml`** — Riemannian tangent space
```yaml
resolution: 64          # spatial features (pooled from tangent space)
transform_type: "riemann"
window: 32              # covariance window in samples
step: 1                 # stride in samples
```

Cohort-specific settings (file patterns, montage, channel renaming) go in `data/<cohort>.yaml`.

## Transforms

### 2D Topomap

Interpolates EEG onto a flat grid using MNE's topomap machinery ([Bashivan et al. 2015](https://arxiv.org/abs/1511.06448)). The grid is `sqrt(resolution) x sqrt(resolution)` pixels, then flattened.

### 3D Spherical Harmonics

Projects EEG sensor data onto spherical harmonic basis functions via least-squares ([SHTOOLS](https://shtools.github.io/SHTOOLS/pyshexpandlsq.html)). The maximum degree `l_max = sqrt(resolution) - 1`.

### Riemannian Tangent Space

Sliding-window covariance matrices projected to the tangent space at the identity via the log-Euclidean framework ([Sabbagh et al. 2020](https://www.sciencedirect.com/science/article/pii/S1053811920303797)). The signal is reflect-padded so the time dimension is preserved. Tangent space features are adaptively pooled to match `resolution`.

## Roadmap

### Data handling

- [X] PyTorch dataset integration
- [X] YAML config files (global + cohorts)
- [X] HDF5 export
- [X] Integrate as a Transform operator in the Dataset class (see [this tutorial](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#transforms))

### Tokenization schemes

- [X] 2D Topomap generation ([Bashivan et al. 2015](https://arxiv.org/abs/1511.06448))
- [X] 3D Spherical harmonics ([SHTOOLS](https://shtools.github.io/SHTOOLS/pyshexpandlsq.html))
- [X] Riemannian tangent space ([Sabbagh et al. 2020](https://www.sciencedirect.com/science/article/pii/S1053811920303797))

### 3D Source reconstruction

- [ ] Spherical model ([Yao 2001](https://mne.tools/1.1/auto_tutorials/preprocessing/55_setting_eeg_reference.html#using-an-infinite-reference-rest))
- [ ] Surface template ([Gross et al. 2001](https://mne.tools/1.1/auto_examples/inverse/dics_source_power.html#compute-source-power-using-dics-beamformer))
- [ ] Volumic template ([Gramfort et al. 2013](https://mne.tools/1.1/auto_examples/inverse/compute_mne_inverse_volume.html))
- [ ] Individual anatomy morphed onto a template ([Avants et al. 2008](https://mne.tools/1.1/auto_examples/inverse/morph_volume_stc.html#sphx-glr-auto-examples-inverse-morph-volume-stc-py))

### Pure statistical representation

- [ ] T-PHATE method ([code](https://github.com/KrishnaswamyLab/TPHATE) and [paper](https://www.nature.com/articles/s43588-023-00419-0)) and beyond (e.g., [GSTH](https://github.com/KrishnaswamyLab/GSTH))

### Visualization

- [ ] Train a model for plotting different representations (e.g., a "cubic brain") of the data based on the latent space.


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
