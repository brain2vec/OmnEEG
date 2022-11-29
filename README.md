# OmnEEG

OmnEEG (pronounce OmnI-I-G) allows you to feed seamlessly multiple large and heterogeneous EEG datasets into your PyTorch models.

## Roadmap

### Data handling

[X] PyTorch dataset integration

[X] YAML config files (global + cohorts)

[ ] HDF5 export

[ ] Integrate as a Transform operator in the Dataset class (see [this tutorial](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#transforms))

### 2D Interpolation

[ ] Topomap generation ([Bashivan et al. 2015](https://arxiv.org/abs/1511.06448); see [this class](https://github.com/mne-tools/mne-python/blob/0ec28e9ad8e234ea51266644ae1ac35a2bc11f46/mne/viz/topomap.py#L629))

### 3D Spherical harmonics

[ ] Extract spherical coordinates of sensors (see [this class](https://mne.tools/dev/generated/mne.bem.fit_sphere_to_headshape.html), [these classes](https://github.com/mne-tools/mne-python/blob/35e466f3fbb71cc7b976ae1a88b97409adabf694/mne/transforms.py#L1001), and [that library](https://shtools.github.io/SHTOOLS/pyshexpandlsq.html))

### 3D Source reconstruction

[ ] Spherical model ([Yao 2001](https://mne.tools/1.1/auto_tutorials/preprocessing/55_setting_eeg_reference.html#using-an-infinite-reference-rest))

[ ] Surface template ([Groß et al. 2001](https://mne.tools/1.1/auto_examples/inverse/dics_source_power.html#compute-source-power-using-dics-beamformer))

[ ] Volumic template ([Gramfort et al. 2013](https://mne.tools/1.1/auto_examples/inverse/compute_mne_inverse_volume.html))

[ ] Individual anatomy morphed onto a template ([Avants et al. 2008](https://mne.tools/1.1/auto_examples/inverse/morph_volume_stc.html#sphx-glr-auto-examples-inverse-morph-volume-stc-py))

### Pure statistical representation

[ ] Check Riemanian geometry approaches ([Sabbagh et al. 2020](https://denis-engemann.de/publication/sabbagh_generative_2019/))

### Visualization

[ ] Train a model for ploting different representations (e.g., a "cubic brain") of the data based on the latent space.
