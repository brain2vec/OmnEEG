# OmnEEG

OmnEEG (pronounce OmnI-I-G) allows you to feed seamlessly multiple large and heterogeneous EEG datasets into your PyTorch models.

## Roadmap

### Data handling

[X] PyTorch dataset integration

[X] YAML config files (global + cohorts)

[X] HDF5 export

[X] Integrate as a Transform operator in the Dataset class (see [this tutorial](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#transforms))

### 2D Interpolation

[X] Topomap generation ([Bashivan et al. 2015](https://arxiv.org/abs/1511.06448); see [this class](https://github.com/mne-tools/mne-python/blob/0ec28e9ad8e234ea51266644ae1ac35a2bc11f46/mne/viz/top...)

### 3D Spherical harmonics

[X] Extract spherical coordinates of sensors (see [this class](https://mne.tools/dev/generated/mne.bem.fit_sphere_to_headshape.html), [these classes](https://github.com/mne-tools/mne-python/blob/3...)

### 3D Source reconstruction

[ ] Spherical model ([Yao 2001](https://mne.tools/1.1/auto_tutorials/preprocessing/55_setting_eeg_reference.html#using-an-infinite-reference-rest))

[ ] Surface template ([Groß et al. 2001](https://mne.tools/1.1/auto_examples/inverse/dics_source_power.html#compute-source-power-using-dics-beamformer))

[ ] Volumic template ([Gramfort et al. 2013](https://mne.tools/1.1/auto_examples/inverse/compute_mne_inverse_volume.html))

[ ] Individual anatomy morphed onto a template ([Avants et al. 2008](https://mne.tools/1.1/auto_examples/inverse/morph_volume_stc.html#sphx-glr-auto-examples-inverse-morph-volume-stc-py))

### Pure statistical representation

[ ] Check Riemanian geometry approaches ([Sabbagh et al. 2020](https://www.sciencedirect.com/science/article/pii/S1053811920303797))

[ ] Check T-PHATE method ([code](https://github.com/KrishnaswamyLab/TPHATE) and [paper](https://www.nature.com/articles/s43588-023-00419-0)) and beyond (e.g., [GSTH](https://github.com/Krishnaswam...)

### Visualization

[ ] Train a model for ploting different representations (e.g., a "cubic brain") of the data based on the latent space.


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