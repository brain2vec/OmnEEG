#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : io.py
# description     : PyTorch loader for EEG datasets
# author          : Guillaume Dumas
# date            : 2022-11-29
# version         : 1
# usage           : from omneeg.io import EEG
#                   dataset = EEG(cohort='XXX')
# notes           : you need to populate the data folder with YAML files
# python_version  : 3.9
# ==============================================================================

from torch.utils.data import Dataset
import os
from glob import glob
import yaml
import mne
import h5py


class EEG(Dataset):
    def __init__(self, cohort) -> None:
        self.cohort = cohort
        with open('config.yaml') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
            self.sfreq = cfg['sfreq']
            self.duration = cfg['duration']
            self.epochs = cfg['epochs']
            self.data = cfg['data']
            self.overwrite = cfg['overwrite']
            self.info = None
        with open(os.path.join(self.data, f'{cohort}.yaml')) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            self.regexp = data['regexp']
            self.create_epochs = data['create_epochs']
            self.rename_channels = data['rename_channels']
            self.set_montage = data['set_montage']
        self.filenames = sorted(glob(self.regexp))

    def __getitem__(self, i):
        filename = self.filenames[i]
        output = os.path.join(self.data, self.cohort,
            filename[self.regexp.find('*'):].replace('/', '_')[:-4]+'.h5')
        if not os.path.exists(output) or self.overwrite:
            if filename[-3:] == 'fif':
                eeg = mne.read_epochs(filename, preload=True)
            elif filename[-3:] == 'mff':
                eeg = mne.io.read_raw_egi(filename, preload=True)
            elif filename[-3:] == 'edf':
                eeg = mne.io.read_raw_edf(filename, preload=True)
            else:
                print('File format not supported')
            if self.rename_channels:
                mne.rename_channels(eeg.info, self.rename_channels)
            if self.set_montage:
                montage = mne.channels.make_standard_montage(self.set_montage)
                eeg.set_montage(montage)
            eeg.resample(sfreq=self.sfreq)
            if self.create_epochs:
                eeg = mne.make_fixed_length_epochs(
                    eeg, duration=self.duration, preload=True)
            eeg.pick_types(meg=False, eeg=True, eog=False)
            if not os.path.exists(os.path.dirname(output)):
                os.makedirs(os.path.dirname(output))
            out = eeg[:self.epochs].get_data()
            with h5py.File(output, "w") as f:
                f.create_dataset("data",
                                 data=out,
                                 compression="gzip",
                                 compression_opts=9)
        else:
            with h5py.File(output, "r") as f:
                out = f['data'][:]
        return out

    def __len__(self):
        return len(self.filenames)
