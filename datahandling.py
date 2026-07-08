import torch
from torch import nn, optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from torch.distributions.normal import Normal
import math
import pathlib as path
import h5py
from torch.utils.data import DataLoader

import logging

logger = logging.getLogger(__name__)


class H5SpecDataset(torch.utils.data.Dataset):
    
    def __init__(self, data_path, split, flux_type="normalized_flux_cont", standardize = True,preload="none", device = "cpu"):
        self.data_path = data_path
        self.split = split
        self.flux_type = flux_type
        self.standardize = standardize
        self.preload = preload
        self.device = device

        if self.flux_type == "normalized_flux_cont":
            mean_key = "norm_mean_cont"
            std_key = "norm_std_cont"
        elif self.flux_type == "raw_flux":
            mean_key = "raw_mean"
            std_key = "raw_std"
        elif self.flux_type == "normalized_flux_med":
            mean_key = "norm_mean_med"
            std_key = "norm_std_med"
        elif self.flux_type == "log_scale_flux":
            mean_key = "norm_mean_log"
            std_key = "norm_std_log"
        else:
            logger.info("WARNING: INVALID flux type, defaulting to normalized_flux_cont")
            self.flux_type = "normalized_flux_cont"
            mean_key = "norm_mean_cont"
            std_key = "norm_std_cont"
        
        self.hf = None
        self.data = None
        self.mask = None

        self.redshifts = None
        self.snr = None

        # dont need to initialise as None, because code is unconditional
        with h5py.File(self.data_path, "r") as hf:
            self.l = hf.attrs["wavelengths"][:]
            self.mean = float(hf.attrs[mean_key])
            self.std = float(hf.attrs[std_key])
            self.len = hf[self.split][self.flux_type].shape[0]
            self.n_pixels = hf[self.split][self.flux_type].shape[1]

        # if preload is set, load the data into RAM or VRAM
        if self.preload in ("ram", "vram"):
            with h5py.File(self.data_path, "r") as hf:
                # dump everything in PAGEABLE ram
                arr = hf[self.split][self.flux_type][:]
            # *contiguous* in memory (easier for ram to handle)
            # data is in pageable, not pinned, RAM
            data = torch.from_numpy(np.ascontiguousarray(arr)).float()
            if self.standardize:
                data = (data - self.mean) / self.std
            # store mask so can switch back nans to 0 before get item (ram, vram only)
            self.mask = ~torch.isnan(data)
            # switch nans back to 0
            data = torch.nan_to_num(data, nan = 0.0)
            # note to me: it is sort of weird to preload to VRAM, we would usually assume
            # the data wouldnt fit in VRAM, but mine is small enough so it does and we can
            # do this
            if self.preload == "vram":
                assert self.device != "cpu", "Cannot preload to VRAM if device is CPU"
                # move to VRAM
                data = data.to(self.device) # non_blocking is false because data moves in one
                self.mask = self.mask.to(self.device)
            self.data = data

    def __len__(self):

        return self.len

    def __getitem__(self, idx):
        
        if self.preload in ("ram", "vram"):
            # get sample, mask from RAM/VRAM
            sample = self.data[idx]
            mask =  self.mask[idx]
            return sample, mask
        else:
            # open h5 and retrieve sample, mask
            # (lazy loading. only open h5 file when start accessing it
            # only needed for streaming, not preloading)
            if self.hf is None:
                self.hf = h5py.File(self.data_path, "r")
            sample = self.hf[self.split][self.flux_type][idx]
            if self.standardize:
                sample = (sample - self.mean)/ self.std
            sample = torch.from_numpy(sample)
            mask = ~torch.isnan(sample)
            sample = torch.nan_to_num(sample, nan = 0.0)
            # make sure sample is float32 (best for Pytorch, also I think what is in the updated h5)
            sample = sample.float()

            return sample, mask

    def _get_redshift(self):
        # CAUTION: only use with a non-shuffled loader
        if self.redshifts is None:
            if self.hf is None:
                self.hf = h5py.File(self.data_path, "r")
            try:
                self.redshifts = np.array(self.hf[self.split]["redshift"])
            except KeyError:
                return None
        return self.redshifts

    def _get_snr(self):
        # CAUTION: only use with a non-shuffled loader
        if self.snr is None:
            if self.hf is None:
                self.hf = h5py.File(self.data_path, "r")
            try:
                self.snr = np.array(self.hf[self.split]["SNR"])
            except KeyError:
                return None
        return self.snr


# # new function to make dataloader to ensure that correct loader settings are used
# # for the correct loading of the dataset (streaming vs preloading, ram vs vram etc)
def make_dataloader(dataset, batch_size = 32, num_workers = None, prefetch_factor = None):#

    # get if preloading or streaming
    preload = dataset.preload
    # if no preloading, need pinning so that can use non_blocking = True when moving
    # batch to GPU
    if preload == "none":
        pin_memory = True
        workers = num_workers if num_workers is not None else 4
        persistent_workers = workers > 0
        prefetch_factor = prefetch_factor if prefetch_factor is not None else 4
    # if ram preloading, needed so that can use non_blocking = True when moving
    # batch to GPU, but no workers since not taking samples from disk.
    elif preload == "ram":
        pin_memory = True
        workers = 0
        persistent_workers = False
        prefetch_factor = None # not relevant for preload
    else: # if vram
        pin_memory = False # default, but to be safe
        workers = 0 # default, but to be safe
        persistent_workers = False # default, but to be safe
        prefetch_factor = None # not relevant for preload


    return DataLoader(dataset, batch_size=batch_size, shuffle=True,pin_memory=pin_memory,
                      num_workers=workers, persistent_workers=persistent_workers,
                      prefetch_factor=prefetch_factor)
        
        

class SpecDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        sample = self.data[idx]

        return sample, sample
