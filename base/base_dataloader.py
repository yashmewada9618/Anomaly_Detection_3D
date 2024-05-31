"""
Author: Yash Mewada
Date: 21st May 2024
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd
import os
import glob

# import io
from skimage import io
from abc import abstractmethod


class BaseDataLoader(DataLoader):
    """
    Base class for data loaders
    """

    def __init__(self, path, root_dir, transform=None):

        self.files = []
        full_path = os.path.join(root_dir, path)
        if os.path.isdir(full_path):
            pcd_files = glob.glob(os.path.join(full_path, "*.pcd"))
            if pcd_files:
                self.files.extend(pcd_files)
            else:
                for subdir in os.listdir(full_path):
                    subdir_path = os.path.join(full_path, subdir)
                    if os.path.isdir(subdir_path):
                        npy_files = glob.glob(os.path.join(subdir_path, "*.npy"))
                        self.files.extend(npy_files)

        else:
            raise ValueError(f"The path {full_path} does not seems to be a dataset.")
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.files)

    @abstractmethod
    def __getitem__(self, idx):
        """
        Just a basic structure to be implemented later
        """
        raise NotImplementedError
