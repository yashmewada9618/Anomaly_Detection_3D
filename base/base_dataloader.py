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

# import io
from skimage import io
from abc import abstractmethod


class BaseDataLoader(DataLoader):
    """
    Base class for data loaders
    """

    def __init__(self, csv_file, root_dir, transform=None):
        self.csv_dataframe = pd.read_csv(
            os.path.join(root_dir, f"data/div2k/{csv_file}")
        )
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.csv_dataframe)

    @abstractmethod
    def __getitem__(self, idx):
        """
        Just a basic structure to be implemented later
        """
        raise NotImplementedError
