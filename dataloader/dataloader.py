"""
Author: Yash Mewada
Date: 21st May 2024
"""

from base.base_dataloader import BaseDataLoader
from skimage import io
import numpy as np
import open3d as o3d
import os
from tqdm import tqdm
import torch
from skimage import io, transform
from torch.utils.data import Dataset


### Data loader class that parses the CSV, abstraction of BaseDataLoader
class ModelNet10(BaseDataLoader):
    def __init__(self, path, root_dir, scale=2, transform=None):
        super().__init__(path=path, root_dir=root_dir, transform=transform)
        self.scale = scale
        print("[+] Model Net 10 Data Loaded with rows: ", super().__len__())

    ### Get item returns an image for specific idx
    def __getitem__(self, idx):
        pcd = o3d.io.read_point_cloud(self.files[idx])
        return torch.tensor(np.array(pcd.points)).float()


class MvTec3D(BaseDataLoader):
    def __init__(self, path, root_dir, scale=2, transform=None):
        super().__init__(path=path, root_dir=root_dir, transform=transform)
        self.scale = scale
        print("[+] MvTec Data Loaded with rows: ", super().__len__())

    ### Get item returns an image for specific idx
    def __getitem__(self, idx):
        pcd = np.load(self.files[idx])
        return torch.tensor(pcd).float()


class MVTec3DDataset(Dataset):
    def __init__(self, base_dir, split, num_points=16000, resize_shape=(400, 400, 3)):
        """
        Initializes the MVTec 3D dataset.

        Args:
            base_dir (str): Base directory containing the dataset.
            split (str): Dataset split to use (e.g., 'train', 'test').
            num_points (int): Number of points to sample from the point cloud.
            resize_shape (tuple): Desired shape to resize the images.
        """
        self.base_dir = base_dir
        self.split = split
        self.num_points = num_points
        self.resize_shape = resize_shape
        self.tiff_files = self._collect_tiff_files()

    def _collect_tiff_files(self):
        """
        Collects all .tiff files from the dataset directory.

        Returns:
            list: List of file paths to .tiff files.
        """
        tiff_files = []
        for category in os.listdir(self.base_dir):
            xyz_dir = os.path.join(self.base_dir, category, self.split, "good", "xyz")
            if os.path.isdir(xyz_dir):
                tiff_files.extend(
                    os.path.join(xyz_dir, f)
                    for f in os.listdir(xyz_dir)
                    if f.endswith(".tiff")
                )
        return tiff_files

    def __len__(self):
        return len(self.tiff_files)

    def __getitem__(self, idx):
        tiff_file = self.tiff_files[idx]
        tiff_image = io.imread(tiff_file)

        # Resize the image
        resized_image = transform.resize(
            tiff_image, self.resize_shape, anti_aliasing=True
        )

        # Convert to point cloud
        point_cloud = resized_image.reshape(-1, 3)

        # Randomly sample points if more than num_points
        if point_cloud.shape[0] > self.num_points:
            indices = np.random.choice(
                point_cloud.shape[0], self.num_points, replace=False
            )
            point_cloud = point_cloud[indices]

        return torch.tensor(point_cloud, dtype=torch.float32)
