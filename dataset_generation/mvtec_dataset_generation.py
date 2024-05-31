z#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import json
import torch
from tifffile import tifffile
from tqdm import tqdm
import open3d as o3d


"""
Author: Yash Mewada
Date: 21st May 2024
"""


def farthest_point_sampling(points, sample_nums):
    N, _ = points.shape
    sampled_indices = torch.zeros(sample_nums, dtype=torch.long)
    distances = torch.full((N,), float("inf"), device=points.device)

    # Initialize the first point randomly
    sampled_indices[0] = torch.randint(0, N, (1,), device=points.device)
    farthest_point = points[sampled_indices[0]].unsqueeze(0)

    for i in range(1, sample_nums):
        # Update distances with the minimum distance to the current farthest point
        dist_to_farthest_point = torch.norm(points - farthest_point, dim=1)
        distances = torch.min(distances, dist_to_farthest_point)

        # Select the next farthest point
        sampled_indices[i] = torch.argmax(distances)
        farthest_point = points[sampled_indices[i]].unsqueeze(0)

    return points[sampled_indices]


def transform_tiff(tiff_file, cam_matrix):
    """Transform the tiff file to a point cloud using the camera matrix.

    Args: Path to the tiff file, camera matrix
    Returns: Open3D point cloud object
    """
    # generate camera matrix from json file
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tiff_file = torch.from_numpy(tiff_file.reshape(-1, 3).astype(np.float32)).to(device)
    cx = cam_matrix["cx"]
    cy = cam_matrix["cy"]
    sx = cam_matrix["sx"]
    sy = cam_matrix["sy"]
    focus = cam_matrix["focus"]
    dist = cam_matrix["kappa"]

    x = (tiff_file[:, 0] - cx) * sx
    y = (tiff_file[:, 1] - cy) * sy
    z = tiff_file[:, 2] * focus

    R = x**2 + y**2
    F = 1 + dist * R

    nx = x * F
    ny = y * F

    new_points = torch.stack((nx, ny, z), dim=1)
    new_points = new_points[new_points[:, 2] != 0]

    return new_points


def generate_pcs_fromtiff(dataset_path, output_path, train_val="train"):
    os.makedirs(output_path, exist_ok=True)

    mvtec_objects = os.listdir(dataset_path)
    print(f"[+] Objects found: {mvtec_objects}")

    for obj in tqdm(mvtec_objects):
        obj_path = os.path.join(dataset_path, obj)
        # print(f"[+] Processing object: {obj_path}")
        if not os.path.isdir(obj_path):
            continue
        os.makedirs(os.path.join(output_path, train_val, obj), exist_ok=True)
        cam_matrix_path = os.path.join(
            obj_path, "calibration", "camera_parameters.json"
        )
        with open(cam_matrix_path, "r") as f:
            cam_matrix = json.load(f)
        tiff_file_path = os.path.join(obj_path, train_val, "good", "xyz")

        for file in tqdm(os.listdir(tiff_file_path)):
            if file.endswith(".tiff"):
                tiff_file = tifffile.imread(os.path.join(tiff_file_path, file))
            transformed_pts = transform_tiff(tiff_file, cam_matrix)
            pcd_path = os.path.join(
                output_path, train_val, obj, f"{obj}_{file.split('.')[0]}"
            )
            pcd = farthest_point_sampling(transformed_pts, 2048)
            pcd = pcd.detach().cpu().numpy()
            np.save(pcd_path, pcd)


if __name__ == "__main__":
    # Path to the dataset
    dataset_path = (
        "/home/mewada/pivot_submission/Anomaly_Detection_3D/datasets/MvTec_3D/"
    )
    output_path = "/home/mewada/pivot_submission/Anomaly_Detection_3D/datasets/mvtec_point_clouds/"

    generate_pcs_fromtiff(dataset_path, output_path, train_val="train")
    # generate_pcs_fromtiff(dataset_path, output_path, train_val="validation")
