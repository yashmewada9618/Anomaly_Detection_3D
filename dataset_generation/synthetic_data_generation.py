#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import pymeshlab
import open3d as o3d
from tqdm import tqdm

"""
Author: Yash Mewada
Date: 21st May 2024

Description: This script is used to generate synthetic data first by scaling, rotating randomly and translating randomly.
            selected 10 sample objects into a unit bounding box using pymeshlab library.
"""


def scale_to_unit_bbox(mesh_path):
    """Scale the longest side of the mesh's bounding box to 1.

    Args: Path to the mesh file
    Returns: MeshSet object with the scaled mesh
    """

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(mesh_path)
    bb = ms.current_mesh().bounding_box()
    scaled_factor = 1 / max(bb.dim_x(), bb.dim_y(), bb.dim_z())

    # Apply the scale filter
    ms.apply_filter(
        "compute_matrix_from_scaling_or_normalization",
        unitflag=True,
    )
    # print(f"[+] Scaled by {scaled_factor}")
    return ms


def rotate_mesh(ms):
    """Objects are rotated around each 3D axis with angles
        sampled uniformly from the interval [0, 2pi]

    Args: Scale mesh object
    Returns: MeshSet object with the rotated mesh
    """

    rotations = np.degrees(np.random.uniform(0, 2 * np.pi, 3))

    # Apply the rotation filter across the X, Y, and Z axes
    ms.apply_filter(
        "compute_matrix_from_rotation",
        rotaxis="X axis",
        angle=rotations[0],
    )
    ms.apply_filter(
        "compute_matrix_from_rotation",
        rotaxis="Y axis",
        angle=rotations[1],
    )
    ms.apply_filter(
        "compute_matrix_from_rotation",
        rotaxis="Z axis",
        angle=rotations[2],
    )
    # print(f"[+] Rotated by {rotations} degrees")

    # Return the rotated mesh
    return ms


def translate_mesh(ms):
    """Objects are translated along each 3D axis with values
        sampled uniformly from the interval [-3, 3]^3

    Args: Rotated mesh object
    Returns: MeshSet object with the translated mesh
    """

    translation = np.random.uniform(-3, 3, 3)

    # Apply the translation filter
    ms.apply_filter(
        "compute_matrix_from_translation",
        traslmethod="XYZ translation",
        axisx=translation[0],
        axisy=translation[1],
        axisz=translation[2],
    )
    # print(f"[+] Translated by {translation}")
    return ms


def mesh2pointcloud(mesh, mesh_file, uniform=True):
    """Convert a mesh to a point cloud.

    Args: MeshSet object, Uniform sampling flag uses Poisson disk sampling when False (Time consuming)
    Returns: Farthest sampled Point clouds
    """
    pcd_file_name = mesh_file.replace(".off", ".pcd")
    pcd_file_path = os.path.join(pcd_file_name.replace("Scenes", "Original_PCD"))

    # convert pymeshlab mesh to open3d mesh
    m = mesh.current_mesh()
    open3d_mesh = o3d.geometry.TriangleMesh()
    open3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(m.vertex_matrix()))
    open3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(m.face_matrix()))

    pcd = (
        open3d_mesh.sample_points_uniformly(number_of_points=100000)
        if uniform
        else open3d_mesh.sample_points_poisson_disk(number_of_points=5500)
    )
    fps_points = farthest_point_sampling(pcd.points, 16000, pcd_file_path)
    # o3d.io.write_point_cloud(pcd_file_path, pcd)

    return fps_points


def load_modelnet10_models(base_dir):
    """Load all OFF files from ModelNet10 dataset.

    Args: Path to the ModelNet10 directory
    Returns: List of paths to the OFF files
    """
    models = []
    categories = [
        d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
    ]
    for category in categories:
        category_path = os.path.join(base_dir, category, "train")
        for file_name in os.listdir(category_path):
            if file_name.endswith(".off"):
                file_path = os.path.join(category_path, file_name)
                models.append(file_path)
    return models


def farthest_point_sampling(pc, n_samples, pcd_file_path):
    """Farthest point sampling algorithm to select n_samples points from the point cloud.

    Args: Point cloud, Number of samples
    Returns: Selected points
    """

    fps_file_path = pcd_file_path.replace("Original_PCD", "FPS_PCD")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)

    # Open3d has inbuilt FPS algorithm
    # more details: https://www.open3d.org/docs/latest/tutorial/t_geometry/pointcloud.html
    fps_points = pcd.farthest_point_down_sample(n_samples)
    o3d.io.write_point_cloud(fps_file_path, fps_points)

    return fps_points


def save_point_clouds(pcds, path):
    for i, pcd in enumerate(pcds):
        o3d.io.write_point_cloud(f"{path}_{i}.pcd", pcd)


def main():
    base_dir = "datasets/ModelNet10/"  # Path to the ModelNet10 directory
    path_to_output_scene = (
        "datasets/pretrained_dataset/Scenes/"  # Path to save the transformed models
    )
    path_to_train_pcds = (
        "datasets/pretrained_dataset/train/"  # Path to save the train point clouds
    )
    path_to_val_pcds = (
        "datasets/pretrained_dataset/val/"  # Path to save the validation point clouds
    )
    models = load_modelnet10_models(base_dir)

    # Randomly select 10 models
    selected_models = random.sample(models, 10)
    train_point_clouds = []
    val_point_clouds = []

    num_train_pcds = 500
    num_val_pcds = 30

    print(num_train_pcds // len(selected_models))
    # Load, scale and save the selected models
    for model_path in tqdm(selected_models):
        # print(f"[+] Loading {model_path.split('/')[-1]}")
        scaled_mesh = scale_to_unit_bbox(model_path)

        # Generate 500 train point clouds
        for _ in tqdm(range(num_train_pcds // len(selected_models))):
            scaled_rotated_mesh = rotate_mesh(scaled_mesh)
            transformed_mesh = translate_mesh(scaled_rotated_mesh)
            filename = (
                f"{path_to_output_scene}transformed_{os.path.basename(model_path)}"
            )
            transformed_mesh.save_current_mesh(filename)
            pcd = mesh2pointcloud(transformed_mesh, filename, uniform=True)
            # print(
            #     f"[+] Tranformed and generated train point cloud {os.path.basename(model_path)}_{k}"
            # )
            train_point_clouds.append(pcd)

        save_point_clouds(
            train_point_clouds,
            f"{path_to_train_pcds}{os.path.basename(model_path).replace('.off', '')}",
        )
        # print(f"[+] Saved {num_train_pcds} train point clouds")

        # Generate 25 validation point clouds
        for _ in tqdm(range(num_val_pcds // len(selected_models))):
            scaled_rotated_mesh = rotate_mesh(scaled_mesh)
            transformed_mesh = translate_mesh(scaled_rotated_mesh)
            filename = (
                f"{path_to_output_scene}transformed_{os.path.basename(model_path)}"
            )
            transformed_mesh.save_current_mesh(filename)
            pcd = mesh2pointcloud(transformed_mesh, filename, uniform=True)
            # print(
            #     f"[+] Tranformed and generated val point cloud {os.path.basename(model_path)}_{l}"
            # )
            val_point_clouds.append(pcd)

        save_point_clouds(
            val_point_clouds,
            f"{path_to_val_pcds}{os.path.basename(model_path).replace('.off', '')}",
        )
        # print(f"[+] Saved {num_val_pcds} validation point clouds")

        train_point_clouds = []
        val_point_clouds = []
        # print(f"[+] Saved {filename}")
        print()

    print("[+] Synthetic Data Generation Completed")


if __name__ == "__main__":
    main()
