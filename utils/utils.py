"""
Author: Yash Mewada
Date: 21st May 2024
Description: This script contains utility functions used in the training and inference of the teacher and student models.
"""

import torch
from tqdm import tqdm
import numpy as np


class Colors:
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"


def knn(points, k, batch_size=50):
    """
    Calculate k-nearest neighbors for each point in a batch of point clouds in chunks.

    Args:
    - points (torch.Tensor): Input point cloud tensor of shape (num_points, num_dimensions).
    - k (int): Number of nearest neighbors to find.
    - batch_size (int): Number of points to process at a time.

    Returns:
    - knn_points (torch.Tensor): Nearest neighbor points tensor of shape (num_points, k, num_dimensions).
    - knn_indices (torch.Tensor): Nearest neighbor indices tensor of shape (num_points, k).
    - knn_dists (torch.Tensor): Pairwise distance tensor of shape (num_points, k).
    """
    _, num_points, num_dimensions = points.size()
    points = points.squeeze(0)
    knn_points = []
    knn_indices = []
    knn_dists = []

    # Process points in batches
    for i in range(0, num_points, batch_size):
        end = min(i + batch_size, num_points)
        batch_points = points[i:end]

        # Calculate pairwise distances for the batch
        x1 = batch_points.unsqueeze(1).expand(-1, num_points, -1)
        x2 = points.unsqueeze(0).expand(end - i, -1, -1)
        diff = x2 - x1
        norm = torch.norm(diff, dim=-1)

        # Get k-nearest neighbors
        batch_knn_dists, batch_knn_indices = torch.topk(
            norm, k=k + 1, largest=False, sorted=True
        )
        batch_knn_dists = batch_knn_dists[
            :, 1:
        ]  # Exclude the first distance (distance to itself)

        batch_knn_indices = batch_knn_indices[
            :, 1:
        ]  # Exclude the first index (distance to itself)

        # Ensure indices are within bounds
        valid_indices_mask = (batch_knn_indices >= 0) & (batch_knn_indices < num_points)
        batch_knn_indices = batch_knn_indices * valid_indices_mask

        batch_knn_points = torch.gather(
            points.unsqueeze(0).expand(end - i, num_points, num_dimensions),
            1,
            batch_knn_indices.unsqueeze(-1).expand(end - i, k, num_dimensions),
        )

        knn_points.append(batch_knn_points)
        knn_indices.append(batch_knn_indices)
        knn_dists.append(batch_knn_dists)

        del x1, x2, diff, norm
        # torch.cuda.empty_cache()

    knn_points = torch.cat(knn_points, dim=0).unsqueeze(0)
    knn_indices = torch.cat(knn_indices, dim=0).unsqueeze(0)
    knn_dists = torch.cat(knn_dists, dim=0).unsqueeze(0)
    # points = points.unsqueeze(0)

    return knn_points, knn_indices, knn_dists


def get_receptive_fields(points, num_samples, k1, k2, sampled_indices, batch_size=50):
    """
    Get 2-layer k-nearest neighbors for randomly sampled points from a point cloud.

    Args:
    - points (torch.Tensor): Input point cloud tensor of shape (num_points, num_dimensions).
    - num_samples (int): Number of points to sample from the point cloud.
    - k1 (int): Number of nearest neighbors for the first layer.
    - k2 (int): Number of nearest neighbors for the second layer.
    - batch_size (int): Number of points to process at a time for KNN.

    Returns:
    - output (torch.Tensor): Output tensor of shape (num_samples, k1 * k2, num_dimensions).
    """
    B, _, num_dimensions = points.size()

    # Get first layer of k-nearest neighbors
    _, knn1_indices, _ = knn(points, k1, batch_size)
    # exit()

    # Initialize the output tensor
    output = torch.zeros((B, num_samples, k1 * k2, num_dimensions))

    for i in range(num_samples):
        first_layer_neighbors = knn1_indices[:, sampled_indices[i]]

        second_layer_neighbors_list = []
        for neighbor in first_layer_neighbors:
            _, second_layer_knn_indices, _ = knn(points, k2, batch_size)
            second_layer_neighbors = second_layer_knn_indices[:, neighbor].squeeze(0)
            second_layer_neighbors_list.append(second_layer_neighbors)

        second_layer_neighbors_list = torch.stack(second_layer_neighbors_list)
        second_layer_neighbors_list = second_layer_neighbors_list.reshape(-1)
        second_layer_points = points[:, second_layer_neighbors_list]

        output[0][i] = second_layer_points

    # Normalize the output by subtracting the mean
    mean_points = output.mean(dim=2, keepdim=True)
    output -= mean_points

    return output


def compute_geometric_data(x, knn_points):
    differences = x.unsqueeze(2) - knn_points
    l2_norms = torch.norm(differences, dim=-1, keepdim=True)
    output = torch.cat([differences, l2_norms], dim=-1)
    return output


def compute_scaling_factor(point_cloud, k):
    """
    Calculate the scaling factor s for normalizing the point cloud.

    Args:
    - point_cloud (numpy.ndarray): Input point cloud array of shape (N, D),
      where N is the number of points and D is the dimensionality of each point.
    - k (int): Number of nearest neighbors to consider.

    Returns:
    - s (float): Scaling factor for normalizing the point cloud.
    """
    _, _, distances = knn(point_cloud, k)

    # Compute the average distance over all points and their k-nearest neighbors
    avg_distance = torch.sum(distances).item()

    # Calculate the scaling factor
    s = avg_distance / (k * point_cloud.shape[1])

    return s


def farthest_point_sampling(point_cloud, num_points):
    """
    Perform farthest point sampling on a point cloud.

    Args:
        point_cloud (np.ndarray): Input point cloud of shape (N, 3), where N is the number of points.
        num_points (int): Number of points to sample.

    Returns:
        np.ndarray: Indices of the sampled points.
    """
    N, _ = point_cloud.shape
    sampled_indices = np.zeros(num_points, dtype=int)
    distances = np.ones(N) * 1e10  # Initialize distances to a large value

    # Randomly choose the first point
    first_index = np.random.choice(N)
    sampled_indices[0] = first_index
    farthest_point = point_cloud[first_index]

    for i in range(1, num_points):
        # Calculate squared Euclidean distances from the farthest point
        dist = np.sum((point_cloud - farthest_point) ** 2, axis=1)
        # Update the distances to keep the minimum distance to the sampled points
        distances = np.minimum(distances, dist)
        # Choose the farthest point
        farthest_index = np.argmax(distances)
        sampled_indices[i] = farthest_index
        farthest_point = point_cloud[farthest_index]

    return sampled_indices


def get_params(teacher_model, train_data_loader, s):
    """
    Compute the mean and standard deviation of the features produced by the teacher model.

    Args:
        teacher_model (torch.nn.Module): The pretrained teacher model used for feature extraction.
        train_data_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        s (float): Scaling factor used for normalizing the point cloud data.

    Returns:
        torch.Tensor: Mean of the extracted features over the entire training set.
        torch.Tensor: Standard deviation of the extracted features over the entire training set.
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    teacher_model.eval()
    features = []
    k = 8
    with torch.no_grad():
        for item in tqdm(train_data_loader):
            item = item.to(device) / s
            temp_indices = torch.randperm(item.size(1))[:5000]
            item = item[:, temp_indices, :]
            knn_points, indices, _ = knn(item, k)
            geom_feat = compute_geometric_data(item, knn_points)
            teacher_out = teacher_model(item, geom_feat, indices)
            features.append(teacher_out)
            del teacher_out
    features = torch.cat(features, dim=0)
    return features.mean(dim=0), features.std(dim=0)
