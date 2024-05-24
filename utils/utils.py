"""
Author: Yash Mewada
Date: 21st May 2024
"""

import torch


def knn(x, k=8):
    batch_size, num_points, num_features = x.size()
    inner = torch.bmm(x, x.transpose(2, 1))
    squared_norm = torch.sum(x**2, dim=-1, keepdim=True)
    pairwise_distances = squared_norm + squared_norm.transpose(2, 1) - 2 * inner
    pairwise_distances = torch.clamp(pairwise_distances, min=0.0)
    _, knn_indices = pairwise_distances.topk(k=k, dim=-1, largest=False, sorted=True)
    knn_points = torch.gather(
        x.unsqueeze(1).expand(batch_size, num_points, num_points, num_features),
        2,
        knn_indices.unsqueeze(-1).expand(batch_size, num_points, k, num_features),
    )
    return knn_points, knn_indices, pairwise_distances


def compute_geometric_data(x, knn_points, pairwise_distances, k=8):
    print(x.shape, knn_points.shape)
    differences = x.unsqueeze(2) - knn_points
    l2_norms = torch.norm(differences, dim=-1, keepdim=True)
    output = torch.cat([differences, l2_norms], dim=-1)
    return output


def compute_s_value(all_data, knn_points, knn_indices, k):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_data = all_data.to(device)
    knn_indices = knn_indices.to(device)

    num_pointclouds, num_points, _ = all_data.shape

    # Flatten the indices for easier gathering
    flat_indices = knn_indices.view(-1, k)

    # Gather the k-nearest neighbors for each point
    knn_points = all_data.view(-1, 3)[flat_indices]

    # Expand all_data for broadcasting
    all_data_expanded = all_data.unsqueeze(2).expand(num_pointclouds, num_points, k, 3)

    # Calculate the pairwise distances
    distances = torch.norm(
        all_data_expanded - knn_points.view(num_pointclouds, num_points, k, 3), dim=-1
    )

    # Calculate the normalization factor s
    s = distances.mean().item()

    return s


def compute_receptive_field(points, k, L):
    """
    Compute the receptive field R(p) for each point p in the input data.

    Args:
    - data (torch.Tensor): [number of point clouds, num_points, 3] input point clouds
    - k (int): number of nearest neighbors
    - L (int): total number of LFA blocks

    Returns:
    - receptive_fields (list): list of receptive fields for each point in each point cloud
    """
    batch_size, num_points, num_features, _ = points.size()

    knn_points, _, _ = knn(points, k)
    receptive_field = torch.zeros_like(points)

    # Iterate over each point in the batch
    for b in range(batch_size):
        # Iterate over each point in the point cloud
        for p in range(num_points):
            # Initialize list to store indices of points in the receptive field
            rf_indices = [p]
            # Iterate over each nearest neighbor level
            for level in range(2):  # Assuming 2 levels for simplicity, can be modified
                # Get indices of nearest neighbors at current level
                neighbors_indices = knn_points[b, p, :, 0].long()
                # Add indices of neighbors to receptive field indices list
                rf_indices.extend(neighbors_indices.tolist())

            # Gather points from data using indices to form receptive field
            receptive_field[b, p] = torch.mean(points[b, rf_indices], dim=0)
    print(receptive_field.shape)
    return receptive_field
