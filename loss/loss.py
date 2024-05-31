"""
Author: Yash Mewada
Date: 21st May 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChampherLoss(nn.Module):
    """ """

    def __init__(self):
        super().__init__()
        ### Dictionary for required layer

    """
    Whenever object is invoked this object is called, and loss is returned based on the 
    hyperparameters defined.
    """

    def __call__(self, normalized_receptive_fields, reconstructed_points):
        """
        Compute the Chamfer Distance between two point clouds pc1 and pc2.

        Args:
        - pc1: Tensor of shape (B, N, 3) representing the first point cloud.
        - pc2: Tensor of shape (B, M, 3) representing the second point cloud.

        Returns:
        - dist: Chamfer distance between the point clouds.
        """

        _, _, num_reconstructed_points, _ = reconstructed_points.shape
        _, _, num_neighbors, _ = normalized_receptive_fields.shape

        # Reshape tensors to the format (batch_size * num_points_selected, num_points, 3)
        reconstructed_points = reconstructed_points.permute(0, 1, 3, 2).reshape(
            -1, num_reconstructed_points, 3
        )
        normalized_receptive_fields = normalized_receptive_fields.reshape(
            -1, num_neighbors, 3
        )
        # Compute pairwise distances between all points
        reconstructed_points = reconstructed_points.unsqueeze(2)
        normalized_receptive_fields = normalized_receptive_fields.unsqueeze(1)
        diff = reconstructed_points - normalized_receptive_fields
        dist = torch.sum(diff**2, dim=-1)

        # Compute the minimum distance from each point in reconstructed_points to normalized_receptive_fields
        min_dist_x_to_y, _ = torch.min(dist, dim=2)
        min_dist_y_to_x, _ = torch.min(dist, dim=1)

        # Chamfer Distance is the mean of these minimum distances
        chamfer_distance = torch.mean(min_dist_x_to_y) + torch.mean(min_dist_y_to_x)
        return chamfer_distance


class AnomalyScoreLoss(nn.Module):
    """ """

    def __init__(self):
        super().__init__()
        ### Dictionary for required layer

    """
    Whenever object is invoked this object is called, and loss is returned based on the 
    hyperparameters defined.
    """

    def __call__(self, teacher_features, student_features):
        """
        Compute the norm between the teacher and student features.

        Args:
        - pc1: Tensor of shape (B, N, 3) representing the first point cloud.
        - pc2: Tensor of shape (B, M, 3) representing the second point cloud.

        Returns:
        - dist: Chamfer distance between the point clouds.
        """

        # return (
        #     torch.norm(teacher_features - student_features, dim=1).sum()
        #     / teacher_features.shape[1]
        # )

        return F.mse_loss(teacher_features, student_features)
