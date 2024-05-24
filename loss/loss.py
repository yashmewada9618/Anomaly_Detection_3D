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

    def __call__(self, receptive_pc, decoder_pc):
        """
        Compute the Chamfer Distance between two point clouds pc1 and pc2.

        Args:
        - pc1: Tensor of shape (B, N, 3) representing the first point cloud.
        - pc2: Tensor of shape (B, M, 3) representing the second point cloud.

        Returns:
        - dist: Chamfer distance between the point clouds.
        """

        pc1_expand = receptive_pc.unsqueeze(2)  # (B, N, 1, 3)
        pc2_expand = decoder_pc.unsqueeze(1)  # (B, 1, M, 3)

        dist = torch.sum((pc1_expand - pc2_expand) ** 2, dim=-1)  # (B, N, M)

        dist1 = torch.min(dist, dim=2)[0]  # (B, N)
        dist2 = torch.min(dist, dim=1)[0]  # (B, M)

        chamfer_dist = torch.mean(dist1) + torch.mean(dist2)
        return chamfer_dist
