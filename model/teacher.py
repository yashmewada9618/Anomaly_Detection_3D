"""
Author: Yash Mewada
Date: 21st May 2024
"""

import torch
import torch.nn as nn
from base.base_model import BaseModel


class SharedMLP(BaseModel):
    def __init__(self, in_features, out_features):
        super(SharedMLP, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = self.fc(x)
        x = self.activation(x)
        return x


class LocalFeatureAggregation(BaseModel):
    def __init__(self, d_lfa):
        super(LocalFeatureAggregation, self).__init__()
        self.shared_mlp = SharedMLP(4, d_lfa)
        self.output_dim = d_lfa

    def forward(self, features, geom_features, neighbor_indices):
        # print(features.shape, "Features")
        _, num_points, _ = features.size()
        transformed_features = self.shared_mlp(geom_features)

        # Expand indices for gathering neighbor features
        exp_idx = neighbor_indices.unsqueeze(-1).expand(-1, -1, -1, features.size(-1))
        f_neighbors = torch.gather(
            features.unsqueeze(1).expand(-1, num_points, -1, -1), 2, exp_idx
        )

        # Concatenate transformed geometric features with neighbor features
        combined_features = torch.cat([transformed_features, f_neighbors], dim=-1)
        aggregated_features = combined_features.mean(dim=2)
        return aggregated_features


class ResidualBlock(BaseModel):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.initial_mlp = SharedMLP(dim, dim // 4)
        self.local_agg_1 = LocalFeatureAggregation(dim // 4)
        self.local_agg_2 = LocalFeatureAggregation(dim // 2)
        self.final_mlp = SharedMLP(dim, dim)
        self.identity_mlp = SharedMLP(dim, dim)

    def forward(self, input_tensor, geometric_features, neighbor_indices):
        residual = input_tensor
        x = self.initial_mlp(input_tensor)
        x = self.local_agg_1(x, geometric_features, neighbor_indices)
        x = self.local_agg_2(x, geometric_features, neighbor_indices)
        x = self.final_mlp(x)
        x += self.identity_mlp(residual)
        return x


class TeacherModel(BaseModel):
    def __init__(self, feature_dim):
        super(TeacherModel, self).__init__()
        self.feature_dim = feature_dim
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(self.feature_dim) for _ in range(4)]
        )

    def forward(self, input_data, geometric_data, neighbor_indices):
        feature_vector = torch.zeros(
            input_data.shape[0], input_data.shape[1], self.feature_dim
        ).to(input_data.device)
        for block in self.residual_blocks:
            feature_vector = block(feature_vector, geometric_data, neighbor_indices)
        return feature_vector
