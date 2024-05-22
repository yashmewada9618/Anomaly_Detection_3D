import torch
import torch.nn as nn


class SharedMLP(nn.Module):
    def __init__(self, in_features, out_features):
        super(SharedMLP, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.activation = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        x = self.fc(x)
        x = self.activation(x)
        return x


class LocalFeatureAggregation(nn.Module):
    def __init__(self, d_lfa):
        super(LocalFeatureAggregation, self).__init__()
        self.shared_mlp = SharedMLP(4, d_lfa)
        self.output_dim = d_lfa

    def forward(self, features, geom_features, neighbor_indices):
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


class ResidualBlock(nn.Module):
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


class TeacherModel(nn.Module):
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


# # Check if the model works by running a forward pass with dummy data
# if __name__ == "__main__":
#     # Parameters
#     batch_size = 8
#     num_points = 1024
#     f_dim = 64  # Feature dimension of the Teacher model
#     d_lfa = 32  # Dimension for Local Feature Aggregation
#     k = 16  # Number of nearest neighbors

#     # Initialize the Teacher model
#     model = TeacherModel(f_dim)

#     # Create dummy input tensors
#     data = torch.randn(batch_size, num_points, 3)  # Example data tensor (B, N, 3)
#     geom_features = torch.randn(
#         batch_size, num_points, k, 4
#     )  # Geometric features (B, N, k, 4)
#     indices = torch.randint(
#         0, num_points, (batch_size, num_points, k)
#     )  # Nearest neighbor indices (B, N, k)

#     # Move model and data to GPU if available
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     data = data.to(device)
#     geom_features = geom_features.to(device)
#     indices = indices.to(device)

#     # Forward pass
#     output = model(data, geom_features, indices)

#     # Output shape
#     print("Output shape:", output.shape) # returns torch.Size([8, 1024, 64])
