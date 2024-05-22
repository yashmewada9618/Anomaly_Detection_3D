import torch
import torch.nn as nn
import torch.nn.functional as F
from Teacher import TeacherModel


class StudentModel(TeacherModel):
    def __init__(self, f_dim):
        super(StudentModel, self).__init__(f_dim)
        self.f_dim = f_dim
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# Check if the model works by running a forward pass with dummy data
# if __name__ == "__main__":
#     # Parameters
#     batch_size = 8
#     num_points = 1024
#     f_dim = 64  # Feature dimension of the Teacher model
#     d_lfa = 32  # Dimension for Local Feature Aggregation
#     k = 16  # Number of nearest neighbors

#     # Initialize the Teacher model
#     model = StudentModel(f_dim)

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
#     print("Output shape:", output.shape)  # returns torch.Size([8, 1024, 64])
