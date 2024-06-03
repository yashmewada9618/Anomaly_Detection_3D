"""
Author: Yash Mewada
Date: 21st May 2024
Description: This script contains the decoder network used to decode the teacher feature vectors from the pointcloud.
"""

import torch.nn as nn
from base.base_model import BaseModel


class DecoderNetwork(BaseModel):
    def __init__(self, input_dim=64, hidden_dim=128, output_dim=1024):
        super(DecoderNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Define layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(
            hidden_dim, output_dim * 3
        )  # Output 3 coordinates for each of the 1024 points

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.05)

    def forward(self, x):
        # Forward pass through the network
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, self.output_dim, 3)  # Reshape to (batch_size, output_dim, 3)
        return x
