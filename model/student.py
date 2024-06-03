"""
Author: Yash Mewada
Date: 21st May 2024
Description: This script contains the student model used to learn the feature representations from the pointcloud.
"""

import torch.nn as nn
import torch.nn.functional as F
from model.teacher import TeacherModel


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
