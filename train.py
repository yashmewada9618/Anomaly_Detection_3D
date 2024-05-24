import os
from model.teacher import TeacherModel
from model.student import StudentModel
from model.decoder import DecoderNetwork
import torch
import torch.nn as nn
import open3d as o3d
from utils.utils import (
    compute_geometric_data,
    knn,
    compute_receptive_field,
    compute_s_value,
)


def load_dataset(dataset_path):
    point_clouds = []
    for file in os.listdir(dataset_path):
        pcd = o3d.io.read_point_cloud(os.path.join(dataset_path, file))
        point_clouds.append(torch.tensor(np.array(pcd.points), dtype=torch.float32))
    return point_clouds


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.rand(32, 1024, 3).to(device)
    teacher = TeacherModel(64).to(device)
    student = StudentModel(64).to(device)
    decoder = DecoderNetwork().to(device)

    if __name__ == "__main__":
        knn_points, indices, distances = knn(data, 8)
        geom_data = compute_geometric_data(data, knn_points, distances)
        out = teacher(data, geom_data, indices)
        sampled_indices = torch.randint(0, 1024, (32, 1)).to(
            device
        )  # Generate random indices for each batch
        sampled_points = torch.gather(
            out, 1, sampled_indices.view(32, 1, 1).expand(32, 1, 64)
        )  # Gather sampled points
        dec_out = decoder(sampled_points)
        print(out.shape)
        print(dec_out.shape)

    # Load the dataset
    # training_dataset_path = (
    #     "/home/mewada/Anomaly_Detection_3D/dataset_generation/pretrained_dataset/train/"
    # )
    # train_pcs = load_dataset(training_dataset_path)
    # validation_dataset_path = (
    #     "/home/mewada/Anomaly_Detection_3D/dataset_generation/pretrained_dataset/val/"
    # )
    # val_pcs = load_dataset(validation_dataset_path)
