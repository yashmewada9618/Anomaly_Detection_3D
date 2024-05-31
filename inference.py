import torch
import open3d as o3d
import numpy as np
from model import decoder, teacher, student
import os
from dataloader.dataloader import MvTec3D
from tqdm import tqdm
from utils.utils import (
    compute_geometric_data,
    knn,
)


def load_model(model, model_path):
    model_state_dict = torch.load(model_path)
    if model == "teacher":
        model.load_state_dict(model_state_dict["teacher"])
    elif model == "decoder":
        model.load_state_dict(model_state_dict["decoder"])
    else:
        model.load_state_dict(model_state_dict)
    print(f"[+] Loaded pretrained {model} model from {model_path}")


def calculate_mu_sigma(teacher_model, train_data_loader, device):
    teacher_model.eval()
    features = []
    with torch.no_grad():
        for item in tqdm(train_data_loader):
            item = item.to(device)
            knn_points, indices, distances = knn(item, 8)
            geom_feat = compute_geometric_data(item, knn_points, distances)
            teacher_out = teacher_model(item, geom_feat, indices)
            features.append(teacher_out)
    features = torch.cat(features, dim=0)
    return features.mean(dim=0), features.std(dim=0)


def get_point_cloud_from_tiff(tiff_path, num_points=5000):
    point_cloud = np.load(tiff_path).reshape(-1, 3)
    if len(point_cloud) > num_points:
        indices = np.random.choice(len(point_cloud), num_points, replace=False)
        point_cloud = point_cloud[indices]
    return point_cloud


def inference_with_decoder(teacher, student, decoder, point_cloud, mu, sigma, device):
    teacher.eval()
    student.eval()
    decoder.eval()
    with torch.no_grad():
        item = point_cloud.to(device)
        knn_points, indices, distances = knn(item, 8)
        geom_feat = compute_geometric_data(item, knn_points, distances)
        teacher_features = teacher(item, geom_feat, indices)
        student_features = student(item, geom_feat, indices)

        norm_teacher_features = (teacher_features - mu) / sigma
        errors = torch.norm(student_features - norm_teacher_features, dim=-1)
        reconstructed_point_cloud = decoder(student_features)
    return (
        errors.squeeze(0).cpu().numpy(),
        reconstructed_point_cloud.squeeze(0).cpu().numpy(),
    )


def visualize_point_cloud_anomaly(
    point_cloud, anomaly_scores, threshold, title="Point Cloud with Anomalies"
):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    colors = np.zeros((len(anomaly_scores), 3))
    colors[anomaly_scores <= threshold] = [1, 0, 0]  # Red for anomalies
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd], window_name=title)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_path = "/home/mewada/pivot_submission/Anomaly_Detection_3D/datasets/mvtec_point_clouds/"
    teacher_model_path = (
        "/home/mewada/pivot_submission/Anomaly_Detection_3D/weights/exp2.pt"
    )
    student_model_path = (
        "/home/mewada/pivot_submission/Anomaly_Detection_3D/weights/exp_student_v1.pt"
    )

    teacher_model = teacher.TeacherModel(feature_dim=64).to(device)
    student_model = student.StudentModel(f_dim=64).to(device)
    decoder_model = decoder.DecoderNetwork(input_dim=64, output_dim=1024).to(device)

    load_model(teacher_model, teacher_model_path)
    load_model(student_model, student_model_path)
    load_model(decoder_model, teacher_model_path)

    train_dataset = MvTec3D("train", scale=1, root_dir=root_path)
    mu, sigma = calculate_mu_sigma(teacher_model, train_dataset, device)

    tiff_path = "./datasets/MvTec_3D/peach/test/hole/xyz/010.npy"
    point_cloud = torch.tensor(
        get_point_cloud_from_tiff(tiff_path), dtype=torch.float32
    )

    anomaly_scores, reconstructed_point_cloud = inference_with_decoder(
        teacher_model, student_model, decoder_model, point_cloud, mu, sigma, device
    )

    threshold = np.mean(anomaly_scores)  # You can choose your threshold here
    visualize_point_cloud_anomaly(point_cloud.cpu().numpy(), anomaly_scores, threshold)
