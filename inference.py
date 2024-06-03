"""
Author: Yash Mewada
Date: 29th May 2024
Description: This script is used to infer the anomaly scores for a given point cloud using the trained teacher and student models.
"""

import torch
import open3d as o3d
import numpy as np
from model.teacher import TeacherModel
from model.student import StudentModel
from skimage import io
from utils.utils import compute_geometric_data, knn, farthest_point_sampling, Colors


def get_anomaly_scores(teacher, student, point_cloud, exp_name):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mu = torch.tensor(np.load(f"s_factors/{exp_name}_mu.npy")).to(device)
    sigma = torch.tensor(np.load(f"s_factors/{exp_name}_sigma.npy")).to(device)
    s_factor = torch.tensor(np.genfromtxt(f"s_factors/s_factor_{exp_name}.txt")).to(
        device
    )
    print(f"[+] S Factor: {s_factor} for {exp_name}")

    point_cloud = (point_cloud / s_factor).unsqueeze(0)
    teacher.eval()
    student.eval()
    with torch.no_grad():
        item = point_cloud.to(device)
        knn_points, indices, _ = knn(item, 8)
        geom_feat = compute_geometric_data(item, knn_points)
        teacher_features = teacher(item, geom_feat, indices)
        student_features = student(item, geom_feat, indices)
        norm_teacher_features = (teacher_features - mu) / sigma
        errors = torch.norm(student_features - norm_teacher_features, dim=-1)
    errors = errors.squeeze(0).cpu().numpy()
    anomaly_scores_normalized = (errors - np.min(errors)) / (
        np.max(errors) - np.min(errors)
    )
    print(f"[+] Anomaly scores: {anomaly_scores_normalized[0]}")
    print(f"[+] Anomaly scores shape: {anomaly_scores_normalized.shape}")
    return anomaly_scores_normalized[0]


def get_point_cloud(tiff_path, num_points=5000):
    point_cloud = io.imread(tiff_path).reshape(-1, 3)
    if len(point_cloud) < num_points:
        raise ValueError(
            f"Number of points in the point cloud is less than the number of points to sample. Number of points: {len(point_cloud)}"
        )
    indices = farthest_point_sampling(point_cloud, num_points)
    point_cloud = point_cloud[indices]
    return point_cloud


def visualize_anomaly(
    point_cloud,
    anomaly_scores_normalized,
    percentage=95,
):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    colors = np.zeros((anomaly_scores_normalized.shape[0], 3))
    threshold = np.percentile(anomaly_scores_normalized, percentage)
    print(f"[+] Threshold: {threshold}")
    print(
        f"[+] Number of anomalied points: {np.count_nonzero(anomaly_scores_normalized > threshold)}"
    )

    colors[anomaly_scores_normalized > threshold] = [1, 0, 0]
    colors[anomaly_scores_normalized <= threshold] = [0, 0, 1]

    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd], window_name="Anomalies")


if __name__ == "__main__":
    teacher_model_path = "weights/test_pretrain1.pt"
    student_model_path = "weights/test_student1.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.cuda.empty_cache()

    teacher = TeacherModel(feature_dim=64).to(device)
    teacher.load_state_dict(torch.load(teacher_model_path)["teacher"])

    student = StudentModel(f_dim=64).to(device)
    student.load_state_dict(torch.load(student_model_path)["student"])

    root_path = "datasets/mvtec_point_clouds/"
    exp_name = "test_student1"

    tiff_path = "datasets/MvTec_3D/carrot/test/crack/xyz/001.tiff"
    point_cloud = get_point_cloud(tiff_path)
    point_cloud = torch.tensor(point_cloud, dtype=torch.float32).to(device)

    anomaly_scores = get_anomaly_scores(
        teacher, student, point_cloud, exp_name=exp_name
    )

    visualize_anomaly(
        point_cloud.detach().cpu(), anomaly_scores, 95
    )  # computes the 95th percentile of the normalized anomaly scores
