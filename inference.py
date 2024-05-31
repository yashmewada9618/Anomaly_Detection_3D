import torch
import open3d as o3d
import numpy as np
from model import decoder, teacher, student
import pickle
from skimage import io
import os
from dataloader.dataloader import MvTec3D, ModelNet10
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.utils import (
    compute_geometric_data,
    knn,
    compute_s_value,
    receptive_field,
    get_receptive_fields,
    receptive_field_,
    get_receptive_field_1,
    compute_local_receptive_field,
    compute_receptive_field_1,
)


def load_model(model, model_path, device):
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"Model loaded from {model_path}")
    else:
        raise FileNotFoundError(f"Model weights file not found at {model_path}")


def calculate_mu_sigma(teacher_model, train_data_loader, device):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model.eval()
    features = []
    chunk_size = 1024
    k = 8
    with torch.no_grad():
        for item in tqdm(train_data_loader):
            item = item.to(device)[:, :chunk_size, :]
            knn_points, indices, distances = knn(item, k)
            geom_feat = compute_geometric_data(item, knn_points, distances)
            teacher_out = teacher_model(item, geom_feat, indices)
            features.append(teacher_out)
    features = torch.cat(features, dim=0)
    return features.mean(dim=0), features.std(dim=0)


def inference_with_decoder(teacher, student, decoder, point_cloud, mu, sigma, device):
    teacher.eval()
    student.eval()
    decoder.eval()
    point_cloud = point_cloud.to(device).unsqueeze(0)

    with torch.no_grad():

        item = point_cloud.to(device)[:, :1024, :]
        knn_points, indices, distances = knn(item, 8)
        geom_feat = compute_geometric_data(item, knn_points, distances)
        teacher_features = teacher(item, geom_feat, indices)
        student_features = student(item, geom_feat, indices)

        norm_teacher_features = (teacher_features - mu) / sigma
        errors = torch.norm(student_features - norm_teacher_features, dim=-1)
        errors = torch.norm(student_features - teacher_features, dim=-1)

        reconstructed_point_cloud = decoder(student_features)
        teacher_point_cloud = decoder(teacher_features)

    return (
        errors.squeeze(0).cpu().numpy(),
        reconstructed_point_cloud.squeeze(0).cpu().numpy(),
        teacher_point_cloud.squeeze(0).cpu().numpy(),
    )


def visualize_point_cloud(point_cloud, title="Point Cloud"):
    point_cloud = point_cloud.reshape(-1, 3)
    print(point_cloud.shape)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    o3d.visualization.draw_geometries([pcd], window_name=title)


def get_loader(path):
    with open(path, "rb") as f:
        train_loader = pickle.load(f)
    return train_loader


def farthest_point_sampling(point_cloud, num_points):
    num_points_input = point_cloud.shape[0]
    center_index = np.random.choice(num_points_input, 1)
    center_point = point_cloud[center_index]
    distances = np.sum((point_cloud - center_point) ** 2, axis=1)
    sampled_indices = [center_index[0]]
    for _ in range(num_points - 1):
        farthest_index = np.argmax(distances)
        sampled_indices.append(farthest_index)
        farthest_point = point_cloud[farthest_index]
        distances = np.minimum(
            distances, np.sum((point_cloud - farthest_point) ** 2, axis=1)
        )
    return sampled_indices


def compute_average_distance(point_clouds):
    distances = []
    for pc in point_clouds:
        for i in range(pc.shape[0]):
            for j in range(i + 1, pc.shape[0]):
                distances.append(np.linalg.norm(pc[i] - pc[j]))
    return np.mean(distances)


def normalize_point_cloud(point_cloud):
    average_distance = 1.0
    return point_cloud / average_distance


def get_point_cloud_from_tiff(tiff_path):
    num_points = 5000
    point_cloud = io.imread(tiff_path)
    point_cloud = point_cloud.reshape(-1, 3)
    if len(point_cloud) > num_points:
        indices = farthest_point_sampling(point_cloud, num_points)
        point_cloud = point_cloud[indices]
    normalized_point_cloud = normalize_point_cloud(point_cloud)
    return normalized_point_cloud


def visualize_point_cloud_anomaly(
    point_cloud,
    anomaly_scores_normalized,
    title="Point Cloud with Anomalies",
    threshold=0.873,
):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    colors = np.zeros((anomaly_scores_normalized.shape[0], 3))
    print(
        "points less than threshold", np.shape(anomaly_scores_normalized <= threshold)
    )

    colors[anomaly_scores_normalized > threshold] = [1, 1, 0]
    colors[anomaly_scores_normalized <= threshold] = [0, 0, 0]

    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd], window_name=title)


def load_model(model, model_path):
    if os.path.exists(model_path):
        if model == "teacher":
            model.load_state_dict(torch.load(model_path)["teacher"])
            print(f"[+] Loaded pretrained teacher model from {model_path}")
        elif model == "decoder":
            model.load_state_dict(torch.load(model_path)["decoder"])
            print(f"[+] Loaded pretrained decoder model from {model_path}")
        else:
            model.load_state_dict(torch.load(model_path))
            print(f"[+] Loaded pretrained {model} from {model_path}")
    else:
        raise FileNotFoundError(
            f"[!] Pretrained teacher model not found at {model_path}"
        )


if __name__ == "__main__":
    teacher_model_path = (
        "/home/mewada/pivot_submission/Anomaly_Detection_3D/weights/exp2.pt"
    )
    student_model_path = (
        "/home/mewada/pivot_submission/Anomaly_Detection_3D/weights/exp_student_v1.pt"
    )
    # decoder_model_path = "best_decoder0.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher = teacher.TeacherModel(feature_dim=64).to(device)
    teacher.load_state_dict(torch.load(teacher_model_path)["teacher"])

    student = student.StudentModel(f_dim=64).to(device)
    student.load_state_dict(torch.load(student_model_path)["student"])

    decoder = decoder.DecoderNetwork(input_dim=64, output_dim=1024).to(device)
    decoder.load_state_dict(torch.load(teacher_model_path)["decoder"])

    root_path = "/home/mewada/pivot_submission/Anomaly_Detection_3D/datasets/mvtec_point_clouds/"

    train_ = MvTec3D("train", scale=1, root_dir=root_path)
    train_dataset = torch.utils.data.DataLoader(
        train_, batch_size=1, pin_memory=True, shuffle=True
    )
    exp_name = "exp_student_v0"
    # mu, sigma = calculate_mu_sigma(teacher, train_dataset, device)
    mu = torch.tensor(np.load(f"weights/{exp_name}_mu_1024.npy")).to(device)
    sigma = torch.tensor(np.load(f"weights/{exp_name}_sigma_1024.npy")).to(device)

    # tiff_path = './mvtec_3d_anomaly_detection/cookie/test/crack/xyz/002.tiff' #0.9
    # tiff_path = './mvtec_3d_anomaly_detection/cable_gland/test/thread/xyz/001.tiff' # 0.903
    tiff_path = "./datasets/MvTec_3D/peach/test/hole/xyz/010.tiff"  # 0.873
    point_cloud = get_point_cloud_from_tiff(tiff_path)
    point_cloud = torch.tensor(point_cloud, dtype=torch.float32)

    anomaly_scores, reconstructed_point_cloud, teacher_point_cloud = (
        inference_with_decoder(
            teacher, student, decoder, point_cloud, mu, sigma, device
        )
    )

    print("Anomaly scores for each point in the point cloud:", anomaly_scores)

    original_point_cloud_np = point_cloud.cpu().numpy()

    # visualize_point_cloud(original_point_cloud_np, title="Original Point Cloud")
    # visualize_point_cloud(reconstructed_point_cloud, title="Student Point Cloud")
    # visualize_point_cloud(teacher_point_cloud, title="Teacher Point Cloud")

    anomaly_scores_normalized = (anomaly_scores - np.min(anomaly_scores)) / (
        np.max(anomaly_scores) - np.min(anomaly_scores)
    )
    threshold = np.mean(anomaly_scores)
    threshold2 = np.median(anomaly_scores)
    threshold3 = np.percentile(anomaly_scores, 75)
    print(threshold, threshold2, threshold3)

    visualize_point_cloud_anomaly(point_cloud, anomaly_scores, threshold=threshold3)
    visualize_point_cloud_anomaly(point_cloud, anomaly_scores, threshold=threshold2)
    visualize_point_cloud_anomaly(point_cloud, anomaly_scores, threshold=threshold)
