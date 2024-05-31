import os
import open3d as o3d
import numpy as np
import torch.utils
from tqdm import tqdm
from model.teacher import TeacherModel
from model.student import StudentModel
from model.decoder import DecoderNetwork
from torch.utils.data import DataLoader
from loss.loss import ChampherLoss, AnomalyScoreLoss
import torch.nn.functional as F
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from dataloader.dataloader import MvTec3D, ModelNet10, MVTec3DDataset
from utils.utils import (
    compute_geometric_data,
    compute_scaling_factor,
    knn,
    compute_s_value,
    receptive_field,
    get_receptive_fields,
    receptive_field_,
    get_receptive_field_1,
    compute_local_receptive_field,
    compute_receptive_field_1,
    get_avg_distance,
)


def get_params(teacher_model, train_data_loader, s):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    teacher_model.eval()
    features = []
    chunk_size = 1024
    with torch.no_grad():
        for item in tqdm(train_data_loader):
            item = item.to(device) / s
            knn_points, indices, distances = knn(item, k)
            geom_feat = compute_geometric_data(item, knn_points, distances)
            teacher_out = teacher_model(item, geom_feat, indices)
            features.append(teacher_out)
    features = torch.cat(features, dim=0)
    return features.mean(dim=0), features.std(dim=0)


def train(
    pretrained_teacher_path,
    training_dataset,
    validation_dataset,
    f_dim,
    exp_name,
    num_epochs=250,
    lr=1e-3,
    weight_decay=1e-6,
    k=8,
):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    print(f"[+] Using {device} device")
    teacher = TeacherModel(feature_dim=f_dim).to(device)
    student = StudentModel(f_dim=f_dim).to(device)
    # Reser student params
    student.apply(
        lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None
    )
    if os.path.exists(pretrained_teacher_path):
        teacher.load_state_dict(torch.load(pretrained_teacher_path)["teacher"])
        print(f"[+] Loaded pretrained teacher model from {pretrained_teacher_path}")
    else:
        raise FileNotFoundError(
            f"[!] Pretrained teacher model not found at {pretrained_teacher_path}"
        )
    optimizer = torch.optim.Adam(
        list(student.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    ######### For Normalization #########
    s_factor = 19.104289397122123
    # count = 0
    # s_factor = 0
    # for item in tqdm(training_dataset):
    #     item = item.to(device)
    #     s_factor += compute_scaling_factor(item, k)
    #     count += 1
    # s_factor /= count * k
    # print(f"[+] S Factor: {s_factor}")
    # with open("s_factor_mvtec3d.txt", "w") as f:
    #     f.write(str(s_factor))

    best_val_loss = float("inf")
    losses = []

    # mu, sigma = get_params(teacher, training_dataset, s_factor)
    # np.save(f"weights/{exp_name}_mu_1024.npy", mu.cpu().numpy())
    # np.save(f"weights/{exp_name}_sigma_1024.npy", sigma.cpu().numpy())

    mu = torch.tensor(np.load(f"weights/{exp_name}_mu_1024.npy")).to(device)
    sigma = torch.tensor(np.load(f"weights/{exp_name}_sigma_1024.npy")).to(device)
    # print(f"[+] Mean: {mu}, Sigma: {sigma}")
    count = 0
    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0.0
        teacher.eval()
        student.train()

        for item in tqdm(training_dataset):
            item = item.to(device)
            optimizer.zero_grad()
            knn_points, indices, distances = knn(item, k)
            geom_feat = compute_geometric_data(item, knn_points, distances)
            teacher_out = teacher(item, geom_feat, indices)
            student_out = student(item, geom_feat, indices)
            print(f"Teacher: {indices}")

            norm_teacher = (teacher_out - mu) / sigma
            # norm_teacher = teacher_out
            loss = AnomalyScoreLoss()(norm_teacher, student_out)
            # print(f"Loss: {teacher_out}")
            # print(f" MSE loss: {student_out}")
            loss1 = F.mse_loss(norm_teacher, student_out)
            print(f"Loss: {loss.item()}, MSE loss: {loss1.item()}")
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            count += 1
        epoch_loss /= len(training_dataset)
        losses.append(epoch_loss)
        print(f"Epoch {epoch + 1} Loss: {epoch_loss}")

        val_loss = 0.0
        student.eval()

        with torch.no_grad():
            for item in validation_dataset:
                item = item.to(device) / s_factor
                knn_points, indices, distances = knn(item, k)
                geom_feat = compute_geometric_data(item, knn_points, distances)
                teacher_out = teacher(item, geom_feat, indices)
                student_out = student(item, geom_feat, indices)
                norm_teacher = (teacher_out - mu) / sigma
                # norm_teacher = teacher_out
                loss1 = F.mse_loss(norm_teacher, student_out)
                loss = AnomalyScoreLoss()(norm_teacher, student_out)
                print(f"Validation Loss: {loss.item()}, MSE Loss: {loss1.item()}")
                val_loss += loss.item()
        val_loss /= len(validation_dataset)
        print(f"Epoch {epoch + 1} Validation Loss: {val_loss}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "student": student.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "loss": best_val_loss,
                },
                f"weights/{exp_name}.pt",
            )
            print(f"[+] Saved best model at weights/{exp_name}.pt")

    plt.plot(losses)
    plt.title("Training Loss for Student Model")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(f"{exp_name}.png")
    plt.show()


if __name__ == "__main__":
    f_dim = 64
    num_epochs = 100
    lr = 1e-3
    weight_decay = 1e-5
    k = 8
    batch_size = 1
    exp_name = "exp_student_v2_t1"

    # root_path = "/home/mewada/pivot_submission/Anomaly_Detection_3D/datasets/mvtec_point_clouds/"
    root_path = "/home/mewada/pivot_submission/Anomaly_Detection_3D/datasets/MvTec_3D/"
    pretrained_teacher_path = (
        "/home/mewada/pivot_submission/Anomaly_Detection_3D/weights/exp3.pt"
    )

    # train_ = MvTec3D("train", scale=1, root_dir=root_path)
    # train_dataloader = torch.utils.data.DataLoader(
    #     train_, batch_size=batch_size, pin_memory=True, shuffle=True
    # )
    # val_ = MvTec3D("validation", scale=1, root_dir=root_path)
    # val_dataloader = torch.utils.data.DataLoader(
    #     train_, batch_size=batch_size, pin_memory=True, shuffle=True
    # )

    train_dataset = MVTec3DDataset(num_points=1024, base_dir=root_path, split="train")
    val_dataset = MVTec3DDataset(
        num_points=1024, base_dir=root_path, split="validation"
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )
    train(
        pretrained_teacher_path,
        train_dataloader,
        val_dataloader,
        f_dim,
        exp_name,
        num_epochs,
        lr,
        weight_decay,
        k,
    )
