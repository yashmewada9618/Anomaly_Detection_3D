import os
import open3d as o3d
import numpy as np
from model.teacher import TeacherModel
from model.student import StudentModel
from model.decoder import DecoderNetwork
from torch.utils.data import DataLoader
from loss.loss import ChampherLoss
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.utils import (
    compute_geometric_data,
    knn,
    compute_scaling_factor,
    compute_receptive_field_1,
    compute_receptive_fields,
    get_two_layer_knn,
)
from dataloader.dataloader import ModelNet10

"""
Author: Yash Mewada
Date: 21st May 2024
"""


def train(
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

    teacher = TeacherModel(feature_dim=f_dim).to(device)
    decoder = DecoderNetwork(input_dim=f_dim, output_dim=f_dim).to(device)
    all_train_loss = []
    all_val_loss = []
    optimizer = torch.optim.Adam(
        list(teacher.parameters()) + list(decoder.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    ######### For Normalization #########
    s_factor = 8.904362992404094
    count = 0
    s_factor = 0
    for item in tqdm(training_dataset):
        item = item.to(device)
        s_factor += compute_scaling_factor(item, k)
        count += 1
    s_factor /= count * k
    print(f"[+] S Factor: {s_factor}")
    with open("s_factor.txt", "w") as f:
        f.write(str(s_factor))
    # # exit()
    # s_factor = 9.294224145263849
    best_val_loss = float("inf")
    # chunk_size = 1024
    loss_values = []
    val_losses = []
    teacher.train()
    decoder.train()
    #####################################
    for epoch in tqdm(range(num_epochs)):
        train_loss = []
        # np.random.shuffle(training_data.numpy())
        point_features = []
        epoch_loss = 0.0
        for item in tqdm(training_dataset):
            # print(f"[+] Processing point cloud {i} in epoch {epoch + 1}")
            optimizer.zero_grad()
            # item = item[:, :chunk_size, :]
            item = item.to(device) / s_factor
            B, N, D = item.size()
            knn_points, indices, distances = knn(item, k)
            # print(indices.shape, "Knn points shape")
            # print(distances.shape, "Distances shape")
            # print(knn_points.shape, "Knn Points")
            geom_feat = compute_geometric_data(item, knn_points, distances)
            teacher_out = teacher(item, geom_feat, indices)
            teacher_batch_size, teacher_num_points, teacher_num_features = (
                teacher_out.shape
            )
            # print(teacher_out.shape, "Teacher Out")
            # Randomly sample points
            sampled_indices = torch.randperm(N)[:16].to(device)
            # print(sampled_indices, "Sampled Indices")
            sampled_features = teacher_out[:, sampled_indices, :]
            # print(sampled_features.shape, "Sampled Features")
            decoder_out = decoder(sampled_features).unsqueeze(0)
            norm_recep_fields = get_two_layer_knn(
                item,
                num_samples=16,
                k1=8,
                k2=8,
                sampled_indices=sampled_indices,
                batch_size=128,
            ).to(device)
            # print(norm_recep_fields.shape, "Receptor Out")
            # print(decoder_out.shape, "Decoder Out")

            # # print(chosen_3d_points.shape, "Chosen 3D Points")
            # # norm_recep_fields = get_receptive_field_1(chosen_3d_points, indices, 3).to(
            # #     device
            # # )
            # norm_recep_fields = compute_receptive_fields(
            #     chosen_3d_points, sampled_indices, 3
            # )
            # norm_recep_fields_tensor = torch.stack(norm_recep_fields).to(device)
            # recep_fields = [
            #     rp - torch.mean(rp, dim=0) for rp in norm_recep_fields_tensor
            # ]
            # rp_tensor = torch.stack(recep_fields).to(device)
            # print(norm_recep_fields.shape, "Receptor Out")

            # loss = torch.zeros(1, device=device)
            # for i in range(len(chosen_3d_points)):

            loss = ChampherLoss()(norm_recep_fields, decoder_out)
            # loss /= len(chosen_3d_points)
            point_features.append(teacher_out)
            train_loss.append(loss.item())

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(training_dataset)
        loss_values.append(epoch_loss)
        # print(torch.vstack(point_features).mean(dim=0).shape, "Point Features")
        print(f"Epoch {epoch + 1}, Combined Training Loss: {epoch_loss}")

        val_loss = 0.0
        teacher.eval()
        decoder.eval()

        with torch.no_grad():
            for item in tqdm(validation_dataset):
                # item = item[:, :chunk_size, :]
                item = item.to(device) / s_factor
                knn_points, indices, distances = knn(item, k)
                geom_feat = compute_geometric_data(item, knn_points, distances)
                teacher_out = teacher(item, geom_feat, indices)
                teacher_batch_size, teacher_num_points, teacher_num_features = (
                    teacher_out.shape
                )
                # sampled_indices = torch.randint(
                #     0,
                #     teacher_num_points,
                #     (batch_size, 16),
                #     device=teacher_out.device,
                # )
                # chosen_points = torch.gather(
                #     teacher_out,
                #     1,
                #     sampled_indices.unsqueeze(-1).expand(-1, -1, teacher_out.size(-1)),
                # )
                # chosen_3d_points = torch.gather(
                #     item,
                #     1,
                #     sampled_indices.unsqueeze(-1).expand(-1, -1, item.size(-1)),
                # )
                # decoder_out = decoder(chosen_points).unsqueeze(0)
                # norm_recep_fields = compute_receptive_fields(
                #     chosen_3d_points, sampled_indices, 3
                # )
                # norm_recep_fields_tensor = torch.stack(norm_recep_fields).to(device)
                # recep_fields = [
                #     rp - torch.mean(rp, dim=0) for rp in norm_recep_fields_tensor
                # ]
                # rp_tensor = torch.stack(recep_fields).to(device)
                val_sampled_indices = torch.randperm(N)[:16].to(device)
                # print(sampled_indices, "Sampled Indices")
                val_sampled_features = teacher_out[:, val_sampled_indices, :]
                # print(sampled_features.shape, "Sampled Features")
                val_decoder_out = decoder(val_sampled_features).unsqueeze(0)
                val_norm_recep_fields = get_two_layer_knn(
                    item,
                    num_samples=16,
                    k1=8,
                    k2=8,
                    sampled_indices=val_sampled_indices,
                    batch_size=128,
                ).to(device)

                val_loss += ChampherLoss()(
                    val_norm_recep_fields, val_decoder_out
                ).item()
        val_loss /= len(validation_dataset)
        val_losses.append(val_loss)
        print(f"Epoch {epoch + 1}, Validation Loss: {val_loss}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "teacher": teacher.state_dict(),
                    "decoder": decoder.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "loss": best_val_loss,
                },
                f"weights/{exp_name}.pt",
            )

        teacher.train()
        decoder.train()

    plt.plot(loss_values, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    f_dim = 64
    num_epochs = 250
    lr = 1e-3
    weight_decay = 1e-6
    k = 8
    batch_size = 1
    exp_name = "exp3"
    # Load the dataset
    root_path = (
        "/home/mewada/Anomaly_Detection_3D/dataset_generation/pretrained_dataset/"
    )
    train_ = ModelNet10("train", scale=1, root_dir=root_path)
    train_dataset = torch.utils.data.DataLoader(
        train_, batch_size=batch_size, pin_memory=True, shuffle=True
    )
    val_ = ModelNet10("val", scale=1, root_dir=root_path)
    val_dataset = torch.utils.data.DataLoader(
        train_, batch_size=batch_size, pin_memory=True, shuffle=True
    )
    train(train_dataset, val_dataset, f_dim, exp_name, num_epochs, lr, weight_decay, k)
