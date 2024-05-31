import os
import open3d as o3d
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from model.teacher import TeacherModel
from model.student import StudentModel
from loss.loss import AnomalyScoreLoss
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataloader.dataloader import MVTec3DDataset
from utils.utils import compute_geometric_data, compute_scaling_factor, knn


def get_params(teacher_model, train_data_loader, s, k):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    teacher_model.eval()
    features = []
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

    optimizer = torch.optim.Adam(student.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = GradScaler()

    # For Normalization
    s_factor = sum(
        compute_scaling_factor(item.to(device), k) for item in tqdm(training_dataset)
    ) / len(training_dataset)
    print(f"[+] S Factor: {s_factor}")
    with open("s_factor_mvtec3d.txt", "w") as f:
        f.write(str(s_factor))

    best_val_loss = float("inf")
    losses = []

    mu, sigma = get_params(teacher, training_dataset, s_factor, k)
    np.save(f"weights/{exp_name}_mu.npy", mu.cpu().numpy())
    np.save(f"weights/{exp_name}_sigma.npy", sigma.cpu().numpy())

    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0.0
        teacher.eval()
        student.train()

        for item in tqdm(training_dataset):
            item = item.to(device) / s_factor
            optimizer.zero_grad()

            with autocast():
                knn_points, indices, distances = knn(item, k)
                geom_feat = compute_geometric_data(item, knn_points, distances)
                teacher_out = teacher(item, geom_feat, indices)
                student_out = student(item, geom_feat, indices)
                norm_teacher = (teacher_out - mu) / sigma
                loss = AnomalyScoreLoss()(norm_teacher, student_out)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            epoch_loss += loss.item()

        epoch_loss /= len(training_dataset)
        losses.append(epoch_loss)
        print(f"Epoch {epoch + 1} Loss: {epoch_loss}")

        # Validation step
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
                loss = AnomalyScoreLoss()(norm_teacher, student_out)
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

    root_path = "datasets/MvTec_3D/"
    pretrained_teacher_path = "weights/exp4.pt"

    train_dataset = MVTec3DDataset(num_points=16000, base_dir=root_path, split="train")
    val_dataset = MVTec3DDataset(
        num_points=16000, base_dir=root_path, split="validation"
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    print(f"[+] Training on {len(train_dataset)} samples")
    print(f"[+] Validating on {len(val_dataset)} samples")

    # train(
    #     pretrained_teacher_path,
    #     train_dataloader,
    #     val_dataloader,
    #     f_dim,
    #     exp_name,
    #     num_epochs,
    #     lr,
    #     weight_decay,
    #     k,
    # )
