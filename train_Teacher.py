"""
Author: Yash Mewada
Date: 25th May 2024
Description: This script is used to train the teacher model using the ModelNet10 dataset.
"""

from model.teacher import TeacherModel
from model.decoder import DecoderNetwork
from loss.loss import ChampherLoss
import torch
import matplotlib.pyplot as plt
import open3d as o3d
from tqdm import tqdm
from dataloader.dataloader import ModelNet10

from torch.cuda.amp import (
    GradScaler,
    autocast,
)  # Mixed precision training (In testing phase)
from torch.utils.tensorboard import SummaryWriter
from utils.utils import (
    compute_geometric_data,
    knn,
    compute_scaling_factor,
    get_receptive_fields,
    Colors,
)


def train(
    training_dataset,
    validation_dataset,
    f_dim,
    exp_name,
    num_epochs=250,
    lr=1e-3,
    weight_decay=1e-6,
    k=8,
    chunks=5000,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"{Colors.MAGENTA}[+] Device: {device}{Colors.RESET}")
    torch.cuda.empty_cache()

    teacher = TeacherModel(feature_dim=f_dim).to(device)
    decoder = DecoderNetwork(input_dim=f_dim, output_dim=f_dim).to(device)

    optimizer = torch.optim.Adam(
        list(teacher.parameters()) + list(decoder.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    ######### For Normalization #########
    print(f"[+] Using {chunks} points out of {training_dataset.dataset[0].size()}")
    s_factor = sum(
        compute_scaling_factor(
            item[:, torch.randperm(item.size(1))[:chunks], :].to(device), k
        )
        for item in tqdm(training_dataset)
    ) / len(training_dataset)
    print(f"{Colors.MAGENTA}[+] S Factor: {s_factor}{Colors.RESET}")
    with open(f"s_factors/s_factor_{exp_name}.txt", "w") as f:
        f.write(str(s_factor))

    best_val_loss = float("inf")
    loss_values = []
    val_losses = []
    writer = SummaryWriter(log_dir=f"runs/{exp_name}")
    scaler = GradScaler()

    ############### Run the training loop ######################
    for epoch in tqdm(range(num_epochs)):
        teacher.train()
        decoder.train()
        epoch_loss = 0.0

        for item in tqdm(training_dataset):
            optimizer.zero_grad()
            item = item.to(device) / s_factor

            # randomly sample 8000 points from the point cloud
            temp_indices = torch.randperm(item.size(1))[:chunks]
            item = item[:, temp_indices, :]
            B, N, D = item.size()

            with autocast():  # Mixed precision training
                knn_points, indices, _ = knn(item, k, batch_size=item.size(1))
                geom_feat = compute_geometric_data(item, knn_points)
                teacher_out = teacher(item, geom_feat, indices)

                # Ensure critical parts run in FP32
                with torch.cuda.amp.autocast(enabled=False):
                    teacher_out = teacher_out.float()

                sampled_indices = torch.randperm(N)[:16].to(device)
                sampled_features = teacher_out[:, sampled_indices, :].float()
                decoder_out = decoder(sampled_features).unsqueeze(0).float()
                norm_recep_fields = (
                    get_receptive_fields(
                        item,
                        num_samples=16,
                        k1=8,
                        k2=8,
                        sampled_indices=sampled_indices,
                        batch_size=item.size(1),
                    )
                    .to(device)
                    .float()
                )
                loss = ChampherLoss()(norm_recep_fields, decoder_out)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # loss.backward()
            # optimizer.step()
            # scaler.update()

            epoch_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), max_norm=1.0)

        epoch_loss /= len(training_dataset)
        loss_values.append(epoch_loss)
        writer.add_scalar("Training Loss", epoch_loss, epoch)
        print(
            f"{Colors.CYAN}[+] Epoch: {epoch+1}/{num_epochs}, Loss: {epoch_loss}{Colors.RESET}"
        )

        val_loss = 0.0
        teacher.eval()
        decoder.eval()

        # Validation step
        with torch.no_grad():
            for item in tqdm(validation_dataset):
                item = item.to(device) / s_factor
                temp_indices = torch.randperm(item.size(1))[:chunks]
                item = item[:, temp_indices, :]

                knn_points, indices, _ = knn(item, k)
                geom_feat = compute_geometric_data(item, knn_points)
                teacher_out = teacher(item, geom_feat, indices)

                # Ensure critical parts run in FP32
                with torch.cuda.amp.autocast(enabled=False):
                    teacher_out = teacher_out.float()

                val_sampled_indices = torch.randperm(N)[:16].to(device)
                val_sampled_features = teacher_out[:, val_sampled_indices, :]
                val_decoder_out = decoder(val_sampled_features).unsqueeze(0)
                val_norm_recep_fields = get_receptive_fields(
                    item,
                    num_samples=16,
                    k1=8,
                    k2=8,
                    sampled_indices=val_sampled_indices,
                    batch_size=item.size(1),
                ).to(device)

                val_loss += ChampherLoss()(
                    val_norm_recep_fields, val_decoder_out
                ).item()

        val_loss /= len(validation_dataset)
        val_losses.append(val_loss)
        writer.add_scalar("Validation Loss", val_loss, epoch)

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

    writer.close()
    print(f"{Colors.GREEN}[+] Training completed!{Colors.RESET}")

    # Plot the losses
    plt.plot(loss_values, label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"runs/{exp_name}/train_loss.png")
    plt.show()
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"runs/{exp_name}/val_loss.png")
    plt.show()


if __name__ == "__main__":
    f_dim = 64
    num_epochs = 250
    lr = 1e-3
    weight_decay = 1e-6
    k = 8
    batch_size = 1
    exp_name = "amp1"
    chunks = 5000

    # Load the dataset
    root_path = "datasets/pretrained_dataset/"
    train_ = ModelNet10("train", scale=1, root_dir=root_path)
    train_dataset = torch.utils.data.DataLoader(
        train_, batch_size=batch_size, pin_memory=True, shuffle=True
    )
    val_ = ModelNet10("val", scale=1, root_dir=root_path)
    val_dataset = torch.utils.data.DataLoader(
        val_, batch_size=batch_size, pin_memory=True, shuffle=False
    )

    params = (
        train_dataset,
        val_dataset,
        f_dim,
        exp_name,
        num_epochs,
        lr,
        weight_decay,
        k,
        chunks,
    )

    print(f"[+] Training on {len(train_dataset)} samples")
    print(f"[+] Validating on {len(val_dataset)} samples")
    train(*params)
