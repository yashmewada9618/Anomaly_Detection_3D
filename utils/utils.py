"""
Author: Yash Mewada
Date: 21st May 2024
"""

import torch
from tqdm import tqdm
import numpy as np
from scipy.spatial import distance


def knn_function(points, k):
    """
    Find the k nearest neighbors for each point in the point cloud.

    Args:
        points (torch.Tensor): Input point cloud of shape (1, 1024, 3).
        k (int): Number of nearest neighbors.

    Returns:
        torch.Tensor: KNN points of shape (1, 1024, k, 3).
        torch.Tensor: KNN indices of shape (1, 1024, k).
        torch.Tensor: KNN distances of shape (1, 1024, k).
    """
    batch_size, num_points, _ = points.shape
    distances = torch.cdist(points, points, p=2)  # Compute pairwise distances
    knn_dists, knn_indices = distances.topk(
        k, largest=False
    )  # Find k smallest distances
    knn_points = points[0, knn_indices[0]].unsqueeze(0)  # Gather KNN points

    return knn_points, knn_indices, knn_dists


def knn(points, k, batch_size=50):
    """
    Calculate k-nearest neighbors for each point in a batch of point clouds in chunks.

    Args:
    - points (torch.Tensor): Input point cloud tensor of shape (num_points, num_dimensions).
    - k (int): Number of nearest neighbors to find.
    - batch_size (int): Number of points to process at a time.

    Returns:
    - knn_points (torch.Tensor): Nearest neighbor points tensor of shape (num_points, k, num_dimensions).
    - knn_indices (torch.Tensor): Nearest neighbor indices tensor of shape (num_points, k).
    - knn_dists (torch.Tensor): Pairwise distance tensor of shape (num_points, k).
    """
    _, num_points, num_dimensions = points.size()
    points = points.squeeze(0)
    knn_points = []
    knn_indices = []
    knn_dists = []

    # Process points in batches
    for i in range(0, num_points, batch_size):
        end = min(i + batch_size, num_points)
        batch_points = points[i:end]

        # Calculate pairwise distances for the batch
        x1 = batch_points.unsqueeze(1).expand(-1, num_points, -1)
        x2 = points.unsqueeze(0).expand(end - i, -1, -1)
        diff = x2 - x1
        norm = torch.norm(diff, dim=-1)

        # Get k-nearest neighbors
        batch_knn_dists, batch_knn_indices = torch.topk(
            norm, k=k + 1, largest=False, sorted=True
        )
        batch_knn_dists = batch_knn_dists[
            :, 1:
        ]  # Exclude the first distance (distance to itself)
        batch_knn_indices = batch_knn_indices[
            :, 1:
        ]  # Exclude the first index (distance to itself)

        # Ensure indices are within bounds
        valid_indices_mask = (batch_knn_indices >= 0) & (batch_knn_indices < num_points)
        batch_knn_indices = batch_knn_indices * valid_indices_mask

        batch_knn_points = torch.gather(
            points.unsqueeze(0).expand(end - i, num_points, num_dimensions),
            1,
            batch_knn_indices.unsqueeze(-1).expand(end - i, k, num_dimensions),
        )

        knn_points.append(batch_knn_points)
        knn_indices.append(batch_knn_indices)
        knn_dists.append(batch_knn_dists)

        del x1, x2, diff, norm
        # torch.cuda.empty_cache()

    knn_points = torch.cat(knn_points, dim=0).unsqueeze(0)
    knn_indices = torch.cat(knn_indices, dim=0).unsqueeze(0)
    knn_dists = torch.cat(knn_dists, dim=0).unsqueeze(0)
    # points = points.unsqueeze(0)

    return knn_points, knn_indices, knn_dists


def get_two_layer_knn(points, num_samples, k1, k2, sampled_indices, batch_size=50):
    """
    Get 2-layer k-nearest neighbors for randomly sampled points from a point cloud.

    Args:
    - points (torch.Tensor): Input point cloud tensor of shape (num_points, num_dimensions).
    - num_samples (int): Number of points to sample from the point cloud.
    - k1 (int): Number of nearest neighbors for the first layer.
    - k2 (int): Number of nearest neighbors for the second layer.
    - batch_size (int): Number of points to process at a time for KNN.

    Returns:
    - output (torch.Tensor): Output tensor of shape (num_samples, k1 * k2, num_dimensions).
    """
    B, num_points, num_dimensions = points.size()

    # Get first layer of k-nearest neighbors
    _, knn1_indices, _ = knn(points, k1, batch_size)
    # exit()

    # Initialize the output tensor
    output = torch.zeros((B, num_samples, k1 * k2, num_dimensions))

    for i in range(num_samples):
        first_layer_neighbors = knn1_indices[:, sampled_indices[i]]

        second_layer_neighbors_list = []
        for neighbor in first_layer_neighbors:
            _, second_layer_knn_indices, _ = knn(points, k2, batch_size)
            second_layer_neighbors = second_layer_knn_indices[:, neighbor].squeeze(0)
            second_layer_neighbors_list.append(second_layer_neighbors)

        second_layer_neighbors_list = torch.stack(second_layer_neighbors_list)
        second_layer_neighbors_list = second_layer_neighbors_list.reshape(-1)
        second_layer_points = points[:, second_layer_neighbors_list]

        # Normalize by subtracting the mean of the receptive field
        mean_point = torch.mean(second_layer_points, dim=0)
        normalized_receptive_field = second_layer_points - mean_point

        output[0][i] = normalized_receptive_field

    return output


# def knn(x, k=8):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     batch_size, num_points, num_features = x.shape
#     x = x.to(device)

#     knn_indices_list = []
#     knn_points_list = []
#     pairwise_distances_list = []
#     chunk_size = 1

#     for batch_idx in range(0, batch_size, chunk_size):
#         x_batch = x[batch_idx].unsqueeze(0)  # Get the current batch
#         x_batch_1 = x[batch_idx].unsqueeze(1)
#         diff = x_batch_1 - x_batch
#         norm = torch.norm(diff, dim=2)
#         _, knn_indices = torch.topk(norm, k=k + 1, largest=False, sorted=True)
#         knn_indices = knn_indices[:, 1:]  # Exclude the first index (distance to itself)
#         knn_points = torch.gather(
#             x_batch.unsqueeze(1).expand(1, num_points, num_points, num_features),
#             2,
#             knn_indices.unsqueeze(-1).expand(1, num_points, k, num_features),
#         )

#         # Move to CPU to free up GPU memory
#         knn_indices = knn_indices.cpu()
#         knn_points = knn_points.cpu()
#         norm = norm.cpu()

#         knn_indices_list.append(knn_indices)
#         knn_points_list.append(knn_points)
#         pairwise_distances_list.append(norm)
#         print(f"Batch {batch_idx + 1}/{batch_size} processed.")

#         # Clear GPU cache
#         del x_batch, norm, knn_indices, knn_points, diff, x_batch_1
#         torch.cuda.empty_cache()

#     knn_indices = torch.cat(knn_indices_list, dim=0)
#     knn_points = torch.cat(knn_points_list, dim=0)
#     pairwise_distances = torch.cat(pairwise_distances_list, dim=0)

#     return knn_points, knn_indices, pairwise_distances


def compute_geometric_data(x, knn_points, pairwise_distances, k=8):
    differences = x.unsqueeze(2) - knn_points
    l2_norms = torch.norm(differences, dim=-1, keepdim=True)
    output = torch.cat([differences, l2_norms], dim=-1)
    return output


def compute_s_value(all_data, knn_points, knn_indices, k):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_data = all_data.to(device)
    knn_indices = knn_indices.to(device)

    num_pointclouds, num_points, _ = all_data.shape

    # Flatten the indices for easier gathering
    flat_indices = knn_indices.view(-1, k)

    # Gather the k-nearest neighbors for each point
    knn_points = all_data.view(-1, 3)[flat_indices]

    # Expand all_data for broadcasting
    all_data_expanded = all_data.unsqueeze(2).expand(num_pointclouds, num_points, k, 3)

    # Calculate the pairwise distances
    distances = torch.norm(
        all_data_expanded - knn_points.view(num_pointclouds, num_points, k, 3), dim=-1
    )

    # Calculate the normalization factor s
    s = distances.mean().item()

    return s


def compute_scaling_factor(point_cloud, k):
    """
    Calculate the scaling factor s for normalizing the point cloud.

    Args:
    - point_cloud (numpy.ndarray): Input point cloud array of shape (N, D),
      where N is the number of points and D is the dimensionality of each point.
    - k (int): Number of nearest neighbors to consider.

    Returns:
    - s (float): Scaling factor for normalizing the point cloud.
    """
    _, _, distances = knn(point_cloud, k)

    # Exclude the first nearest neighbor (the point itself)
    distances = distances[:, 1:]

    # Compute the average distance over all points and their k-nearest neighbors
    avg_distance = torch.mean(distances).item()

    # Calculate the scaling factor
    s = 1 / avg_distance

    return s


def knn_dixit(x, k):
    xx = torch.sum(x**2, dim=2, keepdim=True)
    pairwise_distance = xx + xx.transpose(2, 1) - 2 * torch.matmul(x, x.transpose(2, 1))
    idx = pairwise_distance.topk(k=k, dim=-1, largest=False)[1]
    return idx


def get_avg_distance(points):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    avg_distances = []
    for point in tqdm(points):
        point = point.to(device)[:, :4000, :]
        indices = knn_dixit(point, 8)
        point = point.squeeze(0)
        for p in range(point.shape[0]):
            neighbors = point[indices[0, p, :]]
            dist = torch.norm(point[p] - neighbors, dim=1)
            avg_distances.extend(dist.tolist())
    print(torch.mean(torch.tensor(avg_distances)).item())
    print(sum(avg_distances) / len(avg_distances))

    return sum(avg_distances) / len(avg_distances)


def receptive_field(points, L, knn_function, k=8):
    """
    Compute the receptive fields for each point in the point cloud.

    Args:
        points (torch.Tensor): Input point cloud of shape (1, 1024, 3).
        L (int): Number of LFA blocks.
        k (int): Number of points in the receptive field.
        knn_function (callable): Function that takes in points and k, and returns knn_points, knn_indices, knn_dists.

    Returns:
        torch.Tensor: Receptive fields of shape (1, 1024, k, 3).
    """
    batch_size, num_points, _ = points.shape
    assert batch_size == 1, "Batch size must be 1 for this function."

    points = points.squeeze(0)  # Remove batch dimension for easier processing

    # Initialize the receptive fields with the points themselves
    receptive_fields = points.unsqueeze(1).expand(-1, k, -1)  # Shape: (1024, 64, 3)

    for _ in range(L):
        knn_points_list = []
        for i in range(num_points):
            knn_points, _, _ = knn_function(points.unsqueeze(0), k)
            knn_points_list.append(knn_points.squeeze(0))

        # Concatenate all KNN points and retain only unique points
        knn_points_combined = torch.cat(knn_points_list, dim=1)
        unique_knn_points, indices = torch.unique(
            knn_points_combined, dim=1, return_inverse=True
        )

        # Take the first k unique points
        if unique_knn_points.shape[1] > k:
            unique_knn_points = unique_knn_points[:, :k, :]

        receptive_fields = unique_knn_points

    # Compute the mean of each receptive field and subtract it
    mean_receptive_fields = receptive_fields.mean(dim=1, keepdim=True)
    receptive_fields -= mean_receptive_fields

    # Add back the batch dimension
    receptive_fields = receptive_fields.unsqueeze(0)

    return receptive_fields


def compute_receptive_fields(points, knn_idx, L):
    B, N, _ = points.shape
    receptive_fields = []
    for i in range(N):
        rf = set([i])
        for l in range(1, L + 1):
            next_hop = set()
            for q_idx in rf:
                if isinstance(q_idx, int) and q_idx >= N:
                    continue
                knn_q = knn_idx[:, q_idx]
                knn_q = knn_q.long()
                next_hop.update(knn_q.tolist())
            rf.update(next_hop)
        rf = [idx for idx in rf if isinstance(idx, int) and idx < N]
        receptive_fields.append(points[:, rf, :])
    max_rf_size = max(len(rf) for rf in receptive_fields)
    padded_fields = [
        torch.nn.functional.pad(rf, (0, 0, 0, max_rf_size - rf.size(1), 0, 0))
        for rf in receptive_fields
    ]
    return padded_fields


def compute_receptive_field_1(indices, selected_indices, data_points, layers=8):
    """
    compute_receptive_field(indices, selected_indices, layers=8) computes the receptive field
    for selected points using k-nearest neighbors.

    Args:
    - indices (torch tensor): Shape [Number of point clouds, num_points, k] --> Indices of the nearest neighbors.
    - selected_indices (torch.Tensor): Shape [Number of point clouds, num_selected] --> Indices of the points to reconstruct.
    - layers (int): Number of layers to expand the receptive field

    Returns:
    - R_p (torch tensor): Shape [batch_size, num_selected, variable_neighbors] --> Receptive fields calculated for l layers.
    """
    batch_size, num_points, k = indices.shape
    _, num_selected = selected_indices.shape

    # Initialize R_p with the nearest neighbors for each selected point
    R_p = indices[
        torch.arange(batch_size).unsqueeze(-1), selected_indices
    ]  # Shape [batch_size, num_selected, k]

    for i in range(layers - 1):
        new_neighbors = []

        for batch_idx in range(batch_size):
            batch_new_neighbors = []
            for point_idx in range(num_selected):
                current_neighbors = R_p[batch_idx, point_idx]

                # Get the neighbors of the neighbors
                expanded_neighbors = indices[batch_idx, current_neighbors]
                expanded_neighbors = expanded_neighbors.flatten()

                # Combine current neighbors with new neighbors
                all_neighbors = torch.cat([current_neighbors, expanded_neighbors])

                # Filter out duplicates
                unique_neighbors = torch.unique(all_neighbors)
                batch_new_neighbors.append(unique_neighbors)

            new_neighbors.append(batch_new_neighbors)

        # Pad to have a consistent side
        padded_new_neighbors = []
        max_neighbors = max(neigh.size(0) for batch in new_neighbors for neigh in batch)
        for batch in new_neighbors:
            padded_batch = [
                torch.nn.functional.pad(
                    neigh, (0, max_neighbors - neigh.size(0)), value=-1
                )
                for neigh in batch
            ]
            padded_new_neighbors.append(torch.stack(padded_batch))

        # Update R_p with new neighbors
        R_p = torch.stack(padded_new_neighbors)

    receptive_fields = R_p
    batch_size, num_points, variable_neighbors = receptive_fields.shape
    _, _, num_features = data_points.shape

    valid_mask = receptive_fields != -1
    valid_indices = torch.where(
        valid_mask, receptive_fields, torch.zeros_like(receptive_fields)
    )

    # Gather valid data points
    valid_data_points = torch.gather(
        data_points.unsqueeze(2).expand(-1, -1, variable_neighbors, -1),
        1,
        valid_indices.unsqueeze(-1).expand(-1, -1, -1, num_features),
    )

    # Mask out the invalid points
    valid_data_points = valid_data_points * valid_mask.unsqueeze(-1).float()

    # Compute the sum of valid data points and the count of valid points
    sum_valid_points = valid_data_points.sum(dim=2)
    count_valid_points = valid_mask.sum(dim=2, keepdim=True).float()

    mean_points = sum_valid_points / count_valid_points
    mean_points = mean_points.unsqueeze(
        2
    )  # Shape [batch_size, num_points, 1, num_features]

    # Normalize the receptive fields
    normalized_receptive_fields = valid_data_points - mean_points

    return mean_points, normalized_receptive_fields


def scale_dataset(x, s):
    scaled_data = []
    for item in x:
        scaled_item = item / s
        scaled_data.append(scaled_item)
    return torch.stack(scaled_data)


def get_rf_speed(closest_indices, chosen_idxs, num_decoded_points=1024, max_iters=8):
    """
    Compute the receptive fields for each chosen index based on closest_indices.

    Args:
        closest_indices (torch.Tensor): Precomputed indices of nearest neighbors of shape (1, N, k).
        chosen_idxs (torch.Tensor): Indices of points for which to compute receptive fields of shape (1, num_chosen, num_neighbors).
        num_decoded_points (int): Number of points to include in the receptive field.
        max_iters (int): Maximum number of iterations to expand the receptive field.

    Returns:
        list[torch.Tensor]: List of tensors containing receptive field indices for each chosen index.
    """
    N = closest_indices.shape[1]
    receptive_fields = []
    for chosen_idx in chosen_idxs.squeeze(0):  # Squeeze batch dimension
        receptive_field = {chosen_idx.item()}
        prev_receptive_field = receptive_field
        count = 0
        while len(receptive_field) < num_decoded_points and count < max_iters:
            new_receptive_field = set()
            for idx in prev_receptive_field:
                if idx < N:  # Check that idx is within bounds
                    new_receptive_field.update(closest_indices[0, idx].tolist())
            receptive_field.update(new_receptive_field)
            prev_receptive_field = new_receptive_field
            count += 1
        receptive_field = list(receptive_field)[:num_decoded_points]
        receptive_fields.append(torch.tensor(receptive_field, dtype=torch.long))

    return receptive_fields


def get_receptive_fields(
    points, closest_indices, chosen_idxs, num_decoded_points=1024, max_iters=8
):
    """
    Compute the receptive fields for each chosen index based on points and closest_indices.

    Args:
        points (torch.Tensor): Input point cloud of shape (1, N, 3).
        closest_indices (torch.Tensor): Precomputed indices of nearest neighbors of shape (1, N, k).
        chosen_idxs (torch.Tensor): Indices of points for which to compute receptive fields of shape (1, num_chosen, num_neighbors).
        num_decoded_points (int): Number of points to include in the receptive field.
        max_iters (int): Maximum number of iterations to expand the receptive field.

    Returns:
        torch.Tensor: Receptive fields of shape (1, num_chosen, num_decoded_points, 3).
    """
    receptive_fields_indices = get_rf_speed(
        closest_indices, chosen_idxs, num_decoded_points, max_iters
    )
    receptive_fields = [points[0, rf] for rf in receptive_fields_indices]
    return torch.stack(receptive_fields).unsqueeze(0).view(-1, 16, 64, 3)


# # Example usage
# points = torch.rand(1, 16000, 3)  # Shape (B, N, 3)
# closest_indices = torch.randint(0, 16000, (1, 16000, 8))  # Shape (B, N, k)
# chosen_idxs = torch.randint(0, 16000, (1, 16, 64))  # Shape (B, num_chosen, 64)

# receptive_fields = get_receptive_fields(points, closest_indices, chosen_idxs)
# print(receptive_fields.shape)  # Should print: torch.Size([1, 16, 1024, 3])


def receptive_field_(points, k, L):
    """
    Compute the receptive field for each point in the input point cloud.

    Args:
        points (torch.Tensor): A tensor of shape (B, N, 3) containing the 3D coordinates of the points.
        k (int): The number of nearest neighbors to consider.
        L (int): The number of layers in the network.

    Returns:
        list: A list of length B, where each element is a list of length N containing sets of indices
              representing the receptive field for each point in the batch.
    """
    B, N, _ = points.shape
    receptive_fields = [
        [set([i]) for i in range(N)] for _ in range(B)
    ]  # Initialize with the points themselves

    for _ in range(L):
        new_receptive_fields = [[set() for _ in range(N)] for _ in range(B)]
        for b in range(B):
            for i in range(N):
                print(points[b].shape)
                knn_indices = knn(points[b], k)[i]
                for j in knn_indices:
                    new_receptive_fields[b][i].update(receptive_fields[b][j])
        receptive_fields = new_receptive_fields

    return receptive_fields


def knn_(points, k):
    """
    Compute the k nearest neighbors for each point in the input point cloud.

    Args:
        points (torch.Tensor): A tensor of shape (N, 3) containing the 3D coordinates of the points.
        k (int): The number of nearest neighbors to consider.

    Returns:
        torch.Tensor: A tensor of shape (N, k) containing the indices of the k nearest neighbors for each point.
    """
    # Implement your k-nearest neighbor algorithm here
    # For simplicity, we'll assume a naive approach
    distances = torch.cdist(points, points, p=2)
    knn_indices = distances.argsort()[:, 1 : k + 1]
    return knn_indices


# def get_receptive_field_1(sampled_indices, points, k, num_lfa_blocks):
#     """
#     Computes the receptive field for each sampled point in a list of sampled indices.

#     Args:
#         sampled_indices (list of int): The list of indices of sampled points for which to compute the receptive fields.
#         points (torch.Tensor): Input point cloud tensor of shape (num_points, num_dimensions).
#         k (int): Number of nearest neighbors to find.
#         num_lfa_blocks (int): The number of LFA (Local Feature Aggregation) blocks in the network.

#     Returns:
#         list of set: A list where each set contains the indices representing the receptive field of each sampled point.
#     """
#     # Convert points to batch format expected by knn function
#     points = points.unsqueeze(2)  # Add batch dimension

#     receptive_fields = []

#     for point_idx in sampled_indices:
#         receptive_field = set([point_idx])
#         current_layer = set([point_idx])

#         for i in range(num_lfa_blocks):
#             print(i, "Iteration")
#             next_layer = set()
#             current_layer_list = list(current_layer)

#             # Select points corresponding to current layer indices
#             print(points.shape, current_layer_list, "Points")
#             if i == 0:
#                 current_layer_points = points[:, current_layer_list[0].tolist(), :]
#             else:
#                 current_layer_points = points[:, current_layer_list, :]

#             # Get k-nearest neighbors for points in current_layer
#             current_layer_points = current_layer_points.squeeze(2)
#             print(current_layer_points.shape, "Current Layer Points")
#             _, knn_indices, _ = knn(current_layer_points, k)

#             # Update next_layer with the neighbors
#             for neighbors in knn_indices.squeeze():
#                 next_layer.update(neighbors.tolist())

#             # Update receptive field and current layer
#             receptive_field.update(next_layer)
#             current_layer = next_layer
#         receptive_fields.append(receptive_field)

#     return receptive_fields


def get_receptive_field_1(sampled_indices, points, k, num_lfa_blocks):
    """
    This function computes the receptive field R(p) for a set of sampled points in a point cloud.

    Args:
        sampled_indices (list of int): The list of indices of sampled points for which to compute the receptive fields.
        points (torch.Tensor): Input point cloud tensor of shape (num_points, num_dimensions).
        k (int): Number of nearest neighbors to find.
        num_lfa_blocks (int): The number of LFA blocks in the network.

    Returns:
        torch.tensor: A tensor where each row contains the indices representing the receptive field of a sampled point.
    """
    print(points.shape, "Points Before")
    # points = points.unsqueeze(2)
    # print(points.shape, "Points After")
    # Initialize receptive fields with sampled points themselves
    receptive_fields = [indice for indice in sampled_indices]

    # Iterate over LFA blocks to progressively expand receptive fields
    for _ in range(num_lfa_blocks):
        new_receptive_fields = []
        for point_idx in receptive_fields:
            # Get nearest neighbors for the current point
            # new_points = [points[x] for x in point_idx.tolist()]
            print(points.shape, point_idx.tolist(), "Points")
            _, neighbor_indices, _ = knn(
                points[:, point_idx, :], k
            )  # Use range for batch indices
            # Add nearest neighbors of current point to receptive field
            new_receptive_fields.extend(neighbor_indices.tolist())

        print(new_receptive_fields, "New Receptive Fields")
        print(np.shape(new_receptive_fields), "New Receptive Fields")

        # Remove duplicates and update receptive fields
        receptive_fields = list(set(new_receptive_fields))

    # Convert receptive fields to tensor
    return torch.tensor(list(receptive_fields))


def get_receptive_field_1(points, knn_idx, L):
    B, N, _ = points.shape
    receptive_fields = []
    for i in range(N):
        rf = set([i])
        for l in range(1, L + 1):
            next_hop = set()
            for q_idx in rf:
                if isinstance(q_idx, int) and q_idx >= N:
                    continue
                knn_q = knn_idx[0, q_idx, :]
                knn_q = knn_q.long()
                next_hop.update(knn_q.tolist())
            rf.update(next_hop)
        rf = [idx for idx in rf if isinstance(idx, int) and idx < N]
        receptive_fields.append(points[:, rf, :])
    max_rf_size = max(len(rf) for rf in receptive_fields)
    padded_fields = [
        torch.nn.functional.pad(rf, (0, 0, 0, max_rf_size - rf.size(1), 0, 0))
        for rf in receptive_fields
    ]
    padded_fields_tensor = torch.stack(padded_fields)
    return padded_fields_tensor


def compute_local_receptive_field(point_cloud, knn_indices, points, sampled_indices, k):
    """
    compute_local_receptive_field calculates the local receptive field for a given set of sampled points.

    Inputs:
    - points (torch.Tensor): Tensor of shape [num_points, 3] representing the input point cloud.
    - sampled_indices (torch.Tensor): Tensor of shape [num_sampled_points] representing the indices of the sampled points.
    - k (int): The number of nearest neighbors to consider for each point.

    Outputs:
    - receptive_field (torch.Tensor): Tensor of shape [num_sampled_points, k**2, 3] representing the receptive field for each sampled point.
    """
    print(point_cloud.shape, points.shape, sampled_indices.shape, k, "Inputs")
    num_points = points.size(0)
    B, num_sampled_points = sampled_indices.size()
    # sampled_indices = sampled_indices.detach().cpu().numpy()

    def knn_rp(pc, k):
        # Step 1: Find the k-nearest neighbors of the sampled point
        D = torch.tensor(distance.squareform(torch.pdist(pc).detach().cpu().numpy()))
        closest_indices = torch.argsort(D, axis=1)
        closest_points = point_cloud[closest_indices[:, 1 : k + 1]]
        print(closest_points.shape, closest_indices.shape, "Closest Points and Indices")
        return closest_points, closest_indices[:, 1 : k + 1]

    # Initialize the receptive field tensor
    receptive_field = torch.zeros(
        (B, num_sampled_points, k**2, 3), device=points.device
    )

    rfs = []

    for i, sampled_idx in enumerate(sampled_indices):

        knn_l1_points, knn_indices_l1 = knn_rp(points.squeeze(0), k)
        print(
            knn_l1_points.shape, knn_indices_l1.shape, "KNN Points and Indices for L1"
        )
        # knn_points_l1, knn_indices_l1, _ = knn(points, k=k)
        # Step 2: For each of these k-nearest neighbors, find their k-nearest neighbors
        all_knn_indices_l2 = []
        for j in range(k):
            knn_l2_points, knn_indices_l2 = knn_rp(knn_l1_points[j], k)
            all_knn_indices_l2.append(knn_indices_l2)
            print(
                knn_l1_points[j].shape,
                knn_indices_l2.shape,
                "KNN Points and Indices for L2",
            )
        print(len(all_knn_indices_l2), "All KNN Indices L2")

        # Flatten the list of neighbor indices and ensure uniqueness
        all_knn_indices_l2 = torch.unique(torch.cat(all_knn_indices_l2))

        # Ensure the receptive field has exactly k**2 points (in case of duplicates, select the first k**2 points)
        all_knn_indices_l2 = all_knn_indices_l2[: k**2]

        # Ensure the receptive field has exactly k**2 points
        if all_knn_indices_l2.size(0) < k**2:
            # If less than k**2, pad with repeated elements
            all_knn_indices_l2 = torch.cat(
                [
                    all_knn_indices_l2,
                    knn_indices_l1,
                ]
            )
        else:
            # If more than k**2, trim to k**2
            all_knn_indices_l2 = all_knn_indices_l2[: k**2]

        # Store the points corresponding to these indices in the receptive field tensor
        receptive_field[i] = points[all_knn_indices_l2]
    # exit()

    return receptive_field
