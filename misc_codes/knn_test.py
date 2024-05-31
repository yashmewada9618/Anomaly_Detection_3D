import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
import open3d as o3d


def compute_knn_graph(points, k):
    """
    Compute the k-nearest neighbor graph for a set of points.

    Parameters:
    points (numpy.ndarray): The input point cloud of shape (N, D), where N is the number of points and D is the dimension.
    k (int): The number of nearest neighbors to find for each point.

    Returns:
    knn_indices (numpy.ndarray): An array of shape (N, k) containing the indices of the k-nearest neighbors for each point.
    knn_distances (numpy.ndarray): An array of shape (N, k) containing the distances to the k-nearest neighbors for each point.
    """
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(points)
    knn_distances, knn_indices = nbrs.kneighbors(points)

    return knn_indices, knn_distances


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

    knn_points = torch.cat(knn_points, dim=0)
    knn_indices = torch.cat(knn_indices, dim=0)
    knn_dists = torch.cat(knn_dists, dim=0)
    # points = points.unsqueeze(0)

    return knn_points, knn_indices, knn_dists


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
    # Number of points in the point cloud
    num_points = point_cloud.shape[0]

    # Fit the Nearest Neighbors model
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(point_cloud)

    # Find the k-nearest neighbors for each point
    kdistances, indices = nbrs.kneighbors(point_cloud)
    print(1 / np.mean(kdistances[:, 1:]))

    knn_points, knn_indices, distances = knn(torch.tensor(point_cloud).unsqueeze(0), k)

    # Exclude the first nearest neighbor (the point itself)
    distances = distances[:, 1:]

    # Compute the average distance over all points and their k-nearest neighbors
    avg_distance = torch.mean(distances)

    # Calculate the scaling factor
    s = 1 / avg_distance

    return s


def get_two_layer_knn(points, num_samples, k1, k2, batch_size=50):
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
    _, num_points, num_dimensions = points.size()

    # Randomly sample points
    sampled_indices = torch.randperm(num_points)[:num_samples]

    # Get first layer of k-nearest neighbors
    _, knn1_indices, _ = knn(points, k1, batch_size)
    # exit()

    # Initialize the output tensor
    output = torch.zeros((num_samples, k1 * k2, num_dimensions))

    for i in range(num_samples):
        first_layer_neighbors = knn1_indices[sampled_indices[i]]

        second_layer_neighbors_list = []
        for neighbor in first_layer_neighbors:
            _, second_layer_knn_indices, _ = knn(points, k2, batch_size)
            second_layer_neighbors = second_layer_knn_indices[neighbor].squeeze(0)
            second_layer_neighbors_list.append(second_layer_neighbors)

        second_layer_neighbors_list = torch.stack(second_layer_neighbors_list)
        second_layer_neighbors_list = second_layer_neighbors_list.reshape(-1)
        second_layer_points = points[:, second_layer_neighbors_list]

        output[i] = second_layer_points

    # Normalize by subtracting the mean of the receptive field
    mean_points = output.mean(dim=2, keepdim=True)
    output -= mean_points

    return output


# Example usage
if __name__ == "__main__":
    # Generate some random points (for example, 4000 points in 3D space)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pcd = o3d.io.read_point_cloud("datasets/pretrained_dataset/train/bed_0254_1.pcd")
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points))
    o3d.visualization.draw_geometries([pcd])

    points = torch.tensor(np.asarray(pcd.points))

    points = points.to(device)
    points = points.unsqueeze(0)  # Add a batch dimension
    print("Points:", points)

    # k = 8  # Number of nearest neighbors to compute
    # _, knn_indices, knn_distances = knn(points, k, 128)

    # # Print the results
    # print("k-NN Indices:\n", knn_indices[0])
    print("Points shape:", points.shape)
    output = get_two_layer_knn(points, num_samples=28, k1=8, k2=8, batch_size=128)
    print("Output shape:", output)

    # Visualize the receptive fields points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(output[0].cpu().numpy())
    o3d.visualization.draw_geometries([pcd])
