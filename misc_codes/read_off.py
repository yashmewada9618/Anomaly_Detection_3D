import copy
from matplotlib import scale
from matplotlib.scale import scale_factory
import numpy as np
import matplotlib.pyplot as plt
import pymeshlab
import open3d as o3d
from sympy import fps

ms = pymeshlab.MeshSet()
ms.load_new_mesh(
    "/home/mewada/Documents/Anomaly_Detection_3D/ModelNet10/bathtub/test/bathtub_0107.off"
)
bb = ms.current_mesh().bounding_box()


# mesh = o3d.io.read_triangle_mesh(
#     "/home/mewada/Documents/Anomaly_Detection_3D/ModelNet10/bathtub/test/bathtub_0107.off"
# )
m = ms.current_mesh()


open3d_mesh = o3d.geometry.TriangleMesh()
open3d_mesh.vertices = o3d.utility.Vector3dVector(m.vertex_matrix())
open3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(m.face_matrix()))
pcd = open3d_mesh.sample_points_uniformly(number_of_points=100000)
pc = o3d.geometry.PointCloud()
pc.points = o3d.utility.Vector3dVector(pcd.points)
fps_points = pc.farthest_point_down_sample(2048)


# bbox = mesh.get_axis_aligned_bounding_box()
# bbox_extent = bbox.get_extent()
# print(f"Bounding box by open3d: {bbox_extent}")
# scale_factor = 1 / max(bbox_extent)
# scale_idx = np.argmax(bbox_extent)
# S = [
#     bbox_extent[0] * scale_factor,
#     bbox_extent[1] * scale_factor,
#     bbox_extent[2] * scale_factor,
# ]
# mesh.compute_vertex_normals()
# angles = np.random.uniform(0, 2 * np.pi, size=3)
# R = o3d.geometry.get_rotation_matrix_from_xyz(angles)
# mesh.rotate(R, center=(0, 0, 0))
o3d.visualization.draw_geometries([fps_points])
# print(S)


min_x, min_y, min_z = bb.min()
max_x, max_y, max_z = bb.max()
print(f"Bounding box by pymeshlab: {bb.min()} {bb.max()}")

vertices = np.array(
    [
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ]
)
edges = [
    [vertices[0], vertices[1], vertices[3], vertices[2], vertices[0]],
    [vertices[4], vertices[5], vertices[7], vertices[6], vertices[4]],
    [vertices[0], vertices[4]],
    [vertices[1], vertices[5]],
    [vertices[2], vertices[6]],
    [vertices[3], vertices[7]],
]


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
for edge in edges:
    xs, ys, zs = zip(*edge)
    ax.plot(xs, ys, zs, color="b")

# Set the aspect ratio to be equal
ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1

# Set labels
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

scaling_factor = 1 / max(bb.dim_x(), bb.dim_y(), bb.dim_z())
ms.apply_filter(
    "compute_matrix_from_scaling_or_normalization",
    unitflag=True,
)
# ms.save_current_mesh(
#     "/home/mewada/Documents/Anomaly_Detection_3D/ModelNet10/bathtub/test/bathtub_0107_scaled.off"
# )

print(scaling_factor)
new_bb = ms.current_mesh().bounding_box()
print(bb.dim_x(), bb.dim_y(), bb.dim_z())
print(
    new_bb.dim_x(),
    new_bb.dim_y(),
    new_bb.dim_z(),
)

# new bounding box
new_vertices = vertices * scaling_factor
new_edges = [
    [
        new_vertices[0],
        new_vertices[1],
        new_vertices[3],
        new_vertices[2],
        new_vertices[0],
    ],
    [
        new_vertices[4],
        new_vertices[5],
        new_vertices[7],
        new_vertices[6],
        new_vertices[4],
    ],
    [new_vertices[0], new_vertices[4]],
    [new_vertices[1], new_vertices[5]],
    [new_vertices[2], new_vertices[6]],
    [new_vertices[3], new_vertices[7]],
]

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
for edge in new_edges:
    xs, ys, zs = zip(*edge)
    ax.plot(xs, ys, zs, color="b")

# Set the aspect ratio to be equal
ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1

# Set labels
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")


# Display the plot
plt.show()
