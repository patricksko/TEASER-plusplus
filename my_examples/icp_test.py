import open3d as o3d
import numpy as np
import copy

def print_matrix(matrix):
    print("Rotation matrix:")
    print("    | {:6.3f} {:6.3f} {:6.3f} |".format(matrix[0, 0], matrix[0, 1], matrix[0, 2]))
    print("R = | {:6.3f} {:6.3f} {:6.3f} |".format(matrix[1, 0], matrix[1, 1], matrix[1, 2]))
    print("    | {:6.3f} {:6.3f} {:6.3f} |".format(matrix[2, 0], matrix[2, 1], matrix[2, 2]))
    print("Translation vector:")
    print("t = < {:6.3f}, {:6.3f}, {:6.3f} >\n".format(matrix[0, 3], matrix[1, 3], matrix[2, 3]))

cad_path = "/home/skoumal/dev/BlenderProc/resources/Object_Data/Legoblock/models_cad/obj_000018.ply"
# Load point cloud
source = o3d.io.read_triangle_mesh(cad_path)
source.compute_vertex_normals()
source = source.sample_points_uniformly(number_of_points=40000)
target = copy.deepcopy(source)

# Apply known transformation
theta = np.pi / 8
trans_init = np.eye(4)
trans_init[0:2, 0:2] = [[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta),  np.cos(theta)]]
trans_init[2, 3] = 0.4  # Translate along Z

print("Applying initial transformation:")
print_matrix(trans_init)

transformed = copy.deepcopy(source).transform(trans_init)
o3d.visualization.draw_geometries(
    [source, transformed])
# ICP registration
threshold = 0.08
reg_p2p = o3d.pipelines.registration.registration_icp(
    transformed, target, threshold, np.eye(4),
    o3d.pipelines.registration.TransformationEstimationPointToPoint()
)


print_matrix(reg_p2p.transformation)

# Visualize
source.paint_uniform_color([0, 1, 0])       # white
transformed.paint_uniform_color([0, 0, 1])  # green
aligned = copy.deepcopy(transformed).transform(reg_p2p.transformation)
aligned.paint_uniform_color([1, 0, 0])      # red

o3d.visualization.draw_geometries(
    [source, transformed, aligned],
    window_name="ICP Registration",
    zoom=0.5,
    front=[0, 0, -1],
    lookat=[0, 0, 0],
    up=[0, -1, 0]
)
