import numpy as np
import open3d as o3d
import cv2
import json
from teaserpp_python import _teaserpp as tpp
import copy
import time
import glob
import os



def load_camera_intrinsics(scene_camera_path, frame_id, image_width, image_height):
    """
    Returns Open3D PinholeCameraIntrinsic object from BlenderProc camera data.
    """
    if isinstance(frame_id, int):
        frame_id = f"{frame_id}"

    with open(scene_camera_path, "r") as f:
        cam_data = json.load(f)

    if frame_id not in cam_data:
        raise ValueError(f"Frame ID {frame_id} not found in scene_camera.json")

    cam_K = cam_data[frame_id]["cam_K"]
    fx, fy = cam_K[0], cam_K[4]
    cx, cy = cam_K[2], cam_K[5]

    depth_scale = cam_data[frame_id]["depth_scale"]
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=image_width,
        height=image_height,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy
    )
    return intrinsic, depth_scale

def get_pointcloud(depth_path, rgb_path, scene_camera_path, mask_path):
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    color_raw = cv2.imread(rgb_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    
    binary_mask = (mask == 255).astype(np.uint8)
    mask_pixels = np.sum(binary_mask)

    if mask_pixels == 0:
        print("WARNING: No pixels selected by mask!")
        return None
    
    color_raw = cv2.cvtColor(color_raw, cv2.COLOR_BGR2RGB)
    
    h, w = depth_raw.shape
    intrinsic, depth_scale = load_camera_intrinsics(scene_camera_path, 0, w, h)
    depth_raw = depth_raw * depth_scale
    
    masked_depth = np.where(binary_mask, depth_raw, 0.0)
    masked_color = np.where(binary_mask[:, :, None], color_raw, 0)

    valid_depth_mask = (masked_depth > 0.01) & (masked_depth < 10.0)  
    masked_depth = np.where(valid_depth_mask, masked_depth, 0.0)
    masked_color = np.where(valid_depth_mask[:, :, None], masked_color, 0)
    
    depth_o3d = o3d.geometry.Image(masked_depth)
    color_o3d = o3d.geometry.Image(masked_color)
    
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d,
        depth_scale=1.0,
        convert_rgb_to_intensity=False
    )
    
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)

    # Transform coordinate system
    pcd.transform([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])
    
    # Remove outliers
    if len(pcd.points) > 0:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    return pcd


def uniform_downsample_farthest_point(pcd, target_points=2000):
    """
    Farthest Point Sampling (FPS) for uniform downsampling.
    Good spatial distribution, moderate computation time.
    """
    points = np.asarray(pcd.points)
    n_points = len(points)
    print(f"Number of points: {n_points}, target: {target_points}")

    target_points = int(target_points)
    if n_points <= target_points:
        return pcd
    
    # Initialize with random point
    selected_indices = [np.random.randint(n_points)]
    distances = np.full(n_points, np.inf)
    
    for i in range(1, target_points):
        # Update distances to nearest selected point
        last_point = points[selected_indices[-1]]
        new_distances = np.linalg.norm(points - last_point, axis=1)
        distances = np.minimum(distances, new_distances)
        
        # Select point with maximum distance to any selected point
        next_idx = np.argmax(distances)
        selected_indices.append(next_idx)
        
        # Set distance of selected point to 0
        distances[next_idx] = 0
    
    return pcd.select_by_index(selected_indices)



# Modified preprocessing function
def preprocess_point_cloud_uniform(pcd, target_points=500):
    """
    Preprocess point cloud with uniform downsampling to exactly target_points.
    
    Methods:
    - 'random': Fast random sampling
    - 'grid': Grid-based uniform sampling
    - 'kmeans': K-means clustering (most uniform, slowest)
    - 'farthest_point': Farthest Point Sampling (good balance)
    - 'adaptive_voxel': Adaptive voxel size (fastest, approximate)
    """
    if len(pcd.points) == 0:
        print("Empty point cloud!")
        return None, None
        
    print(f"Original points: {len(pcd.points)}")

    pcd_down = uniform_downsample_farthest_point(pcd, target_points)
    
    print(f"Downsampled points: {len(pcd_down.points)}")
    
    # Ensure we have enough points for feature computation
    if len(pcd_down.points) < 10:
        print("Too few points after downsampling!")
        return None, None
    
    # Estimate normals with adaptive parameters
    nn_param = min(30, len(pcd_down.points) // 2)  # Adaptive neighbor count
    radius = 0.05  # You might want to adjust this based on your data scale
    
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=nn_param)
    )
    
    # Compute FPFH features
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius*2.5, max_nn=100)
    )
    
    print(f"FPFH feature dimension: {fpfh.dimension()}")
    return pcd_down, fpfh

def get_correspondences(pcd1_down, pcd2_down, fpfh1, fpfh2, distance_threshold=0.05):
    # Try different distance thresholds if needed
    for dist_thresh in [distance_threshold, distance_threshold*2, distance_threshold*0.5]:
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            pcd1_down, pcd2_down, fpfh1, fpfh2,
            mutual_filter=True,
            max_correspondence_distance=dist_thresh,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=3,  # Reduced from 4
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist_thresh)
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 1000)
        )
        
        if len(result.correspondence_set) >= 3:  # Minimum for registration
            return result.correspondence_set

    return result.correspondence_set

def run_teaser(source, target, correspondences):
    if len(correspondences) < 3:
        return np.eye(4), []
    
    src_corr = np.array([source.points[i] for i, _ in correspondences]).T
    tgt_corr = np.array([target.points[j] for _, j in correspondences]).T
    
    params = tpp.RobustRegistrationSolver.Params()
    params.cbar2 = 1
    params.noise_bound = 0.001
    params.estimate_scaling = False
    params.rotation_estimation_algorithm = tpp.RotationEstimationAlgorithm.GNC_TLS
    params.rotation_gnc_factor = 1.4
    params.rotation_max_iterations = 100
    params.rotation_cost_threshold = 1e-12
    
    solver = tpp.RobustRegistrationSolver(params)
    solver.solve(src_corr, tgt_corr)
    sol = solver.getSolution()
    
    R_inliers = solver.getRotationInliers()
    R = np.array(sol.rotation)
    t = np.array(sol.translation).reshape((3, 1))
    
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3:] = t
    
    return T, R_inliers

def find_best_template_teaser(scene_pcd, template_pcds, target_points=500):
    # Preprocess scene
    scene_down, scene_fpfh = preprocess_point_cloud_uniform(scene_pcd, target_points)
    
    best_inliers = -1
    best_idx = -1
    best_transform = np.eye(4)
    
    for idx, template_pcd in enumerate(template_pcds):
        # Preprocess template
        template_down, template_fpfh = preprocess_point_cloud_uniform(template_pcd, target_points)
        if template_down is None:
            continue
        
        # Get correspondences and run TEASER++
        correspondences = get_correspondences(template_down, scene_down, template_fpfh, scene_fpfh)
        T, R_inliers = run_teaser(template_down, scene_down, correspondences)
        
        inlier_count = len(R_inliers) if R_inliers is not None else 0
        print(f"Template {idx}: {len(correspondences)} corr, {inlier_count} inliers")
        # Append to file
        with open("inlier_counts.txt", "a") as f:
            f.write(f"Template {idx}: {len(correspondences)} correspondences, {inlier_count} inliers\n")
        
        if inlier_count > best_inliers:
            best_inliers = inlier_count
            best_idx = idx
            best_transform = T
    
    return best_idx, best_transform, best_inliers

def try_manual_alignment(cad_pcd, scene_pcd, scale_ratio):
    """Try manual initial alignment based on centroids, scale, and PCA rotation"""
    print("\n=== TRYING MANUAL ALIGNMENT ===")

    cad_aligned = copy.deepcopy(cad_pcd)

    # 1. Center both point clouds
    cad_center = cad_aligned.get_center()
    scene_center = scene_pcd.get_center()
    cad_aligned.translate(-cad_center)
    cad_aligned.translate(scene_center)

    # 2. Scale
    if abs(scale_ratio - 1.0) > 0.1:
        print(f"Applying scale factor: {scale_ratio:.3f}")
        cad_aligned.scale(scale_ratio, center=scene_center)

 
    
    return cad_aligned

def center_pointcloud(pcd):
    """Center pointcloud at origin"""
    centered = pcd.translate(-pcd.get_center())
    return centered



if __name__ == "__main__":
    # Load images
    depth_path = "/home/skoumal/dev/BlenderProc/my_examples/output_blenderproc_full/bop_data/Legoblock/train_pbr/000000/depth/000005.png"
    rgb_path   = "/home/skoumal/dev/BlenderProc/my_examples/output_blenderproc_full/bop_data/Legoblock/train_pbr/000000/rgb/000005.jpg"
    mask_path = "/home/skoumal/dev/BlenderProc/my_examples/output_blenderproc_full/bop_data/Legoblock/train_pbr/000000/mask/000005_000000.png"
    scene_camera_path = "/home/skoumal/dev/BlenderProc/my_examples/output_blenderproc/bop_data/Legoblock/train_pbr/000000/scene_camera.json"
    
    #Read CAD Path
    cad_path = "/home/skoumal/dev/TEASER-plusplus/build/python/lego_views/"

    # Read all ply files
    ply_files = sorted(glob.glob(os.path.join(cad_path, "*.ply")))

    pointclouds = []
    for ply_file in ply_files:
        pcd = o3d.io.read_point_cloud(ply_file)
        pointclouds.append(pcd)
        print(f"Loaded: {ply_file} with {len(pcd.points)} points")
    
    # for pcd in pointclouds:
    #     o3d.visualization.draw_geometries([pcd.paint_uniform_color([1, 0, 0])])

    scene_pcd = get_pointcloud(depth_path, rgb_path, scene_camera_path, mask_path=mask_path)
    #scene_pcd_centered = scene_pcd.translate(-scene_pcd.get_center())


    if scene_pcd is None or len(scene_pcd.points) == 0:
        print("Failed to generate scene point cloud!")
        exit(1)
    
    start_time = time.time()
    best_idx, best_transform, inliers = find_best_template_teaser(scene_pcd, pointclouds, target_points=200)
    print(f"Best match: Template {best_idx} with {inliers} rotation inliers")

# Apply final transformation
    if best_idx >= 0:
        final_template = copy.deepcopy(pointclouds[best_idx])
        final_template.transform(best_transform)
        
    scale_ratio = 1
    cad_pcd = final_template
    end_time=time.time()
    print(end_time-start_time)
    print("----------------------------")
    #cad_aligned = try_manual_alignment(cad_pcd, scene_pcd, scale_ratio)
    o3d.visualization.draw_geometries([scene_pcd.paint_uniform_color([1, 0, 0]),
                                        cad_pcd.paint_uniform_color([0, 1, 0])])
    exit()
    print("\n=== TRYING TEASER++ WITH ALIGNED CAD ===")
    best_correspondences = []
    best_voxel_size = None
    
    voxel_size = 0.001
    print(f"\nTrying voxel size: {voxel_size}")
    
    cad_down, cad_fpfh = preprocess_point_cloud_uniform(cad_aligned, 500)
    scene_down, scene_fpfh = preprocess_point_cloud_uniform(scene_pcd, 500)
    
    if cad_down is None or scene_down is None:
        exit()
    start_time = time.time() 
    correspondences = get_correspondences(cad_down, scene_down, cad_fpfh, scene_fpfh, 
                                        distance_threshold=voxel_size*10)
    
    if len(correspondences) > len(best_correspondences):
        best_correspondences = correspondences
        best_voxel_size = voxel_size
   
    if len(best_correspondences) >= 3:
        print(f"\nBest result: {len(best_correspondences)} correspondences with voxel size {best_voxel_size}")
        
        T = run_teaser(cad_down, scene_down, best_correspondences)
        
        # Apply transformation and visualize
        cad_final = copy.deepcopy(cad_aligned)
        cad_final.transform(T)
        cad_final_vis = cad_final.paint_uniform_color([1, 0, 0])
        end_time = time.time()
        
        scene_vis = copy.deepcopy(scene_pcd)
        scene_vis = scene_vis.paint_uniform_color([0, 1, 0])
        o3d.visualization.draw_geometries([cad_final_vis, scene_vis], 
                                        window_name="TEASER++ Result")
        

        # Duration
        elapsed = end_time - start_time
        print(f"Elapsed time: {elapsed:.4f} seconds")
    else:
        print(f"\nTEASER++ failed. Best attempt: {len(best_correspondences)} correspondences")
        exit()
        