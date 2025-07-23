import open3d as o3d
import numpy as np
import cv2
import json
import os
import copy

def load_matrix(path):
    """Load camera matrix from JSON or NPY file"""
    if path.endswith(".json"):
        with open(path, 'r') as f:
            data = json.load(f)
            # Handle different JSON structures
            if 'matrix' in data:
                return np.array(data['matrix'])
            elif 'intrinsic_matrix' in data:
                return np.array(data['intrinsic_matrix'])
            else:
                return np.array(data)  # Assume direct matrix data
    elif path.endswith(".npy"):
        return np.load(path)
    else:
        raise ValueError("Unsupported file format. Use .json or .npy")

def depth_image_to_pointcloud(depth_image, intrinsic_matrix):
    """Convert depth image to point cloud using camera intrinsics"""
    height, width = depth_image.shape
    
    # Extract camera parameters from matrix
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    
    # Create grid of coordinates
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    
    # Convert to 3D coordinates
    z = depth_image.astype(np.float32) / 1000.0  # mm to meters
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # Filter valid points
    valid_mask = z > 0
    points = np.stack((x[valid_mask], y[valid_mask], z[valid_mask]), axis=-1)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def load_and_crop_pointcloud(depth_path, intrinsic_matrix, min_bound, max_bound):
    """Load depth image and convert to cropped point cloud"""
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_image is None:
        raise ValueError(f"Could not load depth image: {depth_path}")
    
    pcd = depth_image_to_pointcloud(depth_image, intrinsic_matrix)
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    return pcd.crop(bbox)

def icp_registration_pcd(source, target, trans_init, threshold):
    """Perform ICP registration between two point clouds"""
    reg_result = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=500))
    return reg_result.transformation

def visualize_transform_result(depth_pcd, loader_pcd, transform_matrix):
    """
    Visualize the transformation result with given matrix
    :param depth_pcd: The target point cloud (depth data)
    :param loader_pcd: The source point cloud (loader model)
    :param transform_matrix: 4x4 transformation matrix to apply
    """
    # Create copies for visualization
    depth_vis = copy.deepcopy(depth_pcd)
    loader_vis = copy.deepcopy(loader_pcd)
    
    # Apply colors
    depth_vis.paint_uniform_color([0, 0, 1])  # Blue for depth point cloud
    loader_vis.paint_uniform_color([1, 0, 0])  # Red for loader model
    
    # Apply the transformation
    loader_vis.transform(transform_matrix)
    
    # Visualize
    o3d.visualization.draw_geometries([depth_vis, loader_vis], 
                                     window_name="Transformation Result")

def main():
    # Path configuration
    data_dir = r"D:\Desktop\450\project_code\Autonomous-Wheel-Loader-Perception\data\sequence1"
    depth_intr_path = os.path.join(data_dir, "dep_intr.json")
    depth_path = os.path.join(data_dir, "depth", "depth_0160.png")
    loader_model_path = r"D:\Desktop\450\project_code\Autonomous-Wheel-Loader-Perception\data\loader_model.ply"
    
    # Load camera intrinsics
    depth_intrinsic = load_matrix(depth_intr_path)
    
    # Verify intrinsics
    print("Depth Camera Intrinsics:")
    print(depth_intrinsic)

    # Processing parameters
    min_bound = [-0.1, -0.1, 0.2]
    max_bound = [0.2, 0.3, 0.35]
    icp_threshold = 0.02  # 2cm

    # Example transformation matrix (replace with your actual matrix)
    example_transform = np.array([
        [1,      0,     0,   0],
        [0,      2,     0,   0.07],
        [0,      0,     1,   0.015],
        [0,      0,     0,   1]
    ])

    # Load and process depth frame
    print(f"Processing depth frame: {os.path.basename(depth_path)}")
    depth_pcd = load_and_crop_pointcloud(depth_path, depth_intrinsic, min_bound, max_bound)
    
    # Load loader model
    loader_pcd = o3d.io.read_point_cloud(loader_model_path)
    if not loader_pcd.has_points():
        raise ValueError(f"Could not load loader model: {loader_model_path}")

    # Visualize the transformation result
    print("\nVisualizing transformation result...")
    visualize_transform_result(depth_pcd, loader_pcd, example_transform)

#     {
#   "matrix": [
#         [1,      0,     0,   0],
#         [0,      2,     0,   0.07],
#         [0,      0,     1,   0.015],
#         [0,      0,     0,   1]
#   ]
# }


if __name__ == "__main__":
    main()