import open3d as o3d
import numpy as np
import cv2
import json
import os

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

def load_and_process_frame(depth_path, intrinsic_matrix, min_bound, max_bound):
    """Load and process a single frame"""
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_image is None:
        raise ValueError(f"Could not load depth image: {depth_path}")
    
    pcd = depth_image_to_pointcloud(depth_image, intrinsic_matrix)
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    return pcd.crop(bbox)

def pairwise_registration(source, target, threshold=0.02):
    """Perform ICP registration between two point clouds"""
    trans_init = np.identity(4)
    reg_result = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
    return reg_result.transformation

def main():
    # Path configuration
    data_dir = r"D:\Desktop\450\project_code\Autonomous-Wheel-Loader-Perception\data\loader"
    rgb_intr_path = os.path.join(data_dir, "cam_intr.json")
    depth_intr_path = os.path.join(data_dir, "dep_intr.json")
    depth_files = [
        os.path.join(data_dir, "depth", "depth_0000.png"),
        os.path.join(data_dir, "depth", "depth_0030.png"),
        os.path.join(data_dir, "depth", "depth_0060.png"),
        os.path.join(data_dir, "depth", "depth_0090.png")
    ]
    output_path = os.path.join(data_dir, "merged_pointcloud.ply")

    # Load camera intrinsics
    rgb_intrinsic = load_matrix(rgb_intr_path)
    depth_intrinsic = load_matrix(depth_intr_path)
    
    # Verify intrinsics
    print("RGB Camera Intrinsics:")
    print(rgb_intrinsic)
    print("\nDepth Camera Intrinsics:")
    print(depth_intrinsic)

    # Processing parameters
    min_bound = [-0.1, -0.1, 0.25]
    max_bound = [0.2, 0.2, 0.35]
    colors = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]]  # Different colors for each frame

    # Load and process all frames
    pcds = []
    for i, depth_path in enumerate(depth_files):
        print(f"Processing frame {i+1}/{len(depth_files)}: {os.path.basename(depth_path)}")
        try:
            pcd = load_and_process_frame(depth_path, depth_intrinsic, min_bound, max_bound)
            pcd.paint_uniform_color(colors[i])
            pcds.append(pcd)
        except Exception as e:
            print(f"Error processing {depth_path}: {str(e)}")
            continue

    # Visualize initial alignment
    o3d.visualization.draw_geometries(pcds, window_name="Initial Alignment")

    # Sequential registration
    transformations = [np.identity(4)]  # First frame has identity transform
    
    for i in range(1, len(pcds)):
        transformation = pairwise_registration(pcds[i], pcds[i-1])
        transformations.append(transformations[i-1] @ transformation)  # Cumulative transform

    # Apply transformations and merge
    merged_pcd = o3d.geometry.PointCloud()
    for i, pcd in enumerate(pcds):
        pcd_registered = pcd.transform(transformations[i])
        merged_pcd += pcd_registered

        # Print transformation info
        print(f"\nTransformation for frame {i}:")
        print(transformations[i])

    # Save and visualize results
    o3d.io.write_point_cloud(output_path, merged_pcd)
    print(f"\nMerged point cloud saved to: {output_path}")
    
    o3d.visualization.draw_geometries([merged_pcd], window_name="Final Merged Point Cloud")

if __name__ == "__main__":
    main()