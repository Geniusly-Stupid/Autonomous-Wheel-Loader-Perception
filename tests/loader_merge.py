import open3d as o3d
import numpy as np
import cv2

def depth_image_to_pointcloud(depth_image, fx, fy, cx, cy):
    """Convert depth image to point cloud"""
    height, width = depth_image.shape
    points = []
    
    # Create a grid of pixel coordinates
    u = np.arange(0, width)
    v = np.arange(0, height)
    u, v = np.meshgrid(u, v)
    
    # Convert to 3D coordinates
    z = depth_image.astype(np.float32) / 1000.0  # Convert from mm to meters
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # Stack coordinates and remove invalid points
    valid_mask = z > 0
    points = np.stack((x[valid_mask], y[valid_mask], z[valid_mask]), axis=-1)
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    return pcd

def load_and_crop_pointcloud(depth_path, fx, fy, cx, cy, min_bound, max_bound):
    """Load depth image and return cropped point cloud"""
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_image is None:
        raise ValueError(f"Could not load image from {depth_path}")
    
    point_cloud = depth_image_to_pointcloud(depth_image, fx, fy, cx, cy)
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    return point_cloud.crop(bbox)

def pairwise_registration(source, target, threshold=0.02):
    """Register two point clouds using ICP"""
    trans_init = np.identity(4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
    return reg_p2p.transformation

# Camera intrinsic parameters
fx = 415.924011
fy = 415.202179  
cx = 316.016968  
cy = 241.359283  

# Bounding box parameters
min_bound = [-0.1, -0.1, 0.2]
max_bound = [0.5, 0.5, 0.35]

# Load and crop all point clouds
pcds = []
paths = [
    r"D:\Desktop\450\project_code\Autonomous-Wheel-Loader-Perception\data\loader\depth\depth_0000.png",
    r"D:\Desktop\450\project_code\Autonomous-Wheel-Loader-Perception\data\loader\depth\depth_0030.png",
    r"D:\Desktop\450\project_code\Autonomous-Wheel-Loader-Perception\data\loader\depth\depth_0060.png",
    r"D:\Desktop\450\project_code\Autonomous-Wheel-Loader-Perception\data\loader\depth\depth_0090.png"
]

colors = [
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
]

for i, path in enumerate(paths):
    pcd = load_and_crop_pointcloud(path, fx, fy, cx, cy, min_bound, max_bound)
    pcd.paint_uniform_color(colors[i])
    pcds.append(pcd)

# Visualize initial alignment
print("Initial alignment")
o3d.visualization.draw_geometries(pcds, window_name="Before Registration")

# Perform sequential registration (each frame to previous frame)
transformations = [np.identity(4)]  # First frame has no transformation

for i in range(1, len(pcds)):
    # Register current frame to previous frame
    transformation = pairwise_registration(pcds[i], pcds[i-1])
    transformations.append(transformations[i-1] @ transformation)  # Cumulative transformation

# Apply transformations to all point clouds
registered_pcds = []
for i, pcd in enumerate(pcds):
    pcd_registered = pcd.transform(transformations[i])
    registered_pcds.append(pcd_registered)

# Combine all registered point clouds
merged_pcd = o3d.geometry.PointCloud()
for pcd in registered_pcds:
    merged_pcd += pcd

# Visualize results
print("After registration")
o3d.visualization.draw_geometries(registered_pcds, window_name="All Registered Frames")
o3d.visualization.draw_geometries([merged_pcd], window_name="Merged Point Cloud")

# Save merged point cloud
output_path = r"D:\Desktop\450\project_code\Autonomous-Wheel-Loader-Perception\data\loader\merged_pointcloud.ply"
o3d.io.write_point_cloud(output_path, merged_pcd)
print(f"Merged point cloud saved to {output_path}")

# Print transformation matrices
for i, trans in enumerate(transformations):
    print(f"Transformation matrix for frame {i}:")
    print(trans)