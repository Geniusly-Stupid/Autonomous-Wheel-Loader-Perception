import argparse
import open3d as o3d
import json
import numpy as np
import torch
import cv2
import trimesh
import time
import os
import glob
import threading
from queue import Queue
from src.slam.fusion import TSDFVolumeTorch
from src.slam.dataset.tum_rgbd import TUMDataset
from src.slam.tracker import ICPTracker
from src.slam.utils import load_config, get_volume_setting, get_time
from src.volume_estimation.registration import icp_registration_pcd
from src.volume_estimation.volume_estimation import VolumeEstimator


def detect_loader(pcd, min_bound, max_bound, threshold):
    """
    Detect if a loader is present in the given point cloud volume.
    
    This function checks if the number of points in the given bounding box exceeds a threshold,
    and if so, considers the loader as detected.

    Args:
    - pcd (PointCloud): The input point cloud to check.
    - min_bound (list of floats): Minimum bound of the bounding box (x, y, z).
    - max_bound (list of floats): Maximum bound of the bounding box (x, y, z).
    - threshold (float): The minimum number of points required inside the bounding box to consider the loader detected.
    
    Returns:
    - bool: True if the loader is detected (i.e., points inside the bounding box exceed the threshold), False otherwise.
    """
    # Step 1: Create the Axis Aligned Bounding Box (AABB)
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

    # Step 2: Crop the point cloud using the bounding box
    cropped_pcd = pcd.crop(bbox)

    # Step 3: Voxelization (downsample the point cloud)
    voxel_size = 0.01  # Voxel size can be adjusted based on your needs
    voxel_pcd = cropped_pcd.voxel_down_sample(voxel_size)

    # Step 4: Check if the number of points in the voxelized point cloud exceeds the threshold
    if len(voxel_pcd.points) >= threshold:
        return True  # Loader detected
    else:
        return False  # Loader not detected

class SLAM:
    def __init__(self, cam_intr, margin, device, args):
        """
        Initialize the SLAM system.
        
        This initializes the SLAM system, including the TSDF volume and ICP tracker.

        Args:
        - cam_intr (array): Camera intrinsic parameters (focal length, principal point).
        - voxel_size (float): Voxel size for the TSDF volume.
        - margin (float): Margin for the TSDF volume.
        - device (str): The device (CPU or GPU) to use for computation.
        """
        self.cam_intr = cam_intr  # Camera intrinsics
        self.device = device
        self.args = args
        vol_dims, vol_origin, voxel_size = get_volume_setting(args)

        # Initialize TSDF Volume
        self.tsdf_volume = TSDFVolumeTorch(voxel_dim=vol_dims, 
                                           origin=vol_origin, 
                                           voxel_size=voxel_size, 
                                           device=device, 
                                           margin=margin,
                                           fuse_color=True)
        self.frame_index = 0
        self.icp_tracker = ICPTracker(args=self.args, device=self.device)  # Initialize ICPTracker
        
        # Store previous depth image for tracking
        self.prev_depth = None
        self.prev_rgb = None
        self.cam_pose = torch.eye(4, device=self.device).float()

    def process_frame(self, rgb_frame, depth_frame, cam_pose, color_img, obs_weight=1.0):
        """
        Process a single frame in the SLAM system.
        
        This method converts RGB and depth frames to tensors, integrates them into the TSDF volume,
        and updates the system.

        Args:
        - rgb_frame (array): RGB image frame.
        - depth_frame (array): Depth image frame.
        - cam_pose (array): Camera pose for the current frame.
        - obs_weight (float): The weight of the observation for integration (default is 1.0).
        """
        # Convert rgb and depth frames to torch tensors
        rgb = self.preprocess_image(rgb_frame)  # Convert RGB to tensor
        depth = self.preprocess_depth(depth_frame)  # Convert depth to tensor
        
        # Perform tracking if we have a previous frame
        if self.prev_depth is not None:
            # Convert camera intrinsics to tensor
            K = torch.tensor(self.cam_intr[:3, :3], device=self.device).float()
            
            # Perform tracking using ICPTracker
            rel_pose = self.icp_tracker(self.prev_depth, depth, K)
            
            # Update camera pose
            self.cam_pose = rel_pose @ self.cam_pose
        
        # Store current frame for next tracking
        self.prev_depth = depth
        self.prev_rgb = rgb
        
        # Perform integration of this frame into the TSDF volume
        self.tsdf_volume.integrate(depth_im=depth, 
                                   cam_intr=self.cam_intr, 
                                   cam_pose=cam_pose, 
                                   obs_weight=obs_weight,
                                   color_img=color_img)
        
        self.frame_index += 1

    def preprocess_image(self, rgb_frame):
        """
        This method converts the RGB frame to a torch tensor and moves it to the appropriate device (GPU/CPU).
        """
        # Convert to float32 and normalize to [0, 1]
        rgb_frame = rgb_frame.astype(np.float32) / 255.0
        return torch.tensor(rgb_frame).float().to(self.device).float()

    def preprocess_depth(self, depth_frame):
        """
        This method converts the depth frame to a torch tensor and moves it to the appropriate device (GPU/CPU).
        """
        # Convert to float32 and ensure values are in meters
        depth_frame = depth_frame.astype(np.float32) / 1000.0
        return torch.tensor(depth_frame).float().to(self.device).float()

    def get_tsdf(self):
        """
        Get the current TSDF volume.
        """
        return self.tsdf_volume
    
    def get_current_pose(self):
        """
        Get the current pose.
        """
        return self.cam_pose
        


class FrameProcessor:
    def __init__(self, trans_init_loader, cam_intr, depth_intr, loader_pcd, loader_pcd_icp, args, rgb_image, depth_image):
        """
        Initialize the frame processor for handling frames and processing them through the SLAM system.
        
        Args:
        - trans_init_loader: the initial loader transformation matrix.
        """
        self.index = 0  # Start with the first frame
        self.rgb_images = []  # To store RGB images
        self.depth_images = []  # To store depth images
        self.point_clouds = []  # Store processed point clouds
        self.args = args
        
        # volume estimation setting
        self.min_bound_tracker = [-0.1, -0.1, 0.2]
        self.max_bound_tracker = [0.5, 0.5, 0.35]
        self.threshold = 1  # Threshold for loader detection
        
        self.loader_pcd = loader_pcd
        self.loader_pcd_icp = loader_pcd_icp
        
        self.icp_threshold = 0.02
        
        self.is_alpha_shape = False
        
        self.trans_init_loader = trans_init_loader
        self.cam_intr = cam_intr
        self.depth_intr = depth_intr
        print("cam_intr: \n", cam_intr)
        print("depth_intr: \n", depth_intr)

        # Set up depth camera intrinsics using provided matrix
        self.depth_intrinsics = o3d.camera.PinholeCameraIntrinsic()
        self.depth_intrinsics.set_intrinsics(width=depth_image.shape[1], height=depth_image.shape[0],
                                        fx=self.depth_intr[0, 0], fy=self.depth_intr[1, 1],
                                        cx=self.depth_intr[0, 2], cy=self.depth_intr[1, 2])

        # Set up RGB camera intrinsics using provided matrix
        self.rgb_intrinsics = o3d.camera.PinholeCameraIntrinsic()
        self.rgb_intrinsics.set_intrinsics(width=rgb_image.shape[1], height=rgb_image.shape[0],
                                      fx=self.cam_intr[0, 0], fy=self.cam_intr[1, 1],
                                      cx=self.cam_intr[0, 2], cy=self.cam_intr[1, 2])
        
        # SLAM setting
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.slam_system = SLAM(
            cam_intr=self.cam_intr,  # Camera intrinsics
            margin=3.0, 
            device=self.device,
            args=self.args
        )
        self.cam_pose = np.eye(4)
        
        # result
        self.tsdf_volume = None
        self.is_frame_loader = False
        self.loader_volume = 0.0
        self.loader_processed_pcd = None
    
    def apply_transformation(self, pcd, registration_result):
        """
        Apply transformation to a point cloud and return the transformed point cloud
        
        Args:
            pcd: Open3D point cloud
            registration_result: Either a 4x4 numpy array or an ICP RegistrationResult
            
        Returns:
            Transformed point cloud
        """
        # Create a copy of the point cloud
        pcd_copy = o3d.geometry.PointCloud(pcd)
        
        # Extract transformation matrix from registration result if needed
        if hasattr(registration_result, 'transformation'):
            transformation = registration_result.transformation
        else:
            transformation = registration_result
        
        # Ensure transformation is numpy array
        if isinstance(transformation, torch.Tensor):
            transformation = transformation.cpu().numpy()
        
        # Apply transformation
        pcd_copy.transform(transformation)
        return pcd_copy
    
    def read_next_frame(self, rgb_image, depth_image, rgb_intr_matrix, depth_intr_matrix):
        # Store the images in memory for future reference
        self.rgb_images.append(rgb_image)
        self.depth_images.append(depth_image)

        # Convert input images to numpy arrays if they aren't already
        rgb_image = np.array(rgb_image)
        depth_image = np.array(depth_image)

        # Convert depth values from millimeters to meters
        depth_image = depth_image.astype(np.float32) / 1000.0

        # Create Open3D image objects from numpy arrays
        depth_image_o3d = o3d.geometry.Image(depth_image)
        rgb_image_o3d = o3d.geometry.Image(rgb_image)

        # Generate point cloud from depth image using depth camera parameters
        pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_image_o3d, self.depth_intrinsics)

        # Get point coordinates from the point cloud
        points = np.asarray(pcd.points)
        height, width, _ = rgb_image.shape
        
        # Calculate projection of 3D points onto RGB image plane:
        # Using perspective projection equations with RGB camera intrinsics
        fx = rgb_intr_matrix[0, 0]  # Focal length in x direction
        fy = rgb_intr_matrix[1, 1]  # Focal length in y direction
        cx = rgb_intr_matrix[0, 2]  # Principal point x coordinate
        cy = rgb_intr_matrix[1, 2]  # Principal point y coordinate
        
        # Project 3D points to 2D image coordinates
        u = (points[:, 0] * fx / points[:, 2]) + cx  # x-coordinate in image
        v = (points[:, 1] * fy / points[:, 2]) + cy  # y-coordinate in image
        
        # Convert projected coordinates to integer pixel indices
        u = np.round(u).astype(int)
        v = np.round(v).astype(int)
        
        # Initialize color array (will be black for invalid points)
        colors = np.zeros_like(points)
        
        # Create mask for points that project within the RGB image bounds and have valid depth
        valid_mask = (u >= 0) & (u < width) & (v >= 0) & (v < height) & (points[:, 2] > 0)
        
        # Convert BGR image to RGB color format (OpenCV uses BGR by default)
        rgb_image_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        
        # Assign colors from RGB image to valid points (normalized to [0,1] range)
        colors[valid_mask] = rgb_image_rgb[v[valid_mask], u[valid_mask]] / 255.0

        # Set the colors for the point cloud
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Store point cloud for later use
        self.point_clouds.append(pcd)

        # Increment frame counter for next call
        self.index += 1

        return pcd

    def process_next_frame(self, rgb_image, depth_image):
        """
        Process the next frame in the SLAM system, including ICP registration and volume estimation.
        
        This method handles processing of a single frame, including performing ICP registration, updating
        the TSDF volume, and performing volume estimation for loader detection.

        Args:
        - rgb_image (array): The current RGB image.
        - depth_image (array): The current depth image.
        """
        pcd = self.read_next_frame(rgb_image, depth_image, self.cam_intr, self.depth_intr)

        # -------------------
        # SLAM 
        # --------------------
        
        # # Process SLAM with the current frame
        # self.slam_system.process_frame(rgb_image, depth_image, self.cam_pose, color_img=rgb_image)

        # # Get the updated camera pose from SLAM system (you'll need to implement this)
        # self.cam_pose = self.slam_system.get_current_pose()
        
        # # STEP 4: Return the TSDF volume or other relevant information
        # self.tsdf_volume = self.slam_system.get_tsdf()
        
        # ---------------------------
        # VOLUME_ESTIMATION
        # ----------------------------
        # Step 1: Detect loader presence
        if detect_loader(pcd, self.min_bound_tracker, self.max_bound_tracker, self.threshold):
            self.is_frame_loader = True

            # Step 2: ICP registration between loader model and current frame
            loader_poses = icp_registration_pcd(pcd, self.loader_pcd_icp, self.trans_init_loader, self.icp_threshold)

            # Step 3: Project loader model based on the pose and crop it
            transformed_loader_pcd = self.apply_transformation(pcd, loader_poses)
            o3d.visualization.draw_geometries(
                [transformed_loader_pcd + self.loader_pcd],
                window_name="Loader Processed Point Cloud"
            )

            # Crop the transformed loader model
            bbox = o3d.geometry.AxisAlignedBoundingBox(self.min_bound_tracker, self.max_bound_tracker)
            cropped_loader_pcd = transformed_loader_pcd.crop(bbox)
            
            # Step 4: Merge the cropped_loader_pcd with the original loader_pcd
            merged_pcd = cropped_loader_pcd + self.loader_pcd  # You can also use self.loader_pcd.extend(cropped_loader_pcd.points)

            # Step 4.1: Visualize the merged point cloud
            self.loader_processed_pcd = merged_pcd

            # Step 4: Volume estimation for the cropped loader point cloud
            points = np.asarray(merged_pcd.points)
            if self.is_alpha_shape:
                self.loader_volume, _ = VolumeEstimator.estimate_alpha_shape(points, alpha=10)
            else:
                self.loader_volume = VolumeEstimator.estimate_convex_hull(points)
        else:
            # if not detected, set zero
            self.is_frame_loader = False
            self.loader_volume = 0.0
            self.loader_processed_pcd = None
        
        return

    def get_last_pcd(self):
        """
        Retrieve the last processed point cloud.
        
        Returns:
        - open3d.geometry.PointCloud: The point cloud of the last processed frame
        """
        return self.point_clouds[-1] if self.point_clouds else None
    
    def get_stored_frame(self):
        """
        Retrieve the last frame of stored RGB and depth images.

        Returns:
        - tuple: A tuple containing the last frame of stored RGB frames and depth frames.
        """
        return self.rgb_images[-1], self.depth_images[-1] if self.rgb_images else (None, None)
    
    def get_tsdf_volume(self):
        """
        Retrieve the current TSDF volume.

        Returns:
        - TSDFVolumeTorch: The TSDF volume object.
        """
        return self.tsdf_volume
    
    def get_is_frame_loader(self):
        """
        Retrieve the loader detection status for the current frame.

        Returns:
        - bool: True if a loader is detected, False otherwise.
        """
        return self.is_frame_loader
    
    def get_loader_volume(self):
        """
        Retrieve the estimated volume of the loader.

        Returns:
        - float: The estimated volume of the loader.
        """
        return self.loader_volume
    
    def get_loader_processed_pcd(self):
        """
        Retrieve the processed loader point cloud which wait to be estimated.

        Returns:
        - pcd
        """
        return self.loader_processed_pcd

def preload_next_frame(rgb_path, depth_path, queue):
    if os.path.exists(rgb_path) and os.path.exists(depth_path):
        rgb_image = cv2.imread(rgb_path)
        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        queue.put((rgb_image, depth_image))
    else:
        queue.put((None, None))

def main(rgb_folder, depth_folder, trans_init_path, cam_intr_path, depth_intr_path, loader_pcd_path, loader_pcd_icp_path, args):
    # Load camera intrinsics
    def load_matrix(path):
        if path.endswith(".json"):
            with open(path, 'r') as f:
                return np.array(json.load(f)['matrix'])
        elif path.endswith(".npy"):
            return np.load(path)
        else:
            raise ValueError("Unsupported file format. Use .json or .npy")
    
    rgb_intr = load_matrix(cam_intr_path)
    depth_intr = load_matrix(depth_intr_path)
    trans_init = load_matrix(trans_init_path)
    loader_pcd = o3d.io.read_point_cloud(loader_pcd_path)
    loader_pcd_icp = o3d.io.read_point_cloud(loader_pcd_icp_path)
    
    while(True):
        # Initial preload
        rgb_path = os.path.join(rgb_folder, f"color_0000.jpg")
        depth_path = os.path.join(depth_folder, f"depth_0000.png")
        # Get preloaded frame
        rgb_image = cv2.imread(rgb_path)
        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if rgb_image is None or depth_image is None:
            print(f"Waiting for frame 0000 to appear...")
            time.sleep(0.05)
            continue
        break
    
    # Initialize frame processor with matrices
    processor = FrameProcessor(
        trans_init_loader=trans_init,
        cam_intr=rgb_intr,
        depth_intr=depth_intr,
        loader_pcd=loader_pcd,
        loader_pcd_icp=loader_pcd_icp,
        args=args,
        rgb_image=np.array(rgb_image),
        depth_image=np.array(depth_image)
    )
    
    # Process each frame
    frame_idx = 0
    max_idx = 9999  # You can increase this if your range goes beyond 9999
    prefetch_queue = Queue(maxsize=1)

    try:
        threading.Thread(target=preload_next_frame, args=(rgb_path, depth_path, prefetch_queue)).start()

        while frame_idx <= max_idx:
            if rgb_image is None or depth_image is None:
                print(f"Waiting for frame {frame_idx:04d} to appear...")
                time.sleep(0.05)
                continue

            print(f"Processing frame {frame_idx:04d}")
            start_time = time.time()

            # Start preloading the next frame while we process this one
            next_idx = frame_idx + 1
            if next_idx <= max_idx:
                next_rgb = os.path.join(rgb_folder, f"color_{next_idx:04d}.jpg")
                next_depth = os.path.join(depth_folder, f"depth_{next_idx:04d}.png")
                threading.Thread(target=preload_next_frame, args=(next_rgb, next_depth, prefetch_queue)).start()

            # Process current frame
            processor.process_next_frame(rgb_image, depth_image)

            # Get final results
            if processor.get_is_frame_loader():
                loader_volume = processor.get_loader_volume()
                # loader_processed_pcd = processor.get_loader_processed_pcd()
                # coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
                # o3d.visualization.draw_geometries(
                #     [loader_processed_pcd, coord_frame],
                #     window_name="Loader Processed Point Cloud"
                # )
                print(f"    Loader volume: {loader_volume:.6f} cubic meters")
            else:
                print("Loader not detected.")

            print(f"Frame processed in {time.time() - start_time:.2f} seconds")
            frame_idx += 1
            
    except KeyboardInterrupt:
        print("Stopped real-time processing.")
    

if __name__ == "__main__":
    folder_name = "sequence1"
    
    parser = argparse.ArgumentParser(description="Process RGBD frames for SLAM and volume estimation")
    parser.add_argument("--config", type=str, default=r"data\{}\{}.yaml".format(folder_name, folder_name), help="Path to config file.")
    parser.add_argument("--follow_camera", action="store_true", help="Make view-point follow the camera motion")
    
    # Parse and load configuration
    parsed_args = parser.parse_args()
    args = load_config(parsed_args)
    
    main(
        rgb_folder=r"data\{}\processed\color".format(folder_name),
        depth_folder=r"data\{}\processed\depth".format(folder_name),
        trans_init_path=r"data\{}\trans_init.json".format(folder_name),
        cam_intr_path=r"data\{}\cam_intr.json".format(folder_name),
        depth_intr_path=r"data\{}\dep_intr.json".format(folder_name),
        loader_pcd_path=r"data\loader_model.ply",
        loader_pcd_icp_path=r"data\loader_model_version_2.ply",
        args=args
    )