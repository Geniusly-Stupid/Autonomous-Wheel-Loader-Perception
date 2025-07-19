import argparse
import open3d as o3d
import json
import numpy as np
import torch
import cv2
import trimesh
import time
import os
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
    def __init__(self, rgb_folder, depth_folder, cam_intr, voxel_size, margin, device):
        """
        Initialize the SLAM system.
        
        This initializes the SLAM system, including the TSDF volume and ICP tracker.

        Args:
        - rgb_folder (str): Path to the folder containing RGB images.
        - depth_folder (str): Path to the folder containing depth images.
        - cam_intr (array): Camera intrinsic parameters (focal length, principal point).
        - voxel_size (float): Voxel size for the TSDF volume.
        - margin (float): Margin for the TSDF volume.
        - device (str): The device (CPU or GPU) to use for computation.
        """
        self.rgb_folder = rgb_folder
        self.depth_folder = depth_folder
        self.cam_intr = cam_intr  # Camera intrinsics
        self.device = device

        # Initialize TSDF Volume
        self.tsdf_volume = TSDFVolumeTorch(voxel_dim=[128, 128, 128], 
                                           origin=[0, 0, 0], 
                                           voxel_size=voxel_size, 
                                           device=device, 
                                           margin=margin)
        self.frame_index = 0
        self.icp_tracker = ICPTracker()  # Initialize ICPTracker

    def process_frame(self, rgb_frame, depth_frame, cam_pose, obs_weight=1.0):
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
        
        # Perform integration of this frame into the TSDF volume
        self.tsdf_volume.integrate(depth_im=depth, 
                                   cam_intr=self.cam_intr, 
                                   cam_pose=cam_pose, 
                                   obs_weight=obs_weight)
        
        self.frame_index += 1

    def preprocess_image(self, rgb_frame):
        """
        This method converts the RGB frame to a torch tensor and moves it to the appropriate device (GPU/CPU).
        """
        return torch.tensor(rgb_frame).float().to(self.device)

    def preprocess_depth(self, depth_frame):
        """
        This method converts the depth frame to a torch tensor and moves it to the appropriate device (GPU/CPU).
        """
        return torch.tensor(depth_frame).float().to(self.device)

    def icp_registration(self, source_pcd, target_pcd):
        """
        Perform ICP registration between two point clouds.

        This method uses ICPTracker to register the source point cloud to the target point cloud and
        returns the resulting transformation.

        Args:
        - source_pcd (PointCloud): Source point cloud.
        - target_pcd (PointCloud): Target point cloud.

        Returns:
        - array: Transformation matrix resulting from ICP registration.
        """
        return self.icp_tracker.register(source_pcd, target_pcd)

    def get_tsdf(self):
        """
        Get the current TSDF volume.
        """
        return self.tsdf_volume.get_volume()


class FrameProcessor:
    def __init__(self, trans_init_loader_path):
        """
        Initialize the frame processor for handling frames and processing them through the SLAM system.
        
        Args:
        - trans_init_loader_path (str): Path to the initial loader transformation matrix (JSON/NPY).
        """
        self.trans_init_loader_path = trans_init_loader_path
        self.index = 0  # Start with the first frame
        self.rgb_images = []  # To store RGB images
        self.depth_images = []  # To store depth images
        
        # volume estimation setting
        self.min_bound_tracker = [-0.1, 0.04, 0.2]
        self.max_bound_tracker = [0.1, 0.12, 0.4]
        
        self.loader_pcd = None
        
        self.icp_threshold = 0.02
        
        self.is_alpha_shape = False
        
        self.trans_init_loader = None
        if self.trans_init_loader_path.endswith(".json"):
            with open(self.trans_init_loader_path, 'r') as f:
                self.trans_init_loader = np.array(json.load(f))
        elif self.trans_init_loader_path.endswith(".npy"):
            self.trans_init_loader = np.load(self.trans_init_loader_path)
        else:
            raise ValueError("Unsupported transformation file format. Use .json or .npy")
        
        # SLAM setting
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.slam_system = SLAM(
            rgb_folder="path_to_rgb_folder", 
            depth_folder="path_to_depth_folder", 
            cam_intr=self.cam_intr,  # Camera intrinsics
            voxel_size=0.01, 
            margin=3.0, 
            device=self.device
        )
        
        # result
        self.tsdf_volume = None
        self.is_frame_loader = False
        self.loader_volume = 0.0
    
    def apply_transformation(self, pcd, transformation):
        """
        This method applies the transformation matrix to the given point cloud and returns the transformed point cloud.
        """
        pcd_copy = pcd.copy()
        pcd_copy.transform(transformation)
        return pcd_copy

    def read_next_frame(self, rgb_image, depth_image):
        """
        Read the next frame of RGB and depth images and process them into a point cloud.
        
        This method converts the input RGB and depth images into a point cloud and increments the frame index.

        Args:
        - rgb_image (array): The RGB image.
        - depth_image (array): The depth image.

        Returns:
        - PointCloud: The point cloud generated from the RGB and depth images.
        """
        # Store the images in memory
        self.rgb_images.append(rgb_image)
        self.depth_images.append(depth_image)

        # Create a point cloud from the depth image
        depth_image_o3d = o3d.geometry.Image(np.array(depth_image))
        rgb_image_o3d = o3d.geometry.Image(np.array(rgb_image))

        # Define intrinsic parameters for a typical RGB-D camera
        width, height = 640, 480
        fx, fy = 525.0, 525.0  # Focal lengths (in pixels)
        cx, cy = width / 2, height / 2  # Principal point (usually in the center)

        # Create an intrinsics object
        depth_intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
        
        # Create point cloud from depth image
        pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_image_o3d, depth_intrinsics)

        # Increment the index to read the next frame on the next call
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
        pcd = self.read_next_frame(rgb_image, depth_image)

        # -------------------
        # SLAM 
        # --------------------
        # STEP 1: Get ICP Transformation
        if len(self.rgb_images) > 1:
            # Perform ICP registration using SLAM system
            previous_pcd = self.rgb_images[-2]  # The previous RGB image's point cloud
            icp_transformation = self.slam_system.icp_registration(previous_pcd, pcd)  # SLAM system's ICP
        else:
            icp_transformation = np.eye(4)  # First frame, no registration

        # STEP 2: Apply ICP transformation to current frame
        cam_pose = icp_transformation

        # STEP 3: Process SLAM with the updated camera pose
        self.slam_system.process_frame(rgb_image, depth_image, cam_pose)

        # STEP 4: Return the TSDF volume or other relevant information
        self.tsdf_volume = self.slam_system.get_tsdf()
        
        # ---------------------------
        # VOLUME_ESTIMATION
        # ----------------------------
        # Step 1: Detect loader presence
        if detect_loader(pcd, self.min_bound_tracker, self.max_bound_tracker, self.threshold):
            self.is_frame_loader = True

            # Step 2: ICP registration between loader model and current frame
            loader_poses = icp_registration_pcd(pcd, self.loader_pcd, self.trans_init_loader, self.icp_threshold)

            # Step 3: Project loader model based on the pose and crop it
            transformed_loader_pcd = self.apply_transformation(self.loader_pcd, loader_poses)

            # Crop the transformed loader model
            bbox = o3d.geometry.AxisAlignedBoundingBox(self.min_bound_tracker, self.max_bound_tracker)
            cropped_loader_pcd = transformed_loader_pcd.crop(bbox)

            # Step 4: Volume estimation for the cropped loader point cloud
            points = np.asarray(cropped_loader_pcd.points)
            if self.is_alpha_shape:
                self.loader_volume, _ = VolumeEstimator.estimate_alpha_shape(points, alpha=10)
            else:
                self.loader_volume = VolumeEstimator.estimate_convex_hull(points)
        else:
            # if not detected, set zero
            self.is_frame_loader = False
            self.loader_volume = 0.0
        
        return

    def get_stored_frame(self):
        """
        Retrieve the last frame of stored RGB and depth images.

        Returns:
        - tuple: A tuple containing the last frame of stored RGB frames and depth frames.
        """
        return self.rgb_images[-1], self.depth_images[-1]
    
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

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="ICP Registration for Two Point Clouds")
    parser.add_argument("--trans_init", type=str, default=r"data\trans_init.json", help="Path to initial transformation matrix (JSON/NPY/TXT)")
    args = parser.parse_args()

    # Initialize FrameProcessor with the provided arguments
    processor = FrameProcessor(trans_init_loader_path=args.trans_init)

    # Simulate reading RGB and depth frames (use real frames in practical application)
    rgb_image = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)  # Random RGB image
    depth_image = (np.random.rand(480, 640) * 255).astype(np.uint16)  # Random depth image

    # Process the next frame
    processor.process_next_frame(rgb_image, depth_image)

    # Get the processed TSDF volume from SLAM system
    tsdf_volume = processor.get_tsdf_volume()

    # Visualize the TSDF volume using Open3D
    o3d.visualization.draw_geometries([tsdf_volume])

    # Get stored frames and print the count of stored frames
    rgb_frames, depth_frames = processor.get_stored_frames()
    print(f"Stored {len(rgb_frames)} RGB frames and {len(depth_frames)} Depth frames")

    # Check if loader is detected in the current frame and print the loader volume
    if processor.get_is_frame_loader():
        print(f"Loader detected! Estimated loader volume: {processor.get_loader_volume()} cubic units")
    else:
        print("No loader detected in the current frame.")

