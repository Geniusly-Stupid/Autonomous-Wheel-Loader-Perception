# kinfu_gui_streamlit.py

import os
import argparse
import time
import numpy as np
import torch
import open3d as o3d

import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s] %(filename)s:%(lineno)d - %(message)s'
)

from .fusion import TSDFVolumeTorch
from .dataset.tum_rgbd import TUMDatasetOnline
from .tracker import ICPTracker
from .utils import load_config, get_volume_setting


def main_streamlit(config_path, follow_camera=True):
    args = load_config(argparse.Namespace(config=config_path, follow_camera=follow_camera))

    logging.debug("")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TUMDatasetOnline(os.path.join(args.data_root), device, near=args.near, far=args.far, img_scale=0.25)
    vol_dims, vol_origin, voxel_size = get_volume_setting(args)

    logging.debug("")
    map_vol = TSDFVolumeTorch(vol_dims, vol_origin, voxel_size, device, margin=3, fuse_color=True)
    tracker = ICPTracker(args, device)

    logging.debug("")
    curr_pose = dataset[0][2]
    H, W = dataset.H, dataset.W
    n_frames = len(dataset)

    logging.debug("")
    mesh = None

    logging.debug("")

    frame_id = 0
    color0, depth0, pose_gt, K = dataset[frame_id]

    if frame_id > 0:
        depth1, color1, vertex01, normal1, mask1 = map_vol.render_model(
            curr_pose, K, H, W, near=args.near, far=args.far, n_samples=args.n_steps
        )
        T10 = tracker(depth0, depth1, K)
        curr_pose = curr_pose @ T10

    map_vol.integrate(depth0, K, curr_pose, obs_weight=1., color_img=color0)

    logging.debug("")
    mesh = map_vol.to_o3d_mesh()
    return mesh