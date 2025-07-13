import os
import time
import argparse
import numpy as np
import torch
import open3d as o3d
from .fusion import TSDFVolumeTorch
from .dataset.tum_rgbd import TUMDataset
from .tracker import ICPTracker
from .utils import load_config, get_volume_setting, get_time

def main_streamlit(config_path, follow_camera=True):
    import argparse
    from utils import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=config_path)
    parser.add_argument("--follow_camera", action="store_true" if follow_camera else "store_false")
    args = load_config(parser.parse_args())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TUMDataset(os.path.join(args.data_root), device, near=args.near, far=args.far, img_scale=0.25)
    vol_dims, vol_origin, voxel_size = get_volume_setting(args)

    tsdf_vol = TSDFVolumeTorch(vol_dims, vol_origin, voxel_size, device, margin=3, fuse_color=True)
    tracker = ICPTracker(args, device)
    curr_pose = None

    for i in range(len(dataset)):
        color0, depth0, pose_gt, K = dataset[i]
        if i == 0:
            curr_pose = pose_gt
        else:
            depth1, color1, _, _, _ = tsdf_vol.render_model(curr_pose, K, dataset.H, dataset.W,
                                                            near=args.near, far=args.far, n_samples=args.n_steps)
            T10 = tracker(depth0, depth1, K)
            curr_pose = curr_pose @ T10

        tsdf_vol.integrate(depth0, K, curr_pose, obs_weight=1., color_img=color0)

    mesh = tsdf_vol.to_o3d_mesh()
    return mesh