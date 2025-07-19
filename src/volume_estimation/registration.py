import numpy as np
import open3d as o3d
import copy
import argparse
import json

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.0706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_plotly([source_temp, target_temp])

def load_transformation(trans_init_file):
    if trans_init_file.endswith('.json'):
        with open(trans_init_file, 'r') as f:
            trans_init = np.array(json.load(f))
    elif trans_init_file.endswith('.npy'):
        trans_init = np.load(trans_init_file)
    else:
        trans_init = np.loadtxt(trans_init_file)
    return trans_init

def icp_registration(source_path, target_path, trans_init_file, threshold):
    source = o3d.io.read_point_cloud(source_path)
    target = o3d.io.read_point_cloud(target_path)
    
    trans_init = load_transformation(trans_init_file)
    
    print("\nApplying point-to-point ICP...")
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    
    print("\nICP Registration Result:")
    print(reg_p2p)
    print("\nFinal Transformation Matrix:")
    print(reg_p2p.transformation)
    
    draw_registration_result(source, target, reg_p2p.transformation)
    
    return reg_p2p

def icp_registration_pcd(source, target, trans_init, threshold):
    
    print("\nApplying point-to-point ICP...")
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    
    print("\nICP Registration Result:")
    print(reg_p2p)
    print("\nFinal Transformation Matrix:")
    print(reg_p2p.transformation)
    
    draw_registration_result(source, target, reg_p2p.transformation)
    
    return reg_p2p

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ICP Registration for Two Point Clouds")
    parser.add_argument("--source", type=str, required=True, help="Path to source point cloud (e.g., source.pcd)")
    parser.add_argument("--target", type=str, required=True, help="Path to target point cloud (e.g., target.pcd)")
    parser.add_argument("--trans_init", type=str, required=True, help="Path to initial transformation matrix (JSON/NPY/TXT)")
    parser.add_argument("--threshold", type=float, required=True, help="ICP threshold (max correspondence distance)")
    
    args = parser.parse_args()
    
    icp_result = icp_registration(
        args.source, 
        args.target, 
        args.trans_init, 
        args.threshold
    )
