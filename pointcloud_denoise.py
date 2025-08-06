import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN

def smart_denoise_point_cloud(pcd, preserve_edges=True):
    """
    Hybrid point cloud denoising: conservative statistical outlier removal
    followed by optional edge preservation via clustering.
    Returns a new point cloud.
    """
    original = np.asarray(pcd.points)

    # Step 1: statistical outlier removal
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=2.0)

    if preserve_edges:
        all_idx = np.arange(len(original))
        dropped_idx = np.setdiff1d(all_idx, ind)
        if len(dropped_idx) > 0:
            dropped = original[dropped_idx]
            db = DBSCAN(eps=0.05, min_samples=10).fit(dropped)
            labels, counts = np.unique(db.labels_, return_counts=True)
            valid = labels[counts > 100]
            good_parts = []
            for lb in valid:
                if lb == -1:
                    continue
                pts = dropped[db.labels_ == lb]
                if pts.shape[0] > 0:
                    good_parts.append(pts)
            if good_parts:
                restored = np.vstack(good_parts)
                combined = np.vstack((np.asarray(cl.points), restored))
                cl = o3d.geometry.PointCloud()
                cl.points = o3d.utility.Vector3dVector(combined)

    # Step 2: distance-based filtering
    pts = np.asarray(cl.points)
    if pts.size:
        dist = np.linalg.norm(pts, axis=1)
        maxd = np.percentile(dist, 95) * 1.5
        mask = dist < maxd
        filtered = pts[mask]
        cl = o3d.geometry.PointCloud()
        cl.points = o3d.utility.Vector3dVector(filtered)

    return cl

def visualize_comparison(original_pcd, denoised_pcd):
    """
    Visualizes both original and denoised point clouds in a single view,
    with fresh objects to avoid Open3D caching issues.
    """
    orig = o3d.geometry.PointCloud()
    orig.points = o3d.utility.Vector3dVector(np.asarray(original_pcd.points))
    den = o3d.geometry.PointCloud()
    den.points = o3d.utility.Vector3dVector(np.asarray(denoised_pcd.points))

    orig.paint_uniform_color([1.0, 0.0, 0.0])  # red
    den.paint_uniform_color([0.0, 1.0, 0.0])   # green

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Denoising Comparison", width=800, height=600)
    vis.add_geometry(orig)
    vis.add_geometry(den)
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Smart Point Cloud Denoising Tool")
    parser.add_argument("--input", default="demo.pcd",
                        help="Input point cloud (.pcd) file path")
    parser.add_argument("--output", default=None,
                        help="Output path for denoised point cloud")
    parser.add_argument("--no_edges", action="store_true",
                        help="Disable edge preservation (strict removal)")

    args = parser.parse_args()

    input_path = args.input
    if args.output:
        output_path = args.output
    else:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_denoised.pcd"

    pcd = o3d.io.read_point_cloud(input_path)
    den = smart_denoise_point_cloud(pcd, preserve_edges=not args.no_edges)
    o3d.io.write_point_cloud(output_path, den)
    print(f"Denoised cloud saved to: {output_path}")

    visualize_comparison(pcd, den)
