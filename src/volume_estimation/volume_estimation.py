import numpy as np
from scipy.spatial import ConvexHull
import alphashape
import trimesh
import matplotlib.pyplot as plt
import open3d as o3d


class ShapeGenerator:
    @staticmethod
    def dense_sphere():
        r = 1.0
        phi, theta = np.meshgrid(np.linspace(0, np.pi, 30), np.linspace(0, 2*np.pi, 60))
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        points = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
        volume = 4/3 * np.pi * r**3
        return points, volume

    @staticmethod
    def sparse_sphere():
        points, volume = ShapeGenerator.dense_sphere()
        idx = np.random.choice(len(points), len(points) // 5, replace=False)
        return points[idx], volume

    @staticmethod
    def half_sphere():

        # 1. 先生成完整球面
        r = 1.0
        phi, theta = np.meshgrid(
            np.linspace(0, np.pi, 30),
            np.linspace(0, 2*np.pi, 60)
        )
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        sphere_pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)

        # 2. 取下半球 (z <= 0)
        half_sphere = sphere_pts[sphere_pts[:, 2] <= 0]

        # 3. 生成底座：z=0 平面上的实心圆盘
        num_r = 30    # 径向采样数
        num_ang = 60  # 角度采样数
        radial = np.linspace(0, r, num_r)
        ang = np.linspace(0, 2*np.pi, num_ang)
        R, A = np.meshgrid(radial, ang)
        x_base = R * np.cos(A)
        y_base = R * np.sin(A)
        z_base = np.zeros_like(x_base)
        disk = np.stack([x_base.ravel(), y_base.ravel(), z_base.ravel()], axis=1)

        # 4. 合并点云
        points = np.vstack([half_sphere, disk])

        # 理论体积：V = 1/2 * (4/3 π r^3) = 2/3 π r^3
        volume = 2/3 * np.pi * r**3
        return points, volume

    @staticmethod
    def hollow_sphere():
        points, volume = ShapeGenerator.dense_sphere()
        mask = np.linalg.norm(points[:, :2], axis=1) > 0.4
        return points[mask], volume

    @staticmethod
    def dense_cylinder():
        r = 1.0
        h = 1.0
        theta, z = np.meshgrid(np.linspace(0, 2*np.pi, 60), np.linspace(0, h, 30))
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
        volume = np.pi * r**2 * h
        return points, volume

    @staticmethod
    def sparse_cylinder():
        points, volume = ShapeGenerator.dense_cylinder()
        idx = np.random.choice(len(points), len(points) // 5, replace=False)
        return points[idx], volume
    
    # @staticmethod
    # def ring_bowl():
    #     # 类似圆环碗形状：扁平环状点云
    #     r_outer = 1.0
    #     r_inner = 0.5
    #     height = 0.2
    #     theta, z = np.meshgrid(np.linspace(0, 2*np.pi, 80), np.linspace(-height, height, 10))
    #     r = np.linspace(r_inner, r_outer, 30)

    #     points = []
    #     for ri in r:
    #         x = ri * np.cos(theta)
    #         y = ri * np.sin(theta)
    #         zz = np.tile(z, (1, 1))
    #         points.append(np.stack([x.ravel(), y.ravel(), zz.ravel()], axis=1))

    #     points = np.concatenate(points, axis=0)
    #     return points, 0  # 没有理论体积
    

    # @staticmethod
    # def double_bump():
    #     # Two separate spheres
    #     phi, theta = np.meshgrid(np.linspace(0, np.pi, 20), np.linspace(0, 2*np.pi, 40))
    #     r = 0.5
    #     x = r * np.sin(phi) * np.cos(theta)
    #     y = r * np.sin(phi) * np.sin(theta)
    #     z = r * np.cos(phi)

    #     sphere1 = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1) + np.array([1.0, 0.0, 0.0])
    #     sphere2 = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1) + np.array([-1.0, 0.0, 0.0])
    #     return np.concatenate([sphere1, sphere2], axis=0), None
    
    # @staticmethod
    # def indented_torus():
    #     # 参数化 torus：大半径 R，小半径 r
    #     R = 1.0
    #     r = 0.3
    #     u = np.linspace(0, 2 * np.pi, 60)
    #     v = np.linspace(0, 2 * np.pi, 30)
    #     u, v = np.meshgrid(u, v)

    #     # 创建凹陷：将一部分区域的小半径缩小
    #     r_mod = r * np.ones_like(u)
    #     dent_mask = (u > np.pi/2) & (u < np.pi)
    #     r_mod[dent_mask] *= 0.5

    #     x = (R + r_mod * np.cos(v)) * np.cos(u)
    #     y = (R + r_mod * np.cos(v)) * np.sin(u)
    #     z = r_mod * np.sin(v)

    #     points = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)

    #     # 理论体积（未凹陷）为 2 * pi^2 * R * r^2
    #     volume = 2 * np.pi**2 * R * r**2
    #     return points, volume

    # @staticmethod
    # def dented_sphere():
    #     phi, theta = np.meshgrid(np.linspace(0, np.pi, 40), np.linspace(0, 2*np.pi, 80))
    #     r = 1.0

    #     x = r * np.sin(phi) * np.cos(theta)
    #     y = r * np.sin(phi) * np.sin(theta)
    #     z = r * np.cos(phi)

    #     # 在底部凹陷区域人为压缩 z 值
    #     z[z < -0.5] *= 0.5

    #     points = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)

    #     # 理论体积为完整球体体积
    #     volume = 4/3 * np.pi * r**3
    #     return points, volume

    @staticmethod
    def get_all_shapes():
        return {
            "Dense Sphere": ShapeGenerator.dense_sphere,
            "Sparse Sphere": ShapeGenerator.sparse_sphere,
            "Half Sphere": ShapeGenerator.half_sphere,
            "Hollow Sphere": ShapeGenerator.hollow_sphere,
            # "Dense Cylinder": ShapeGenerator.dense_cylinder,
            # "Sparse Cylinder": ShapeGenerator.sparse_cylinder,
            # "Ring Bowl": ShapeGenerator.ring_bowl,
            # "Double Bump": ShapeGenerator.double_bump,
            # "Indented Torus": ShapeGenerator.indented_torus,
            # "Dented Sphere": ShapeGenerator.dented_sphere,
        }


class VolumeEstimator:
    @staticmethod
    def estimate_convex_hull(points):
        return ConvexHull(points).volume

    @staticmethod
    def estimate_alpha_shape(points, alpha=5):
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        mesh =o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
        tri = trimesh.Trimesh(np.asarray(mesh.vertices), np.array(mesh.triangles))
        
        return tri.volume, mesh.is_watertight()

# ================== Main ================== #
if __name__ == "__main__":
    shape_funcs = ShapeGenerator.get_all_shapes()

    fig = plt.figure(figsize=(18, 10))
    bar_labels, bar_theory, bar_convex, bar_alpha, bar_watertight = [], [], [], [], []

    for i, (name, func) in enumerate(shape_funcs.items()):
        points, true_volume = func()
        vol_convex = VolumeEstimator.estimate_convex_hull(points)
        vol_alpha, is_watertight = VolumeEstimator.estimate_alpha_shape(points, alpha=10)

        # 3D scatter plot
        ax = fig.add_subplot(2, 3, i+1, projection='3d')
        ax.scatter(points[:,0], points[:,1], points[:,2], s=1)
        ax.set_title(name)
        ax.axis('off')

        # collect data
        bar_labels.append(name)
        bar_theory.append(true_volume if true_volume else 0)
        bar_convex.append(vol_convex)
        bar_alpha.append(vol_alpha)
        bar_watertight.append(is_watertight)
        
    # Print volume comparison for each shape
    print("\nVolume Comparison:")
    print(f"{'Shape':<20} {'True Volume':<15} {'Convex Hull':<15} {'Alpha Shape':<15} {'Watertight':<15}")
    print("-" * 80)
    for i, name in enumerate(bar_labels):
        true_vol = f"{bar_theory[i]:.4f}" if bar_theory[i] else "N/A"
        print(f"{name:<20} {true_vol:<15} {bar_convex[i]:.4f} {bar_alpha[i]:.4f} {bar_watertight[i]}")


    plt.tight_layout()
    plt.show()
