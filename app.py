import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

from src.slam.dataset import preprocess
from src.slam import kinfu
from src.slam import kinfu_gui_streamlit
from src.visualization import visualize_ply
from src import volume_estimation

CONFIG_PATH = "src/slam/configs/fr1_room.yaml"
SAVE_DIR = "data/reconstruct/slam"
PLY_PATH = os.path.join(SAVE_DIR, "mesh.ply")

st.set_page_config(page_title="SLAM & Volume Estimation App", layout="wide")
st.title("🤖 SLAM 重建 & 点云体积估计可视化")

st.sidebar.header("功能选择")
options = [
    "1️⃣ 数据预处理",
    "2️⃣ 运行 KinFu 重建",
    "3️⃣ 逐帧 SLAM 可视化 (Streamlit)",
    "4️⃣ 加载并可视化 PLY 文件",
    "5️⃣ 点云体积估计示例"
]
choice = st.sidebar.radio("请选择功能", options)

# ================== 数据预处理 ==================
if choice == options[0]:
    st.subheader("数据预处理")
    if st.button("开始执行"):
        preprocess.main(CONFIG_PATH)
        st.success("✅ 预处理完成")

# ================== KinFu 重建 ==================
elif choice == options[1]:
    st.subheader("运行 KinFu 重建")
    if st.button("开始重建"):
        kinfu.main(CONFIG_PATH, SAVE_DIR)
        st.success("✅ 重建完成")
        if os.path.exists(PLY_PATH):
            st.info("已生成 mesh.ply，可在【加载并可视化 PLY 文件】查看")

# ================== Frame-by-Frame SLAM 可视化 ==================
elif choice == options[2]:
    st.subheader("逐帧 SLAM 可视化 (Streamlit)")
    follow = st.checkbox("跟随相机视角", value=True)
    if st.button("开始可视化"):
        mesh = kinfu_gui_streamlit.main_streamlit(CONFIG_PATH, follow_camera=follow)
        if mesh is not None:
            pcd = mesh.sample_points_uniformly(number_of_points=5000)

            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False)
            vis.add_geometry(mesh)
            vis.poll_events()
            vis.update_renderer()
            img = vis.capture_screen_float_buffer(True)
            vis.destroy_window()

            plt.imshow(np.asarray(img))
            plt.axis('off')
            st.pyplot(plt)
            st.success("✅ 可视化完成")
        else:
            st.error("❌ 没有生成 Mesh")

# ================== 加载并可视化 PLY 文件 ==================
elif choice == options[3]:
    st.subheader("加载并显示 PLY 文件")
    if os.path.exists(PLY_PATH):
        mesh = visualize_ply.load_ply(PLY_PATH)
        if mesh and not mesh.is_empty():
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False)
            vis.add_geometry(mesh)
            vis.poll_events()
            vis.update_renderer()
            img = vis.capture_screen_float_buffer(True)
            vis.destroy_window()

            plt.imshow(np.asarray(img))
            plt.axis('off')
            st.pyplot(plt)
            st.success("✅ PLY 文件可视化完成")
        else:
            st.error("❌ mesh.ply 文件为空")
    else:
        st.warning("请先运行重建步骤生成 mesh.ply")

# ================== 点云体积估计 ==================
elif choice == options[4]:
    st.subheader("点云体积估计示例")

    shape_options = list(volume_estimation.ShapeGenerator.get_all_shapes().keys())
    shape_choice = st.selectbox("选择形状", shape_options)
    alpha = st.slider("Alpha shape 参数", 1, 30, 10)

    if st.button("生成并估计体积"):
        points, true_volume = volume_estimation.ShapeGenerator.get_all_shapes()[shape_choice]()
        vol_convex = volume_estimation.VolumeEstimator.estimate_convex_hull(points)
        vol_alpha, is_watertight = volume_estimation.VolumeEstimator.estimate_alpha_shape(points, alpha=alpha)

        st.write(f"理论体积: {true_volume:.4f}" if true_volume else "理论体积: N/A")
        st.write(f"Convex Hull 估计体积: {vol_convex:.4f}")
        st.write(f"Alpha Shape 估计体积: {vol_alpha:.4f} (Watertight: {is_watertight})")

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
        ax.set_title(shape_choice)
        ax.axis('off')
        st.pyplot(fig)