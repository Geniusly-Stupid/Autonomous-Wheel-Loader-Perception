import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

from src.slam.dataset import preprocess
from src.slam import kinfu_gui_streamlit
from src.visualization import visualize_ply
from src.volume_estimation import volume_estimation

import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s] %(filename)s:%(lineno)d - %(message)s'
)

CONFIG_PATH = "src/slam/configs/new_dataset.yaml"
SAVE_DIR = "data/reconstruct/slam"
PLY_PATH = os.path.join(SAVE_DIR, "mesh.ply")

st.set_page_config(page_title="SLAM & Volume Estimation App", layout="wide")
st.title("🤖 SLAM Reconstruction & Point Cloud Volume Estimation")

st.sidebar.header("Function Selection")
options = [
    "1️⃣ Data Preprocessing",
    "2️⃣ Run KinFu Reconstruction",
    "3️⃣ Frame-by-Frame SLAM Visualization (Streamlit)",
    "4️⃣ Load and Visualize PLY File",
    "5️⃣ Point Cloud Volume Estimation Example"
]
choice = st.sidebar.radio("Please select a function", options)

# ================== Data Preprocessing ==================
if choice == options[0]:
    st.subheader("Data Preprocessing")
    if st.button("Start"):
        preprocess.main(CONFIG_PATH)
        st.success("✅ Preprocessing completed")

# ================== KinFu Reconstruction ==================
elif choice == options[1]:
    st.subheader("Run KinFu Reconstruction")
    if st.button("Start Reconstruction"):
        kinfu.main(CONFIG_PATH, SAVE_DIR)
        st.success("✅ Reconstruction completed")
        if os.path.exists(PLY_PATH):
            st.info("mesh.ply generated. You can view it in [Load and Visualize PLY File]")

# ================== Frame-by-Frame SLAM Visualization ==================
elif choice == options[2]:
    st.subheader("Frame-by-Frame SLAM Visualization (Streamlit)")
    follow = st.checkbox("Follow camera view", value=True)

    if st.button("Start Visualization"):
        st.info("⚙️ Reconstructing mesh from SLAM frames...")
        mesh = kinfu_gui_streamlit.main_streamlit(CONFIG_PATH, follow_camera=follow)

        if mesh is not None and not mesh.is_empty():
            st.success("✅ Mesh reconstructed. Launching Open3D WebRTC viewer...")

            # Start WebRTC viewer
            logging.debug("")
            o3d.visualization.webrtc_server.enable_webrtc()
            logging.debug("")
            o3d.visualization.draw(mesh)
            logging.debug("")

            st.info("🟢 Open3D WebRTC viewer launched in browser tab or embed.")
        else:
            st.error("❌ Mesh reconstruction failed or mesh is empty.")

# ================== Load and Visualize PLY File ==================
elif choice == options[3]:
    st.subheader("Load and Visualize PLY File")
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
            st.success("✅ PLY file visualization completed")
        else:
            st.error("❌ mesh.ply file is empty")
    else:
        st.warning("Please run the reconstruction step first to generate mesh.ply")

# ================== Point Cloud Volume Estimation ==================
elif choice == options[4]:
    st.subheader("Point Cloud Volume Estimation Example")

    shape_options = list(volume_estimation.ShapeGenerator.get_all_shapes().keys())
    shape_choice = st.selectbox("Select shape", shape_options)
    alpha = st.slider("Alpha shape parameter", 1, 30, 10)

    if st.button("Generate and Estimate Volume"):
        points, true_volume = volume_estimation.ShapeGenerator.get_all_shapes()[shape_choice]()
        vol_convex = volume_estimation.VolumeEstimator.estimate_convex_hull(points)
        vol_alpha, is_watertight = volume_estimation.VolumeEstimator.estimate_alpha_shape(points, alpha=alpha)

        st.write(f"Theoretical volume: {true_volume:.4f}" if true_volume else "Theoretical volume: N/A")
        st.write(f"Convex Hull estimated volume: {vol_convex:.4f}")
        st.write(f"Alpha Shape estimated volume: {vol_alpha:.4f} (Watertight: {is_watertight})")

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
        ax.set_title(shape_choice)
        ax.axis('off')
        st.pyplot(fig)