# Autonomous-Wheel-Loader-Perception

## Pipeline
```shell
python src\slam\dataset\preprocess.py --config src\slam\configs\fr1_room.yaml
python src\slam\kinfu.py --config src\slam\configs\fr1_room.yaml --save_dir data\reconstruct\slam
python src\slam\kinfu_gui.py --config src\slam\configs\fr1_room.yaml # Frame-by-frame SLAM Visualization
python src\visualization\visualize_ply.py data\reconstruct\slam\mesh.ply # Final Result Visualization
```