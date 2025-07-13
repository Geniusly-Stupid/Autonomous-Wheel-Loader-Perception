# Autonomous-Wheel-Loader-Perception


## Code Structure

```
Autonomous-Wheel-Loader-Perception/
├── README.md                         # Project overview and instructions
├── data/                             # Sensor data and calibration files
├── src/                              # Main source code
│   ├── config/                       # Configuration files and parameters
│   │   └── config.yaml
│   ├── preprocessing/                # Preprocessing module
│   ├── slam/                         # SLAM module
│   ├── volume_estimation/            # Volume Estimation module
│   │   └── registration.py
│   │   └── cleaning.py
│   │   └── volume_estimation.py
│   ├── visualization/                # Visualization utilities
├── tests/                            # Unit and integration tests
├── main.py                           # Entry point for running the full pipeline
├── demo.py                           # Visualization the full pipeline
└── requirements.txt                  # Python dependency list
```

## Pipeline
```shell
python src/slam/dataset/preprocess.py --config src/slam/configs/fr1_room.yaml
python src/slam/kinfu.py --config src/slam/configs/fr1_room.yaml --save_dir data/reconstruct/slam
python src/slam/kinfu_gui.py --config src/slam/configs/fr1_room.yaml # Frame-by-frame SLAM Visualization
python src/visualization/visualize_ply.py data/reconstruct/slam/mesh.ply # Final Result Visualization
```

## Reference

- [KinectFusion-python](https://github.com/shiyoung77/KinectFusion-python): A Python implementation of KinectFusion.
