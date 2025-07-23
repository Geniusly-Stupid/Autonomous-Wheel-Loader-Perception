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
### SLAM
```shell
python src\slam\dataset\preprocess.py --config src\slam\configs\new_dataset.yaml
python src\slam\kinfu_gui.py --config src\slam\configs\new_dataset.yaml # Frame-by-frame SLAM Visualization
```
### Volume Estimation
```shell
python main.py
```

## Reference

- [KinectFusion-python](https://github.com/shiyoung77/KinectFusion-python): A Python implementation of KinectFusion.
