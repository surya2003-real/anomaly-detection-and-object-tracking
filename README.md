# Anomaly Detection and Object Tracking (ADOT)
## Introduction
ADOT model offers you the features for detecting anomalies in various public places such as airports, railway stations, bus stations, etc. It also provides the feature of tracking the detected objects.

To train the model read the [README.md](./train/README.md) file in the train folder.
Here I will explain how to use the model for detecting anomalies and tracking objects.

## Getting Started
So let's get started to run the model for detecting anomalies and tracking objects at realtime.
## Requirements
- Python
- Tensorflow
- OpenCV
- Numpy
- Matplotlib
- Pillow
- Scikit-learn
- argparse

## Usage
### Detecting Anomalies
To detect anomalies in a video, run the following command:
```
cd realtime_anomaly_detection
```
```
py adot.py --video_path <path to video> --threshold <threshold value>
```
For feeding images through webcam, run the following command, put 0 in place of video path:
```
py adot.py --video_path 0 --threshold <threshold value>
```
Note: The threshold value should be between 0 and 1. The default value is 0.02. For larger videos with more field of view, the threshold value should be greater(about 0.2). For smaller videos confined to a small area, the threshold value should be smaller(about 0.02).

The code will run and display the video with the detected anomalies in the realtime.

## Model Architecture
### Live Data Capture: 
Begin by capturing real-time environmental data using a Raspberry Pi camera.​

### Human Detection with "BoT-SORT": 
Employ "BoT-SORT" to identify human subjects and track their movements. This system is effective even when humans are partially obscured.​

### Pose Estimation using YOLOV8-Pose: 
Use YOLOV8 to determine the poses of detected humans, providing detailed information about their body orientations using 17 keypoints which detects eyes, hands, etc.​

### Anomaly Detection with LSTM Autoencoders: 
Apply LSTM autoencoders for time-series analysis to predict the future positions and poses of humans. Anomalies are detected when prediction errors surpass a defined threshold.​

### Visual Alarm Activation: 
Upon anomaly detection, trigger a visual alarm (e.g., blinking an LED light) by sending a signal through the WiFi module.​
## Authors
* **[Suryansh Goel](www.github.com/surya2003-real)**

