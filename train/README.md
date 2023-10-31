# Object Tracking and Anomaly Detection
## Introduction
We have built an object tracking an detection system using a Raspberry Pi 3B+ and a Pi Camera. The system is able to detect and track objects in real time and can detect anomalies in the video stream. It is also able to detect and track multiple objects at the same time and is able to detect anomalies in the video stream and can send an alert to the user if an anomaly is detected. 
## Getting Started
### Prerequisites
* Raspberry Pi 3B+
* Pi Camera
* Python
* OpenCV
* Numpy
* Tensorflow
* Keras
## Usage
First go to the train directory:
```
cd train
```
### Object Tracking
1. Run the object tracking script
```
py model.py
```
2. Create dataset for training
```
py model2.py
py arrange.py
```
### Training
1. Run the training script
```
py LSTM_autoencoder.py
```
### Anomaly Detection
1. Run the anomaly detection script
```
py anomaly_detection.py
```
## Description of Architecture
### Object Tracking
1. The object tracking system uses a pre-trained YOLOv8 pose model to detect objects in the video stream.
2. The detected objects are then tracked using the BoT-SORT tracker.
3. The system is able to track multiple objects at the same time.
### Anomaly Detection
1. The anomaly detection system uses an LSTM autoencoder to detect anomalies in the video stream.
2. The LSTM autoencoder is trained on the MOT dataset.
3. The system is able to detect anomalies in the video stream and can send an alert to the user if an anomaly is detected.

## Authors
* **[Suryansh Goel](github.com/surya2003-real)**

