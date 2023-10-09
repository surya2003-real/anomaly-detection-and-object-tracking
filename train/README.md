# Object Tracking and Anomaly Detection
## Introduction
We have built an object tracking an detection system using a Raspberry Pi 3B+ and a Pi Camera. The system is able to detect and track objects in real time and can detect anomalies in the video stream. It is also able to detect and track multiple objects at the same time and is able to detect anomalies in the video stream and can send an alert to the user if an anomaly is detected. 
## Getting Started
### Prerequisites
* Raspberry Pi 3B+
* Pi Camera
* Python 3.7
* OpenCV 4.1.0
* Numpy 1.16.4
* Tensorflow 1.14.0
* Keras 2.2.4

### Installation
1. Install OpenCV 4.1.0
```
pip install opencv-python==4.1.0
```
2. Install Numpy 1.16.4
```
pip install numpy==1.16.4
```
3. Install Tensorflow 1.14.0
```
pip install tensorflow==1.14.0
```
4. Install Keras 2.2.4
```
pip install keras==2.2.4
```
## Usage
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
2. The LSTM autoencoder is trained on the CUHK Avenue dataset.
3. The system is able to detect anomalies in the video stream and can send an alert to the user if an anomaly is detected.

## Authors
* **[Suryansh Goel](github.com/surya2003-real)**

