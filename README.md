# Real-Time-Object-Detection-on-Raspberry-pi-using-Tensorflow-2
Applying Object Detection Algorithm on devices like raspberry pi can be difficult especially when real time object detection is to be performed.

1. Image Collection Jupyter Notebook: Set Up Files and Floder and used to collect dataset, label and annotate using bounding box Technique.
2. Training and Detection Jupyter Notebook: Set up Tensorflow files, installation of Tensorflow, downloading pre-trained model from Tensorflow 2 Model Zoo, creating Label Map, construting Tensorflow record for training and test dataset, customizing congiguration files according to needs, train and evaluate model and performed detection on windows machine, converting model to TFLite format therefore model can run on raspberry pi.

Camera: Mobile Phone camera is used with rasoberry pi for detection using ip connection and andriod app IP Webcam.

In This project hand sign are detected based on Brazilian sign language alphabet taken from mandeley dataset platform. Link: https://data.mendeley.com/datasets/k4gs3bmx5k/5

To run detection on raspberry pi, connect camera to pi and clone PI folder into it and detect tflite file and insert or move your own TFlite file and customized label according to your needs. But prior to it, install tensorflow 2 on your raspberry pi systems.
