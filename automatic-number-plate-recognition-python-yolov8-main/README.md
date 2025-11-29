# automatic-number-plate-recognition-python-yolov8
A real-time Automatic Number Plate Recognition (ANPR) system built using YOLOv8, OpenCV, and EasyOCR/Tesseract.
This project detects vehicle number plates from images or live video streams and extracts the alphanumeric text with high accuracy. It also includes automatic parking slot allocation and entry logging.

🔍 Project Overview

This project uses YOLOv8 for fast and accurate number plate detection and integrates OCR to extract plate numbers.
Once a plate is detected, the system:

Reads the number plate text using OCR

Allocates a parking slot automatically

Stores vehicle entry details (plate number, slot, timestamp)

Maintains logs in Excel/CSV format

It can be used in smart parking, toll systems, security gates, and traffic monitoring.

🚀 Features

Real-time number plate detection using YOLOv8

OCR-based text extraction with EasyOCR / Tesseract

Automatic parking slot assignment

Entry logs stored in Excel/CSV

Fast inference (~50 ms per frame)

Supports webcam & video input

Works for Indian number plates & other formats

🛠️ Tech Stack

YOLOv8 (Ultralytics)

Python

OpenCV

EasyOCR / Tesseract OCR

Pandas for logging

Numpy

## models

A Yolov8 pretrained model was used to detect vehicles.

A licensed plate detector was used to detect license plates. The model was trained with Yolov8 using [this dataset](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4) and following this [step by step tutorial on how to train an object detector with Yolov8 on your custom data](https://github.com/computervisioneng/train-yolov8-custom-dataset-step-by-step-guide). 

The trained model is available in my [Patreon](https://www.patreon.com/ComputerVisionEngineer).


