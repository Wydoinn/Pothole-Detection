# Pothole Detection


## Overview

This project implements an advanced pothole detection system using computer vision and deep learning techniques. By leveraging the YOLOv8-small model, we've created a robust and efficient solution for identifying and localizing potholes in road images and videos.


## Demo

[Demp Video](https://github.com/user-attachments/assets/83ee4342-de83-43a4-9d01-b5713b117c8e)


## Key Features

- YOLOv8-small Model: Utilizes the compact yet powerful YOLOv8-small architecture for real-time object detection and segmentation.
- Multi-format Input: Processes both images and videos for versatile application.
- Real-time Detection: Achieves fast inference times, suitable for mobile and edge devices.
- User-friendly Interface: Implemented with Streamlit for easy interaction.


## Technology Stack

- Deep Learning Framework: YOLO (You Only Look Once) v8
- Computer Vision: OpenCV and Supervision
- Data Processing: NumPy
- UI: Streamlit


## Setup and Installation

1. Clone the repository:
   ```
   git clone https://github.com/Wydoinn/Pothole-Detection.git
   cd pothole-detection
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```
   streamlit run app.py
   ```


## Usage

- Use the Streamlit app to upload images or videos for pothole detection.
- Adjust confidence thresholds and other parameters as needed. 
