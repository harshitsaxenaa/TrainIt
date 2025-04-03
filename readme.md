Real-Time Suspicious Activity Detection & Tracking System
Github: https://github.com/harshitsaxenaa/TrainIt

# Overview:

This project aims to develop a real-time surveillance system capable of detecting humans, weapons, suspicious activities, and facial expressions to identify possible threats. The system can be deployed in banks, CCTV surveillance, public spaces, airports, and other security-sensitive areas to enhance safety and prevent crimes.

# Applications:

1. In public places, home CCTVs, surveillance areas, buses etc. to detect ill-intentions and acts.
2. This can be extended to be used in forests and animal reserves to detect acts like animal poaching and hunting.
3. It also finds applications in markets, jewellery shops, etc.

# Features:

--> Human Detection – Identifies people in video feeds using human detection and marks boundaries.

--> Weapon Detection – Recognizes guns, knives, and other suspicious objects.

--> Pose Estimation – Detects aggressive actions and unusual body movements like running, stealing, unusual roaming etc.

--> Facial Expression Analysis – Classifies emotions and expressions like fear, tensed etc. to detect potential threats.

--> Real-Time Tracking – Monitors individuals over time to track movements.

--> Anomaly Detection – Flags suspicious behavior based on predefined patterns.

# Technologies Used:

--> Programming Language:
Python

--> Frameworks & Libraries:
YOLOv5 (fine tuned on datasets) – Object detection (humans, weapons, objects)

--> MediaPipe – Human pose estimation

--> TensorFlow/Keras – CNN-based facial expression recognition

--> OpenCV – Video processing and real-time feed handling

--> PyTorch – Model training and inference

--> Scikit-learn – Data preprocessing and model evaluation

# Hardware Requirements:

--> CPU – Minimum Intel i5 or AMD Ryzen 5

--> GPU – Recommended NVIDIA GPU with CUDA support for faster inference

--> RAM – At least 8GB (16GB recommended)

# Dataset Information:

1. Object Detection (Humans & Weapons):

1.1 Weapon Dataset: https://www.kaggle.com/datasets/raghavnanjappan/weapon-dataset-for-yolov5, https://www.kaggle.com/code/kumariritika/framing

1.2 Human Detection Dataset: https://www.kaggle.com/datasets/constantinwerner/human-detection-dataset

1.3 General Object Detection Dataset: https://www.kaggle.com/datasets/constantinwerner/human-detection-dataset

2. Facial Expression Recognition
   
2.1 Face Expression Dataset: https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset

# System Architecture:
Video Feed Capture:

Capture real-time video from CCTV/IP cameras using OpenCV.

Human & Weapon Detection:

Use YOLOv5 (fine tuned) for detecting humans and weapons.

Pose Estimation for Activity Recognition:

Use MediaPipe to analyze body movements and detect suspicious activities.

Facial Expression Recognition:

Classify emotions such as anger, fear, or stress using a CNN-based model.

Tracking & Anomaly Detection:

Implement DeepSORT tracking to monitor individuals and flag unusual behaviors.

Real-Time Alert System:

Generate alerts when a suspicious action, object, or expression is detected.

# GitHub Repo link: https://github.com/harshitsaxenaa/TrainIt

## Installation & Setup
Step 1: Clone the Repository

git clone "https://github.com/harshitsaxenaa/TrainIt"

cd TrainIt

Step 2: Create a Virtual Environment

python -m venv venv

source venv/bin/activate   # On macOS/Linux

venv\Scripts\activate      # On Windows


Step 3: Install Dependencies

To install dependencies, run:

pip install -r requirements.txt



Execution Instructions:
Running the System:

python predict.py (for webcam based)

or

python predict_video.py (for video) #Edit the video path in the script


# Expected Output:

--> The system will display real-time detection of humans and weapons with bounding boxes.

--> Weapons and suspicious objects will be highlighted.

--> Anomalous activities will trigger alerts.

--> The pose of person shall be detected and displayed using pointers and markers and tracked for unusual movements.

