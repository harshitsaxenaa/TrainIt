import cv2
import numpy as np
import torch
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from ultralytics import YOLO
from keras.models import model_from_json, Sequential
from keras.saving import register_keras_serializable

register_keras_serializable()(Sequential)

@register_keras_serializable()
class CustomSequential(Sequential):
    pass

# Load YOLO models
weapon_model = torch.hub.load("ultralytics/yolov5", "custom", path="weapon_model_best.pt", force_reload=False)  # YOLOv5 weapon detection
human_model = YOLO("human_detection_model_best.pt") 

# Load Facial Expression Model
with open("face_expression.json", "r") as json_file:
    loaded_model_json = json_file.read()
expression_model = model_from_json(loaded_model_json, custom_objects={"Sequential": Sequential})
expression_model.build(input_shape=(None, 48, 48, 1))
expression_model.load_weights("face_expression.h5")

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force MediaPipe to use CPU
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load video instead of webcam
video_path = "input_video.mp4.mp4"  # Change this to your video file path
cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)  # Use FFMPEG for better decoding

if not cap.isOpened():
    print(f"❌ Error: Could not open {video_path}")
    exit()

frame_count = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("✅ Video processing complete.")
        break  # Exit loop when video ends

    frame_count += 1
    print(f"📹 Processing Frame {frame_count}")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_copy = frame.copy()

    # Human detection
    human_results = human_model(frame_copy)
    for result in human_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0].item())
            cls = int(box.cls[0].item())
            if cls == 0 and confidence > 0.5:  # Class 0 = Human
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Human ({confidence:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Weapon detection
    weapon_results = weapon_model(frame_copy)
    for det in weapon_results.pred[0]:
        x1, y1, x2, y2, confidence, cls = det.tolist()
        label = weapon_model.names[int(cls)]
        if confidence > 0.5:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(frame, f"{label} ({confidence:.2f})", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    #  Face Detection + Expression Recognition
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        roi_gray = gray_frame[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1) / 255.0

        expression_prediction = expression_model.predict(roi_gray)
        expression_label = np.argmax(expression_prediction)
        expression_text = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"][expression_label]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f"{expression_text}", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Pose Detection using MediaPipe
    pose_results = pose.process(frame_rgb)
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display frame count on screen
    cv2.putText(frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show frame
    cv2.imshow("Video Detection", frame)

    # Ensure proper video playback speed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
