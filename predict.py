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
print("🔵 Loading YOLO models...")
weapon_model = torch.hub.load("ultralytics/yolov5", "custom", path="weapon_model_best.pt", force_reload=False)  # YOLOv5 weapon detection
human_model = YOLO("human_detection_model_best.pt") 
print("✅ YOLO models loaded.")

# Load Facial Expression Model
print("🔵 Loading Facial Expression Model...")
with open("face_expression.json", "r") as json_file:
    loaded_model_json = json_file.read()
expression_model = model_from_json(loaded_model_json, custom_objects={"Sequential": Sequential})
expression_model.build(input_shape=(None, 48, 48, 1))
expression_model.load_weights("face_expression.h5")
print("✅ Facial Expression Model loaded.")

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force MediaPipe to use CPU
import mediapipe as mp

# Initialize MediaPipe Pose
print("🔵 Initializing MediaPipe Pose...")
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
print("✅ MediaPipe Pose initialized.")

# Open webcam
print("🔵 Starting webcam...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set frame size
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

prev_keypoints = None  # Store previous frame keypoints for motion analysis

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Failed to capture frame.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_copy = frame.copy()

    # Process Pose Detection
    results = pose.process(frame_rgb)
    action = "Normal"

   
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

    # Detect Weapons using YOLOv5
    weapon_results = weapon_model(frame_copy)
    for det in weapon_results.pred[0]:
        x1, y1, x2, y2, confidence, cls = det.tolist()
        label = weapon_model.names[int(cls)]
        if confidence > 0.5:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(frame, f"{label} ({confidence:.2f})", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Facial Expression Detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (48, 48)) / 255.0  # Resize and normalize
        face_roi = np.expand_dims(face_roi, axis=[0, -1])
        expression_preds = expression_model.predict(face_roi)
        expression_label = np.argmax(expression_preds)
        expression_text = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"][expression_label]
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, expression_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # If human pose detected, draw skeleton & detect movement
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                                  mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))

        keypoints = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
        nose_y = keypoints[mp_pose.PoseLandmark.NOSE.value][1]
        hand_left_y = keypoints[mp_pose.PoseLandmark.LEFT_WRIST.value][1]
        hand_right_y = keypoints[mp_pose.PoseLandmark.RIGHT_WRIST.value][1]

        if hand_left_y < nose_y and hand_right_y < nose_y:
            action = "Hiding"
        
        if prev_keypoints is not None:
            movement = np.linalg.norm(keypoints - prev_keypoints, axis=1)
            if np.mean(movement) > 0.08:
                action = "Running"
        
        prev_keypoints = keypoints

    # Display action label
    cv2.putText(frame, action, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Show frame
    cv2.imshow("Webcam Detection", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
