import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load Model
model = load_model("facial_expression_model.h5")

# Define Class Labels
class_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Load Image
image_path = "sample.jpg"  # Change this to any image of a face
image = cv2.imread(image_path)
image_resized = cv2.resize(image, (48, 48))
image_resized = image_resized / 255.0
image_resized = np.expand_dims(image_resized, axis=0)

# Predict Emotion
predictions = model.predict(image_resized)
emotion = class_labels[np.argmax(predictions)]

# Show Image with Label
cv2.putText(image, emotion, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.imshow("Prediction", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
