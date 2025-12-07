import cv2
import numpy as np
import tensorflow as tf
from utils import detect_face

import os
model_path = os.path.join(os.getcwd(), "models", "liveness_model.keras")
model = tf.keras.models.load_model(model_path)

# model = tf.keras.models.load_model("../models/liveness_model.keras")
cap = cv2.VideoCapture(0)




print("[INFO] Starting real-time face liveness detection...")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    face = detect_face(frame)

    
    # pred = model.predict(face)[0]
    # print("Predictions:", pred)
    # label = "real" if pred[0] > pred[1] else "fake"
    # confidence = max(pred)
    # print(f"Label: {label}, Confidence: {confidence:.2f}")

    if face is not None:
        face_input = np.expand_dims(face.astype("float") / 255.0, axis=0)
        prediction = model.predict(face_input, verbose=0)[0][0]

        label = "REAL" if prediction > 0.15 else "FAKE"
        color = (0, 255, 0) if label == "REAL" else (0, 0, 255)

        cv2.putText(frame, f"{label} ({prediction*100:.2f}%)", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.rectangle(frame, (20, 20), (620, 460), color, 2)

    cv2.imshow("Face Liveness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
