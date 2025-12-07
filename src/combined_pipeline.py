import cv2
import numpy as np
import tensorflow as tf
import dlib
import random
from fer.fer import FER
from eye_blink_detection import BlinkDetector


# ------------------------------
# Load ML Model (MobileNet)
# ------------------------------
model = tf.keras.models.load_model("models/liveness_mobilenet.keras")

# ------------------------------
# Face Detector
# ------------------------------
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ------------------------------
# Dlib Blink Detector
# ------------------------------
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()
blink_detector = BlinkDetector(predictor, detector)

# ------------------------------
# Expression Detector
# ------------------------------
expression_detector = FER(mtcnn=True)

# Random expression challenge
expressions = ["happy", "surprise", "neutral"]
TARGET_EXPRESSION = random.choice(expressions)

print(f"[INFO] Expression challenge: SHOW {TARGET_EXPRESSION.upper()} 😄")

# ------------------------------
# Webcam
# ------------------------------
cap = cv2.VideoCapture(0)

blink_pass = False
expression_pass = False
model_pass = False


# ------------------------------
# Utility: Blur background except face
# ------------------------------
def blur_background(frame, x, y, w, h):
    blurred = cv2.GaussianBlur(frame, (55, 55), 0)
    blurred[y:y+h, x:x+w] = frame[y:y+h, x:x+w]
    return blurred


# ------------------------------
# Main Loop
# ------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        (x, y, w, h) = faces[0]

        # Blur everywhere except face
        frame = blur_background(frame, x, y, w, h)

        face_roi = frame[y:y+h, x:x+w]

        # ------------------------------
        # 1️⃣ Liveness (CNN Texture Check)
        # ------------------------------
        face_resized = cv2.resize(face_roi, (224, 224))
        norm = face_resized.astype("float32") / 255.0
        norm = np.expand_dims(norm, axis=0)

        pred = model.predict(norm, verbose=0)[0][0]
        model_pass = pred > 0.5  # threshold

        # ------------------------------
        # 2️⃣ Blink Detection
        # ------------------------------
        if not blink_pass:
            blink_pass = blink_detector.detect_blink(frame)

        # ------------------------------
        # 3️⃣ Expression Challenge
        # ------------------------------
        if not expression_pass:
            exp_result = expression_detector.top_emotion(face_roi)
            if exp_result is not None:
                detected_exp, score = exp_result
                if detected_exp == TARGET_EXPRESSION:
                    expression_pass = True

        # ------------------------------
        # Combine decision
        # ------------------------------
        if model_pass and blink_pass and expression_pass:
            main_label = "REAL ✔"
            color = (0, 255, 0)
        else:
            main_label = "FAKE ✖"
            color = (0, 0, 255)

        # ------------------------------
        # Draw face box
        # ------------------------------
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        cv2.putText(frame, main_label, (x, y-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # ------------------------------
        # Stage indicators
        # ------------------------------
        y0 = y + h + 30

        cv2.putText(frame,
            f"Texture: {'PASS' if model_pass else 'FAIL'}",
            (x, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0,255,0) if model_pass else (0,0,255), 2)

        cv2.putText(frame,
            f"Blink: {'PASS' if blink_pass else 'FAIL'}",
            (x, y0+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0,255,0) if blink_pass else (0,0,255), 2)

        cv2.putText(frame,
            f"Expression ({TARGET_EXPRESSION}): {'PASS' if expression_pass else 'FAIL'}",
            (x, y0+60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0,255,0) if expression_pass else (0,0,255), 2)

    cv2.imshow("Liveness Detection (Advanced)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
