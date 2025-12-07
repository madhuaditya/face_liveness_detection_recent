# combined_advanced_liveness.py
import cv2
import tensorflow as tf
import numpy as np
import dlib
import random

from eye_blink_detection import BlinkDetector
from expression_detection import detect_expression
from depth_detection import DepthDetector
from reflection_screen_detector import detect_screen_reflection,detect_moire


# Load CNN model (texture spoof detector)
model = tf.keras.models.load_model("models/liveness_model.keras")

# Haar face detector
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Blink + Landmark Modules
predictor_path = "models/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)
detector = dlib.get_frontal_face_detector()

blink_detector = BlinkDetector(predictor, detector)
depth_detector = DepthDetector(predictor_path=predictor_path)

# Expression challenge
expressions = ["happy", "neutral"]
target_expression = random.choice(expressions)
print(f"[INFO] Show expression → {target_expression.upper()}")

cap = cv2.VideoCapture(0)

blink_done = False
expression_done = False
depth_pass = False
screen_pass = False
cnn_pass = False


def blur_background(frame, x, y, w, h):
    blurred = cv2.GaussianBlur(frame, (55, 55), 30)
    focused = blurred.copy()
    focused[y:y+h, x:x+w] = frame[y:y+h, x:x+w]
    return focused


while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        ### --- FACE ROI --- ###
        face_roi = frame[y:y+h, x:x+w]
        final_display = blur_background(frame, x, y, w, h)

        ### 1️⃣ CNN TEXTURE CHECK ###
        resized = cv2.resize(face_roi, (64, 64))
        normalized = resized.astype("float32") / 255.0
        normalized = np.expand_dims(normalized, axis=0)

        pred = model.predict(normalized)[0][0]
        print(pred)

        cnn_pass = pred > 0.03  # 1 = real, 0 = fake

        ### 2️⃣ BLINK CHECK ###
        if not blink_done:
            blink_done = blink_detector.detect_blink(frame)

        ### 3️⃣ EXPRESSION CHALLENGE ###
        if not expression_done:
            exp = detect_expression(frame)
            if exp == target_expression:
                expression_done = True

        ### 4️⃣ DEPTH CHECK ###
        depth_score, depth_real = depth_detector.get_depth_score(frame)
        depth_pass = depth_real

        ### 5️⃣ SCREEN REFLECTION TEST ###
        screen_real = detect_screen_reflection(face_roi) # detect_screen_texture(face_roi)
        screen_pass = screen_real
        is_moire = detect_moire(frame)

        ### Combine logic ###
        all_pass = blink_done and expression_done and depth_pass and screen_pass and cnn_pass and is_moire

        color = (0, 255, 0) if all_pass else (0, 0, 255)
        text = "REAL FACE ✓" if all_pass else "FAKE / SPOOF ✗"

        cv2.rectangle(final_display, (x, y), (x+w, y+h), color, 3)
        cv2.putText(final_display, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        ### STAGE STATUS PANEL ###
        panel_y = y + h + 20
        cv2.putText(final_display, f"Blink: {'✔' if blink_done else '✘'}",
                    (x, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0,255,0) if blink_done else (0,0,255), 2)

        cv2.putText(final_display, f"Expression({target_expression}): {'✔' if expression_done else '✘'}",
                    (x, panel_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0,255,0) if expression_done else (0,0,255), 2)

        cv2.putText(final_display, f"Depth 3D: {'✔' if depth_pass else '✘'}",
                    (x, panel_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0,255,0) if depth_pass else (0,0,255), 2)

        cv2.putText(final_display, f"Texture Real: {'✔' if screen_pass and is_moire else '✘'}",
                    (x, panel_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0,255,0) if screen_pass else (0,0,255), 2)

        cv2.putText(final_display, f"CNN: {'✔' if cnn_pass else '✘'}",
                    (x, panel_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0,255,0) if cnn_pass else (0,0,255), 2)

        frame = final_display

    cv2.imshow("Advanced Liveness Detector (AI Shield)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
