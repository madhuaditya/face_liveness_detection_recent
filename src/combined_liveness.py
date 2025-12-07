# # combined_liveness.py
# import cv2
# import tensorflow as tf
# import numpy as np
# import dlib
# from eye_blink_detection import detect_blink
# from expression_detection import detect_expression

# # Load liveness model
# model = tf.keras.models.load_model("models/liveness_model.keras")

# # Load face detector
# face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# # Dlib setup for blink detection
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# detector = dlib.get_frontal_face_detector()

# cap = cv2.VideoCapture(0)
# print("[INFO] Starting camera...")

# blink_done = False
# expression_done = False
# target_expression = "happy"  # You can randomize this

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_detector.detectMultiScale(gray, 1.3, 5)

#     for (x, y, w, h) in faces:
#         face = frame[y:y+h, x:x+w]
#         resized = cv2.resize(face, (128, 128))
#         normalized = resized.astype("float32") / 255.0
#         normalized = np.expand_dims(normalized, axis=0)

#         preds = model.predict(normalized)
#         label = "real" if preds[0][0] > 0.5 else "fake"

#         # Blink detection
#         if not blink_done:
#             blink_done = detect_blink(frame, predictor, detector)
#             if blink_done:
#                 print("[INFO] Blink detected ✅")

#         # Expression detection
#         if not expression_done:
#             exp = detect_expression(frame)
#             if exp == target_expression:
#                 expression_done = True
#                 print(f"[INFO] Expression '{target_expression}' detected ✅")

#         # Combine results
#         if label == "real" and blink_done and expression_done:
#             text = "REAL ✅"
#             color = (0, 255, 0)
#         else:
#             text = "FAKE ❌"
#             color = (0, 0, 255)

#         cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
#         cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

#     cv2.imshow("Combined Liveness Detection", frame)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()


# combined_liveness.py
# import cv2
# import tensorflow as tf
# import numpy as np
# import dlib
# import random
# # from eye_blink_detection import detect_blink
# from expression_detection import detect_expression
# from eye_blink_detection import BlinkDetector


# # Load CNN liveness model
# model = tf.keras.models.load_model("models/liveness_model.keras")

# # Load face detector
# face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# # Dlib setup for blink detection
# predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
# detector = dlib.get_frontal_face_detector()
# blink_detector = BlinkDetector(predictor, detector)

# # Expression challenge options
# expressions = ["happy", "surprise", "neutral"]
# target_expression = random.choice(expressions)
# print(f"[INFO] Please show expression: {target_expression.upper()} 😄")

# cap = cv2.VideoCapture(0)
# print("[INFO] Starting camera...")

# blink_done = False
# expression_done = False
# blinked = False

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_detector.detectMultiScale(gray, 1.3, 5)

#     for (x, y, w, h) in faces:
#         face = frame[y:y+h, x:x+w]
#         resized = cv2.resize(face, (64, 64))
#         normalized = resized.astype("float32") / 255.0
#         normalized = np.expand_dims(normalized, axis=0)

#         # CNN texture check
#         preds = model.predict(normalized)
#         label = "real" if preds[0][0] > 0 else "fake"

#         # Blink detection
#         if not blinked:
#             # blink_done = detect_blink(frame, predictor, detector)
#             blinked = blink_detector.detect_blink(frame)

#             if blinked:
#                 print("[INFO] Blink detected ✅")
#             else :
#                 print("Blink detection fail")

#         # Expression detection
#         if not expression_done:
#             exp = detect_expression(frame)
#             if exp == target_expression:
#                 expression_done = True
#                 print(f"[INFO] Expression '{target_expression}' detected ✅")
#             else :
#                 print("Expression detection fail")

#         # Combine all checks

#         # print("here i have come till")

#         if label == "real" and blinked and expression_done:
#             text = "REAL "
#             color = (0, 255, 0)

#         else:
#             text = f"FAKE X (Need: blink + {target_expression})"
#             color = (0, 0, 255)

#         cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
#         cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

#     cv2.imshow("Advanced Liveness Detection", frame)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2
import tensorflow as tf
import numpy as np
import dlib
import random
from expression_detection import detect_expression
from eye_blink_detection import BlinkDetector

# --- Load Models ---
model = tf.keras.models.load_model("models/liveness_model.keras")

# --- Face Detector ---
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# --- Dlib for Blink Detection ---
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()
blink_detector = BlinkDetector(predictor, detector)

# --- Expression Challenge ---
expressions = ["happy", "surprise", "neutral"]
target_expression = random.choice(expressions)
print(f"[INFO] Please show expression: {target_expression.upper()} 😄")

# --- Camera Setup ---
cap = cv2.VideoCapture(0)
print("[INFO] Starting camera...")

blink_done = False
expression_done = False
blinked = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for mirror effect
    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    display_frame = frame.copy()
    blurred = cv2.GaussianBlur(frame, (55, 55), 50)

    for (x, y, w, h) in faces:
        # Focus only on the face
        mask = np.zeros_like(frame, dtype=np.uint8)
        cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), -1)
        focused = np.where(mask == np.array([255, 255, 255]), frame, blurred)

        # Crop face for liveness CNN
        face = frame[y:y+h, x:x+w]
        resized = cv2.resize(face, (64, 64))
        normalized = resized.astype("float32") / 255.0
        normalized = np.expand_dims(normalized, axis=0)
        face_input = np.expand_dims(face.astype("float") / 255.0, axis=0)
        prediction = model.predict(face_input, verbose=0)[0][0]

        label = "REAL" if prediction > 0.15 else "FAKE"

        # CNN Texture Liveness
        # preds = model.predict(normalized)
        # label = "real" if preds[0][0] > 0.1 else "fake"

        # --- Blink Detection ---
        if not blink_done:
            blinked = blink_detector.detect_blink(frame)
            if blinked:
                blink_done = True
                print("[INFO] Blink detected [✓]")

        # --- Expression Detection ---
        if not expression_done:
            exp = detect_expression(frame)
            if exp == target_expression:
                expression_done = True
                print(f"[INFO] Expression '{target_expression}' detected [✓]")

        # --- Status Color & Text ---
        if label == "real" and blink_done and expression_done:
            text = "REAL [✓]"
            color = (0, 255, 0)
        else:
            text = f"Pending: "
            if not blink_done:
                text += "Blink [✕] "
            if not expression_done:
                text += f"Expr({target_expression}) [✕]"
            color = (0, 0, 255)

        # --- Draw Face Focus ---
        cv2.rectangle(focused, (x, y), (x + w, y + h), color, 3)
        cv2.putText(focused, text, (x, y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Overlay current stage
        status_text = [
            f"Face: {label.upper()}",
            f"Blink: {'pass' if blink_done else 'trying'}",
            f"Expression ({target_expression}): {'pass' if expression_done else 'trying'}"
        ]

        for i, line in enumerate(status_text):
            cv2.putText(focused, line, (10, 40 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        display_frame = focused

    # Show output
    cv2.imshow("Advanced Liveness Detection", display_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

