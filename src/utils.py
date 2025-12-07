import cv2
import numpy as np

def detect_face(frame):
    """Detects face in frame and returns cropped face."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return None

    (x, y, w, h) = faces[0]
    face = frame[y:y+h, x:x+w]
    return cv2.resize(face, (64, 64))
