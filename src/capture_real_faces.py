# capture_real_faces.py
import cv2
import os
import time
import random
from datetime import datetime

# Paths
base_dir = "dataset"
splits = ["train", "val", "test"]
subfolder = "real"

# Create directories if not exist
for split in splits:
    path = os.path.join(base_dir, split, subfolder)
    os.makedirs(path, exist_ok=True)

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
print("[INFO] Starting webcam... please look at the camera.")
time.sleep(2)

count = 0
max_images = 1000   # total images to collect

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (256, 256))

        # Decide folder (train/val/test)
        # rand = random.random()
        # if rand < 0.7:
        #     split = "train"
        # elif rand < 0.9:
        #     split = "val"
        # else:
        #     split = "test"
        split = "test"

        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        cv2.imwrite(os.path.join(base_dir, split, subfolder, filename), face)
        count += 1

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Captured: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Capture Real Face Data", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q') or count >= max_images:
        break

cap.release()
cv2.destroyAllWindows()
print(f"[INFO] Saved {count} face images in train/val/test folders.")
