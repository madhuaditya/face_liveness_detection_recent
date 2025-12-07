

import cv2
from ultralytics import YOLO

# Load YOLO device detector
device_detector = YOLO("yolov8n.pt")

FORBIDDEN_CLASSES = ["cell phone", "laptop", "tv", "monitor"]

def detect_phone_or_screen(frame):
    results = device_detector(frame, verbose=False)[0]
    for box in results.boxes:
        cls = int(box.cls[0])
        label = results.names[cls]
        if label in FORBIDDEN_CLASSES:
            return True, label
    return False, None


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Step 1 — reject immediately if phone/laptop detected
    phone_present, label = detect_phone_or_screen(frame)
    if phone_present:
        cv2.putText(frame, f"REJECT (Detected: {label})", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
        cv2.imshow("Liveness", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Step 2 — Continue with face liveness...
    # face detection
    # depth check
    # anti-spoof CNN
    # blink detection
    # (Your existing liveness code here)

    cv2.imshow("Liveness", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
