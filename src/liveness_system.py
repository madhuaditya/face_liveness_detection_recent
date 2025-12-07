import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
import dlib
from scipy.spatial import distance
from collections import deque

# ==============================================================
# 1. ---- Phone / Screen Detection (YOLO) ------------------------
# ==============================================================

device_detector = YOLO("models/yolov8n.pt")
FORBIDDEN_CLASSES = ["cell phone", "laptop", "tv", "monitor"]

def detect_phone_or_screen(frame):
    results = device_detector(frame, verbose=False)[0]
    for box in results.boxes:
        cls = int(box.cls[0])
        label = results.names[cls]
        if label in FORBIDDEN_CLASSES:
            return True, label
    return False, None


# ==============================================================
# 2. ---- Depth Detector (MiDaS ONNX) ---------------------------
# ==============================================================

class MiDaSDepth:
    def __init__(self, model_path="models/midas_v21_small_256.onnx"):
        self.session = ort.InferenceSession(
            model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.input = self.session.get_inputs()[0].name
        self.output = self.session.get_outputs()[0].name

    def preprocess(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        return np.expand_dims(img, axis=0)

    def get_depth_score(self, face_roi):
        inp = self.preprocess(face_roi)
        pred = self.session.run([self.output], {self.input: inp})[0]
        depth = pred.squeeze()
        depth_norm = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX)
        stddev = float(np.std(depth_norm))
        is_real = stddev > 0.27
        return stddev, is_real


# ==============================================================
# 3. ---- Spoof Detection CNN (ONNX instead of TensorFlow) ------
# ==============================================================

class SpoofNetONNX:
    def __init__(self, model_path="models/liveness_model.onnx"):
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.input = self.session.get_inputs()[0].name
        self.output = self.session.get_outputs()[0].name

    def predict(self, face):
        # Correct preprocessing for your ONNX model
        face = cv2.resize(face, (64, 64))  # expected size
        face = face.astype(np.float32) / 255.0
        face = np.expand_dims(face, axis=0)  # (1, 64, 64, 3)

        pred = self.session.run([self.output], {self.input: face})[0]
        return float(pred[0][0])


# ==============================================================
# 4. ---- Blink Detector (dlib EAR) ------------------------------
# ==============================================================

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

class BlinkDetector:
    def __init__(self, predictor_path="models/shape_predictor_68_face_landmarks.dat"):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

        self.blink_thresh = 0.21      # fixed threshold
        self.consec_frames = 2        # stable blink
        self.counter = 0
        self.ear_history = deque(maxlen=3)
        self.blinked = False

    def detect_blink(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)

        if len(rects) == 0:
            return False

        rect = rects[0]

        # face too small → unreliable landmarks
        if rect.width() < 80:
            return False
        
        # print("hello eye detection")

        shape = self.predictor(gray, rect)
        shape = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

        leftEye = shape[36:42]
        rightEye = shape[42:48]

        ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
        self.ear_history.append(ear)

        smooth_ear = sum(self.ear_history) / len(self.ear_history)

        if smooth_ear < self.blink_thresh:
            self.counter += 1
        else:
            if self.counter >= self.consec_frames:
                self.blinked = True

            # self.counter = 0
        # print(self.counter)
        detected = self.blinked
        self.blinked = False  # RESET AFTER EACH FRAME
        return detected




# ==============================================================
# 5. ---- Expression Detection (FER) -----------------------------
# ==============================================================

try:
    from fer.fer import FER
except:
    from fer import FER

expression_detector = FER(mtcnn=True)

def get_expression(frame):
    try:
        result = expression_detector.detect_emotions(frame)
        if not result:
            return "neutral"
        emotions = result[0]["emotions"]
        return max(emotions, key=emotions.get)
    except:
        return "neutral"


# ==============================================================
# 6. ---- Main Liveness System ----------------------------------
# ==============================================================

print("[INFO] Initializing models...")
depth_model = MiDaSDepth()
spoof_model = SpoofNetONNX()
blink_detector = BlinkDetector()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

print("[INFO] Starting unified liveness detection...")
depth_real=False
spoof_real = False
blinked = False
depth_real =False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ---------------- STEP 1: Phone/Screen detection ----------------
    phone_present, device_label = detect_phone_or_screen(frame)
    if phone_present:
        cv2.putText(frame, f"REJECT: {device_label} detected", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.imshow("Liveness", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    # ---------------- STEP 2: Face detection ------------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    if len(faces) == 0:
        cv2.putText(frame, "No Face Detected", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.imshow("Liveness", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    x, y, w, h = faces[0]
    face_roi = frame[y:y+h, x:x+w]

    # ---------------- STEP 3: Depth check ---------------------------
    depth_score, depth_real_Temp = depth_model.get_depth_score(face_roi)
    depth_real = depth_real_Temp

    # ---------------- STEP 4: Spoof classifier ----------------------
    prob_real = spoof_model.predict(face_roi)
    spoof_real = prob_real > 0.50

    # ---------------- STEP 5: Blink check ---------------------------
    if not blinked :
        blinked = blink_detector.detect_blink(frame)
        # print(blinked)

    # ---------------- STEP 6: Expression (optional) -----------------
    expression = get_expression(frame)

    # ---------------- FINAL DECISION LOGIC ---------------------------
    if not depth_real:
        decision = "REJECT: Fake depth"
        color = (0, 0, 255)
    elif not spoof_real:
        decision = "REJECT: Spoof detected"
        color = (0, 0, 255)
    elif not blinked:
        decision = "REJECT: No blink"
        color = (0, 0, 255)
    else:
        decision = "ACCEPT: LIVE PERSON"
        color = (0, 255, 0)

    # ---------------- Display Output Panel ---------------------------
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    cv2.putText(frame, decision, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, f"Depth: {depth_score:.3f}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, .8, color, 2)
    cv2.putText(frame, f"Spoof Prob: {prob_real*100:.1f}%", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, .8, color, 2)
    cv2.putText(frame, f"Blink: {blinked}", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, .8, color, 2)
    cv2.putText(frame, f"Expr: {expression}", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, .8, color, 2)

    cv2.imshow("Unified Liveness System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
