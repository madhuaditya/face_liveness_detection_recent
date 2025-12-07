# # depth_midas.py
# import cv2
# import torch
# import numpy as np

# class MiDaSDepthDetector:
#     def __init__(self):
#         print("[INFO] Loading MiDaS model...")

#         self.midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
#         self.midas.to("cuda" if torch.cuda.is_available() else "cpu").eval()

#         self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

#     def get_depth_score(self, frame):
#         # Convert frame to RGB
#         img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         # Preprocess
#         input_batch = self.transform(img).to("cuda" if torch.cuda.is_available() else "cpu")

#         # Inference
#         with torch.no_grad():
#             prediction = self.midas(input_batch)

#         depth_map = prediction.squeeze().cpu().numpy()

#         # Normalize
#         depth_norm = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)

#         # Extract face depth variation
#         std_dev = np.std(depth_norm)

#         # REAL FACE: expected depth variation > 0.05
#         is_real = std_dev > 0.05

#         return std_dev, is_real


# if __name__ == "__main__":
#     cap = cv2.VideoCapture(0)
#     detector = MiDaSDepthDetector()

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         score, is_real = detector.get_depth_score(frame)
#         label = "REAL" if is_real else "FAKE"
#         color = (0, 255, 0) if is_real else (0, 0, 255)

#         cv2.putText(frame, f"Depth Score: {score:.3f}", (10, 40),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
#         cv2.putText(frame, f"Depth Verdict: {label}", (10, 80),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

#         cv2.imshow("MiDaS Depth Detector", frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# import cv2
# import numpy as np
# import onnxruntime as ort

# class ONNXMiDaS:
#     def __init__(self, model_path="midas_v21_small_256.onnx"):
#         print("[INFO] Loading MiDaS ONNX model...")

#         self.session = ort.InferenceSession(
#             model_path,
#             providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
#         )

#         # Get input and output names for inference
#         self.input_name = self.session.get_inputs()[0].name
#         self.output_name = self.session.get_outputs()[0].name

#     def preprocess(self, frame):
#         img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         img = cv2.resize(img, (256, 256))
#         img = img.astype(np.float32) / 255.0
#         img = img.transpose(2, 0, 1)
#         img = np.expand_dims(img, axis=0)
#         return img

#     def get_depth_score(self, frame):
#         inp = self.preprocess(frame)

#         pred = self.session.run([self.output_name], {self.input_name: inp})[0]
#         depth = pred.squeeze()

#         depth_norm = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX)

#         # Variation detection
#         stddev = float(np.std(depth_norm))
#         is_real = stddev > 0.05   # adjust if needed

#         return stddev, is_real


# if __name__ == "__main__":
#     cap = cv2.VideoCapture(0)
#     detector = ONNXMiDaS("models/midas_v21_small_256.onnx")

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         score, is_real = detector.get_depth_score(frame)

#         label = "REAL" if is_real else "FAKE"
#         color = (0,255,0) if is_real else (0,0,255)

#         cv2.putText(frame, f"Depth Score: {score:.3f}", (10, 40),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
#         cv2.putText(frame, f"Verdict: {label}", (10, 80),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

#         cv2.imshow("ONNX MiDaS Depth", frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

import cv2
import numpy as np
import onnxruntime as ort

class ONNXMiDaS:
    def __init__(self, model_path="midas_v21_small_256.onnx"):
        print("[INFO] Loading MiDaS ONNX model...")

        self.session = ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        return img

    def get_depth_score(self, face_roi):
        inp = self.preprocess(face_roi)

        pred = self.session.run([self.output_name], {self.input_name: inp})[0]
        depth = pred.squeeze()

        depth_norm = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX)

        stddev = float(np.std(depth_norm))
        is_real = stddev > 0.27   # adjust threshold

        return stddev, is_real


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    detector = ONNXMiDaS("models/midas_v21_small_256.onnx")

    # Load face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        if len(faces) > 0:
            (x, y, w, h) = faces[0]  # first face
            face_roi = frame[y:y+h, x:x+w]

            score, is_real = detector.get_depth_score(face_roi)
            label = "REAL" if is_real else "FAKE"
            color = (0,255,0) if is_real else (0,0,255)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"Depth Score: {score:.3f}", (x, y-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, label, (x, y-50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            cv2.putText(frame, "No Face Detected", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.imshow("ONNX MiDaS Depth", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# from ultralytics import YOLO
# import cv2
# import numpy as np
# from onnx_midas import ONNXMiDaS   # your class

# # Load YOLO face detector
# face_yolo = YOLO("yolov8n-face.pt")

# # Load MiDaS
# midas = ONNXMiDaS("models/midas_v21_small_256.onnx")

# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Run YOLO face detection
#     results = face_yolo(frame, verbose=False)

#     if len(results[0].boxes) > 0:
#         box = results[0].boxes[0].xyxy[0].cpu().numpy()
#         x1, y1, x2, y2 = map(int, box)

#         face_roi = frame[y1:y2, x1:x2]

#         # Depth test on face only
#         score, is_real = midas.get_depth_score(face_roi)

#         color = (0,255,0) if is_real else (0,0,255)
#         label = "REAL" if is_real else "FAKE"

#         # Draw box
#         cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
#         cv2.putText(frame, f"{label} ({score:.3f})", (x1, y1-10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

#     cv2.imshow("YOLO + MiDaS Liveness", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
