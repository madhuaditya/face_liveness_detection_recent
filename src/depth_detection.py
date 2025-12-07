# depth_detection.py
import dlib
import cv2
import numpy as np

class DepthDetector:
    def __init__(self, predictor_path="models/shape_predictor_68_face_landmarks.dat"):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

    def get_depth_score(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)

        if len(rects) == 0:
            return 0, False

        rect = rects[0]
        shape = self.predictor(gray, rect)
        points = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])

        # Nose and eye distance for depth estimation
        nose = points[30]
        left_eye = points[36]
        right_eye = points[45]

        eye_dist = np.linalg.norm(left_eye - right_eye)
        nose_eye_avg = (np.linalg.norm(nose - left_eye) + np.linalg.norm(nose - right_eye)) / 2

        depth_ratio = eye_dist / nose_eye_avg

        # Real faces usually give ratio between 1.3 – 1.9
        is_real = 1.2 < depth_ratio < 2.1

        return depth_ratio, is_real


# # depth_midas.py
# import cv2
# import torch
# import numpy as np
# import torchvision.transforms as transforms
# import time

# class DepthDetector:
#     def __init__(self):
#         # Load MiDaS model
#         self.model_type = "DPT_Hybrid"  # best speed + quality

#         self.midas = torch.hub.load("intel-isl/MiDaS", self.model_type)
#         self.midas.eval()

#         # Load transforms
#         self.transform = torch.hub.load("intel-isl/MiDaS", "transforms")
#         if self.model_type in ["DPT_Large", "DPT_Hybrid"]:
#             self.transform = self.transform.dpt_transform
#         else:
#             self.transform = self.transform.small_transform

#         print("[INFO] MiDaS Depth Model Loaded")

#     def get_depth_score(self, face_crop):
#         """
#         face_crop: cropped face (BGR image)
#         returns: depth_variation(float), is_real(bool)
#         """

#         img = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
#         input_batch = self.transform(img).unsqueeze(0)

#         with torch.no_grad():
#             prediction = self.midas(input_batch)

#         prediction = torch.nn.functional.interpolate(
#             prediction.unsqueeze(1),
#             size=img.shape[:2],
#             mode="bicubic",
#             align_corners=False
#         ).squeeze()

#         depth_map = prediction.cpu().numpy()

#         # Normalize for variation calculation
#         depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)

#         variation = np.std(depth_norm)

#         # Threshold:
#         #   REAL face -> depth variation ~ 0.12–0.35
#         #   PHOTO/SCREEN -> depth variation ~ 0.01–0.04
#         is_real = variation > 0.10

#         return variation, is_real
