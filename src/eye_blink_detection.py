# # eye_blink_detection.py
# import cv2
# import dlib
# from scipy.spatial import distance

# # Calculate EAR
# def eye_aspect_ratio(eye):
#     A = distance.euclidean(eye[1], eye[5])
#     B = distance.euclidean(eye[2], eye[4])
#     C = distance.euclidean(eye[0], eye[3])
#     ear = (A + B) / (2.0 * C)
#     return ear

# def detect_blink(frame, predictor, detector, blink_thresh=0.22):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     rects = detector(gray, 0)
#     blinked = False

#     for rect in rects:
#         shape = predictor(gray, rect)
#         shape = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
#         leftEye = shape[36:42]
#         rightEye = shape[42:48]

#         leftEAR = eye_aspect_ratio(leftEye)
#         rightEAR = eye_aspect_ratio(rightEye)
#         ear = (leftEAR + rightEAR) / 2.0

#         if ear < blink_thresh:
#             blinked = True
#         return blinked
#     return blinked




# # import cv2
# # import dlib
# # from scipy.spatial import distance
# # from collections import deque

# # # Calculate Eye Aspect Ratio (EAR)
# # def eye_aspect_ratio(eye):
# #     A = distance.euclidean(eye[1], eye[5])
# #     B = distance.euclidean(eye[2], eye[4])
# #     C = distance.euclidean(eye[0], eye[3])
# #     ear = (A + B) / (2.0 * C)
# #     return ear

# # # Blink detection with consecutive frame logic
# # class BlinkDetector:
# #     def __init__(self, predictor, detector, blink_thresh=0.22, consec_frames=3):
# #         self.predictor = predictor
# #         self.detector = detector
# #         self.blink_thresh = blink_thresh
# #         self.consec_frames = consec_frames
# #         self.counter = 0
# #         self.blinked = False
# #         self.ear_history = deque(maxlen=10)  # smooth small noise

# #     def detect_blink(self, frame):
# #         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# #         rects = self.detector(gray, 0)

# #         for rect in rects:
# #             shape = self.predictor(gray, rect)
# #             shape = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
# #             leftEye = shape[36:42]
# #             rightEye = shape[42:48]

# #             leftEAR = eye_aspect_ratio(leftEye)
# #             rightEAR = eye_aspect_ratio(rightEye)
# #             ear = (leftEAR + rightEAR) / 2.0

# #             # Add to history and smooth EAR
# #             self.ear_history.append(ear)
# #             smooth_ear = sum(self.ear_history) / len(self.ear_history)

# #             # Check if blink started
# #             if smooth_ear < self.blink_thresh:
# #                 self.counter += 1
# #             else:
# #                 # Blink confirmed if closed for enough consecutive frames
# #                 if self.counter >= self.consec_frames:
# #                     self.blinked = True
# #                 self.counter = 0

# #             return self.blinked  # Return after first face detected

# #         return False  # No face detected

import cv2
import dlib
from scipy.spatial import distance
from collections import deque

# --- Eye aspect ratio calculation ---
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


# --- Blink Detector Class ---
class BlinkDetector:
    def __init__(self, predictor, detector, blink_thresh=0.22, consec_frames=3):
        """
        predictor: dlib.shape_predictor object
        detector: dlib.get_frontal_face_detector object
        blink_thresh: EAR threshold for blink detection
        consec_frames: number of consecutive frames below threshold to confirm a blink
        """
        self.predictor = predictor
        self.detector = detector
        self.blink_thresh = blink_thresh
        self.consec_frames = consec_frames
        self.counter = 0
        self.blinked = False
        self.ear_history = deque(maxlen=10)

    def detect_blink(self, frame):
        """Returns True if a blink is detected in the frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)
        blink_detected = False

        for rect in rects:
            shape = self.predictor(gray, rect)
            shape = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
            leftEye = shape[36:42]
            rightEye = shape[42:48]

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            self.ear_history.append(ear)
            smooth_ear = sum(self.ear_history) / len(self.ear_history)

            if smooth_ear < self.blink_thresh:
                self.counter += 1
            else:
                if self.counter >= self.consec_frames:
                    blink_detected = True
                self.counter = 0

            # Reset blink status after detection to allow next blink
            if blink_detected:
                self.blinked = True
            else:
                self.blinked = False

            return self.blinked
        return False


# --- Main Camera Loop ---
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

    blink_detector = BlinkDetector(predictor, detector)

    print("[INFO] Starting blink detection... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        blinked = blink_detector.detect_blink(frame)

        text = "Blink Detected!" if blinked else "Watching..."
        color = (0, 255, 0) if blinked else (0, 0, 255)
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("Blink Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
