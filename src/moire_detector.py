# # moire_detector.py
# import cv2
# import numpy as np

# class MoireDetector:
#     def __init__(self):
#         print("[INFO] Moiré detector initialized.")

#     def detect_moire(self, frame):
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Apply FFT (frequency scan)
#         fft = np.fft.fft2(gray)
#         fft_shift = np.fft.fftshift(fft)
#         magnitude = np.log(np.abs(fft_shift) + 1)

#         # High-frequency patterns = Moiré = phone screen
#         high_freq_energy = np.mean(magnitude[:, :20]) + np.mean(magnitude[:, -20:])

#         # Threshold (good starting value)
#         is_screen = high_freq_energy > 12.0

#         return high_freq_energy, is_screen


# if __name__ == "__main__":
#     cap = cv2.VideoCapture(0)
#     detector = MoireDetector()

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         score, is_screen = detector.detect_moire(frame)
#         label = "SCREEN (FAKE)" if is_screen else "NO SCREEN (REAL)"
#         color = (0, 0, 255) if is_screen else (0, 255, 0)

#         cv2.putText(frame, f"Moiré Score: {score:.2f}", (10, 40),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
#         cv2.putText(frame, label, (10, 80),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

#         cv2.imshow("Moiré Pattern Detection", frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# moire_detector.py
import cv2
import numpy as np

class MoireDetector:
    def __init__(self):
        print("[INFO] Improved Moiré detector ready.")

    def detect_moire(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Resize for stable FFT response
        gray = cv2.resize(gray, (320, 320))

        # FFT transform
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.log(np.abs(fft_shift) + 1)

        # Measure energy in *diagonal frequency spikes*, typical for screen refresh
        h, w = magnitude.shape

        # Screen moiré usually appears at these bands
        left_band = magnitude[:, 5:25]
        right_band = magnitude[:, w-25:w-5]

        hf_left = np.mean(left_band)
        hf_right = np.mean(right_band)

        high_freq_energy = (hf_left + hf_right) / 2

        # Instead of fixed threshold → dynamic threshold
        dynamic_threshold = np.mean(magnitude) * 1.6

        is_screen = high_freq_energy > dynamic_threshold

        return high_freq_energy, dynamic_threshold, is_screen

is_screen = False

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    detector = MoireDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        score, threshold, is_screen = detector.detect_moire(frame)
        label = "SCREEN (FAKE)" if is_screen else "NO SCREEN (REAL)"
        color = (0, 0, 255) if is_screen else (0, 255, 0)

        cv2.putText(frame, f"Moiré Score: {score:.2f}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Threshold: {threshold:.2f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, label, (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Improved Moiré Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
