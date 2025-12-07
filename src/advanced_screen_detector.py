# advanced_screen_detector.py
import cv2
import numpy as np

class ScreenDetector:
    def __init__(self):
        print("[INFO] Advanced screen detector loaded.")

    def detect_screen(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- (1) SHARPNESS (phones are overly sharp)
        lap = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharp_screen = lap > 350   # phone screens have high artificial sharpness

        # --- (2) SPECULAR HIGHLIGHTS (light reflections on glass)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        bright_pixels = np.sum(v > 240)
        reflection_screen = bright_pixels > 400  # many bright reflections → glass

        # --- (3) SATURATION PEAK (OLED pixel boosting)
        s = hsv[:, :, 1]
        sat_score = np.mean(s)
        oled_screen = sat_score > 110  # OLED screens oversaturate faces

        # --- (4) ROLLING SHUTTER BAND DETECTION (video replay)
        fft = np.fft.fft(gray.mean(axis=0))
        fft_mag = np.abs(fft)
        band_freq = fft_mag[5:40].mean()
        rolling_screen = band_freq > 25  # screen flicker

        # FINAL DECISION
        screen_score = (
            (1 if sharp_screen else 0) +
            (1 if reflection_screen else 0) +
            (1 if oled_screen else 0) +
            (1 if rolling_screen else 0)
        )

        is_screen = screen_score >= 2   # at least 2 indicators needed

        return {
            "sharpness": lap,
            "bright_spots": bright_pixels,
            "saturation": sat_score,
            "rolling": band_freq,
            "screen_detected": is_screen,
            "score": screen_score
        }


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    detector = ScreenDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = detector.detect_screen(frame)

        color = (0,255,0)
        label = "REAL FACE"
        if result["screen_detected"]:
            color = (0,0,255)
            label = "PHONE SCREEN / SPOOF"

        cv2.putText(frame, f"SHARP:{result['sharpness']:.1f}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"BRIGHT:{result['bright_spots']}", (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"SAT:{result['saturation']:.1f}", (10,90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"ROLL:{result['rolling']:.1f}", (10,120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.putText(frame, f"{label}", (10,160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)

        cv2.imshow("Advanced Screen Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
