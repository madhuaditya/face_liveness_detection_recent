# reflection_screen_detector.py
import cv2
import numpy as np

# def detect_screen_texture(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Apply Laplacian to detect surface texture
#     lap = cv2.Laplacian(gray, cv2.CV_64F)
#     variance = lap.var()

#     # Real faces have variance > 50
#     # Screens/photos = < 20
#     is_real = variance > 40

#     return variance, is_real

def detect_screen_reflection(face_region):
    hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
    brightness = hsv[:, :, 2]

    # Phones have unusually uniform brightness
    uniformity = brightness.std()

    return uniformity < 18  # low variance = screen

def detect_moire(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    # Moiré patterns create repeated peaks in the frequency domain
    moire_score = np.mean(magnitude_spectrum[50:150, 50:150])

    return moire_score > 18000  # threshold (tune it)


