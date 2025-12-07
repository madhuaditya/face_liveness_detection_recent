import cv2

try:
    # New FER import (for fer>=25)
    from fer.fer import Video, FER
except ImportError:
    # Fallback for older versions
    from fer.fer import  FER

def detect_expression(frame):
    """
    Detects the dominant facial expression in the given frame.
    Returns the expression name or 'neutral' if nothing detected.
    """
    try:
        detector = FER(mtcnn=True)
        result = detector.detect_emotions(frame)
        if result:
            emotions = result[0]["emotions"]
            dominant_emotion = max(emotions, key=emotions.get)
            return dominant_emotion
        else:
            return "neutral"
    except Exception as e:
        print(f"[WARN] Expression detection failed: {e}")
        return "neutral"

# Quick test
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        expression = detect_expression(frame)
        cv2.putText(frame, f"Expression: {expression}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Expression Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
