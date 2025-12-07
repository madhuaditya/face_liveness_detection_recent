import cv2
import random
import time
from fer.fer import FER

# Initialize FER emotion detector
detector = FER(mtcnn=True)

# Define possible expressions to test
EXPRESSIONS = ["happy", "sad", "surprise", "angry", "neutral"]

# Settings
MATCH_THRESHOLD = 0.5  # Confidence threshold
HOLD_TIME = 1          # Seconds to hold correct expression
DISPLAY_TIME = 6       # Max time for each expression

def detect_expression(frame):
    """Detects dominant facial expression."""
    result = detector.detect_emotions(frame)
    if result:
        emotions = result[0]["emotions"]
        top_expression = max(emotions, key=emotions.get)
        confidence = emotions[top_expression]
        return top_expression, confidence
    return None, 0

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot access camera.")
        return

    print("[INFO] Starting expression challenge...")
    time.sleep(1)

    random.shuffle(EXPRESSIONS)
    for target_expr in EXPRESSIONS:
        print(f"\n➡️ Please show expression: {target_expr.upper()}")
        matched_time = 0
        start_time = time.time()

        while time.time() - start_time < DISPLAY_TIME:
            ret, frame = cap.read()
            if not ret:
                break

            expr, conf = detect_expression(frame)

            if expr == target_expr and conf > MATCH_THRESHOLD:
                matched_time += 1
                cv2.putText(frame, f"✅ Matched {expr} ({conf:.2f})", (40, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if matched_time >= HOLD_TIME * 5:
                    print(f"✅ Expression {target_expr.upper()} passed!")
                    break
            else:
                cv2.putText(frame, f"Show: {target_expr.upper()}", (40, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Expression Challenge", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Exiting early.")
                cap.release()
                cv2.destroyAllWindows()
                return

        else:
            print(f"❌ Expression {target_expr.upper()} not detected in time.")

    print("\n🎉 All expression tests completed!")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
