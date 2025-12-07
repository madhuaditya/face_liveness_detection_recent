import cv2, numpy as np, tensorflow as tf
model = tf.keras.models.load_model("models/liveness_mobilenet.keras")
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret: break
    # face detection + crop -> for demo we just center crop
    h,w = frame.shape[:2]
    face = cv2.resize(frame[int(h*0.25):int(h*0.75), int(w*0.25):int(w*0.75)], (224,224))
    x = (face.astype("float32") / 255.0)
    x = np.expand_dims(x,0)
    p = model.predict(x, verbose=0)[0][0]
    print(p)
    label = "REAL" if p>0.5 else "FAKE"
    cv2.putText(frame, f"{label} {p:.3f}", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0) if p>0.5 else (0,0,255), 2)
    cv2.imshow("Demo", frame)
    if cv2.waitKey(1)&0xFF==ord("q"): break
cap.release(); cv2.destroyAllWindows()
