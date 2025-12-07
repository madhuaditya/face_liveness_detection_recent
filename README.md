# Face Liveness Detection (ML + OpenCV + TensorFlow)

## 📘 Overview
Detects whether a face is **real or spoofed** using a CNN model trained on real and fake face images.

## 🚀 Run Steps
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python src/train_model.py
python src/realtime_test.py
