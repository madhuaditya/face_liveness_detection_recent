# Face Liveness Detection (ML + OpenCV + TensorFlow)

## 📘 Overview
Detects whether a face is **real or spoofed** using a CNN model trained on real and fake face images.

## 🚀 Run Steps
```bash
python -m venv venv
venv\Scripts\activate
with requiremnt.txt you need to add dlib in your folder then give that path in the requirements.txt
and then run
pip install -r requirements.txt
create folder 
mkdir logs , dataset , reports , models
and in dataset there will be 3 sub folder name as train text val and each subfolder have 2 classes fake and real
you can use these data sets in you project 
1. https://www.kaggle.com/datasets/minhnh2107/casiafasd
then you need to some pre trained models like 
midas_v21_small.onnx
shape_predictor_68_face_landmarks.dat
yolov8n.pt
ans store these models in models folder
and run command to train you light weight CNN model
python src/train_model.py
then run 
python src/liveness_system.py
and you can see one camara pop will come 
you should have python 3.10 to run this project and pip 25.3 
```

