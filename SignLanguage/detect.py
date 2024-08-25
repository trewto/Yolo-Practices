import os

#os.system("yolo detect predict model=runs/detect/train7/weights/best.pt imgsz=416 conf=0.5 source=0")
#yolo track model=runs/detect/train7/weights/best.pt source=0
#yolo track model=best.pt source=0
from ultralytics import YOLO

# Load the trained model
model = YOLO("runs/detect/train10/weights/best.pt")
#model = YOLO("best.pt")

# Predict using the webcam
results = model.predict(source=0, show=True, conf=0.5)
