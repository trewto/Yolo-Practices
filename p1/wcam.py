
from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv


# Load the YOLO model
#model = YOLO("yolov8n.pt")
"""
# Define class names (these are the COCO dataset class names used by YOLOv8)
class_names = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]
"""

model = YOLO("yolov8s.pt")

# Open the webcam (usually the default webcam is at index 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
n = 0
class_names = model.names

# Display the number of classes and their names
#num_classes = len(class_names)
#print(f"Number of classes: {num_classes}")
#print("Class names:", class_names)


skip_num =2

while True:
    # Read a frame from the webcam
    success, frame = cap.read()
    
    if not success:
        print("Error: Could not read the frame.")
        cap.release()
        exit()

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit the program
        break
    elif key == ord('i'):  # Increase skip_num
        skip_num += 1
        print(f"skip_num increased to {skip_num}")
    elif key == ord('k'):  # Decrease skip_num
        skip_num = max(1, skip_num - 1)  # Ensure skip_num doesn't go below 1
        print(f"skip_num decreased to {skip_num}")

        
    #press q to quit program
    if cv2.waitKey(1):
        if  0xFF == ord('q'):
            break
        
        
    n = n + 1

    #make the fram speedy by skipping every other frame
    if n%skip_num == 0:
        continue

    results= model.track(frame, persist=True,verbose=False)
    boxes = results[0].boxes.xywh.cpu()
    track_ids = results[0].boxes.id
    class_ids = results[0].boxes.cls.int().cpu().tolist()
    annotated_frame = results[0].plot()

    cv2.imshow("YOLOv8 Tracking", annotated_frame)
    

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
