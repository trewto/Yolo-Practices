#for train cli
#yolo mode=train task=detect model=yolov8s.pt data=data.yaml epochs=25 imgsz=224 plots=True

#!yolo task=classify mode=predict model=yolov8n-cls.pt source="images/1.jpg"
#for detect
#yolo task=detect mode=predict model=runs/detect/train7/weights/best.pt source="39.jpeg"