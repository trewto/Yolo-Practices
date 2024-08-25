import os
import cv2
import time
import uuid

IMAGE_PATH = 'CollectedImages'
labels = ['RightHand', 'LeftHand']
number_of_images = 20

# Ensure the base directory exists
os.makedirs(IMAGE_PATH, exist_ok=True)

for label in labels:
    img_path = os.path.join(IMAGE_PATH, label)
    os.makedirs(img_path, exist_ok=True)
    cap = cv2.VideoCapture(0)
    
    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera")
        break
    
    print('Collecting images for {}'.format(label))
    time.sleep(2)  # Initial sleep to allow the camera to warm up
    
    for imgnum in range(number_of_images):
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        imagename = os.path.join(img_path, '{}.jpg'.format(str(uuid.uuid1())))
        cv2.imwrite(imagename, frame)
        cv2.imshow('frame', frame)
        time.sleep(0.5)  # Reduced sleep interval for faster capture
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()

cv2.destroyAllWindows()
