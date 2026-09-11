#Importing the modules
from ultralytics import YOLO
import cv2
#Loading pretrained YOLO model
model=YOLO("yolov8n.pt")
model.predict(source=0,show=True)
#Open webcam
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    if not ret:
        break
    results=model(frame)
    annotated_frame=results[0].plot()
    cv2.imshow("YOLO Webcam Detection",annotated_frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
