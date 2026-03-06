import time
from ultralytics import YOLO
import cv2
model=YOLO("yolov8n.pt")
cap=cv2.VideoCapture(0)
prev_time=0
while True:
    ret,frame=cap.read()
    if not ret:
        break
    results=model(frame,conf=0.5,classes=[0])
    annoatated_frame=results[0].plot()
    person_count=len(results[0].boxes)
    curr_time=time.time()
    fps=1/(curr_time-prev_time) if prev_time!=0 else 0
    prev_time=curr_time
    cv2.putText(annoatated_frame,f"FPS:{int(fps)}",(20,40),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    cv2.putText(annoatated_frame,f"person_count:{person_count}",(20,80),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    cv2.imshow("YOLO Advanced webcam",annoatated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()