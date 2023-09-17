import cv2
from ultralytics import YOLO
import playsound
import random
import random

import cv2
import playsound
from ultralytics import YOLO

cap = cv2.VideoCapture(0)
model = YOLO('/Users/sohabhutta/PycharmProjects/theBackend3/sleepymodel.pt')

while True:
    ret, frame = cap.read()
    results = model(frame)
    count = 45

    box = results[0].boxes
    label = box.cls
    class_labels_strings = [results[0].names[int(labe.cpu().numpy())] for labe in label]
    if "Asleep" in class_labels_strings:
        while count > 0:
            ret, frame = cap.read()
            results = model(frame)
            box = results[0].boxes
            label = box.cls
            class_labels_strings = [results[0].names[int(labe.cpu().numpy())] for labe in label]
            if "Awake" in class_labels_strings:
                break
            annotated_frame = results[0].plot()
            cv2.imshow('Webcam Feed', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            count -= 1
        if count == 0:
            num = random.randint(1,2)
            if num == 1:
                playsound.playsound("/Users/sohabhutta/PycharmProjects/theBackend3/mystery1.mp3", True)
            else:
                playsound.playsound("/Users/sohabhutta/PycharmProjects/theBackend3/mystery2.mp3", True)

    annotated_frame = results[0].plot()
    cv2.imshow('Webcam Feed', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()