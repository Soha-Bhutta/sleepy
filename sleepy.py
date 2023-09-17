import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
from ultralytics import YOLO
import pandas

# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    model = YOLO('/Users/christinazhang/Documents/vscode/hackmit/sleepymodel.pt')
    results = model(frame)

    box = results[0].boxes
    label = box.cls
    class_labels_strings = [results[0].names[int(labe.cpu().numpy())] for labe in label]
    print(class_labels_strings)
    if "Asleep" in class_labels_strings:
        print("Alert: 'asleep' detected!")

    annotated_frame = results[0].plot()

    cv2.imshow('Webcam Feed', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()