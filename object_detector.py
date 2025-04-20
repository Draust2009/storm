import torch
import numpy as np

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.6
model.classes = None

def detect_objects(frame):
    results = model(frame)
    return results
