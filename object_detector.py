import torch
import numpy as np

model = torch.hub.load('yolov5', 'yolov5s', source='local')
model.conf = 0.6
model.classes = None

def detect_objects(frame):
    results = model(frame)
    return results
