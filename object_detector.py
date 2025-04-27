import torch
import cv2
import pyttsx3
import time

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def run_object_detector():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.conf = 0.5

    cam = cv2.VideoCapture(0)

    last_spoken_time = 0
    cooldown_seconds = 3 

    while True:
        ret, frame = cam.read()
        if not ret:
            continue

        results = model(frame)
        detections = results.pandas().xyxy[0]

        if not detections.empty:
            closest_detection = detections.iloc[0] 

            label = closest_detection['name']
            x_center = (closest_detection['xmin'] + closest_detection['xmax']) / 2
            frame_width = frame.shape[1]

            if x_center < frame_width / 3:
                direction = "on the left"
            elif x_center > 2 * frame_width / 3:
                direction = "on the right"
            else:
                direction = "ahead"

            message = f"{label} {direction}"

            current_time = time.time()
            if current_time - last_spoken_time > cooldown_seconds:
                print(f"🧹 Detected: {message}")
                speak(message)
                last_spoken_time = current_time

        if cv2.waitKey(1) == 27: 
            break

    cam.release()
    cv2.destroyAllWindows()
