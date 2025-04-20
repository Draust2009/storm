import cv2
import time
from object_detector import detect_objects
from audio_feedback import speak

cap = cv2.VideoCapture(1)

frame_width = int(cap.get(3))
spoken_labels = set()

relevant_labels = ['person', 'chair', 'couch', 'bed', 'table', 'tv']
last_suggestion_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = detect_objects(frame)
    detections = results.xyxy[0]

    left_count = 0
    center_count = 0
    right_count = 0

    for *box, conf, cls in detections:
        x1, y1, x2, y2 = map(int, box)
        label = results.names[int(cls)]

        if label not in relevant_labels:
            continue

        x_center = (x1 + x2) // 2
        box_area = (x2 - x1) * (y2 - y1)

        if x_center < frame_width / 3:
            position = "left"
            left_count += 1
        elif x_center > (2 * frame_width) / 3:
            position = "right"
            right_count += 1
        else:
            position = "center"
            center_count += 1

        if box_area > 30000:
            distance = "very close"
        elif box_area > 15000:
            distance = "close"
        else:
            distance = "far"

        object_key = f"{label}_{position}"
        if object_key not in spoken_labels:
            spoken_labels.add(object_key)
            speak(f"{label} {distance} on your {position}")

    now = time.time()
    if now - last_suggestion_time > 5:
        zone_counts = {'left': left_count, 'center': center_count, 'right': right_count}
        safest = min(zone_counts, key=zone_counts.get)
        speak(f"Try moving slightly {safest}")
        last_suggestion_time = now

    annotated = results.render()[0]
    cv2.imshow("YOLO - Obstacle Assistant", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if len(spoken_labels) > 30:
        spoken_labels.clear()

cap.release()
cv2.destroyAllWindows()
