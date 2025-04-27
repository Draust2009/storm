import os
import cv2
import face_recognition
import numpy as np
import pyttsx3

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def run_face_recognizer(known_faces_dir="faces"):
    known_encodings = []
    known_names = []

    if not os.path.exists(known_faces_dir):
        print("❗ No registered faces found.")
        return

    for name in os.listdir(known_faces_dir):
        person_dir = os.path.join(known_faces_dir, name)
        if not os.path.isdir(person_dir):
            continue

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(img)

            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(name)

    if not known_encodings:
        print("❗ No face encodings found.")
        return

    cam = cv2.VideoCapture(0)
    print("🔎 Looking for faces...")

    last_spoken_name = None

    while True:
        ret, frame = cam.read()
        if not ret:
            continue

        rgb_frame = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Stranger"

            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]

            if name != last_spoken_name:
                print(f"🧑 {name} detected.")
                speak(name)
                last_spoken_name = name

        if cv2.waitKey(1) == 27:  # ESC key
            break

    cam.release()
    cv2.destroyAllWindows()
