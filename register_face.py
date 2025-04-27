import cv2
import os

def capture_faces(name, save_dir="faces", num_images=5):
    cam = cv2.VideoCapture(0)

    person_dir = os.path.join(save_dir, name)
    os.makedirs(person_dir, exist_ok=True)

    count = 0
    print(f"📸 Capturing {num_images} pictures for {name}...")

    while count < num_images:
        ret, frame = cam.read()
        if not ret:
            continue

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            img_path = os.path.join(person_dir, f"{count}.jpg")
            cv2.imwrite(img_path, face_img)
            print(f"✅ Saved {img_path}")
            count += 1

        if cv2.waitKey(1) == 27:  # ESC key
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"✅ Done registering {name}!")
