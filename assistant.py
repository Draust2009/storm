import logging
import speech_recognition as sr
from face_recognizer import run_face_recognizer
from register_face import capture_faces
from object_detector import run_object_detector
from audio_feedback import speak 

logging.basicConfig(filename="assistant.log", level=logging.INFO, format="%(asctime)s - %(message)s")

def listen_for_command():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        speak("Listening for your command...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        command = recognizer.recognize_google(audio).lower()
        speak(f"You said: {command}")
        return command
    except sr.UnknownValueError:
        speak("Sorry, I couldn't understand that. Please try again.")
        return None
    except sr.RequestError:
        speak("There was an issue with the speech recognition service. Try again later.")
        return None

def start_assistant():
    speak("Welcome to the Visually Impaired Assistant!")

    while True:
        command = listen_for_command()

        if command is None:
            continue

        logging.info(f"User command: {command}")

        if "detect objects" in command:
            speak("Detecting objects...")
            run_object_detector()

        elif "who is nearby" in command or "recognize faces" in command:
            speak("Recognizing faces...")
            run_face_recognizer()

        elif "register face" in command:
            try:
                name = command.split("name")[1].strip()
                logging.info(f"Registering new face: {name}")
                speak(f"Registering the new face: {name}")
                capture_faces(name)
            except IndexError:
                speak("❗ Please say: register face name <name>")

        elif "exit" in command:
            speak("Goodbye!")
            print("Goodbye!")
            break

        else:
            speak("Unknown command. Try again.")

if __name__ == "__main__":
    start_assistant()
