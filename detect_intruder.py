import cv2
import face_recognition
import os
import numpy as np
import playsound
import threading
import time

# 🚨 Alert function with cooldown
last_alert_time = 0
cooldown = 2  # seconds

def play_alert():
    global last_alert_time
    current_time = time.time()
    if current_time - last_alert_time >= cooldown:
        threading.Thread(target=playsound.playsound, args=('alert.wav',), daemon=True).start()
        last_alert_time = current_time

# 📂 Load known faces from 'known_faces' folder
known_faces_dir = 'known_faces'
known_encodings = []
known_names = []

print("[INFO] Loading known faces...")

for filename in os.listdir(known_faces_dir):
    image_path = os.path.join(known_faces_dir, filename)
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)

    if encodings:
        known_encodings.append(encodings[0])
        known_names.append(os.path.splitext(filename)[0])
    else:
        print(f"[WARNING] No face found in {filename}")

# 🎥 Set camera source
# For built-in cam use 0, USB cam might be 1 or 2
# For IP camera use a stream URL like "http://192.168.1.25:8080/video"
camera_source = 2  # ← change this to 1 or an IP stream if needed

video = cv2.VideoCapture(camera_source)
if not video.isOpened():
    print(f"[ERROR] Cannot open camera source: {camera_source}")
    exit()

print("[INFO] Face recognition intruder system started. Press 'q' to quit.")

while True:
    ret, frame = video.read()
    if not ret:
        print("[ERROR] Failed to grab frame")
        break

    # Resize frame for performance
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    names_in_frame = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]

        names_in_frame.append(name)

        if name == "Unknown":
            play_alert()

    # Annotate frame
    for (top, right, bottom, left), name in zip(face_locations, names_in_frame):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    if "Unknown" in names_in_frame:
        print("🔴 Intruder Detected!")

    cv2.imshow("Intruder Detection (Face Recognition)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
