import cv2
import os
import time

person_name = "tanay"
dataset_path = f"dataset/{person_name}"

os.makedirs(dataset_path, exist_ok=True)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

count = 0
max_images = 100

last_capture_time = 0
capture_interval = 0.4   # seconds between captures

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        current_time = time.time()

        if current_time - last_capture_time > capture_interval:
            count += 1
            file_path = f"{dataset_path}/{count}.jpg"
            cv2.imwrite(file_path, face)

            last_capture_time = current_time

        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)

        cv2.putText(frame, f"Images: {count}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("FaceSense Dataset Capture", frame)

    if count >= max_images:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()