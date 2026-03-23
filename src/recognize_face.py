import cv2
from deepface import DeepFace
import os
import csv
from datetime import datetime

dataset_path = "dataset"
attendance_file = "outputs/attendance.csv"

# Create outputs folder
os.makedirs("outputs", exist_ok=True)

# Create attendance file with header
if not os.path.exists(attendance_file):
    with open(attendance_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Time"])

marked_names = set()

def mark_attendance(name):
    if name not in marked_names:
        now = datetime.now()
        time_string = now.strftime("%H:%M:%S")

        with open(attendance_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([name, time_string])

        marked_names.add(name)

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

THRESHOLD = 0.4  # important

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        name = "Unknown"

        try:
            result = DeepFace.find(
                img_path=face,
                db_path=dataset_path,
                enforce_detection=True
            )

            if len(result) > 0 and len(result[0]) > 0:
                distance = result[0]['distance'][0]

                if distance < THRESHOLD:
                    identity = result[0]['identity'][0]
                    name = os.path.basename(os.path.dirname(identity))
                    mark_attendance(name)
                else:
                    name = "Unknown"

        except:
            name = "Unknown"

        # Draw box
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)

        # Show name
        cv2.putText(
            frame,
            name,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

    cv2.imshow("FaceSense - Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()