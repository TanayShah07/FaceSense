import cv2
from deepface import DeepFace

dataset_path = "dataset"

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        result = DeepFace.find(
            img_path=frame,
            db_path=dataset_path,
            enforce_detection=True
        )

        if len(result) > 0 and len(result[0]) > 0:
            identity = result[0]['identity'][0]

            name = identity.split("\\")[1]

            cv2.putText(frame,
                        f"Person: {name}",
                        (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,255,0),
                        2)

    except:
        pass

    cv2.imshow("FaceSense Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()