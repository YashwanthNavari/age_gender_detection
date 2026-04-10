import cv2
import numpy as np

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:
        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h),
                      (0, 255, 0), 2)

        # Crop face
        face = frame[y:y+h, x:x+w]

        # Resize face for CNN
        face_resized = cv2.resize(face, (224, 224))

        # Normalize
        face_normalized = face_resized / 255.0

        cv2.putText(frame, "Face Ready for ML",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 0, 0), 2)

        # Show cropped face
        cv2.imshow("Cropped Face", face_resized)

    cv2.imshow("Webcam - Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
