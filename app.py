import cv2
import numpy as np
import gradio as gr
from download_models import download_models

# Ensure models are downloaded before trying to load them
download_models()

FACE_PROTO = "models/deploy.prototxt"
FACE_MODEL = "models/res10_300x300_ssd_iter_140000.caffemodel"

AGE_PROTO = "models/age_deploy.prototxt"
AGE_MODEL = "models/age_net.caffemodel"

GENDER_PROTO = "models/gender_deploy.prototxt"
GENDER_MODEL = "models/gender_net.caffemodel"

AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)",
               "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
GENDER_LIST = ["Male", "Female"]

# Load networks
try:
    face_net = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)
    age_net = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
    gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)
except Exception as e:
    print(f"Error loading models: {e}")

def detect_age_gender(frame):
    if frame is None:
        return frame

    # The frame we get from Gradio is already a numpy array (RGB)
    # Convert it to BGR for OpenCV
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    h, w = frame_bgr.shape[:2]

    blob = cv2.dnn.blobFromImage(
        frame_bgr, 1.0, (300, 300), (104.0, 177.0, 123.0)
    )

    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.7:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            
            # Ensure boundaries are within frame
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            face = frame_bgr[y1:y2, x1:x2]
            if face.size == 0:
                continue

            face_blob = cv2.dnn.blobFromImage(
                face, 1.0, (227, 227),
                (78.4263377603, 87.7689143744, 114.895847746),
                swapRB=False
            )

            gender_net.setInput(face_blob)
            gender = GENDER_LIST[gender_net.forward()[0].argmax()]

            age_net.setInput(face_blob)
            age = AGE_BUCKETS[age_net.forward()[0].argmax()]

            label = f"{gender}, Age {age}"

            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_bgr, label, (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Convert back to RGB for Gradio
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.BGR2RGB)
    return frame_rgb

# Create Gradio Interface
demo = gr.Interface(
    fn=detect_age_gender,
    inputs=gr.Image(sources=["webcam", "upload"], streaming=False),
    outputs=gr.Image(label="Processed Image"),
    title="Age & Gender Detection",
    description="Capture a photo from your webcam or upload an image to detect faces, estimated age, and gender using OpenCV deep learning.",
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()
