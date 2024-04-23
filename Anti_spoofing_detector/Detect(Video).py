import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

cap=cv2.VideoCapture(0)
#------------------- Load the trained model --------------------------
model = load_model('anti_spoof_model.h5')

class_color={"Fake":(0,0,255),"Real":(0,255,0)}

def detect_faces(model):

    while True:
        ret,frame=cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        label=detect_fake_or_real(frame, model)
        text_size=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.9,1)[0]
        clsColor=class_color[label]

        # --------------- Load the face cascade ---------------------------
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        #------------------------- Detect faces ----------------------------------
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        #----------------- Draw bounding boxes around faces ---------------------------------

        for (x, y, w, h) in faces:
            confidence = w * h / (gray.shape[0] * gray.shape[1])

            if confidence > 0.01:
                cv2.rectangle(frame, (x, y), (x+w, y+h), clsColor, 2)  # Draw red bounding box for fake images
                cv2.rectangle(frame, (x, y - text_size[1]), (x + text_size[0], y), clsColor, -1)
                cv2.putText(frame, label, (x, y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                
        # Display the result
        cv2.imshow("Faces Detected", frame)

        if cv2.waitKey(1) & 0xFF==ord("q"):
            break

def detect_fake_or_real(frame, model):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (150, 150))  # Resize image to match model input size
    img_resized = np.expand_dims(img_resized, axis=0)  # Add batch dimension

    #---------------------Predict using the model-----------------------------------------
    prediction = model.predict(img_resized)
    label=""

    if prediction[0][0] > 0.5:
        label="Fake"

    else:
        label="Real" # Real image

    return label

detect_faces(model)
