import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('anti_spoof_model.h5')
class_color={"Fake":(0,0,255),"Real":(0,255,0)}

def detect_faces(image_path,model):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    label=detect_fake_or_real(image_path, model)
    clsColor=class_color[label]
    # Load the face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw bounding boxes around faces

    for (x, y, w, h) in faces:
        confidence = w * h / (gray.shape[0] * gray.shape[1])

        if confidence > 0.01:
            cv2.rectangle(image, (x, y), (x+w, y+h), clsColor, 2)  # Draw red bounding box for fake images
            cv2.putText(image,label,(x,y),cv2.FONT_HERSHEY_PLAIN,0.9,clsColor,1)

    # Display the result
    cv2.imshow("Faces Detected", image)
    cv2.waitKey(0)

def detect_fake_or_real(image_path, model):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (150, 150))  # Resize image to match model input size
    img_resized = np.expand_dims(img_resized, axis=0)  # Add batch dimension

    # Predict using the model
    prediction = model.predict(img_resized)
    label=""

    if prediction[0][0] > 0.5:
        
        label="Fake"
      # Fake image
    else:
        
        label="Real" # Real image

    return label

# Example usage
image_path = 'WIN_20240419_23_26_43_Pro.jpg'
detect_faces(image_path, model)
