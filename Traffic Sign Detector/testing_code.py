import cv2
import pickle
import csv
import numpy as np

# Load model
with open("model_trained.p", "rb") as f:
    MODEL = pickle.load(f)

# Load label dictionary
labels_dict = {}
with open("./labels/labels.csv") as f:
    csv_data = csv.DictReader(f)
    
    for csv_ in csv_data:
        labels_dict[int(csv_["ClassId"])] = csv_["Name"]

# Function to preprocess image
def preprocess_image(image):
    imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imgEql = cv2.equalizeHist(imgGray)
    imgScale = imgEql / 255
    imgReshape = cv2.resize(imgScale, (32, 32))
    # Convert single channel to three channels by replicating across all channels
    imgReshape = np.stack((imgReshape, imgReshape, imgReshape), axis=-1)
    imgReshape = imgReshape.reshape((1, 32, 32, 3))  # Ensure the image has 3 channels
    return imgReshape

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Unable to capture frame from webcam")
        break

    # Preprocess frame
    p_image = preprocess_image(frame)
    
    # Make prediction
    prediction = MODEL.predict(p_image)
    predicted_label_id = np.argmax(prediction)
    confidence=round(np.max(prediction)*100,2)
    predicted_label = labels_dict.get(predicted_label_id, "Unknown")

    # Overlay prediction on frame
    cv2.putText(frame, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame,f"{confidence}%",(10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
 
    # Display frame
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break