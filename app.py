import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import streamlit as st
from PIL import Image

# Streamlit setup
st.title("Sign Language Prediction using Computer Vision")
st.text("This app captures a live video feed, detects the hand, and predicts the sign language.")

# Create placeholders for video and prediction display
frame_placeholder = st.empty()
prediction_placeholder = st.empty()

# Add a button to stop the webcam feed
stop_button = st.button("Stop Webcam")

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("D:\pythonProject\.venv\Model\keras_model.h5",
                        "D:\pythonProject\.venv\Model\labels.txt")
offset = 20
imgSize = 300
counter = 0

labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
          "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

while True:
    success, img = cap.read()
    if not success:
        st.error("Unable to access the camera.")
        break

    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Ensure the cropping coordinates are within the image bounds
        h_img, w_img, _ = img.shape

        y_start = max(0, y - offset)
        y_end = min(h_img, y + h + offset)
        x_start = max(0, x - offset)
        x_end = min(w_img, x + w + offset)

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        imgCrop = img[y_start:y_end, x_start:x_end]
        imgCropShape = imgCrop.shape

        if imgCropShape[0] > 0 and imgCropShape[1] > 0:  # Ensure imgCrop is not empty
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap: wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap: hCal + hGap, :] = imgResize

            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            st.write(f"Prediction: {labels[index]}")
            print(prediction, index)

            # Draw bounding box and text on the image
            cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 90, y - offset + 50 - 50),
                          (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)

        # Convert the image for displaying in Streamlit
        imgRGB = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(imgRGB)

        # Display the image using Streamlit
        frame_placeholder.image(pil_img, caption="Live Camera Feed", use_column_width=True)

    # Check if stop button is pressed
    if stop_button:
        break

cap.release()
st.text("Webcam stopped.")
