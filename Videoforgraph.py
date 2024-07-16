import os
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt

# Load model
model = load_model("fer.h5")

# Load face cascade
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize figure and axis for plotting
plt.ion()
fig, ax = plt.subplots()
emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
y_pos = np.arange(len(emotions))
emotion_values = np.zeros(len(emotions))

# Main loop
while True:
    ret, test_img = cap.read()  # Capture frame
    if not ret:
        continue
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    for (x, y, w, h) in faces_detected:
        roi_gray = gray_img[y:y + w, x:x + h]  # Crop face region
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        # Predict emotions
        predictions = model.predict(img_pixels)

        # Update emotion values for plotting
        emotion_values = predictions[0]

        # Find max index
        max_index = np.argmax(predictions[0])
        predicted_emotion = emotions[max_index]

        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        resized_img = cv2.resize(test_img, (1000, 700))
        cv2.imshow('Facial emotion analysis', resized_img)

    # Plotting
    ax.clear()
    ax.barh(y_pos, emotion_values, align='center', color='blue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(emotions)
    ax.invert_yaxis()  # Invert y-axis to match the emotions order
    ax.set_xlabel('Probability')
    ax.set_title('Emotion Probabilities')
    plt.pause(0.1)  # Pause for a short while to update plot

    if cv2.waitKey(10) == ord('q'):  # Wait until 'q' key is pressed
        exit()

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
