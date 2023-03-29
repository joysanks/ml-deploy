import cv2
import numpy as np

# Load the face detection cascade classifier
face_cascade = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_alt.xml')

# Load pre-trained face recognition model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trained_model.xml")

# Define function to recognize faces in an image
def recognize_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Loop through detected faces and recognize each one
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        label, confidence = recognizer.predict(face_roi)

        # Draw a rectangle around the face and label it with the recognized name
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, f"{label}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return img

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Loop through frames from the video capture
while True:
    ret, frame = cap.read()

    # If there's an error with the video capture, break out of the loop
    if not ret:
        break

    # Recognize faces in the current frame
    frame = recognize_faces(frame)

    # Show the current frame
    cv2.imshow('Face Recognition', frame)

    # Wait for key press and check if it's the escape key
    if cv2.waitKey(1) == 27:
        break

# Release the video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()