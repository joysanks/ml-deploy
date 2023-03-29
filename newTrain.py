import cv2
import os
import numpy as np

# Load the dataset
dataset_path = 'dataset'
faces = []
labels = []
for folder_name in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder_name)
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        faces.append(image)
        labels.append(int(folder_name))

# Train the face recognition model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))

# Save the trained model to a file
recognizer.write('trained_model.xml')