import cv2
import numpy as np
import os
from flask import Flask, flash, request, redirect, abort, jsonify, session
from flask_cors import CORS, cross_origin
from concurrent.futures import ThreadPoolExecutor
from werkzeug.utils import secure_filename

import face_detect as face_detect
import training_data as training_data

UPLOAD_FOLDER = "./test"
app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
cors = CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/")
def hello_world():
	return "Hello World"

@app.route('/get-attendence', methods=['POST'])
def upload_test_file():
	
	if not request.files:
		return('No file part')
	files = request.files.getlist('files[]')
	dirname = request.filename

	isExist = os.path.exists(dirname)
	
	
	pool = ThreadPoolExecutor(max_workers=1)

	for file in files:
		if file:
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
	flash('files successfully uploaded')
	pool.shutdown(wait=True)
	return "success"

# label = []
# def predict(test_img):
# 	img = cv2.imread(test_img).copy()
# 	print ("\n\n\n")
# 	print ("Face Prediction Running -\-")
# 	face, rect, length = face_detect.face_detect(test_img)
# 	print (len(face), "faces detected.")
# 	for i in range(0, len(face)):
# 		labeltemp, confidence = face_recognizer.predict(face[i])
# 		label.append(labeltemp)
# 	return img, label

# faces, labels = training_data.training_data("training-data")
# face_recognizer = cv2.face.LBPHFaceRecognizer_create()
# face_recognizer.train(faces, np.array(labels))


# Read the test image.
# test_img = "test-data/test.jpg"
# print(test_img)
# predicted_img , label= predict(test_img)
# cv2.destroyAllWindows()
# cv2.waitKey(1)
# cv2.destroyAllWindows()
# print ("Recognized faces = ", label)


if __name__ == "__main__":
    app.secret_key = 'super secret key'
    
    app.debug = True
    app.run()
