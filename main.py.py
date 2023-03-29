import cv2
import numpy as np
import os
import random
import base64,io
from PIL import Image
from flask import Flask, flash, request, redirect, abort, jsonify, session
from flask_cors import CORS, cross_origin
from concurrent.futures import ThreadPoolExecutor
from werkzeug.utils import secure_filename

import face_detect as face_detect
import training_data as training_data

UPLOAD_FOLDER = "./test-folder"
save_path1 = "/training-data"
app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
cors = CORS(app, resources={r"/*": {"origins": "*"}})


@app.route("/")
def hello_world():
	return "Hello World"

@app.route('/upload-train-image', methods=['POST'], strict_slashes = False)
def upload_train_image():
    data = request.get_json()
    base64_images = data['images']
    uploaded_files = []

#     data = request.get_json()
#   base64_images = data['images']
#   uploaded_files = []

#   # Specify the folder path where you want to save the images
#   folder_path = '/path/to/folder/'

#   for base64_image in base64_images:
#     image_data = base64.b64decode(base64_image.split(',')[1])
#     file_name = 'uploaded_image_' + str(len(uploaded_files) + 1) + '.jpg'
#     file_path = os.path.join(folder_path, file_name)
#     with open(file_path, 'wb') as f:
#       f.write(image_data)
#     uploaded_files.append(file_path)

#   return {'message': 'Files uploaded successfully', 'files': uploaded_files}
    folder_path = '/path'

    for base64_image in base64_images:
        print(base64_image)
        image_data = base64.b64decode(base64_image.split(',')[1])
        file_name = 'uploaded_image_' + str(len(uploaded_files) + 1) + '.jpg'
        file_path = os.path.join(folder_path, file_name)
        with open(file_name, 'wb') as f:
            f.write(image_data)
        uploaded_files.append(file_path)

        return {'message': 'Files uploaded successfully', 'files': uploaded_files}
    
    # data = request.json['img']
    # data = data[23:]
    # img = Image.open(io.BytesIO(base64.decodebytes(bytes(data, 'utf-8'))))
    # # filename = str(random.randint(1, 1000000)) + 'my-image.jpeg'
    # filename = 'test.jpg'
    # img.save(filename)
    
    # # print(data)
    # predicted_img , label= predict("./test.jpg")
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)
    # cv2.destroyAllWindows()
    # print ("Recognized faces = ", label)

    # data = {
    #     "schId" : label 
    # }

    # label =[]

    # return jsonify(data)

@app.route('/upload-test-image', methods=['POST'], strict_slashes = False)
def upload_image():
    
    data = request.json['img']
    data = data[23:]
    img = Image.open(io.BytesIO(base64.decodebytes(bytes(data, 'utf-8'))))
    # filename = str(random.randint(1, 1000000)) + 'my-image.jpeg'
    filename = 'test.jpg'
    img.save(filename)
    
    # print(data)
    predicted_img , label= predict("./test.jpg")
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    print ("Recognized faces = ", label)

    data = {
        "schId" : label 
    }

    label =[]

    return jsonify(data)
    

@app.route('/get-attendence', methods=['POST'])
def upload_test_file():

    if request.method == "POST":
        if "test" not in request.files:
            return "no file found"
        
        file = request.files["test"]
        path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(path)

        
        #Read the test image.
        test_img = "test-data/test.jpg"
        print(test_img)
        predicted_img , label= predict(test_img)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        print ("Recognized faces = ", label)

    return label




def predict(test_img):
    label = []
    
    # Load pre-trained face recognition model
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read("trained_model.xml")
    
    img = cv2.imread(test_img).copy()
    print ("\n\n\n")
    print ("Face Prediction Running -\-")
    face, rect, length = face_detect.face_detect(test_img)
    print (len(face), "faces detected.")
    for i in range(0, len(face)):
        labeltemp, confidence = face_recognizer.predict(face[i])
        print(confidence)
        if confidence>50:
            label.append(labeltemp)
        else:
            label.append(-1)
    print(label)
    print(type(label[0]))
    return img, label



@app.route('/train-data', methods=['GET'])
def train():
    faces, labels = training_data.training_data("training-data")
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, np.array(labels))
    face_recognizer.write('trained_model.xml')
    return "train data successfully"




if __name__ == "__main__":
    app.secret_key = 'super secret key'
    
    app.debug = True
    app.run()
