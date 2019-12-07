#Imports
from flask import Flask, json, jsonify, render_template, request
import numpy as np
from DigitRecognition import prediction
import sys
from PIL import Image
import re
import base64
from keras.models import load_model
import io
from io import BytesIO
from flask_cors import CORS, cross_origin
import cv2

app = Flask(__name__)
#stops cors unauthorised notification
cors = CORS(app, resources={r"/": {"origins": "http://localhost:5000"}})

@app.route('/')
def homePage():
    return app.send_static_file('Draw.html')

@app.route('/', methods=['POST'])
def getImage():
    #Retrieving Base64 image data
    imgBase64 = request.values['imageBase64']

    #Strips the image of unnecessary contents(similar  to PIL vid)
    imgStripped = re.sub('^data:image/.+;base64,', '', imgBase64)

    #Decodes the image from Base64
    imgDecoded = base64.b64decode(imgStripped)

    #Converting the decoded image to bytes
    img = Image.open(BytesIO(imgDecoded))

    #Convert the image to grayscale
    img = img.convert('L')

    #Resizes the image to the required size of 28x28 using Bilinear mode
    img = img.resize((28, 28), Image.ANTIALIAS)
    
    #converts image to black and white 
    img = img.point(lambda p: p > 0 and 255)  

    #Convert the image to an array
    myArray = np.ndarray.flatten(np.array(img))

    #reshape the array to a 1d array
    myArray = myArray.reshape(1,784)

    #convert to type float32
    myArray = myArray.astype('float32')
    
    #divide all floats in the array so that they either have a value of 1 or 0
    myArray /= 255

    #Prints out the image in console
    counter = 0
    for i in myArray[0]:
        if i == 1:
           print("0",end="")
        else:
           print(".",end="")
        counter +=1

        if counter == 28:
            print("\n")
            counter = 0
    
    #Calls function prediction in DigitRecognition.py
    myPrediction = prediction(myArray)

    #Prints Prediction
    print("The number mostly resembles a: ")
    print(myPrediction)

    #Sends prediction back to the user.
    return str(myPrediction)
    
if __name__ == "__main__":
    app.run(debug = True, threaded = False)