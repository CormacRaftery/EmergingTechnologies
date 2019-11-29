import numpy as np
import keras
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.models import load_model
from keras.datasets import mnist
import os.path

def build():

    # load data
    (x_train,y_train), (x_test,y_test) = mnist.load_data()

    #reshape the array to a 1d array
    x_test = x_test.reshape(x_test.shape[0], 784)
    x_train = x_train.reshape(x_train.shape[0], 784)

    #set array types to  float32
    x_test = x_test.astype('float32')
    x_train = x_train.astype('float32')

    #convert the array to black and white
    x_test /= 255
    x_train /= 255

    #Adds layers for 10 classes
    y_test = keras.utils.to_categorical(y_test, 10)
    y_train = keras.utils.to_categorical(y_train, 10)

    # create model
    model = Sequential()
    model.add(Dense(16, input_dim=784, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    #compile the model
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    #fit model to the training data
    model.fit(x_train, y_train, batch_size=100, epochs=10, verbose=1)

    #save the model
    model.save("models/model")

    return model

def prediction(image):
    #Create model if not made already
    try:
        model = load_model("models/model")
    except:
        print("Could not load model, creating new")
        model = build()

    #Makes prediction
    labelPredict = model.predict(image)
    
    #returns the index of the latest number in the array which is the prediction
    return labelPredict.argmax()


