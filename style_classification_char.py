import sys
import os
import time
from src.HelperFunctions.helper import *
from scipy import ndimage
from src.SlantNormalization.slantNormalize import *
import cv2 as cv
import math
import random
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.utils import to_categorical

def trainModel(outputfile, xTrain, yTrain, xTest, yTest):

    numClasses = len(yTrain[0])
    x, y = xTrain[0].shape
    xTrain = xTrain.reshape(len(xTrain), x,y,1)
    xTest = xTest.reshape(len(xTest), x,y,1)

    print(xTrain.shape)
    print(x)
    print(y)

    model = Sequential()
    model.add(Conv2D(64, kernel_size=16, activation='relu', input_shape=(x,y,1)))
    model.add(Conv2D(32, kernel_size=16, activation='relu'))
    model.add(Flatten())
    model.add(Dense(numClasses, activation='softmax'))

    print("compiling model...")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print("fitting model...")
    model.fit(xTrain, yTrain, validation_data=(xTest, yTest), epochs=3)




def augment(imgs):
    allChars = []

    maxX = 0
    maxY = 0

    for charImg in imgs:
        for angle in range(-10, 10, 2): # rotation
            rotatedImg = ndimage.rotate(charImg, angle)

            for i in range(0, 40, 3): # deskew over different angles
                
                # pad whitespace to left border
                dst = cv.copyMakeBorder(rotatedImg, 0, 0, 50, 0, cv.BORDER_CONSTANT)
                newImg = deskew(dst, i)
                
                newImg = cropWhiteSpace(newImg) # crop image

                # see what max dimensions are
                x, y = newImg.shape
                if (x > maxX): maxX = x
                if (y > maxY): maxY = y

                allChars.append(newImg)
                # add different croppings


    return allChars, maxX, maxY

def padOnMax(imgs, maxX, maxY):
    return [cv.copyMakeBorder(img, math.ceil((maxX - img.shape[0]) / 2), math.floor((maxX - img.shape[0]) / 2), 
    math.ceil((maxY - img.shape[1]) / 2), math.floor((maxY - img.shape[1]) / 2), cv.BORDER_CONSTANT) for img in imgs]

def split(l, sp = 0.8):
    splitA = int(sp * len(archaicChars))
    return archaicChars[:splitA], archaicChars[splitA:]

archaicFolder = "characters_training/Archaic/"
hasmoneanFolder = "characters_training/Hasmonean/"
herodianFolder = "characters_training/Herodian/"

# make unique list of all characters 
characters = set([])
characters.update([f.path[len(archaicFolder):] for f in os.scandir(archaicFolder) if f.is_dir()]) 
characters.update([f.path[len(hasmoneanFolder):] for f in os.scandir(hasmoneanFolder) if f.is_dir()]) 
characters.update([f.path[len(herodianFolder):] for f in os.scandir(herodianFolder) if f.is_dir()]) 

characters = ["Alef"] # just for trying with only the first character. 

for char in characters:
    print("busy with " + char)
    print("reading images...")
    # read all png images of character
    archaicChars = [getImage(f.path) for f in os.scandir(archaicFolder + char) if f.is_file() and f.path[-3:] == "png"]
    hasmoneanChars = [getImage(f.path) for f in os.scandir(hasmoneanFolder + char) if f.is_file() and f.path[-3:] == "png"]
    herodianChars = [getImage(f.path) for f in os.scandir(herodianFolder + char) if f.is_file() and f.path[-3:] == "png"]

    aTrain, aTest = split(archaicChars)
    haTrain, haTest = split(hasmoneanChars)
    heTrain, heTest = split(herodianChars) 

    augATrain, x1, y1 = augment(aTrain)
    augATest, x2, y2 = augment(aTest)
    augHaTrain, x3, y3 = augment(haTrain)
    augHaTest, x4, y4 = augment(haTest)
    augHeTrain, x5, y5 = augment(heTrain)
    augHeTest, x6, y6 = augment(heTest)

    maxX = max(x1, x2, x3, x4, x5, x6)
    maxY = max(y1, y2, y3, y4, y5, y6)

    augATrain = padOnMax(augATrain, maxX, maxY)
    augATest = padOnMax(augATest, maxX, maxY)
    augHaTrain = padOnMax(augHaTrain, maxX, maxY)
    augHaTest = padOnMax(augHaTest, maxX, maxY)
    augHeTrain = padOnMax(augHeTrain, maxX, maxY)
    augHeTest = padOnMax(augHeTest, maxX, maxY)
    
    yTrain = [0] * len(augATrain) + [1] * len(augHaTrain) + [2] * len(augHeTrain)
    yTrain = to_categorical(yTrain)
    xTrain = augATrain + augHaTrain + augHeTrain

    yTest = [0] * len(augATest) + [1] * len(augHaTest) + [2] * len(augHeTest)
    yTest = to_categorical(yTest)
    xTest = augATest + augHaTest + augHeTest

    z = list(zip(xTrain, yTrain))
    random.shuffle(z)
    xTrain, yTrain = zip(*z)

    trainModel("./models/" + char, np.asarray(xTrain), np.asarray(yTrain), np.asarray(xTest), np.asarray(yTest))