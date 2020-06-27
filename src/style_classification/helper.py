import os
import random
import numpy as np
import cv2 as cv
from statistics import stdev
from src.data_preparation.helper import *
from src.segmentation.slantNormalize import *

def preprocess(imgs):
    allChars = []
    for charImg in imgs:  
        #only get biggest connected component
        nb_components, output, stats, centroids = cv.connectedComponentsWithStats(charImg, connectivity=8)
        sizes = stats[:, -1]

        max_label = 1
        max_size = sizes[1]
        for i in range(2, nb_components):
            if sizes[i] > max_size:
                max_label = i
                max_size = sizes[i]

        img2 = np.zeros(output.shape, np.uint8)
        img2[output == max_label] = 1

        newImg = cropWhiteSpace(img2) # crop image
        allChars.append(newImg)

    return allChars

def getData(char, archaicFolder, hasmoneanFolder, herodianFolder):

    archaicChars = [getImage(f.path) for f in os.scandir(archaicFolder + char) if f.is_file() and f.path[-3:] == "jpg"]
    hasmoneanChars = [getImage(f.path) for f in os.scandir(hasmoneanFolder + char) if f.is_file() and f.path[-3:] == "jpg"]
    herodianChars = [getImage(f.path) for f in os.scandir(herodianFolder + char) if f.is_file() and f.path[-3:] == "jpg"]

    random.shuffle(archaicChars)
    random.shuffle(hasmoneanChars)
    random.shuffle(herodianChars)
    
    x = preprocess(archaicChars) + preprocess(hasmoneanChars) + preprocess(herodianChars)
    y = [0] * len(archaicChars) + [1] * len(hasmoneanChars) + [2] * len(herodianChars)

    return x, y

def zScoreTwoArrs(trainTransformed, testTransformed):
     # calculate z-score features
    numVectors, numFeatures = trainTransformed.shape
    zScores = np.zeros((numFeatures, 2))
    for i in range(numFeatures):
        l = np.concatenate((trainTransformed[:,i], testTransformed[:,i]))
        zScores[i, 0] = sum(l) / numVectors
        zScores[i, 1] = stdev(l)

    for vec in range(numVectors):
        for fet in range(numFeatures):
            trainTransformed[vec, fet] =  (trainTransformed[vec, fet] - zScores[fet, 0]) / zScores[fet, 1]

    numVectors, numFeatures = testTransformed.shape
    for vec in range(numVectors):
        for fet in range(numFeatures):
            testTransformed[vec, fet] =  (testTransformed[vec, fet] - zScores[fet, 0]) / zScores[fet, 1]

    return trainTransformed, testTransformed, zScores
