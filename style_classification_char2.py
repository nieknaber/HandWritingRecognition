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
from skimage.morphology import convex_hull_image
from PIL import Image


def augment(imgs):
    allChars = []

    maxX = 0
    maxY = 0

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

        img2 = np.zeros(output.shape)
        img2[output == max_label] = 1

        newImg = cropWhiteSpace(img2) # crop image
        x, y = newImg.shape
        if (x > maxX): maxX = x
        if (y > maxY): maxY = y
        allChars.append(newImg)

    return allChars, maxX, maxY

def split(l, sp = 0.8):
    splitA = int(sp * len(archaicChars))
    return archaicChars[:splitA], archaicChars[splitA:]

#works
def getBackgroundSet(d):
    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(d, connectivity=8)
    arr = np.zeros(nb_components, np.uint8)

    for i in range(1,nb_components):
        arr[i] = sum([sum(row) for row in output == i])

    # get 4 beggest centroids
    dNumbers = np.argsort(arr)[-4:]
    centroids2 = (centroids[dNumbers])

    # sort to y coordinate
    dNumbers = dNumbers[np.argsort(centroids2[:,1])]
    centroids = centroids[dNumbers]

    # swap middle two if x coordinate demands
    if (centroids[2][0] < centroids[1][0]):
        tmp = centroids[1]
        centroids[1] = centroids[2]
        centroids[2] = centroids[1]

        tmp2 = dNumbers[1]
        dNumbers[1] = dNumbers[2]
        dNumbers[2] = dNumbers[1]

    dSet = [] # contains in order d1 d2 d3 and d4
    for num in dNumbers:
        img = np.zeros(output.shape, np.uint8)
        img[output == num] = 1
        dSet.append(img)

    return dSet

# |Di| / |Hs|
def getFeatureOne(lenHs, dSet):
    f1 = []
    for background in dSet:
        f1.append(sum([sum(row) for row in background]) / lenHs)

    return f1

# Minor(Di) / Major(Di)
def getFeatureTwo(dSet):
    
    f2 = [] 
    for background in dSet:
        contours, _ = cv.findContours(background, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        if (len(contours[0]) >= 5):
            _, (minor, major), _ = cv.fitEllipse(contours[0])
            f2.append(minor / major)
        else:
            f2.append(1.0)

    return f2

def getFeatureThree(dSet):
    var = 5

def getFeatureFour(dSet):

    f4 = []
    for background in dSet:
        contours, _ = cv.findContours(background, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        area = cv.contourArea(contours[0])
        perimeter = cv.arcLength(contours[0])
        f4.append(4 * np.pi * area / perimeter)

    return f4
    


def getFeatures(outputfile, xTrain, yTrain, xTest, yTest):

    for s in [xTrain[0]]:
        t, p = s.shape
        hs = convex_hull_image(s).astype(np.uint8) # convex hull
        lenHs = sum([sum(row) for row in hs])
        d = hs - s.astype(np.uint8) # convex deficiency

        dSet = getBackgroundSet(d) # set D_i from the paper. now contains 4 background sets. number 4 is hardcoded
        f1 = getFeatureOne(lenHs, dSet)
        f2 = getFeatureTwo(dSet)
        f3 = getFeatureThree(dSet)
        f4 = getFeatureFour(dSet)

    # use if you want to display a background set
    #cv.namedWindow("Window", flags=cv.WINDOW_NORMAL)
    #cv.imshow("Window", 255 * dSet[0])
    #cv.waitKey(0)


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
    # print("reading images...")
    # read all png images of character
    if (not(os.path.isdir(archaicFolder + char)) or not(os.path.isdir(hasmoneanFolder + char)) or not(os.path.isdir(herodianFolder + char))):
        print("skipped")
        continue

    archaicChars = [getImage(f.path) for f in os.scandir(archaicFolder + char) if f.is_file() and f.path[-3:] == "jpg"]
    hasmoneanChars = [getImage(f.path) for f in os.scandir(hasmoneanFolder + char) if f.is_file() and f.path[-3:] == "jpg"]
    herodianChars = [getImage(f.path) for f in os.scandir(herodianFolder + char) if f.is_file() and f.path[-3:] == "jpg"]

    aTrain, aTest = split(archaicChars)
    haTrain, haTest = split(hasmoneanChars)
    heTrain, heTest = split(herodianChars) 

    augATrain, x1, y1 = augment(aTrain)
    augATest, x2, y2 = augment(aTest)
    augHaTrain, x3, y3 = augment(haTrain)
    augHaTest, x4, y4 = augment(haTest)
    augHeTrain, x5, y5 = augment(heTrain)
    augHeTest, x6, y6 = augment(heTest)

    yTrain = [0] * len(augATrain) + [1] * len(augHaTrain) + [2] * len(augHeTrain)
    xTrain = augATrain + augHaTrain + augHeTrain

    yTest = [0] * len(augATest) + [1] * len(augHaTest) + [2] * len(augHeTest)
    xTest = augATest + augHaTest + augHeTest

    z = list(zip(xTrain, yTrain))
    random.shuffle(z)
    xTrain, yTrain = zip(*z)

    xTrain = [img for img in xTrain]
    xTest = [img for img in xTest]

    getFeatures("./models/" + char, np.asarray(xTrain), np.asarray(yTrain), np.asarray(xTest), np.asarray(yTest))