import cv2 as cv
import numpy as np
import sys
import os
from skimage.morphology import convex_hull_image

def getBackgroundSet(d, numBackgrounds = 4):
    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(d, connectivity=8)
    arr = np.zeros(nb_components, np.uint8)

    for i in range(1,nb_components):
        arr[i] = sum([sum(row) for row in output == i])

    # get 4 beggest centroids
    dNumbers = np.argsort(arr)
    numBackgrounds = min(len(dNumbers), numBackgrounds)
    dNumbers = dNumbers[-numBackgrounds:]
    centroids2 = (centroids[dNumbers])

    # sort to y coordinate
    dNumbers = dNumbers[np.argsort(centroids2[:,1])]
    centroids = centroids[dNumbers]

    # swap middle two if x coordinate demands
    if (numBackgrounds > 3 and centroids[2][0] < centroids[1][0]):
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
    return []

def getFeatureFour(dSet):

    f4 = []
    for background in dSet:
        contours, _ = cv.findContours(background, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        area = cv.contourArea(contours[0])
        perimeter = cv.arcLength(contours[0], True)
        if (perimeter != 0):
            f4.append(4 * np.pi * area / perimeter)
        else:
            f4.append(0)

    return f4
    
def getFeatureFive(dSet):

    f5 = []
    for background in dSet:
        out = cv.moments(background)
        update = [out["nu11"], out["nu02"], out["nu20"], out["nu21"], out["nu12"]]
        f5.extend(update)

    return f5

def getFeatureSix(b, lenHs):
    x, y = b.shape
    return [x * y / lenHs]

def getFeatureSeven(b):
    contours, _ = cv.findContours(b, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    if (len(contours[0]) >= 5):
        _, (minor, major), _ = cv.fitEllipse(contours[0])
        return [minor / major]
    else:
        return [1.0]

def getFeatureEight(b):
    return [b.shape[1] / b.shape[0]]

def getFeatures(x, num):

    numBackgrounds = num

    allFeatures = []
    for b in x:
        features = []
        hs = convex_hull_image(b).astype(np.uint8) # convex hull
        lenHs = sum([sum(row) for row in hs])
        d = hs - b # convex deficiency
        dSet = getBackgroundSet(d, numBackgrounds) # set D_i from the paper. now contains 4 background sets. number 4 is hardcoded

        features.extend(getFeatureOne(lenHs, dSet))
        features.extend(getFeatureTwo(dSet))
        features.extend(getFeatureThree(dSet))
        features.extend(getFeatureFour(dSet))
        features.extend(getFeatureFive(dSet))
        features.extend(getFeatureSix(b, lenHs))
        features.extend(getFeatureSeven(b))
        features.extend(getFeatureEight(b))

        allFeatures.append(features)

    return np.array(allFeatures)