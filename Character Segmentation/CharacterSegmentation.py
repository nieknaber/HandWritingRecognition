
import numpy as np
from helper import *

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt



def findWindows():
    binaryImage = getBinaryDummyImage('dummy.jpg')
    # showBinaryImage(binaryImage)

    windows = generateWindows(binaryImage)

    significantWindows = filterWindows(binaryImage, windows)
    windows = significantWindows

    centers = generateAllCenterOfGravities(binaryImage, windows)

    imageRGB = convertToRGBImage(binaryImage)
    windowedImage = addWindows(imageRGB, windows)
    imageWithCenter = drawCentrePoint(windowedImage, centers)
    showRGBImage(windowedImage)

def projectionTransform(image, threshold = 0.01, alongXAxis = True):

    (height, width) = np.shape(image)

    if alongXAxis:
        image = image.T

    histogram = np.zeros(width)
    for index, segment in enumerate(image):
        histogram[index] = sum(segment)
    
    minimumLinePositions = []

    maximum = max(histogram)
    valueThreshold = maximum * threshold
    print("Thresholded Projection Transform at: ", valueThreshold)

    for index, line in enumerate(histogram):
        if line <= valueThreshold:
            minimumLinePositions.append(index)

    return minimumLinePositions

def generateWindows(image, windowSize = 40, stepSize = 1):
    # StepSize >1 is currently broken, but also not necessary...

    (height, width) = np.shape(image)

    listOfWindows = []
    for i in range(int((width-windowSize-1)/stepSize+1)):
        listOfWindows.append( (i*stepSize, windowSize) )

    return listOfWindows

def generateCenterOfGravity(image, window):

    (height, _) = np.shape(image)
    (windowStart, windowSize) = window

    allXs = []
    allYs = []
    for y in range(height):
        for x in range(windowStart, windowStart+windowSize):
            if image[y,x] == 1:
                allXs.append(x)
                allYs.append(y)

    if allXs == []:
        meanX = int(round(windowStart + windowSize/2))
        meanY = int(round(height/2))
    else:
        meanX = int(round(np.mean(allXs),0))
        meanY = int(round(np.mean(allYs),0))        

    return (meanY, meanX)

def generateAllCenterOfGravities(image, windows):

    (height, _) = np.shape(image)

    allGravityPoints = []

    for window in windows:
        center = generateCenterOfGravity(image, window)
        allGravityPoints.append(center)

    return allGravityPoints

def determineSignificantDeltas(image, windows):

    allCenters = generateAllCenterOfGravities(image, windows)

    allXs = []
    for center in allCenters:
        (_, x) = center
        allXs.append(x)

    allDeltas = []
    for i, center in enumerate(allXs):
        if i == 0: continue
        delta = allXs[i] - allXs[i-1]
        allDeltas.append(delta)

    # Confidence interval as a multiple of SD
    threshold = 1 * np.std(allDeltas)

    apartWindows = [windows[0]]
    for i, delta in enumerate(allDeltas):
        if delta > threshold:
            apartWindows.append(windows[i+1])

    return apartWindows

def filterWindows(image, windows, minDensity = 0.06):

    survivingWindows = []

    differentCenteredWindows = determineSignificantDeltas(image, windows)

    previousWindow = None
    for window in differentCenteredWindows:

        include = True

        density = calculatePixelDensityOfWindow(image, window)
        if density < minDensity: include = False

        if previousWindow != None:
            (currentStart,size) = window
            (previousStart,_) = previousWindow
            delta = currentStart - previousStart
            if delta < size*0.4: include = False

        if include: 
            survivingWindows.append(window)
            previousWindow = window

    return survivingWindows