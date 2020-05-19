
import numpy as np

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

def generateWindows(image, windowSize = 80, stepSize = 80):

    (height, width) = np.shape(image)

    listOfWindows = []
    for i in range(int((width-windowSize)/stepSize+1)):
        listOfWindows.append( (i*stepSize, windowSize) )

    return listOfWindows

def generateCenterOfGravity(image, windows):

    (height, _) = np.shape(image)

    allGravityPoints = []

    for window in windows:
        (windowStart, windowSize) = window

        allXs = []
        allYs = []
        for y in range(height):
            for x in range(windowStart, windowStart+windowSize):
                if image[y,x] == 1:
                    allXs.append(x)
                    allYs.append(y)

        meanX = int(round(np.mean(allXs),0))
        meanY = int(round(np.mean(allYs),0))
        allGravityPoints.append((meanY, meanX))

    return allGravityPoints