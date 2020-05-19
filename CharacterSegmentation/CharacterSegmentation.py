
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

def generateWindows(image, windowSize = 80, stepSize = 20):

    (height, width) = np.shape(image)

    listOfWindows = []
    for i in range(int((width-windowSize)/stepSize+1)):
        listOfWindows.append( (i*stepSize, windowSize) )

    print(listOfWindows[-1])
    return listOfWindows