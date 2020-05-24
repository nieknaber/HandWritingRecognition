
import numpy as np
import cv2

def projectionTransform(image, threshold = 0.01, alongXAxis = True):
    (height, width) = np.shape(image)

    if alongXAxis:
        image = image.T
        histogram = np.zeros(width)
    else:
        histogram = np.zeros(height)

    for index, segment in enumerate(image):
        histogram[index] = sum(segment)
    minimumLinePositions = []

    maximum = max(histogram)
    valueThreshold = maximum * threshold
    print("Thresholded Projection Transform at: ", valueThreshold)

    for index, line in enumerate(histogram):
        if line <= valueThreshold:
            #print(line)
            #print(valueThreshold)
            minimumLinePositions.append(index)

    return minimumLinePositions
    
    
def ContourExtraction(image):
    
    im = cv2.imread(image)
    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray,127,255,0)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    print("Found Contours: ", len(contours))

    return contours 