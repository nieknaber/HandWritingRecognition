import cv2 as cv
import numpy as np
# import sys
import math
# import matplotlib.pyplot as plt
from scipy import stats

# Takes image name, performs hough transform. 
#
# Returns Theta, Rows[], Segments [][][]:
#
# Theta is angle of all lines. 
# Rhos are all rho's of polar coordinates of line.
# Segment is a list of all cropped images, from one line to the other. 
# The line that is found in a cropped image has angle theta, and starts in the top left corner.
def lineSegment(filename, invertImg = False):
    
    image = cv.imread(filename)
    dimensions = np.shape(image)
    grayimg = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    if (invertImg): grayimg = ~grayimg

    #Hough transform
    lines = cv.HoughLines(grayimg, 80, np.pi/90, 5, None,0,0,1,3) # dim: (lines, 1, 2) where third dim: rho; theta
    numberOfLines = len(lines)
    mode = stats.mode(lines[:,:,1])
    theta = mode[0]
    
    # # range(numberOfLines - 1)
    rhos = []
    for index in range(numberOfLines):
        if (lines[index][0][1] == theta):
            rhos.append(int(lines[index][0][0]))

    rhos = np.sort(np.array(rhos))
    print(rhos)

    croppedImgs = []
    for i in range(len(rhos) - 1):
        cropped = grayimg[rhos[i]:rhos[i+1], :]
        croppedImgs.append(cropped)

    cropped = grayimg[rhos[-1]:dimensions[0], :]
    croppedImgs.append(cropped)

    return (theta, rhos, croppedImgs)