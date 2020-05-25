import cv2 as cv
import numpy as np
import math
from scipy import stats

# Takes image name, performs hough transform. 
#
# Returns Theta, Rows[], Segments [][][]:
#
# Theta is angle of all lines. 
# Rhos are all rho's of polar coordinates of line.
# Segment is a list of all cropped images, from one line to the other. 
# The line that is found in a cropped image has angle theta, and starts in the top left corner.
def lineSegment(filename):
    
    image = cv.imread(filename)
    dimensions = np.shape(image)
    grayimg = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    ah = 100 # average character height, for now an estimate

    #Hough transform
    lines = cv.HoughLines(~grayimg, int(0.2 * ah), np.pi / 90, 2500, None,0,0,np.pi / 2 - np.pi / 36, np.pi / 2 + np.pi / 36) # dim: (lines, 1, 2) where third dim: rho; theta
    freqs = stats.itemfreq(lines[:,:,1])
    freqs = freqs[freqs[:, 1].argsort()]
    
    i = len(freqs) - 1

    thetaDominant = freqs[len(freqs) - 1, 0]
    while (abs(np.pi / 2 - thetaDominant % np.pi) > np.pi / 36): # theta allowed to deviate 2.5 degrees from 90
        i = i - 1
        thetaDominant = freqs[i, 0]
    
    linePoints = determinePoints(lines, grayimg, dimensions, thetaDominant, 0, True)
    # cv.namedWindow("Window", flags=cv.WINDOW_NORMAL)
    # cv.imshow("Window",grayimg)
    # cv.waitKey(0)

    return (thetaDominant, linePoints, grayimg)

########
# helper functions
def determinePoints(lines, grayimg, dimensions, theta, variation = np.pi / 90, threshold = True):
    linePoints = []

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta2 = lines[i][0][1]
            # print(theta2)
            a = math.cos(theta2)
            b = math.sin(theta2)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + dimensions[1]*(-b)), int(y0 + dimensions[0]*(a)))
            pt2 = (int(x0 - dimensions[1]*(-b)), int(y0 - dimensions[0]*(a)))
            if (threshold):
                if(abs(theta - theta2) <= variation): # 2 degree variation allowed
                    # cv.line(grayimg, pt1, pt2, (0,0,0), 5)
                    linePoints.append((pt1, pt2))
            else:
                # cv.line(grayimg, pt1, pt2, (0,0,0), 5)
                linePoints.append((pt1, pt2))

    return linePoints