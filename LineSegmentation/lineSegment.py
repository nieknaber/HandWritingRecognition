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
    lines = cv.HoughLines(grayimg, int(0.2 * ah), np.pi / 90, 1, None,0,0,1,3) # dim: (lines, 1, 2) where third dim: rho; theta
    freqs = stats.itemfreq(lines[:,:,1])
    freqs = freqs[freqs[:, 1].argsort()]
    
    i = len(freqs) - 1

    thetaDominant = freqs[i, 0]
    while (abs(np.pi / 2 - thetaDominant % np.pi) > np.pi / 36): # theta allowed to deviate 2.5 degrees from 90
        i = i - 1
        thetaDominant = freqs[i, 0]
    
    print(thetaDominant)
    cv.namedWindow("Window", flags=cv.WINDOW_NORMAL)
    drawLines(lines, grayimg, dimensions, thetaDominant, True)
    cv.imshow("Window",grayimg)
    cv.waitKey(0)


    numberOfLines = len(lines)
    # # range(numberOfLines - 1)
    # rhos = []
    # for index in range(numberOfLines):
    #     # if (lines[index][0][1] == theta):
    #     rhos.append(int(lines[index][0][0] / math.sin(lines[index][0][1])))

    # rhos = np.sort(np.array(rhos))
    # print(rhos)

    # croppedImgs = []
    # for i in range(len(rhos) - 1):
    #     cropped = grayimg[rhos[i]:rhos[i+1], :]
    #     croppedImgs.append(cropped)

    # cropped = grayimg[rhos[-1]:dimensions[0], :]
    # croppedImgs.append(cropped)

    # return (thetaDominant, rhos, croppedImgs)



########
# helper functions
def drawLines(lines, grayimg, dimensions, theta, threshold = True):
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
                if(abs(theta - theta2) < np.pi / 90): # 2 degree variation allowed
                    cv.line(grayimg, pt1, pt2, (0,0,0), 5)
            else:
                cv.line(grayimg, pt1, pt2, (0,0,0), 5)