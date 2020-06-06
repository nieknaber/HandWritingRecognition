import cv2 as cv
import numpy as np
import math
from scipy import stats, ndimage
from helper import getImage

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
    grayimg = image
    #grayimg = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    ah = 200 # average character height, for now an estimate

    #Hough transform
    lines = cv.HoughLines(~grayimg, int(0.2 * ah), np.pi / 90, 5000, None,0,0,np.pi / 2 - np.pi / 36, np.pi / 2 + np.pi / 36) # dim: (lines, 1, 2) where third dim: rho; theta
    freqs = stats.itemfreq(lines[:,:,1])
    freqs = freqs[freqs[:, 1].argsort()]
    
    i = len(freqs) - 1

    thetaDominant = freqs[len(freqs) - 1, 0]
    while (abs(np.pi / 2 - thetaDominant) > np.pi / 36): # theta allowed to deviate 2.5 degrees from 90
        i = i - 1
        thetaDominant = freqs[i, 0]
    
    linePoints = determinePoints(lines, grayimg, dimensions, thetaDominant, 0, True)
    # cv.namedWindow("Window", flags=cv.WINDOW_NORMAL)
    # cv.imshow("Window",grayimg)
    # cv.waitKey(0)

    return (thetaDominant, linePoints, grayimg)

def findSlope(filename, n_angles, precision):
    # load image and invert it for the histograms to work
    image = getImage(filename)
    #grayimg = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # initialize array to store results (max projection height)
    slopeResults = np.zeros(n_angles*precision+1)

    # loop over all angles
    for angle in range(0,n_angles*precision+1):
        # print statement to see that stuff is happening cause it's not so fast
        print("Computing angle ",angle/precision-(n_angles/2))
        # rotate image by specified degrees
        rotated = ndimage.rotate(image,angle/precision-(n_angles/2))
        # compute projection profile for this rotation
        rotatedProj = np.sum(rotated,1)
        # find the highest peak (indicating the longest line) and save it to results
        slopeResults[angle] = np.max(rotatedProj)


    bestAngle = np.argmax(slopeResults)/precision-(n_angles/2)
    # Create output image same height as text, 500 px wide
    returnImage = ndimage.rotate(image,bestAngle)
    # compute projections (again) to show for return image
    proj = np.sum(returnImage,1)
    m = np.max(proj)
    w = 500 # width of the graph
    result = np.zeros((proj.shape[0], 500))

    # Draw projection profile for result
    for row in range(image.shape[0]):
        cv.line(result, (0, row), (int(proj[row] * w / m), row), (255, 0, 0), 1)  # int(proj[row]*w/m)

    # return image with added projection profile
    returnImage = np.concatenate((returnImage, result), axis=1)
    asdf = np.zeros((proj.shape[0],1))


    return returnImage, bestAngle


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
            # convert to coordinates on edges of the image
            slope = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])  
            newpt1 = (0, int(pt1[1] - pt1[0] * slope))
            newpt2 = (dimensions[1] - 1, int(pt2[1] + (dimensions[1] - 1 - pt2[0]) * slope ))
            if (threshold):
                if(abs(theta - theta2) <= variation): # 2 degree variation allowed
                    # cv.line(grayimg, newpt1, newpt2, (0,0,0), 5)
                    linePoints.append((newpt1, newpt2))
            else:
                # cv.line(grayimg, newpt1, newpt2, (0,0,0), 5)
                linePoints.append((newpt1, newpt2))

    return linePoints