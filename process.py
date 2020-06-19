import sys
import time
from src.HelperFunctions.helper import *
from src.LineSegmentation.lineSegment import *
from src.LineSegmentation.lineSegmentAStar import *
from src.SlantNormalization.slantNormalize import *

SAVEPICTURES = True # set to false if you do not want to save pictures in between steps

img = getImage("images/05.jpg")
rotatedImage, slope = findSlope(img, 10, 1)
print("Best angle: ", slope)

images = lineSegmentAStar(rotatedImage)
print("len(images)): ",len(images))
for i in range(0,len(images)):
    images[i] = np.transpose(images[i])
    if SAVEPICTURES: cv.imwrite("line_" + str(i) + ".bmp", (1 - images[i]) * 255)
    if SAVEPICTURES: print("saving picture ", i)

slantAngles = []
for i in range(0,len(images)):
    slantAngle, dst = slantNormalize(images[i])
    if SAVEPICTURES: cv.imwrite("line_" + str(i) + "_deslanted.bmp", (1 - dst) * 255)
    images[i] = dst
    slantAngles.append(slantAngle)

# images[] is now the list of slope corrected line segments
# slantAngles[] is the list of slants per line segment (same indexes)
# slope is the line slope angle