import sys
import time
from SlantNormalization.slantNormalize import *
from LineSegmentation.lineSegment import *
from LineSegmentation.lineSegmentAStar import *
from SlantNormalization.slantNormalize import *

SAVEPICTURES = True # set to false if you do not want to save pictures in between steps

img = getImage("images/20.jpg")
rotatedImage, slope = findSlope(img, 10, 1)
print("Best angle: ", slope)

images = lineSegmentAStar(rotatedImage)

for i in range(0,len(images)):
    images[i] = np.transpose(images[i])
    if SAVEPICTURES: cv.imwrite("line_" + str(i) + ".bmp", (1 - images[i]) * 255)

slantAngles = []
for i in range(0,len(images)):
    slantAngle, dst = slantNormalize(images[i])
    if SAVEPICTURES: cv.imwrite("line_" + str(i) + "_deslanted.bmp", (1 - dst) * 255)
    images[i] = dst
    slantAngles.append(slantAngle)

# images[] is now the list of slope corrected line segments
# slantAngles[] is the list of slants per line segment (same indexes)
# slope is the line slope angle