import sys
import time
from LineSegmentation.lineSegment import *
from LineSegmentation.lineSegmentAStar import *

img = getImage("images/20.jpg")
rotatedImage, slope = findSlope(img, 10, 1)
print("Best angle: ", slope)

images = lineSegmentAStar(rotatedImage)

for i in range(0,len(images)):
    images[i] = np.transpose(images[i])
    cv.imwrite("line " + str(i) + ".bmp", images[i] * 255)

# images is now the list of slope corrected line segments