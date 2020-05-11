
import numpy as np
from helper import *
from CharacterSegmentation import *

# Importing the image and converting to binarys
binaryImage = getBinaryDummyImage('dummy.jpg')
showBinaryImage(binaryImage)

# Simple Projection Transform along the X-axis
lines = projectionTransform(binaryImage)
markedImage = addVerticalLinesToImage(binaryImage, lines)
showRGBImage(markedImage)