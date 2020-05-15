
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

# Y-axis Projection Transform before X-axis Preojection Transform. According to paper on Thai characters
line_levels = projectionTransform(binaryImage, threshold = 0.15, alongXAxis = False)
markedImage = addHorizontalLinesToImage(binaryImage,  line_levels)
#showRGBImage(markedImage)
cuttedImage = np.delete(binaryImage, line_levels, axis=0)

# X-axis Preojection Transform
lines = projectionTransform(cuttedImage)
markedImage = addVerticalLinesToImage(cuttedImage, lines)
showRGBImage(markedImage)