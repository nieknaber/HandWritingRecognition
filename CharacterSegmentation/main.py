
import numpy as np
from helper import *
from CharacterSegmentation import *

# Importing the image and converting to binarys
binaryImage = getBinaryDummyImage('dummy.jpg')
showBinaryImage(binaryImage)

windows = generateWindows(binaryImage)
center = generateCenterOfGravity(binaryImage, windows[0])

imageRGB = convertToRGBImage(binaryImage)
windowedImage = addWindows(imageRGB, windows)
imageWithCenter = drawCentrePoint(windowedImage, [center])
showRGBImage(windowedImage)