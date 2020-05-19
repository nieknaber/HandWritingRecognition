
import numpy as np
from helper import *
from CharacterSegmentation import *

# Importing the image and converting to binarys
binaryImage = getBinaryDummyImage('dummy.jpg')
showBinaryImage(binaryImage)

windows = generateWindows(binaryImage)

imageRGB = convertToRGBImage(binaryImage)
windowedImage = addWindows(imageRGB, windows)
showRGBImage(windowedImage)