
import numpy as np
from helper import *
from CharacterSegmentation import *

binaryImage = getBinaryDummyImage('dummy.jpg')
# showBinaryImage(binaryImage)

windows = generateWindows(binaryImage)

significantWindows = filterWindows(binaryImage, windows)
windows = significantWindows

centers = generateAllCenterOfGravities(binaryImage, windows)

imageRGB = convertToRGBImage(binaryImage)
windowedImage = addWindows(imageRGB, windows)
imageWithCenter = drawCentrePoint(windowedImage, centers)
showRGBImage(windowedImage)