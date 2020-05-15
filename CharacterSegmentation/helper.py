
import numpy as np
from PIL import Image

def getBinaryDummyImage(filename):
    image = Image.open(filename)
    (height, width, _) = np.shape(image) # dummy is (180, 1250, 3), i.e. (height, width, colors)

    image = image.load()
    newBinarizedImage = np.zeros((height,width))

    for y in range(height):
        for x in range(width):

            (r, _, _) = image[x,y]

            if r < 125:
                newBinarizedImage[y,x] = 1

    return newBinarizedImage 

def showBinaryImage(image):
    image = convertToRGBImage(image)
    image = Image.fromarray(image, 'RGB')
    image.show()

def showRGBImage(image):
    image = Image.fromarray(image, 'RGB')
    image.show()

def convertToRGBImage(image):
    (height, width) = np.shape(image)

    data = np.zeros((height, width, 3), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            if image[y,x] == 0:
                data[y,x] = [255,255,255]
            else:
                data[y,x] = [0,0,0]

    return data

def addVerticalLinesToImage(image, xPositions):
    
    (height, width) = np.shape(image)
    image = convertToRGBImage(image)

    for position in xPositions:
        for y in range(height):
            image[y,position] = [255,0,0]

    return image 

def addHorizontalLinesToImage(image, yPositions):
    
    (height, width) = np.shape(image)
    image = convertToRGBImage(image)

    for position in yPositions:
        for x in range(width):
            image[position, x] = [255,0,0]

    return image 
    

