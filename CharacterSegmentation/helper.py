
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


### Here are the RGB image functions ###

def showRGBImage(imageRGB):
    imageRGB = Image.fromarray(imageRGB, 'RGB')
    imageRGB.show()

def addVerticalLinesToImage(imageRGB, xPositions):
    
    (height, width, _) = np.shape(imageRGB)

    for position in xPositions:
        for y in range(height):
            imageRGB[y,position] = [255,0,0]

    return imageRGB

def addWindows(imageRGB, windows):

    for window in windows:
        (start, size) = window
        imageRGB = addVerticalLinesToImage(imageRGB, [start, start+size])

    return imageRGB

def drawX(imageRGB, color, size, x, y):
    for i in range(size):
        imageRGB[x-i,y-i] = color
        imageRGB[x+i,y-i] = color
        imageRGB[x-i,y+i] = color
        imageRGB[x+i,y+i] = color
    return imageRGB

def drawMultipleOffsetX(imageRGB, x, y, color = [0,0,255], size = 6, offset = 2):
    imageRGB[x,y] = color
    drawX(imageRGB, color, size, x, y)
    while offset != 0:
        drawX(imageRGB, color, size, x-offset, y)
        drawX(imageRGB, color, size, x+offset, y)
        drawX(imageRGB, color, size, x, y-offset)
        drawX(imageRGB, color, size, x, y+offset)
        offset -= 1
    return imageRGB

def drawCentrePoint(imageRGB, points):

    for point in points:
        (x, y) = point
        imageRGB = drawMultipleOffsetX(imageRGB, x, y)

    return imageRGB