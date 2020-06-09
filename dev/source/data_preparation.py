
import os, cv2
import numpy as np

from os import listdir
from PIL import Image

#####################################################################
## Resizing single or all training images

def resizeImage(imageFileName, newSize):
    (height, width) = newSize
    image = cv2.imread(imageFileName, cv2.IMREAD_UNCHANGED)
    resized = cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)
    return Image.fromarray(resized)

# Convert training data to windows
def resizeAllImages(resizedDimensions, fromLoation, toLocation, type = ".png"):
    characters = os.listdir(fromLoation)
    for character in characters:
        if not character.startswith('.'):
            files = os.listdir(fromLoation + "/" + character)
            for f in files:
                if not f.startswith('.'):
                    if f.endswith(type):
                        imagePath = fromLoation + '/' + character + '/' + f
                        newImage = resizeImage(imagePath, resizedDimensions)
                        newImage.save(toLocation+ '/' +f)

#####################################################################
## Creating and saving the Segments

def createWindowsFromTrainingImage(filename, windowSize):

    image = Image.open(filename)
    image = np.array(image)

    (h,w) = np.shape(image)
    (height, width) = windowSize
    left = image[:,0:width]
    right = image[:,(w-width):]

    return (left, right)

def createFeatureSegments(window, segmentSize):
    segments = []
    for i in range(6):
        for j in range(2):
            segments.append(window[i*16:(i+1)*16,j*16:(j+1)*16])
    return segments

def saveSegmentsAsImages(segments, location):
    i = 1
    for s in segments:
        s = Image.fromarray(s)
        s.save(location + str(i) + ".png")
        i += 1