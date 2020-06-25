
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
def resizeAllImages(resizedDimensions, fromLocation, toLocation, type = ".png"):
    characters = os.listdir(fromLocation)
    for character in characters:
        if not character.startswith('.'):
            files = os.listdir(fromLocation + "/" + character)
            for f in files:
                if not f.startswith('.'):
                    if f.endswith(type):
                        imagePath = fromLocation + '/' + character + '/' + f
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

def createFeatureSegments(window, segmentSize, windowSize):
    (h,w) = segmentSize
    (x,y) = windowSize

    segments = []
    for i in range(int(x/h)):
        for j in range(int(y/w)):
            segments.append(window[i*h:(i+1)*h,j*h:(j+1)*h])
    return segments

def saveSegmentsAsImages(segments, location):
    i = 1
    for s in segments:
        s = Image.fromarray(s)
        s.save(location + str(i) + ".png")
        i += 1

#####################################################################
## Converting all images to directions

def covertResizedSegmentsIntoDirections(fromLocation, segmentSize, windowSize, type = ".png"):
    # segmentSize = (16,16)
    # windowSize = (16*6, 16*3) # 96,32

    data = []

    characters = os.listdir(fromLocation)
    for character in characters:
        if not character.startswith('.'):
            if character.endswith(type):
                label = character.split("_")[0]
                imagePath = fromLocation + '/' + character

                (left, right) = createWindowsFromTrainingImage(imagePath, windowSize)
                leftSegments = createFeatureSegments(left, segmentSize, windowSize)
                # rightSegments = createFeatureSegments(right, segmentSize)

                data.append((leftSegments, label))
                # data.append((rightSegments, label))

    return data

def saveDirectionsToFile(data, toLocation):
    data = np.transpose(np.array(data).flatten())
    np.savetxt(toLocation, data, fmt='%i', delimiter=',')

def loadFromText(fromLocation):
    data = np.genfromtxt(fromLocation, delimiter=',')
    return data.astype(int)

def loadAllData(fromLocation):
    allData = []
    characters = os.listdir(fromLocation)
    for character in characters:
        if not character.startswith('.'):
            label = character.split("_")[0]
            dataPath = fromLocation + '/' + character
            data = loadFromText(dataPath)
            allData.append((data, label))
    
    return allData