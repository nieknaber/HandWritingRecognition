
import numpy as np
from PIL import Image
from helper import *
from CharacterSegmentation import *
import cv2

import os
from os import listdir

## GLOBAL (HEIGHT, WIDTH)

segmentSize = (16,16)
windowSize = (16*6, 16*2) # 96,32
resizedDimensions = (96, 64)
featuresPerSegment = 8
featuresPerWindow = featuresPerSegment * (2*12)

def resizeImage(imageFileName, newSize):
    (height, width) = newSize
    image = cv2.imread(imageFileName, cv2.IMREAD_UNCHANGED)
    resized = cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)
    return Image.fromarray(resized)

# Convert training data to windows
def resizeAllImages(fromLoation = "Herodian", toLocation = "./Resized Herodian/", type = ".png"):
    characters = os.listdir(fromLoation)
    for character in characters:
        if not character.startswith('.'):
            files = os.listdir(fromLoation + "/" + character)
            for f in files:
                if not f.startswith('.'):
                    if f.endswith(type):
                        imagePath = fromLoation + '/' + character + '/' + f
                        newImage = resizeImage(imagePath, resizedDimensions)
                        newImage.save(toLocation+f)

def createWindowsFromTrainingImage(image, windowSize):
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

i = Image.open('./Resized Herodian/Alef_19.png')
i = np.array(i)
(left, right) = createWindowsFromTrainingImage(i,windowSize)
segments = createFeatureSegments(left, segmentSize)

i = 1
for s in segments:
    s = Image.fromarray(s)
    s.save("./Test Segments/" + str(i) + ".png")
    i += 1