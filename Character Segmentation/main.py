
import numpy as np
from PIL import Image
from helper import *
from CharacterSegmentation import *

import os
from os import listdir

# findWindows()

# Constants and parameters
segmentSize = (16,16)
windowSize = (16*6, 16*2)
featuresPerSegment = 8
featuresPerWindow = featuresPerSegment * (2*12)

# Examine training data

# i = Image.open('./monkbrill/Resh/navis-QIrug-Qumran_extr09_2357-line-007-y1=392-y2=514-zone-HUMAN-x=0711-y=0020-w=0032-h=0067-ybas=0040-nink=443-segm=COCOS5cocos.pgm')
# i.show()

characterLocation = "Herodian"
characters = os.listdir(characterLocation)
for character in characters:
    if not character.startswith('.'):
        files = os.listdir(characterLocation + "/" + character)
        for f in files:
            if not f.startswith('.'):
                imagePath = characterLocation + '/' + character + '/' + f
                # print(imagePath)
                image = Image.open(imagePath)
                (height, width) = np.shape(image)
                print(height, width)

