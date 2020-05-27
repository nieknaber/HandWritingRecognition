
import numpy as np
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
arr = os.listdir()
print(arr)