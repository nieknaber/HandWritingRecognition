
from src.dev.source import selective_windowing as window
from src.dev.source import directional_rose as rose
from src.dev.source import training as train
from src.dev.source import helper as h
from src.dev.source import data_preparation as prep
from src.StyleClassification import helper as h2
import numpy as np

def createWindowsFromTrainingImage(image, windowParams):

    (h,w) = np.shape(image)
    windows = []
    for windowParam in windowParams: 
        (height, width) = windowParam
        left = image[:,0:width]
        right = image[:,(w-width):]
        windows.append(left)
        windows.append(right)
    return windows

def getWindows(image, windowSize):
    allWindows = window.generateWindows(image, windowSize)
    windows = window.filterWindows(image, allWindows)
    return createWindowsFromTrainingImage(image, windows)

def getSegments(window, windowSize, segmentSize):
    return prep.createFeatureSegments(window, windowSize, segmentSize)

def classify(segments):
    #something here
    pass


# define params
segmentSize = (30,30)
windowSize = (30*6, 30*3)

# read img. I think this should be a line right?
img = h2.getImage("/home/niek/git/HandWritingRecognition/src/dev/resources/Herodian/Alef/Alef_00.png")

# get windows for img
windows = getWindows(img, 30)
print(np.array(windows).shape)

# for each window get feature segments
segments = []
for w in windows:
    segments.append(getSegments(w, windowSize, segmentSize))

print(segments)

# create list of labels (character names) for all windows
labels = []
for segment in segments:
    labels.append(classify(segment))

# after this the windows and labels are passed to style classifier