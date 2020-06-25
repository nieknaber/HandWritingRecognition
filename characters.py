
from src.dev.source import selective_windowing as window
from src.dev.source import directional_rose as rose
from src.dev.source import training as train
from src.dev.source import helper as h
from src.dev.source import data_preparation as prep
from src.StyleClassification import helper as h2

def getWindows(image, windowSize):
    allWindows = window.generateWindows(image, windowSize)
    return window.filterWindows(image, allWindows)


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
windows = getWindows(img)
print(windows)

# for each window get feature segments
segments = []
for w in windows:
    segments.append(getSegments(w))

# create list of labels (character names) for all windows
labels = []
for segment in segments:
    labels.append(classify(segment))

# after this the windows and labels are passed to style classifier