
from source import selective_windowing as window
from source import directional_rose as rose
from source import training as train

from source import helper as h
from source import data_preparation as prep

def testWindowPipeline():
    binaryImage = h.getBinaryDummyImage('resources/dummy.jpg')
    h.showBinaryImage(binaryImage)

    allWindows = window.generateWindows(binaryImage)
    windows = window.filterWindows(binaryImage, allWindows)

    # The following are only for showcasing!
    centers = window.generateAllCenterOfGravities(binaryImage, windows)
    imageRGB = h.convertToRGBImage(binaryImage)
    windowedImage = h.addWindows(imageRGB, windows)
    imageWithCenter = h.drawCentrePoint(windowedImage, centers)
    h.showRGBImage(windowedImage)

def testFindingBestDirections():
    corr = rose.calculateAutoCorrelationMatrix('resources/test_segments/4.png')
    rose.saveImageToFile(corr * 255, 'resources/test_segments/c4.png')

    sumForDirections = rose.summedCorrelationPerDirection(corr)
    bestDirections = rose.findTopKValues(sumForDirections)

def testDataPreparation():
    segmentSize = (16,16)
    windowSize = (16*6, 16*2) # 96,32

    prep.resizeAllImages((96,64), 'resources/herodian', 'resources/resized_herodian')

    (left, right) = prep.createWindowsFromTrainingImage('resources/resized_herodian/Alef_19.png', windowSize)
    segments = prep.createFeatureSegments(left, segmentSize)
    prep.saveSegmentsAsImages(segments, 'resources/test_segments/')

def testTraining():
    (trainset, testset) = train.get_data()
    net = train.Net()
    train.train_network(net, trainset)
    train.test_network(net, testset)

testTraining()