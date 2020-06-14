
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

    prep.resizeAllImages((96,64-16), 'resources/herodian', 'resources/resized_herodian')

    # (left, right) = prep.createWindowsFromTrainingImage('resources/resized_herodian/Alef_19.png', windowSize)
    # segments = prep.createFeatureSegments(left, segmentSize)
    # prep.saveSegmentsAsImages(segments, 'resources/test_segments/')

def testConvertResizedSegmentsIntoDirections():
    
    data = prep.covertResizedSegmentsIntoDirections('resources/resized_herodian')

    trainData = []

    length = len(data)
    for i, point in enumerate(data):
        (allSegments, label) = point

        linedUpDirections = []
        for segment in allSegments:
            corr = rose.calculateAutoCorrelationMatrix(segment)
            sumForDirections = rose.summedCorrelationPerDirection(corr)
            bestDirections = rose.findTopKValues(sumForDirections)

            linedUpDirections.append(bestDirections)
        
        prep.saveDirectionsToFile(linedUpDirections, "resources/converted_directions/" + str(label) + "_" + str(i) + ".csv")
        
        print(str(i) + "/" + str(length))

def testTraining():
    data = prep.loadAllData('resources/converted_directions')
    (trainset, testset) = train.splitData(data)

    net = train.Net()

    train.train_network(net, trainset)
    train.test_network(net, testset)
        
testTraining()