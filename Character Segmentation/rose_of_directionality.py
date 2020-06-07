
import numpy as np
from PIL import Image
import cv2
from scipy.cluster.vq import kmeans

# def binarizeImage(image):
#     (height, width) = np.shape(image)
#     for y in range(height):
#         for x in range(width):
#             if image[y,x] > 127:
#                 image[y,x] = 255
#             else:
#                 image[y,x] = 0
#     return image

def calculateAutoCorrelationMatrix(image):
    fftImage = np.fft.fft2(image)
    conjImage = fftImage.conj()
    step = np.matmul(fftImage, conjImage)
    step = np.fft.ifft2(step)
    step = np.abs(step)
    corr = np.fft.fftshift(step)
    return corr

def summedCorrelationPerDirection(corr):

    f = 10
    dim = 16

    allSums = []

    for angle in range(1,180):
        values = []
        lut = np.zeros(shape=(dim,dim))

        rad = angle * np.pi / 180
        slope = np.cos(rad)/np.sin(rad)

        for x in range(1,dim*f):
            lineY = slope * (x-(dim*f/2))

            for y in range(1,dim*f):
                offsetY = y - (dim*f/2)
                offsetX = x - (dim*f/2)

                if lineY >= (y-(dim*f/2)) and lineY <= (y-(dim*f/2)+1):
                    foundX = int(np.ceil(x/f))
                    foundY = int(np.ceil(y/f))
                    
                    if lut[foundY-1, foundX-1] == 0:
                        values.append(corr[foundY-1,foundX-1])
                        lut[foundY-1,foundX-1] = 1

        allSums.append((np.sum(values)-np.min(corr))/(np.max(corr)-np.min(corr)))
        print("angle: " + str(angle) + " done!")

    return allSums

def findTopKValues(array, k = 4):
    sortedArray = array.copy()
    sortedArray.sort()
    sortedArray = sortedArray[::-1]
    topK = sortedArray[:k]

    maxIndices = []
    for i in topK:
        for index, item in enumerate(array):
            print(i, index, item)
            if item == i:
                maxIndices.append(index)
                
    maxIndices = list(set(maxIndices))
    maxIndices.sort()
    return maxIndices

def findBestDirection(allDirections):
    directions = 16
    topK = findTopKValues(allDirections, directions)
    print(topK)

image = Image.open('./Test Segments/4.png')
image = np.array(image)
# image = binarizeImage(image)

corr = calculateAutoCorrelationMatrix(image)
corrSum = summedCorrelationPerDirection(corr)

findBestDirection(corrSum)