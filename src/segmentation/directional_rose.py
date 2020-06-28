
import numpy as np
from PIL import Image

#####################################################################
## Helper for file access

def saveImageToFile(image, filename):
    im = Image.fromarray(image)
    im = im.convert('L')
    im.save(filename)

def openImageFromLocation(filename):
    image = Image.open(filename)
    image = np.array(image)
    return image

#####################################################################
## Correlation -> Directions -> Best directions

def calculateAutoCorrelationMatrix(image):
    
    # Calculate the autocorrelation via FFT stuff
    fftImage = np.fft.fft2(image)
    step = fftImage * fftImage.conjugate()
    step = np.fft.ifft2(step).real
    step = np.fft.fftshift(step)

    # Normalize it because of huge values
    maximum = np.max(step)
    minimum = np.min(step)
    denom = maximum - minimum
    if denom == 0:
        denom = 1
    normalized = step.copy()
    (h, w) = np.shape(image)
    for i in range(h):
        for j in range(w):
            normalized[i,j] = ((step[i,j]-minimum)/denom)

    return normalized

# Calculates the summed correlation values for each direction in 180 degrees
# f is the sampling rate of the pixels
def summedCorrelationPerDirection(corr, verbose = False):

    (h, w) = np.shape(corr)
    dim = h

    allSums = []
    for angle in range(1,181):

        values = []
        lut = np.zeros(shape=(dim,dim))

        rad = angle * np.pi / 180
        slope = np.cos(rad)/np.sin(rad)

        for y in range(0,dim):
            for x in range(0,dim):
                xOff = x - dim/2
                yOff = y - dim/2
                lineY = slope * xOff
                lineX = yOff / slope
                
                found = False
                if not (lineY < yOff or lineY > yOff+1):
                    found = True
                
                if not (lineX < xOff or lineX > xOff+1):
                    found = True

                if found and lut[y, x] == 0:
                    values.append(corr[y,x])
                    lut[y,x] = 1
        
        denom = np.max(corr)-np.min(corr)
        if denom == 0:
            denom = 1

        allSums.append((np.sum(values)-np.min(corr))/denom)

        if verbose:
            print("angle: " + str(angle) + " done!")

    return allSums

# Finding the indices of all top K values
def findTopKValues(array, k = 8):
    sortedArray = array.copy()
    sortedArray.sort()
    sortedArray = sortedArray[::-1]
    topK = sortedArray[:k]

    maxIndices = []
    for i in topK:
        for index, item in enumerate(array):
            if item == i:
                maxIndices.append(int(index))
                maxIndices = list(set(maxIndices))
    
    maxIndices = maxIndices[:k]
    maxIndices.sort()
    return maxIndices