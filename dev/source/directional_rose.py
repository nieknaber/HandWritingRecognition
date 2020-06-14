
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
    # image = openImageFromLocation(image_filename)

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
def summedCorrelationPerDirection(corr, f = 5, verbose = False):

    (h, w) = np.shape(corr)

    dim = 0
    if h == w:
        dim = h

    allSums = []

    for angle in range(1,181):
        values = []
        lut = np.zeros(shape=(dim,dim))
        rad = angle * np.pi / 180
        slope = np.cos(rad)/np.sin(rad)

        for x in range(1,dim*f):
            lineY = slope * (x-(dim*f/2))

            for y in range(1,dim*f):
                offsetY = y - (dim*f/2)

                if lineY >= offsetY and lineY <= offsetY+1:
                    foundX = int(np.ceil(x/f))
                    foundY = int(np.ceil(y/f))
                    
                    if lut[foundY-1, foundX-1] == 0:
                        values.append(corr[foundY-1,foundX-1])
                        lut[foundY-1,foundX-1] = 1
                        break

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