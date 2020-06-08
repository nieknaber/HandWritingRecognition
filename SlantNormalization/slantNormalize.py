import cv2 as cv
import numpy as np
from math import atan

# deskews picture with i pixels
# returns deskewed picture
def deskew(img, i):
    
    rows, cols = img.shape
    pts1 = np.float32([[0, 0], [cols - 1,  0], [cols - 1 + i, rows - 1]])
    pts2 = np.float32([[0, 0], [cols - 1, 0], [cols - 1, rows - 1]])
    M = cv.getAffineTransform(pts1,pts2)
    dst = cv.warpAffine(img,M,(cols,rows))
    
    return dst

# crops whitespace from image
def cropWhiteSpace(image):
    coords = cv.findNonZero(newImg)
    x, y, w, h = cv2.boundingRect(coords)
    newImg = newImg[y:y+h, x:x+w] 
    
    return newImg

# deskews over different angles and calculates best based on 
# summed squared vertical density
# returns slant angle + deslanted image 
def slantNormalize(img):
    
    imgs = []
    ss = []
    for i in range(50):
        # pad whitespace to left border
        dst = cv.copyMakeBorder(img, 0, 0, 50, 0, cv.BORDER_CONSTANT)
        newImg = deskew(dst, i)
        
        # crop image
        newImg = cropWhiteSpace(newImg)
        y, x = newImg.shape

        s = 0
        for i in range(x):
            s += sum(newImg[:,i]) ** 2

        imgs.append(newImg)
        ss.append(s)

    finalS = np.argmax(ss)
    finalImg = imgs[finalS]

    y, x = finalImg.shape
    angle = atan(finalS / y)

    return angle, finalImg