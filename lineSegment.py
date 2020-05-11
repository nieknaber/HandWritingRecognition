import cv2 as cv
import numpy as np
import sys
import math
import matplotlib.pyplot as plt
from scipy import stats



filename = 'images/08.jpg'

image = cv.imread(filename)
dimensions=np.shape(image)


#cv.imshow("Image Window", image)
#k = cv.waitKey(0) #press 0 to exit



#Convert image to grayscale
grayimg = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

#invert image (experiment to see if it increases performance)
grayimg = ~grayimg

#Hough transform
lines = cv.HoughLines(grayimg, 80, np.pi/90, 5, None,0,0,1,3)


all_theta = lines[:,:,1]

mode = stats.mode(all_theta)
mode = mode[0]
print(mode)
plt.hist(all_theta, 314)
plt.show()

#Open a window
cv.namedWindow("Window", flags=cv.WINDOW_NORMAL)

#Draw lines
j = 0
if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + dimensions[1]*(-b)), int(y0 + dimensions[0]*(a)))
            pt2 = (int(x0 - dimensions[1]*(-b)), int(y0 - dimensions[0]*(a)))
            if(theta == mode):
                cv.line(grayimg, pt1, pt2, (255,0,155), 5, cv.LINE_AA)
                j = j+1



#Show image
cv.imshow("Window",image)
cv.imshow("Window",grayimg)

k = cv.waitKey(0)


