import sys
from LineSegmentation.lineSegment import *


for i in range(1,9):

    (theta, linePoints, grayImg) = lineSegment("images/0" + str(i) + ".jpg") 
    # theta can be stored so that we can use it later as a feature
    # linePoints contain pairs of two (x,y) points. 
    # A* algorithm can be used to separate lines by using the first point as a start, second as end.
    # grayImg is the binary gray image where 0 = white, 255 = black
