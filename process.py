import sys
from LineSegmentation.lineSegment import *


for i in range(1,9):
    (theta, linePoints, grayImg) = lineSegment("images/0" + str(i) + ".jpg") 

# just to show all segments:
# for segment in segments:
#     cv.namedWindow("Window", flags=cv.WINDOW_NORMAL)
#     cv.imshow("Window", segment)
#     k = cv.waitKey(0)