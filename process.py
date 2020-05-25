import sys
from LineSegmentation.lineSegment import *


lineSegment("images/P423-1-Fg002-R-C01-R01-binarized.jpg")
for i in range(1,9):
    lineSegment("images/0" + str(i) + ".jpg") 

# just to show all segments:
# for segment in segments:
#     cv.namedWindow("Window", flags=cv.WINDOW_NORMAL)
#     cv.imshow("Window", segment)
#     k = cv.waitKey(0)