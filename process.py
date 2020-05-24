import sys
from LineSegmentation.lineSegment import *

(theta, rhos, segments) = lineSegment("images/08.jpg") 

# just to show all segments:
for segment in segments:
    cv.namedWindow("Window", flags=cv.WINDOW_NORMAL)
    cv.imshow("Window", segment)
    k = cv.waitKey(0)