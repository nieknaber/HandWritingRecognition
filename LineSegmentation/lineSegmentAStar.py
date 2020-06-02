import cv2 as cv
import numpy as np
import math
from scipy import stats, ndimage, signal


def lineSegmentAStar(image):
    proj = np.sum(image, 1)
    invproj = np.max(proj) - proj
    peaks = signal.find_peaks(invproj, prominence = 0.2*np.max(proj))
    past_peak = 0
    for peak in peaks[0]:
        if(peak-past_peak>50):
            cv.line(image, (0, peak), (2706, peak), (255, 255, 255), thickness=10)
        past_peak = peak
    cv.namedWindow("Window", flags=cv.WINDOW_NORMAL)
    cv.imshow("Window", image)
    cv.imwrite("segmented-image.png",image)
    k = cv.waitKey(0)
    print("Printing peaks:\n")
    print(peaks)
