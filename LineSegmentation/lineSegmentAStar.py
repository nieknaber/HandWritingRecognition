import cv2 as cv
import numpy as np
import math
from scipy import stats, ndimage, signal


def lineSegmentAStar(image):
    shape = np.shape(image)
    width = shape[1]
    proj = np.sum(image, 1)
    invproj = np.max(proj) - proj
    peaks = signal.find_peaks(invproj, prominence = 0.2*np.max(proj))
    past_peak = 0
    # Initializing AStar
    # Finding the valleys (inverted peaks) that represent lines and run a* algorithm
    j = 0
    points = []
    for peak in peaks[0]:
        if(peak-past_peak>50 and peak < 900):
            #cv.line(image, (0, peak), (2706, peak), (255, 255, 255), thickness=10)
            print("Running a* for peak ",peak)
            points.insert(j, astar(np.transpose(image), (0, peak), (width-501, peak)))
            print("Finished!")

#            for i in range(0, len(points[j])-1):
#                mask[points[i][1]][points[i][0]] = 1
#                cv.line(image, points[i], points[i+1], (255, 0, 0), thickness=1)
                #cv.circle(image,points[i], 1, (255,0,0))

        past_peak = peak
        j += 1
    images = []
    imageT = np.transpose(image)


    for i in range(0,len(points)):
        returnimg = np.zeros((5000, 5000))
        for point in points[i]:
            print("point[0]:", point[0])
            print("point[1]:", point[1])
            returnimg[point[0]][0:point[1]] = imageT[point[0]][0:point[1]]
        coords = cv.findNonZero(returnimg)  # Find all non-zero points (text)
        x, y, w, h = cv.boundingRect(coords)  # Find minimum spanning bounding box
        returnimg = returnimg[y:y + h, x:x + w]  # Crop the image - note we do this on the original image
        images.append(returnimg)

    cv.namedWindow("Window", flags=cv.WINDOW_NORMAL)
    cv.imshow("Window",np.transpose(returnimg))
    #cv.imwrite(image,"detectedLines.png")
    cv.waitKey(0)
    return images


# a star path planning algorithm
from heapq import *


def heuristic(a, b):
    return (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2


def astar(array, start, goal):
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []

    heappush(oheap, (fscore[start], start))

    while oheap:

        current = heappop(oheap)[1]

        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:
                    if array[neighbor[0]][neighbor[1]] == 1:
                        continue
                else:
                    # array bound y walls
                    continue
            else:
                # array bound x walls
                continue

            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue

            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heappush(oheap, (fscore[neighbor], neighbor))
    print("no route found")
    return []