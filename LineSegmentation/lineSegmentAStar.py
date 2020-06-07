import cv2 as cv
import numpy as np
from scipy import signal


def lineSegmentAStar(image):
    shape = np.shape(image)
    width = shape[1]
    proj = np.sum(image, 1)
    invproj = np.max(proj) - proj
    peaks = signal.find_peaks(invproj, prominence = 0.2*np.max(proj))
    past_peak = 0
    # Initializing AStar
    # Finding the valleys (inverted peaks) that represent lines and run a* algorithm
    j = 0           # counter for lines
    points = []     # list to store edges of lines (paths)
    for peak in peaks[0]:
        if(peak-past_peak>50):
            print("Running a* for peak ",peak)
            points.insert(j, astar(np.transpose(image), (0, peak), (width-501, peak)))
        past_peak = peak
        j += 1
    images = []
    imageT = np.transpose(image)
    previous = 0 # set 0 so the first
    # for all lines
    for i in range(0,len(points)):
        newImage = np.zeros((5000, 5000)) #cropped later
        for point in points[i]:
            newImage[point[0]][0:point[1]] = imageT[point[0]][0:point[1]] #abracadabra
        returnImage = newImage - previous
        previous = newImage

        coords = cv.findNonZero(returnImage)          # Find all non-zero points (text)
        x, y, w, h = cv.boundingRect(coords)        # Find minimum spanning bounding box
        returnImage = returnImage[y:y + h, x:x + w]     # Crop the image

        images.append(returnImage)
    return images


# a star path planning algorithm (borrowed from stack overflow)
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