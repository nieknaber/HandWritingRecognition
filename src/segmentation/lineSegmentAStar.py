import cv2 as cv
import numpy as np
from scipy import signal
from PIL import Image


def lineSegmentAStar(image):
    #showBinaryImage(image)
    #sys.exit()
    shape = np.shape(image)
    width = shape[1]
    proj = np.sum(image, 1)
    invproj = np.max(proj) - proj
    peaks = signal.find_peaks(invproj, prominence = 0.2*np.max(invproj)) #0.2 is TUNEABLE
    past_peak = 0
    # Initializing AStar
    # Finding the valleys (inverted peaks) that represent lines and run a* algorithm
    j = 0           # counter for lines
    points = []     # list to store edges of lines (paths)

    # compute average line height because we need to scale the parameters:
    line_counter = 1 # initialized to 1 as we skip the first peak in computation
    all_peaks = 0 # stores the cumulative height of all lines
    for peak in peaks[0]:
        if(peak-past_peak>(0.02*shape[0])):
            if past_peak != 0:
                line_counter += 1
                all_peaks += peak-past_peak
            past_peak = peak
    line_average = all_peaks / line_counter
    print("Line average:", line_average)

    past_peak = 0
    for peak in peaks[0]:
        if(peak-past_peak>(0.02*shape[0])):
            print("Running a* for peak ",peak)
            points.insert(j, astar(np.transpose(image), (0, peak), (width-1, peak), line_average))
            #points.insert(j, astar(np.transpose(image), (0, peak), (width, peak)))

        past_peak = peak
        j += 1
    images = []
    imageT = np.transpose(image)
    previous = 0 # set 0 so the first
    # for all lines
    print(len(points))
    for i in range(0,len(points)+1):
        print(i)
        newImage = np.zeros((5000, 5000)) #cropped later
        if i == len(points):
            shape = np.shape(imageT)
            for point in points[i-1]:
                newImage[point[0]][point[1]:shape[1]] = imageT[point[0]][point[1]:shape[1]]
            returnImage = newImage
        else:
            for point in points[i]:
                newImage[point[0]][0:point[1]] = imageT[point[0]][0:point[1]] #abracadabra
            returnImage = newImage - previous
        print(np.sum(returnImage))
        if(np.sum(returnImage)) < 0.01*np.sum(image): #if a line has less than 1% of all black pixels TUNEABLE
            print("continuing because line at peak ",i," is too short")
            continue
        previous = newImage
        #images.append(returnImage)
        coords = cv.findNonZero(returnImage)            # Find all non-zero points (text)
        x, y, w, h = cv.boundingRect(coords)            # Find minimum spanning bounding box
        images.append(returnImage[y:y + h, x:x + w])    # Crop the image and append to all images
    return images


# a star path planning algorithm (borrowed from stack overflow)
from heapq import *

def heuristic(a, b):
    return (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2

def astar(array, start, goal, line_height):
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
            y_distance = goal[1] - neighbor[1] #This measures diversion from straight line
            if y_distance < 0:
                tentative_g_score += abs(y_distance) * 7.5*line_height # tuneable parameter, penalizes going under straight line
                #print("Going under..")
            if y_distance > 0:
                tentative_g_score += abs(y_distance) * 0 # tuneable parameter, penalizes going over straight line
                #print("going over...")
            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:
                    if array[neighbor[0]][neighbor[1]] == 1:
                        tentative_g_score += 150*line_height # negative penalty for black pixels. TUNEABLE parameter!
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

def showBinaryImage(image):
    image = convertToRGBImage(image)
    image = Image.fromarray(image, 'RGB')
    image.show()


def convertToRGBImage(image):
    (height, width) = np.shape(image)

    data = np.zeros((height, width, 3), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            if image[y,x] == 0:
                data[y,x] = [255,255,255]
            else:
                data[y,x] = [0,0,0]

    return data

