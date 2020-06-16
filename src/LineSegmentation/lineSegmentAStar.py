import cv2 as cv
import numpy as np
from scipy import signal


def lineSegmentAStar(image):
    shape = np.shape(image)
    width = shape[1]
    proj = np.sum(image, 1)
    invproj = np.max(proj) - proj
    peaks = signal.find_peaks(proj, prominence = 0.2*np.max(proj))

    past_peak = 0
    # Initializing AStar
    # Finding the valleys (inverted peaks) that represent lines and run a* algorithm
    j = 0           # counter for lines
    points = []     # list to store edges of lines (paths)
    for peak in peaks[0]:
        peak = peak+70
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
            image[point[1]][point[0]] = 1
            newImage[point[0]][0:point[1]] = imageT[point[0]][0:point[1]] #abracadabra
        returnImage = newImage - previous
        previous = newImage

        #coords = cv.findNonZero(returnImage)          # Find all non-zero points (text)
        #x, y, w, h = cv.boundingRect(coords)        # Find minimum spanning bounding box
        #returnImage = returnImage[y:y + h, x:x + w]     # Crop the image

        images.append(returnImage)
    cv.imwrite("segmented_lines.bmp", image*255)
    return images



def heuristic(a, b):
    return (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2

def get_D(array,a):

    max_range = 500
    for dist in range(1,max_range):
        if array[a[0]][a[1]+dist] != 0 or array[a[0]][a[1]-dist] != 0:
            return 1/(1+dist)

    return 1/1000

def get_D2(array,a):
    closest_dist = 1000
    max_range = 100
    for dist in range(1,max_range):
        if array[a[0]][a[1]+dist] != 0:
            return 1/((1+dist)*(1+dist))
    return 1/100

def get_linear_D(array,a):
    max_range = 100
    for dist in range(1,max_range):
        if array[a[0]][a[1]+dist] or array[a[0]][a[1]-dist] != 0:
            return 100-dist
    return 0

def get_M(array,b):
    if array[b[0]][b[1]] == 1:
        return int(1)
    else:
        return int(0)

def get_N(array,a,b):
    change_x = abs(a[0]-b[0])
    change_y = abs(a[1]-b[1])
    if change_x+change_y == 1:
        return 10
    if change_x+change_y == 2 or change_x+change_y == 3:
        return 14
    else:
        print("get_n not working properly")
        print(change_x)
        print(change_y)
        print(a)
        print(b)

def cost(img, a, b, start_y):
    cd = 10
    D = get_linear_D(img,b)

    cd2 = 50
    D2 = get_D2(img,b)

    cm = 50
    M = get_M(img,b)

    cv = 30
    V = abs(b[1]-start_y)

    cn = 1
    N = get_N(img,a,b)

    return (cd * D + cd2 * D2 + cm * M + cv * V + cn*N)

# Credit for this: Nicholas Swift
# as found at https://medium.com/@nicholas.w.swift/easy-a-star-pathfinding-7e6689c7f7b2
from warnings import warn
import heapq


class Node:
    """
    A node class for A* Pathfinding
    """

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

    def __repr__(self):
        return f"{self.position} - g: {self.g} h: {self.h} f: {self.f}"

    # defining less than for purposes of heap queue
    def __lt__(self, other):
        return self.f < other.f

    # defining greater than for purposes of heap queue
    def __gt__(self, other):
        return self.f > other.f


def return_path(current_node):
    path = []
    current = current_node
    while current is not None:
        path.append(current.position)
        current = current.parent
    return path[::-1]  # Return reversed path


def astar(maze, start, end, allow_diagonal_movement=True):
    """
    Returns a list of tuples as a path from the given start to the given end in the given maze
    :param maze:
    :param start:
    :param end:
    :return:
    """

    # Create start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Heapify the open_list and Add the start node
    heapq.heapify(open_list)
    heapq.heappush(open_list, start_node)

    # Adding a stop condition
    outer_iterations = 0
    max_iterations = (len(maze[0]) * len(maze) // 2)

    # what squares do we search
    adjacent_squares = ((0, -1), (0, 1), (-1, 0), (1, 0),)
    if allow_diagonal_movement:
        #adjacent_squares = ((0, -1), (0, 1), (1, 0), (1, -1), (1, 1),)
        adjacent_squares = ((0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1),)

    # Loop until you find the end
    while len(open_list) > 0:
        outer_iterations += 1

        if outer_iterations > max_iterations:
            # if we hit this point return the path such as it is
            # it will not contain the destination
            warn("giving up on pathfinding too many iterations")
            return return_path(current_node)

            # Get the current node
        current_node = heapq.heappop(open_list)
        closed_list.append(current_node)

        # Found the goal
        if current_node == end_node:
            return return_path(current_node)

        # Generate children
        children = []

        for new_position in adjacent_squares:  # Adjacent squares

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (
                    len(maze[len(maze) - 1]) - 1) or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] != 0:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:
            # Child is on the closed list
            if len([closed_child for closed_child in closed_list if closed_child == child]) > 0:
                continue

            # Create the f, g, and h values
            #print("child pos:", child.position)
            print("node y pos:", current_node.position)
            child.g = current_node.g + 1 #+ cost(maze, current_node.position, child.position, start[1])
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + (
                        (child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h

            # Child is already in the open list
            if len([open_node for open_node in open_list if
                    child.position == open_node.position and child.g > open_node.g]) > 0:
                continue

            # Add the child to the open list
            heapq.heappush(open_list, child)

    warn("Couldn't get a path to destination")
    return None