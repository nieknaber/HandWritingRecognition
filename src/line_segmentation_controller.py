import cv2 as cv
import numpy as np
from scipy import signal
from heapq import *


class LineSegmentationController:

    def __init__(self, negative_y_penalty=7.5, positive_y_penalty=0, passthrough_black_penalty=150, peak_prominence=0.2, min_line_size=0.01):
        self.negative_y_penalty = negative_y_penalty # penalizes going under the ideal (straight) line
        self.passthrough_black_penalty = passthrough_black_penalty # penalizes going through black pixels
        self.positive_y_penalty = positive_y_penalty # penalizes going over the ideal (straight) line
        self.peak_prominence = peak_prominence
        self.min_line_size = min_line_size



    def segment_lines(self, image):
        # Find the peaks of the projection profiles and return their indices (vertically)
        peaks = self.find_peaks(image)

        # Compute the average line height to scale the parameters off of later
        self.avg_line_height = self.get_average_line_height(peaks, image.shape[0])

        # Run the a* pathfinding through the images
        points = self.run_astar(image, peaks)

        # Return the images segmented by the paths
        return self.generate_images(points, image)

    ### PRIVATE ###
    def heuristic(self, a, b):
        return (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2

    def astar(self, array, start, goal, line_height):
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

        close_set = set()
        came_from = {}
        gscore = {start: 0}
        fscore = {start: self.heuristic(start, goal)}
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
                tentative_g_score = gscore[current] + self.heuristic(current, neighbor)
                y_distance = goal[1] - neighbor[1]  # This measures diversion from straight line
                if y_distance < 0:
                    tentative_g_score += abs(
                        y_distance) * self.negative_y_penalty * line_height
                    # print("Going under..")
                if y_distance > 0:
                    tentative_g_score += abs(y_distance) * self.positive_y_penalty
                    # print("going over...")
                if 0 <= neighbor[0] < array.shape[0]:
                    if 0 <= neighbor[1] < array.shape[1]:
                        if array[neighbor[0]][neighbor[1]] == 1:
                            tentative_g_score += self.passthrough_black_penalty * line_height
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
                    fscore[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    heappush(oheap, (fscore[neighbor], neighbor))
        print("no route found")
        return []

    def find_peaks(self, image):
        proj = np.sum(image, 1)
        invproj = np.max(proj) - proj

        return signal.find_peaks(invproj, prominence = self.peak_prominence * np.max(invproj))

    def get_average_line_height(self, peaks, height):
        line_counter = 1  # initialized to 1 as we skip the first peak in computation
        past_peak = 0
        all_peaks = 0  # stores the cumulative height of all lines
        for peak in peaks[0]:
            if (peak - past_peak > 0.02 * height):
                if past_peak != 0:
                    line_counter += 1
                    all_peaks += peak - past_peak
                past_peak = peak
        return all_peaks / line_counter

    def run_astar(self, image, peaks):
        j = 0  # counter for lines
        points = []  # list to store edges of lines (paths)
        past_peak = 0
        for peak in peaks[0]:
            if peak - past_peak > 0.02 * image.shape[0]:
                print("Running a* for peak ", peak)
                points.insert(j, self.astar(np.transpose(image), (0, peak), (image.shape[1] - 1, peak), self.avg_line_height))
            past_peak = peak
            j += 1
        return points

    def generate_images(self, points, image):
        images = []
        imageT = np.transpose(image)  # I forgot why we transpose but everything breaks if we don't
        previous = 0  # Initialize variable that stores the previously processed line
        for i in range(0, len(points) + 1):
            new_image = np.zeros((image.shape[1], image.shape[0]))  # cropped later
            print(new_image.shape)
            if i == len(points):
                shape = np.shape(image)
                for point in points[i - 1]:
                    new_image[point[0]][point[1]:shape[1]] = imageT[point[0]][point[1]:shape[1]]
                return_image = new_image
            else:
                for point in points[i]:
                    new_image[point[0]][0:point[1]] = imageT[point[0]][0:point[1]]  # abracadabra
                return_image = new_image - previous
            if (np.sum(return_image)) < self.min_line_size * np.sum(image):  # if a line has less than 1% of all black pixels
                print("continuing because line at peak ", i, " is too short")
                continue
            previous = new_image
            coords = cv.findNonZero(return_image)  # Find all non-zero points (text)
            x, y, w, h = cv.boundingRect(coords)  # Find minimum spanning bounding box
            images.append(np.transpose(return_image[y:y + h, x:x + w]))  # Crop the image and append to all images
        return images
