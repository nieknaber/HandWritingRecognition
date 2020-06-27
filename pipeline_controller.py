from src.character_classification_controller import Character_Classification_Controller
from src.network_controller import Network_Controller
from src.style_classification_controller import Style_Classifier
from src.line_segmentation_controller import LineSegmentationController

from src.data_preparation import helper as h

import json, sys, os
import numpy as np
from PIL import Image
import cv2 as cv2

class Pipeline_Controller:

    def __init__(self, segment_dim, window_dim, num_directions, data_directories):
        self.segment_dim = segment_dim
        self.window_dim = window_dim
        self.num_directions = num_directions
        self.data_directories = data_directories

        self.cached_lines = './src/cached_data/lines/'
        self.trained_model_directory = './src/cached_data/trained_models/'
        self.cached_characters = './src/cached_data/classified_characters/'

    def network_training(self, epochs):
        
        net = Network_Controller(self.segment_dim, self.window_dim, self.data_directories, self.num_directions, epochs, cached = True)
        net.run_training()
        net.run_testing()

        name = "model_" + str(epochs) + "_" + str(self.segment_dim) + ".pt"
        net.save_network(self.trained_model_directory + name)

    def line_segmentation(self, images):

        for index, image in enumerate(images):
            image = h.getImage(image)

            negative_y_penalty = 7.5
            positive_y_penalty = 0
            passthrough_black_penalty = 150
            peak_prominence = 0.2
            min_line_size = 0.01

            segmenter = LineSegmentationController(negative_y_penalty,positive_y_penalty,passthrough_black_penalty,peak_prominence,min_line_size)
            images = segmenter.segment_lines(image)

            part_counter = 0
            for image in images:
                json.dump(image.astype(int).tolist(), open(self.cached_lines + str(index)+'-'+str(part_counter)+'.json', 'w'))
                part_counter += 1

        print("Line segmentation done, images are saved.")

    def character_classfication(self):

        all_lines = []
        files = os.listdir(self.cached_lines)
        for lines in files:
            if not lines.startswith('.'):
                all_lines.append(lines)

        results = []
        for index, line in enumerate(all_lines):
            line = np.array(json.load(open(self.cached_lines + line, 'r'))).astype(np.uint8)

            num_inputs = self.num_directions * self.window_dim[0] * self.window_dim[1]
            segment_size = (self.segment_dim, self.segment_dim)
            window_size = (self.segment_dim * self.window_dim[0], self.segment_dim * self.window_dim[1])
            model_path = self.trained_model_directory + 'model_200_144.pt'

            cc = Character_Classification_Controller(segment_size, window_size, model_path, num_inputs)
            result = cc.run_classification(line)

            (windows, labels) = result
            results.append(labels)

            json.dump(([window.tolist() for window in windows],labels), open(self.cached_characters + str(index) + '-characters.json', 'w'))

        print(results)
        print("Character classification done, results are saved.")

    def style_classification(self):
        
        all_data = []
        files = os.listdir(self.cached_characters)
        for data in files:
            if not data.startswith('.'):

                result = json.load(open(self.cached_characters + data, 'r'))

                (windows, labels) = result

                windows = np.array(windows)

                newImgs = [img.astype(np.uint8) for img in windows]
                capitalized_labels = [l.capitalize() for l in labels]

                styleClassifier = Style_Classifier("./src/cached_data/knn/char_num_acc_lda_k3.txt", self.data_directories, 3)
                style = styleClassifier.classifyList(newImgs, capitalized_labels)

                print(style)

        print("Style classification done!")
