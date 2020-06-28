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
        self.training_data_cache = './src/cached_data/converted_images_for_training/data.json'
        self.results_directory = './results/'

        self.character_for_name = {
            "alef":"א", "ayin":"ע", "bet":"ב", "dalet":"ד", "gimel":"ג", "he":"ה", "het":"ח", "kaf":"כ", "kaf-final":"ך", "lamed":"ל", "mem":"מ", "mem-medial":"ם", "nun-final":"ן", "nun-medial":"נ", "pe":"פ", "pe-final":"ף", "qof":"ק", "resh":"ר", "samekh":"ס", "shin":"ש", "taw":"ת", "tet":"ט", "tsadi-final":"ץ", "tsadi-medial":"צ", "waw":"ו", "yod":"י", "zayin":"ז"
        }

    def network_training(self, epochs):
        
        net = Network_Controller(
            segment_dim = self.segment_dim, 
            window_dim = self.window_dim, 
            data_directories = self.data_directories, 
            num_directions = self.num_directions, 
            epochs = epochs, 
            cached = False, 
            cache_path = self.training_data_cache,
            verbose = True
        )

        net.run_training()
        net.run_testing()

        name = "model_" + str(epochs) + "_" + str(self.segment_dim) + ".pt"
        net.save_network(self.trained_model_directory + name)

    def line_segmentation(self, images):

        for index, image in enumerate(images):

            name = image.split("/")[-1].split(".")[0]
            image = h.getImage(image)

            segmenter = LineSegmentationController(
                negative_y_penalty = 7.5, 
                positive_y_penalty = 0, 
                passthrough_black_penalty = 150, 
                peak_prominence = 0.2, 
                min_line_size = 0.01
            )
            
            images = segmenter.segment_lines(image)

            part_counter = 1
            for image in images:
                if part_counter < 10:
                    counter_text = "0" + str(part_counter)
                else:
                    counter_text = str(part_counter)
                filename = self.cached_lines + name + "-" + counter_text +'.json'
                json.dump(image.astype(int).tolist(), open(filename, 'w'))
                part_counter += 1

        print("Line segmentation done, images are saved.")

    def character_classfication(self):

        all_lines = []
        files = os.listdir(self.cached_lines)
        for lines in files:
            if not lines.startswith('.'):
                all_lines.append(lines)
        all_lines.sort()

        results = []
        for line in all_lines:

            name = line.split(".")[0]
            line = np.array(json.load(open(self.cached_lines + line, 'r'))).astype(np.uint8)

            num_inputs = self.num_directions * self.window_dim[0] * self.window_dim[1]
            segment_size = (self.segment_dim, self.segment_dim)
            window_size = (self.segment_dim * self.window_dim[0], self.segment_dim * self.window_dim[1])
            model_path = self.trained_model_directory + 'model_200_144.pt'

            cc = Character_Classification_Controller(segment_size, window_size, model_path, num_inputs)
            result = cc.run_classification(line)

            (windows, labels) = result
            results.append(labels)

            original_image = name.split("-")[0]
            filename = original_image + "_characters.txt"
            text = ""
            for l in labels:
                text += self.character_for_name[l]
            self.append_output_to_file(filename, text)

            json.dump(([window.tolist() for window in windows],labels), open(self.cached_characters + name + '.json', 'w'))

            print("Line " + str(name) + " has been analyized.")

        print(results)
        print("Character classification done, results are saved.")

    def style_classification(self):
        
        all_data = []
        files = os.listdir(self.cached_characters)
        files.sort()
        files = [ f for f in files if not f.startswith('.')]
        
        # styles_of_images = {}

        for data in files:

            result = json.load(open(self.cached_characters + data, 'r'))
            (windows, labels) = result
            
            windows = np.array(windows)
            newImgs = [img.astype(np.uint8) for img in windows]
            capitalized_labels = [l.capitalize() for l in labels]

            styleClassifier = Style_Classifier("./src/cached_data/knn/char_num_acc_lda_k3.txt", self.data_directories, 3)
            style = styleClassifier.classifyList(newImgs, capitalized_labels)

            # if original_image in styles_of_images:
            #     previous_styles = styles_of_images[original_image]
            #     previous_styles.append(style)
            #     styles_of_images[original_image] = previous_styles
            # else:
            #     styles_of_images[original_image] = [style]

        # print(styles_of_images)

        # for key in styles_of_images.keys():
        #     vote_dict = {}
        #     voted = styles_of_images[key]
        #     unique = list(set(voted))
        #     for u in unique:
        #         vote_dict[u] = 0
        #     for v in voted:
        #         vote_dict[v] += 1

        #     print(vote_dict)

        #     max = 0
        #     current_best = ""
        #     for vote_key in vote_dict.keys():
        #         if vote_dict[vote_key] > max:    
        #             max = vote_dict[vote_key]
        #             current_best = vote_key
                
        #     styles_of_images[key] = current_best

        print("Style classification done!")
    
    def append_output_to_file(self, filename, content):

        if not os.path.exists(self.results_directory):
            os.makedirs(self.results_directory)

        path = self.results_directory + filename

        if os.path.isfile(path):
            with open(path, 'a') as f:
                f.write("\n" + content)
        else:
            with open(path, 'w') as f:
                f.write(content)


        