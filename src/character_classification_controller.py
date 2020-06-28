
from src.segmentation import selective_windowing as window
from src.segmentation import directional_rose as rose
from src.networks import training as train
from src.data_preparation import helper as h
from src.data_preparation import data_preparation as prep

import numpy as np
import sys

class Character_Classification_Controller:

    def __init__(self, segment_size, window_size, model_path, num_inputs):
        self.segment_size = segment_size
        self.window_size = window_size
        self.model_path = model_path
        self.num_inputs = num_inputs

    ### API #######################################################

    def run_classification(self, line_of_characters):
        self.line_of_characters = self.__resize_image(line_of_characters)
        windows = self.__get_windows()
        segments = self.__get_segments(windows)
        directions_per_window = self.__get_directions(segments)
        classified_data = self.__classify_data(directions_per_window)

        windows_data = []
        for window in windows:
            windows_data.append(self.__get_window_data(window))
        return (windows_data, classified_data)

    ### Private ###################################################

    def __get_windows(self):
        (_, window_width) = self.window_size
        all_windows = window.generateWindows(self.line_of_characters, window_width)
        windows = window.filterWindows(self.line_of_characters, all_windows)
        return windows

    def __resize_image(self, image):
        (height, width) = np.shape(image)
        # h.showBinaryImage(image)
        (newHeight, _) = self.window_size
        resized_image = prep.resizeImage(image, (newHeight, width))
        # h.showBinaryImage(resized_image)
        return resized_image
        
    def __get_segments(self, windows):
        all_segments = []
        for window in windows:
            window_data = self.__get_window_data(window)
            segments = prep.createFeatureSegments(window_data, self.segment_size, self.window_size)
            all_segments.append(segments)

        return all_segments

    def __get_window_data(self, window):
        (start, width) = window
        return self.line_of_characters[:,start:start+width]

    def __get_directions(self, segments):
        directions = []
        for window in segments:
            segments_data = []
            for segment in window:
                best_directions = self.__find_best_directions_for_segment(segment)
                segments_data.append(best_directions)

            directions.append(segments_data)
        return directions

    def __find_best_directions_for_segment(self, segment):
        corr = rose.calculateAutoCorrelationMatrix(segment)
        sum_of_directions = rose.summedCorrelationPerDirection(corr)
        best_directions = rose.findTopKValues(sum_of_directions)
        return best_directions

    def __classify_data(self, samples):
        dummy_model = train.Net(self.num_inputs)
        text = dummy_model.evaluate_samples_with_model(self.model_path, self.num_inputs, samples)
        return text
