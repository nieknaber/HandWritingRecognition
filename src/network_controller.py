
from src.segmentation import selective_windowing as window
from src.segmentation import directional_rose as rose
from src.networks import training as network
from src.data_preparation import data_preparation as prep

import numpy as np
import torch
import random
import json

class Network_Controller():

    def __init__(self, segment_dim, window_dim, data_directories, num_directions, epochs, cached, verbose = True):
        self.segment_dim = segment_dim
        self.window_dim = window_dim

        self.segment_size = (segment_dim, segment_dim)
        self.window_size = (segment_dim*window_dim[0], segment_dim*window_dim[1])
        self.data_directories = data_directories
        self.verbose = verbose

        self.data_path = './src/cached_data/converted_images/data.json'
        self.num_directions = num_directions
        self.epochs = epochs
        self.cached = cached
        self.num_inputs = self.num_directions * self.window_dim[0] * self.window_dim[1]

    def run_training(self):
        data = self.__prepare_data(self.cached)
        (train_data, test_data) = self.__split_data(data)

        self.net = network.Net(self.num_inputs)
        self.net.train_with_data(train_data, self.epochs)

        self.test_data = test_data

    def run_testing(self):
        self.net.test_with_data(self.test_data)

    def save_network(self, toPath):
        torch.save(self.net.state_dict(), toPath)
        if self.verbose: print("Model saved.")

    ### Private methods ################################################
    
    def __prepare_data(self, load_from_file):

        if load_from_file:
            with open(self.data_path, 'r') as fp:
                data = json.load(fp)
                if self.verbose: print("Data was loaded from disk.")
                return data

        data = {}
        for directory in self.data_directories:
            new_data = prep.getResizedImages(self.window_size, directory)
            for key in new_data.keys():
                key = key.lower()
                if key in data:
                    previous_list = data[key]
                    previous_list.extend(new_data[key])
                    data[key] = previous_list
                else:
                    data[key] = new_data[key]

        if self.verbose: print("Data is loaded.")

        num_characters = len(data.keys())
        for key_index, key in enumerate(data.keys()):
            converted_windows = []
            for index, window in enumerate(data[key]):
                num_windows = len(data[key])
                segments = prep.createFeatureSegments(window, self.segment_size, self.window_size)

                lined_up_directions = []
                for segment in segments:
                    corr = rose.calculateAutoCorrelationMatrix(segment)
                    sum_for_directions = rose.summedCorrelationPerDirection(corr)
                    best_directions = rose.findTopKValues(sum_for_directions, self.num_directions)
                    lined_up_directions.extend(best_directions)
                
                converted_windows.append(lined_up_directions)

                if self.verbose: print(str(index+1) + "/" + str(num_windows) + " windows for " + str(key) + " " + str(key_index+1) + "/" + str(num_characters) + " done!")
            
            data[key] = converted_windows

        if not load_from_file:
            with open(self.data_path, 'w') as fp:
                json.dump(data, fp)
                if self.verbose: print("Data has been saved to disk.")
        
        return data

    def __split_data(self, dict_data, split = 0.9):
        data = []
        for key in dict_data.keys():
            for window in dict_data[key]:
                data.extend([(window, key)])

        random.shuffle(data)
        length = len(data)
        trainset = data[:int(split*length)]
        testset = data[int(split*length):]
        return (trainset, testset)    