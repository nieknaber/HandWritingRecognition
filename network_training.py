
from src.dev.source import selective_windowing as window
from src.dev.source import directional_rose as rose
from src.dev.source import training as train
from src.dev.source import helper as h
from src.dev.source import data_preparation as prep

import numpy as np

class Network_Training():

    def __init__(self, segment_size, window_size, model_path, data_directories):
        self.segment_size = segment_size
        self.window_size = window_size
        self.model_path = model_path
        self.data_directories = data_directories

    def run_training(self):
        self.__prepare_data()
    
    def __prepare_data(self):
        data = prep.getResizedImages(self.window_size, self.data_directories[0])
        print(data)
        pass

    def __convert_data_into_directions(self):
        pass

    def __train_network(self):
        pass

    def __test_network(self):
        pass

    def __save_network(self):
        pass


def test_Network_Training():
    
    segment_dim = 30
    window_dim = (6,3)

    segment_size = (segment_dim, segment_dim)
    window_size = (segment_dim*window_dim[0], segment_dim*window_dim[1])
    model_path = './trained_models/model_16x16_k8.pt'

    data_directories = [
         './src/characters_training/Herodian',
         './src/characters_training/Archaic',
         './src/characters_training/Hasmonean'
    ]

    net = Network_Training(segment_size, window_size, model_path, data_directories)
    net.run_training()




test_Network_Training()