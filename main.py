
from src.character_classification_controller import Character_Classification_Controller
from src.network_controller import Network_Controller
from src.style_classification_controller import Style_Classifier

from src.data_preparation import helper as h

import json
import numpy as np

### Global Variable ################################################

segment_dim = 30
window_dim = (6,3)
num_directions = 8

data_directories = [
         './src/resources/original/characters/Archaic',
         './src/resources/original/characters/Hasmonean',
         './src/resources/original/characters/Herodian'
    ]

def test_Network_Training():
    
    epochs = 20

    net = Network_Controller(segment_dim, window_dim, data_directories, num_directions, epochs, cached = True)
    net.run_training()
    net.run_testing()

    name = "model_" + str(epochs) + "_" + str(segment_dim) + ".pt"
    net.save_network('./src/cached_data/trained_models/' + name)

def test_Character_Classfication():

    ### INPUT
    dummy = h.getImage("./src/resources/test_data_lines/line_0.bmp")

    num_inputs = num_directions * window_dim[0] * window_dim[1]
    segment_size = (segment_dim, segment_dim)
    window_size = (segment_dim*window_dim[0], segment_dim*window_dim[1])
    model_path = './src/cached_data/trained_models/model_200_144.pt'

    cc = Character_Classification_Controller(segment_size, window_size, model_path, num_inputs)
    result = cc.run_classification(dummy)

    ### OUTPUT
    (windows, labels) = result

    ### SAVING
    json.dump([window.tolist() for window in windows], open('./src/cached_data/classified_characters/test_windows.json', 'w'))
    json.dump(labels, open('./src/cached_data/classified_characters/test_labels.json', 'w'))

def test_style_classification():

    ### INPUT
    windows = np.array(json.load(open('./src/cached_data/classified_characters/test_windows.json', 'r')))
    labels = json.load(open('./src/cached_data/classified_characters/test_labels.json', 'r'))

    newImgs = [img.astype(np.uint8) for img in windows]
    capitalized_labels = [l.capitalize() for l in labels]
    # newLabels = [label.capitalize() for label in labels]
    # newImgs = [img.astype(np.uint8) for img in imgs]
    # print("doint style classification")
    styleClassifier = Style_Classifier("./src/cached_data/knn/char_num_acc_lda_k3.txt", data_directories, 3)
    style = styleClassifier.classifyList(newImgs, capitalized_labels)
    print(style)
    print(labels[1])

test_style_classification()