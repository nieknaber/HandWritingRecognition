
from src.character_classification_controller import Character_Classification_Controller
from src.network_controller import Network_Controller

from src.data_preparation import helper as h

def test_Network_Training():
    
    segment_dim = 30
    window_dim = (6,3)
    data_directories = [
         './src/resources/original/characters/Herodian',
         './src/resources/original/characters/Archaic',
         './src/resources/original/characters/Hasmonean'
    ]

    num_directions = 8
    epochs = 20

    net = Network_Controller(segment_dim, window_dim, data_directories, num_directions, epochs, cached = True)
    net.run_training()
    net.run_testing()

    name = "model_" + str(epochs) + "_" + str(segment_dim) + ".pt"
    net.save_network('./src/cached_data/trained_models/' + name)


# test_Network_Training()


def test_Character_Classfication():

    segment_dim = 30
    window_dim = (6,3)

    segment_size = (segment_dim, segment_dim)
    window_size = (segment_dim*window_dim[0], segment_dim*window_dim[1])
    dummy = h.getImage("./src/resources/test_data_lines/line_0.bmp")
    model_path = './src/cached_data/trained_models/model_dimension3_250_epochs.pt'

    cc = Character_Classification_Controller(segment_size, window_size, model_path)
    result = cc.run_classification(dummy)

    (windows, labels) = result

    print(windows[0])
    print(labels)
    
test_Character_Classfication()