
from src.dev.source import selective_windowing as window
from src.dev.source import directional_rose as rose
from src.dev.source import training as train
from src.dev.source import helper as h
from src.dev.source import data_preparation as prep
from src.StyleClassification import helper as h2
import numpy as np

class Character_Classification:

    def __init__(self, segment_size, window_size, model_path):
        self.segment_size = segment_size
        self.window_size = window_size
        self.model_path = model_path

    ### API #######################################################

    def run_classification(self, line_of_characters):
        self.line_of_characters = line_of_characters
        windows = self.__get_windows()
        segments = self.__get_segments(windows)
        directions_per_window = self.__get_directions(segments)
        classified_data = self.__classify_data(directions_per_window)

    ### Private ###################################################

    def __get_windows(self):
        (_, window_width) = self.window_size
        all_windows = window.generateWindows(self.line_of_characters, window_width)
        windows = window.filterWindows(self.line_of_characters, all_windows)
        return windows
        
    def __get_segments(self, windows):
        all_segments = []
        for window in windows:
            window_data = self.__get_window_data(window)
            segments = prep.createFeatureSegments(window_data, self.segment_size, self.window_size)
            all_segments.append(segments)

        return all_segments

    def __get_window_data(self, window):
        (start, width) = self.window_size
        window = self.line_of_characters[:,start:start+width]
        return window

    def __get_directions(self, segments):
        directions = []
        for window in segments:
            segments = []
            for segment in window:
                segments.append(self.__find_best_directions_for_segment(segment))
            directions.append(segments)
        return directions

    def __find_best_directions_for_segment(self, segment):
        corr = rose.calculateAutoCorrelationMatrix(segment)
        sum_of_directions = rose.summedCorrelationPerDirection(corr)
        best_directions = rose.findTopKValues(sum_of_directions)
        return best_directions

    def __classify_data(self, directions):
        train.evaluate_directions_with_model(self.model_path, directions)


def test_Character_Classfication():

    segment_size = (30,30)
    window_size = (30*6, 30*3)
    dummy = h2.getImage("./src/dev/resources/dummy.jpg")
    model_path = './trained_models/model_dimension3_250_epochs.pt'

    cc = Character_Classification(segment_size, window_size, model_path)
    cc.run_classification(dummy)
    

test_Character_Classfication()








# def createWindowsFromTrainingImage(image, windowParams):

#     (h,w) = np.shape(image)
#     windows = []
#     for windowParam in windowParams: 
#         (height, width) = windowParam
#         left = image[:,0:width]
#         right = image[:,(w-width):]
#         windows.append(left)
#         windows.append(right)
#     return windows



# def getSegments(window, windowSize, segmentSize):
#     return prep.createFeatureSegments(window, windowSize, segmentSize)

# def classify(segments):
#     #something here
#     pass


# # define params


# # read img. I think this should be a line right?
# img = h2.getImage("/home/niek/git/HandWritingRecognition/src/dev/resources/dummy.jpg")

# # get windows for img
# windows = getWindows(img, 30)
# print(np.array(windows).shape)

# # for each window get feature segments
# segments = []
# for w in windows:
#     segments.append(getSegments(w, segmentSize, windowSize))

# print(segments)

# # create list of labels (character names) for all windows
# labels = []
# for segment in segments:
#     labels.append(classify(segment))

# after this the windows and labels are passed to style classifier

