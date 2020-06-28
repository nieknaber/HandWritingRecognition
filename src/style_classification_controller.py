
from src.style_classification import features
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from src.style_classification.helper import *
import numpy as np

# Class that can be used to classify a document into a style.
# Example:
# ...
# imgList, labels = characterClassify(images)
# classifier = Classifier("char_num_acc_lda_k3.txt", 3)
# style = classifier.classifyList(imgList, labels)
# ...
#
# More elaborate example can be found in commented code at bottom of this file
class Style_Classifier:

    def __init__(self, file, directories, k = 1):
        self.directories = [str(d+"/") for d in directories]
        self.charToNum = {}
        self.charToDat = {}
        self.k = k
        self.parseFile(file)

    # Function to load all training images for the knn classifier
    def readData(self, char):
        self.charToDat[char] = getData(char, self.directories[0], self.directories[1], self.directories[2])

    # Function to parse parameter file
    def parseFile(self, file):
        f = open(file, 'r')
        lines = f.readlines()
        for line in lines:
            words = line.split("\t")
            char = words[0]
            self.charToNum[char] = int(words[1])
            self.readData(char)

    # Function to vote on style of image.
    # Returns list of votes where every window voted once.
    # pos 0: Archaic, pos1: Hasmonean, pos2 Herodian
    def classifyList(self, images, labels):
        assert len(images) == len(labels) # each image needs a character label

        images = np.array(images)
        labels = np.array(labels)

        u = np.unique(np.array(labels))
        styles = []
        keys = self.charToNum.keys()
        for label in u:
            if (not(label in keys)): continue # if character cannot be classified
            indexes = np.where(np.array(labels) == label)
            subset = images[indexes]
            styles.extend(self.classifySingle(subset, label))

        styles = np.array(styles)
        
        results = []
        for i in range(3):
            results.append(np.sum(styles == i))

        return results # results[0] = archaic. results[1] = hasmonean. result[2] = herodian

    # Classify multiple images that are all the same character
    # Returns a list of integers that correspond to style labels
    def classifySingle(self, images, label):
        
        if (not(label in self.charToNum.keys())):
            return [-1]
        
        num = self.charToNum[label]
        
        test = features.getFeatures(preprocess(images), num)
        while (test.ndim != 2 and num != 0):
            num -= 1
            test = features.getFeatures(images, num)
        
        xTrain, yTrain = self.charToDat[label]
        train = features.getFeatures(xTrain, num)

        if (test.shape[1] == train.shape[1]):
            lda = LinearDiscriminantAnalysis()
            trainTr = lda.fit_transform(train, yTrain)
            testTr = lda.transform(test)
            
            trainTr, testTr, _ = zScoreTwoArrs(trainTr, testTr)
            
            knn = KNeighborsClassifier(self.k)
            knn.fit(trainTr, yTrain)
            predictions = knn.predict(testTr)
        else:
            predictions = [-1] * test.shape[0]

        return predictions