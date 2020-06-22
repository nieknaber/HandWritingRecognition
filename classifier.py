from src.StyleClassification.helper import getData, zScoreTwoArrs
from src.StyleClassification import features
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from src.StyleClassification.helper import *
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
class Classifier:

    def __init__(self, file, k = 1):
        self.charToNum = {}
        self.charToDat = {}
        self.k = k
        self.parseFile(file)

    # Function to load all training images for the knn classifier
    def readData(self, char):
        self.charToDat[char] = getData(char, "characters_training/Archaic/", "characters_training/Hasmonean/", "characters_training/Herodian/")

    # Function to parse parameter file
    def parseFile(self, file):
        f = open(file, 'r')
        lines = f.readlines()
        for line in lines:
            words = line.split("\t")
            char = words[0]
            self.charToNum[char] = int(words[1])
            self.readData(char)

    # Function to classify all images with ther character lables.
    # Returns string of the style that most characters classify to.
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
        a = np.sum(styles == 0)
        ha = np.sum(styles == 1)
        he = np.sum(styles == 2)

        if (a >= ha) :
            if (a >= he):
                return "Archaic"
        else:
            if (ha >= he):
                return "Hasmonian"
        
        return "Herodian"

    # Classify multiple images that are all the same character
    # Returns a list of integers that correspond to style labels
    def classifySingle(self, images, label):
        
        if (not(label in self.charToNum.keys())):
            return [-1]
        
        num = self.charToNum[label]

        test = features.getFeatures(preprocess(images), num)
        while (test.ndim != 2 and num != 1):
            num -= 1
            test = features.getFeatures(images, num)

        if (num == 0):
            return [-1]
        
        xTrain, yTrain = self.charToDat[label]
        train = features.getFeatures(xTrain, num)

        lda = LinearDiscriminantAnalysis()
        trainTr = lda.fit_transform(train, yTrain)
        testTr = lda.transform(test)
        
        trainTr, testTr, _ = zScoreTwoArrs(trainTr, testTr)
        
        knn = KNeighborsClassifier(self.k)
        knn.fit(trainTr, yTrain)
        predictions = knn.predict(testTr)

        return predictions

# Example to test classifier:
# classifier = Classifier("char_num_acc_lda_k3.txt", 3)

# characters = [getImage("./characters_training/Archaic/Alef/Alef_00.jpg"), 
# getImage("./characters_training/Archaic/Alef/Alef_04.jpg"),
# getImage("./characters_training/Archaic/Het/Het_00.jpg"), 
# getImage("./characters_training/Archaic/Het/Het_01.jpg"), 
# getImage("./characters_training/Archaic/Gimel/Gimel_00.jpg"),
# getImage("characters_training/Archaic/Kaf/Kaf_00.jpg"),
# getImage("characters_training/Archaic/Kaf/Kaf_02.jpg"),
# getImage("characters_training/Archaic/Mem/Mem_04.jpg"),
# getImage("characters_training/Archaic/Mem/Mem_00.jpg"),
# getImage("characters_training/Archaic/Mem/Mem_02.jpg"),
# getImage("characters_training/Archaic/Qof/Qof_00.jpg"),
# getImage("characters_training/Archaic/Qof/Qof_01.jpg"), 
# getImage("characters_training/Archaic/Pe/Pe_00.jpg")]

# labels = ["Alef"] * 2 + ["Het"] * 2 + ["Gimel"] + ["Kaf"] * 2 + ["Mem"] * 3 + ["Qof"] * 2 + ["Pe"]
# print(classifier.classifyList(characters, labels))
