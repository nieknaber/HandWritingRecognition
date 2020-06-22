from src.StyleClassification.helper import getData, zScoreTwoArrs
from src.StyleClassification import features
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from src.StyleClassification.helper import *
import numpy as np

class Classifier:

    def __init__(self, file, k = 3):
        self.charToNum = {}
        self.charToDat = {}
        self.k = k
        self.parseFile(file)

    def readData(self, char):
        self.charToDat[char] = getData(char, "characters_training/Archaic/", "characters_training/Hasmonean/", "characters_training/Herodian/")

    def parseFile(self, file):
        f = open(file, 'r')
        lines = f.readlines()
        for line in lines:
            words = line.split("\t")
            char = words[0]
            self.charToNum[char] = int(words[1])
            self.readData(char)

    def classifyList(self, images, labels):
        assert len(images) == len(labels) # each image needs a character label

        votes = np.zeros(3)
        for (img, label) in zip(images, labels):
            style = self.classifySingle(img, label)
            s = style[0]
            if (s != -1): votes[style[0]] += 1
            print(style[0])

        print(votes)

    def classifySingle(self, image, label):
        
        if (not(label in self.charToNum.keys())):
            return -1
        
        num = self.charToNum[label]

        test = features.getFeatures(preprocess([image]), num)
        while (test.ndim != 2 and num != 1):
            num -= 1
            test = features.getFeatures([image], num)

        if (num == 0):
            return -1
        
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


classifier = Classifier("char_num_acc_lda_k3.txt")

characters = [getImage("./characters_training/Archaic/Alef/Alef_00.jpg"), 
getImage("./characters_training/Archaic/Alef/Alef_04.jpg"),
getImage("./characters_training/Archaic/Het/Het_00.jpg"), 
getImage("./characters_training/Archaic/Het/Het_01.jpg"), 
getImage("./characters_training/Archaic/Gimel/Gimel_00.jpg"),
getImage("characters_training/Archaic/Kaf/Kaf_00.jpg"),
getImage("characters_training/Archaic/Kaf/Kaf_02.jpg"),
getImage("characters_training/Archaic/Mem/Mem_04.jpg"),
getImage("characters_training/Archaic/Mem/Mem_00.jpg"),
getImage("characters_training/Archaic/Mem/Mem_02.jpg"),
getImage("characters_training/Archaic/Qof/Qof_00.jpg"),
getImage("characters_training/Archaic/Qof/Qof_01.jpg"), 
getImage("characters_training/Archaic/Pe/Pe_00.jpg")]

labels = ["Alef"] * 2 + ["Het"] * 2 + ["Gimel"] + ["Kaf"] * 2 + ["Mem"] * 3 + ["Qof"] * 2 + ["Pe"]

classifier.classifyList(characters, labels)
