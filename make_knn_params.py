import os
import numpy as np
from src.StyleClassification.features import *
from src.StyleClassification.helper import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

archaicFolder = "characters_training/Archaic/"
hasmoneanFolder = "characters_training/Hasmonean/"
herodianFolder = "characters_training/Herodian/"

# make unique list of all characters 
characters = set([])
characters.update([f.path[len(archaicFolder):] for f in os.scandir(archaicFolder) if f.is_dir()]) 
characters.update([f.path[len(hasmoneanFolder):] for f in os.scandir(hasmoneanFolder) if f.is_dir()]) 
characters.update([f.path[len(herodianFolder):] for f in os.scandir(herodianFolder) if f.is_dir()]) 
characters = sorted(list(characters))

f = open("char_num_acc_lda_k11.txt","w+")

for char in characters:
    
    if (not(os.path.isdir(archaicFolder + char)) or not(os.path.isdir(hasmoneanFolder + char)) or not(os.path.isdir(herodianFolder + char))):
        continue

    x, y = getData(char, archaicFolder, hasmoneanFolder, herodianFolder)

    totAccs = []
    for num in range(1,5):

        features = getFeatures(x, num)
        if (features.ndim != 2): # important check! if not it means that background stuff did not work, no classification possible
            totAccs.append(0)
            continue

        accs = []
        for index in range(len(features)):
            trainFeatures = [feature for ind, feature in enumerate(features) if ind != index]
            testFeatures = [features[index]]
            yTrain = [feature for ind, feature in enumerate(y) if ind != index]
            yTest = [y[index]]

            lda = LinearDiscriminantAnalysis()
            trainTransformed = lda.fit_transform(trainFeatures, yTrain)
            testTransformed = lda.transform(testFeatures)
            
            trainTransformed, testTransformed, _ = zScoreTwoArrs(trainTransformed, testTransformed)

            n_neighbors = 11
            knn = KNeighborsClassifier(n_neighbors)
            knn.fit(trainTransformed, yTrain)

            predictions = knn.predict(testTransformed)
            acc = sum(predictions == yTest) / len(predictions)
            accs.append(acc)

        tot = sum(accs) / len(accs)
        print("1-fold accuracy of letter ", char, " with k 11, lda 2: ", tot, " num: ", num)
        totAccs.append(tot)

    f.write(char + "\t" + str(np.argmax(totAccs) + 1) + "\t" + str(max(totAccs)) + "\n")
    f.flush()

f.close()