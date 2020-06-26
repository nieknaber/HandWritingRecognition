import sys
import time
from src.HelperFunctions.helper import *
from src.LineSegmentation.lineSegment import *
from src.LineSegmentation.lineSegmentAStar import *
from src.SlantNormalization.slantNormalize import *
from characters import Character_Classification
from classifier import Classifier

SAVEPICTURES = False # set to false if you do not want to save pictures in between steps

img = getImage("25-Fg001.pbm")
rotatedImage, slope = findSlope(img, 10, 1)
print("Best angle: ", slope)


images = lineSegmentAStar(rotatedImage)
print("Segmenting lines")
for i in range(0,len(images)):
    images[i] = np.transpose(images[i])
    if SAVEPICTURES: cv.imwrite("line_" + str(i) + ".bmp", (1 - images[i]) * 255)
    if SAVEPICTURES: print("saving picture ", i)

# print("doing slant normalization")
# slantAngles = []
# for i in range(0,len(images)):
#     slantAngle, dst = slantNormalize(images[i])
#     if SAVEPICTURES: cv.imwrite("line_" + str(i) + "_deslanted.bmp", (1 - dst) * 255)
#     images[i] = dst
#     slantAngles.append(slantAngle)

# images[] is now the list of slope corrected line segments
# slantAngles[] is the list of slants per line segment (same indexes)
# slope is the line slope angle

name_to_character = {'א': "alef",'ב': "bet",'ג': "gimel",'ד': "dalet", 'ה':"he", "ו":"waw","ז":"zayin","ח":"het","ט":"tet","י":"yod","כ":"kaf","ך":"kaf-final","ל":"lamed","מ":"mem-medial","ם":"mem","נ":"nun-medial","ן":"nun-final","ס":"samekh","ע":"ayin","פ":"pe","ף":"pe-final","צ":"tsadi-medial","ץ":"tsadi-final","ק":"qof","ר":"resh","ש":"shin","ת":"taw", " ":" "}

print("Doing character classification and segmentation")
segment_size = (30,30)
window_size = (30*6, 30*3)
model_path = './trained_models/model_dimension3_500_epochs.pt'
cc = Character_Classification(segment_size, window_size, model_path)
imgs = []
labels = []
cnt = 0
texts=''
with open('img_001_characters.txt', 'w') as f:
    for image in images:
        print(cnt)
        cnt +=1
        img, lab = cc.run_classification(image)
        imgs.extend(img)
        labels.extend(lab)
        sentence=''
        for ch in lab:
            for character, name in name_to_character.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
                if name == ch:
                    sentence += str(character)
        f.write("%s\n" % sentence)
        texts+=(sentence+ '\n')
print(texts)

newLabels = [label.capitalize() for label in labels]
newImgs = [img.astype(np.uint8) for img in imgs]


print("Doing style classification")
styleClassifier = Classifier("char_num_acc_lda_k3.txt", 3)
style = styleClassifier.classifyList(newImgs, newLabels)
print(style)
with open('img_001_style.txt', 'w') as f:
    f.write(style)


