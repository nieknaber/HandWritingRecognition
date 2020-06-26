import sys
import time
from src.HelperFunctions.helper import *
from src.LineSegmentation.lineSegment import *
from src.LineSegmentation.lineSegmentAStar import *
from src.SlantNormalization.slantNormalize import *
from characters import Character_Classification
from classifier import Classifier

SAVEPICTURES = False # set to false if you do not want to save pictures in between steps

img = getImage("images/05.jpg")
rotatedImage, slope = findSlope(img, 10, 1)
print("Best angle: ", slope)


images = lineSegmentAStar(rotatedImage)
print("segmenting lines")
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

print("doing character classification and segmentation")
segment_size = (30,30)
window_size = (30*6, 30*3)
model_path = './trained_models/model_dimension3_500_epochs.pt'
cc = Character_Classification(segment_size, window_size, model_path)
imgs = []
labels = []
cnt = 0
for image in images:
    print(cnt)
    cnt +=1
    img, lab = cc.run_classification(image)
    imgs.extend(img)
    labels.extend(lab)

print("doint style classification")
styleClassifier = Classifier("char_num_acc_lda_k3.txt", 3)
style = styleClassifier.classifyList(imgs, labels)
print(style)


