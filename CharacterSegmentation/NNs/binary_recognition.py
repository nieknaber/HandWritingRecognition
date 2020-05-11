import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.applications import mobilenet
from keras import callbacks
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
import numpy as np
from os import listdir
from keras.optimizers import Adam
import matplotlib.pyplot as plt

def load_images(directories=[]):
    #create a blank list to store the images in
    imgs=[]
    #loop over the list of directories we have been passed
    for directory in directories:
        #loop over the files that are contained in the directory
        for file in listdir(directory):
            #load the image in the directory directory with the filename file
            img = image.load_img(directory+file, target_size=None).resize((32,32))
            #convert it to a numpy array
            img = np.array(img)
            #/255 for data normalization
            #append to the imgs list
            imgs.append(img/255) # so all have the same shape
            
    return imgs

def load_y(directories):
    y_data=[]
    #add [1,0] (0 one hot encoded) to y_data the same amount of times as the number of files in the first directory in direcotories
    for i in range(0,len(listdir(directories[0]))):
        y_data.append([1,0])
    #same but with second direcotry
    for i in range(0,len(listdir(directories[1]))):
        y_data.append([0,1])
    return y_data

dirs=["monkbrill/Alef/","monkbrill/Mem/"]

#creates the Y data
Y = np.array(load_y(dirs))
print(Y.shape)
#loads all the images
X = np.array(load_images(dirs))
print(X.shape)
#tells the user the total number of images
print(len(load_images(dirs)),"images")


#creates train and test data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

#print(x_train.shape)
#print(y_train)

#plt.figure()
#plt.imshow(x_train[5])
#plt.colorbar()
#plt.grid(False)
#plt.show()

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i])
    plt.xlabel(y_train[i])
plt.show()


img_width, img_height = 32, 32
shape = (img_width, img_height, 3)
#load MobileNet pretrained on imagenet with input_shape of the shape variable
mn=mobilenet.MobileNet(weights='imagenet', include_top=False, input_shape=shape) # neeeded to get the data in the right shape
model = Sequential()
model.add(mn)
model.add(Flatten())
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
model.fit(x_train, y_train, epochs = 10, batch_size=32)#, validation_data=(x_test, y_test),callbacks=callbacks_list)

test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)

print('\nTest accuracy:', test_acc)

# https://github.com/qwertpi/catvdogcnn/blob/master/train.py

# https://www.tensorflow.org/tutorials/keras/classification

