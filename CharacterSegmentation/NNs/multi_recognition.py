import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
import numpy as np
from os import listdir
import matplotlib.pyplot as plt

def getBinaryDummyImage(filename):
    im = image.load_img(filename) # changed to keras 
    (height, width, _) = np.shape(im) # dummy is (180, 1250, 3), i.e. (height, width, colors)

    im = im.load()
    newBinarizedImage = np.zeros((height,width))

    for y in range(height):
        for x in range(width):

            (r, _, _) = im[x,y]

            if r < 125:
                newBinarizedImage[y,x] = 1

    return newBinarizedImage 

def load_images(directories=[]):
	#create a blank list to store the images in
	imgs=[]
	#loop over the list of directories we have been passed
	for directory in directories:
		#loop over the files that are contained in the directory
		for file in listdir(directory):
			#load the image in the directory directory with the filename file
			img = getBinaryDummyImage(directory+file)#.resize((32,32))
			#convert it to a numpy array
			img.resize((32,32))
			imgs.append(img)			
	return imgs

def load_y(directories):
	y_data=[]
	#add (0 one hot encoded) to y_data the same amount of times as the number of files in the first directory in direcotories
	for d,directory in enumerate(directories):
		for i in range(0,len(listdir(directories[d]))):
			#array = np.zeros((1, 26))
			#array = np.insert(array, d, 1, axis=1)
			y_data.append(d)
	return y_data

dirs=["monkbrill/Alef/","monkbrill/Ayin/","monkbrill/Bet/","monkbrill/Dalet/","monkbrill/Gimel/","monkbrill/He/","monkbrill/Het/","monkbrill/Kaf/","monkbrill/Kaf-final/","monkbrill/Lamed/","monkbrill/Mem/","monkbrill/Mem-medial/","monkbrill/Nun-final/","monkbrill/Nun-medial/","monkbrill/Pe/","monkbrill/Pe-final/","monkbrill/Qof/","monkbrill/Resh/","monkbrill/Samekh/","monkbrill/Shin/","monkbrill/Taw/","monkbrill/Tet/","monkbrill/Tsadi-final/","monkbrill/Tsadi-medial/","monkbrill/Waw/","monkbrill/Yod/","monkbrill/Zayin/"]

#creates the Y data
Y = np.array(load_y(dirs))
print(Y.shape)
#loads all the images
X = np.array(load_images(dirs))
print(X.shape)
#tells the user the total number of images
#print(len(load_images(dirs)),"images")


#creates train and test data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

print(x_train.shape)
print(len(y_train))

#plt.figure()
#plt.imshow(x_train[5])
#plt.colorbar()
#plt.grid(False)
#plt.show()

plt.figure(figsize=(10,6))
for i in range(25):
	 plt.subplot(5,5,i+1)
	 plt.xticks([])
	 plt.yticks([])
	 plt.grid(False)
	 plt.imshow(x_train[i], cmap=plt.cm.binary)
	 plt.xlabel(y_train[i])
plt.show()

model = keras.Sequential([
	 keras.layers.Flatten(input_shape=(32, 32)), # transforms the format of the images from a two-dimensional array (of 28 by 28 pixels) to a one-dimensional array (of 28 * 28 = 784 pixels)
	 keras.layers.Dense(128, activation='relu'),
     keras.layers.Dense(27)
])

model.compile(optimizer='adam',
			   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
			   metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10)

test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)

print('\nTest accuracy:', test_acc)

# https://github.com/qwertpi/catvdogcnn/blob/master/train.py

# https://www.tensorflow.org/tutorials/keras/classification

