
#Created on Mon May 25 10:18:25 2020#

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Activation
import seaborn as sns
from keras.layers import BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, adam
from keras.utils import np_utils, plot_model

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from PIL import Image
from numpy import *
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

DATADIR = "D:\\yash\\doc\\Intern\\Tech\\Vision Sensing of weld\\Welding Datasets\\Dataset"
DATADIR2 = "D:\\yash\\doc\\Intern\\Tech\\Vision Sensing of weld\\Welding Datasets\\Input_data_resized"
list = os.listdir(DATADIR)
num = size(list)
print(num)
for file in list:
    im = Image.open(DATADIR + '\\' + file)
    
    
    img_rows, img_cols = 100, 100
    img = im.resize((img_rows,img_cols))
    gray = img.convert('L')
    gray.save(DATADIR2 + '\\' + file, "JPEG")
imlist = os.listdir(DATADIR2)

im1 = array(Image.open('Input_data_resized' + '\\' + imlist[0]))
m,n = im1.shape[0:2]
imnbr = len(imlist)

immatrix = array([array(Image.open('input_data_resized'+'\\'+ im2)).flatten()
                  for im2 in imlist],'f')

label = np.ones((num,),dtype = int)
label[0:8]=0
label[8:43]=1
label[43:71]=2

data,Label = shuffle(immatrix,label, random_state=2)
train_data = [data,Label]

img = immatrix[2].reshape(img_rows, img_cols)
plt.imshow(img)
plt.imshow(img, cmap ='gray')
print(train_data[0].shape)
print(train_data[1].shape)

batch_size = 12
nb_classes = 3
nb_epoch = 10
img_rows, img_cols = 100, 100
img_channels = 1
nb_filters = 32
nb_pool = 2
nb_conv = 3

(X,y) = (train_data[0],train_data[1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 4)

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

i=41
plt.imshow(X_train[i, 0], interpolation='nearest')
print("label : ", Y_train[i,:])

class_names = ['OK Weld', 'Porosity','Undercut']

#Training Model#
model = Sequential()
model.add(Convolution2D(nb_filters, nb_conv, nb_conv, border_mode= 'valid', input_shape=( img_rows, img_cols, 1)))
convout1 = Activation('relu')
model.add(convout1)
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
          
convout2 = Activation('relu')
model.add(convout2)
model.add(MaxPooling2D( pool_size = (nb_pool, nb_pool)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.summary()          

model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_test, Y_test))
          
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
           verbose=1, validation_split=0.3)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(X_test,  Y_test, verbose=2)

print(test_acc)
print(model.predict_classes(X_test[1:5]))
print(Y_test[1:5])