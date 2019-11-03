import numpy as np
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Activation, UpSampling2D
from keras.utils import plot_model
import pandas as pd
import os

os.chdir("/Volumes/Files/Onedrive/Masters/Study Materials/Third Semester/Seminar-Recent Trends in Deep Learning")


matrix = pd.read_pickle("Word_Matrices.pkl")
matrix = matrix[:100]

x_train = np.array([])
for i in range(len(matrix)):
    data = np.array(matrix.iloc[i,3], np.array(np.zeros(23)))
    x_train = np.append( x_train, data)

x_train = np.zeros(720000).reshape((100, 24,300))

x_train = x_train.reshape((100, 23,300))

def define_CNN_autoencoder_layers(input_shape, filter_size, pool_size):	
    model = Sequential()
    
    #1st convolution layer
    model.add(Conv2D(16, filter_size, padding='same', input_shape=(100, 24, 300)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))
    
    #2nd convolution layer
    model.add(Conv2D(2,filter_size, padding='same')) # apply 2 filters sized of (3x3)
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))
    
    #here compressed version
    
    #3rd convolution layer
    model.add(Conv2D(2,filter_size, padding='same')) # apply 2 filters sized of (3x3)
    model.add(Activation('relu'))
    model.add(UpSampling2D(filter_size))
    
    #4rd convolution layer
    model.add(Conv2D(16,filter_size, padding='same'))
    model.add(Activation('relu'))
    model.add(UpSampling2D(filter_size))
    
    model.add(Conv2D(1,filter_size, padding='same'))
    model.add(Activation('sigmoid'))

    model.summary()
    model.compile(optimizer='adadelta', loss='binary_crossentropy')
    return model


model = define_CNN_autoencoder_layers((100,24,300), (2,2), (2,2))

model.fit(x_train, x_train, epochs=3)

