import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, UpSampling1D,Conv2D,MaxPooling2D,Conv2DTranspose,Reshape
import tensorflow as tf



def recreate_input_matrix(dataframe, index, batch, dimensions, channel):
    x_train = np.array([])
    for i in range(len(dataframe)):
        x_train = np.append( x_train, dataframe.iloc[i,index])
    
    x_train = x_train.reshape((batch, dimensions, channel))
    return x_train

def recreate_input_matrix_2d(dataframe, index, batch, dimensions, channel):
    x_train = np.array([dataframe.iloc[i,index] for i in range(len(dataframe))]) 
    x_train = x_train.reshape((batch, dimensions, channel,1))
    return x_train

def CNN_autoencoder(x_train, filter_size, pool_size):	
    model = Sequential()
    
    #Encoder Part
    model.add(Conv1D(16, filter_size, padding='same', input_shape=(len(x_train[0]), len(x_train[0][0])), name='input', data_format="channels_last", activation='relu'))
    model.add(MaxPooling1D(pool_size=pool_size, padding='same'))
    
    model.add(Conv1D(16,filter_size, padding='same', activation='relu')) 
    model.add(MaxPooling1D(pool_size=pool_size, padding='same'))
    
    model.add(Conv1D(8,filter_size, padding='same', activation='relu')) 
    model.add(MaxPooling1D(pool_size=3, padding='same'), )  
    model.add(Dense(1 , name='bottleneck'))
   
    #Decoder Part
    model.add(Conv1D(2,filter_size, padding='same', activation='relu')) 
    model.add(UpSampling1D(3))

    model.add(Conv1D(8,filter_size, padding='same',activation='relu'))
    model.add(UpSampling1D(pool_size))
    
    model.add(Conv1D(16,filter_size, padding='same', activation='relu'))
    model.add(UpSampling1D(pool_size))
    
    model.add(Conv1D(299,filter_size, padding='same', activation='sigmoid'))
    model.summary()
    
    model.compile(optimizer='adadelta', loss='binary_crossentropy')
    model.fit(x_train, x_train, epochs=10)
    bottleneck = model.get_layer('bottleneck')
    encoder = Model(input=model.input, output=bottleneck.output)
    
    return model, encoder


def CNN_autoencoder_2D(x_train, filter_size, pool_size):	
    model = Sequential()
    
    #Encoder Part
    model.add(Conv2D(16, filter_size, padding='same', input_shape=(len(x_train[0]), len(x_train[0][0]),1), name='input', data_format="channels_last", activation='relu'))
    model.add(Conv2D(8,filter_size, padding='valid', activation='relu')) 
    model.add(Conv2D(4,filter_size, padding='valid', activation='relu'))
    #model.add(MaxPooling2D(pool_size=(24,300), padding='same'))
    model.add(Flatten())
    model.add(Dense(5 , name='bottleneck'))
    
    #Unflatten 
    model.add(Reshape((1, 1, 5)))
   
    #Decoder Part
    model.add(Conv2DTranspose(2, (3,3), padding='valid', activation='relu')) 
    model.add(Conv2DTranspose(4,pool_size, padding='valid',activation='relu'))
    model.add(Conv2DTranspose(8,pool_size, padding='valid', activation='relu'))
    model.add(Conv2DTranspose(1,pool_size, padding='valid', activation='sigmoid'))
    model.summary()
    
    model.compile(optimizer='adadelta', loss='mse')
    model.fit(x_train, x_train, epochs=5)
    bottleneck = model.get_layer('bottleneck')
    encoder = Model(input=model.input, output=bottleneck.output)
    
    return model, encoder

def CNN_autoencoder_2D_em(x_train, filter_size, vocab_size,feature_dimension_size,max_sequence_length,embedding_matrix, type):
    gpus = tf.config.experimental.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(gpus[0], True)
    cluster=5
    if type=="Amazon" :
        adjustment = 1
    else:
        adjustment = 0

    model = Sequential()
    
    #Encoder Part
    model.add(Embedding(vocab_size, feature_dimension_size, input_length=max_sequence_length, trainable=False, weights=[embedding_matrix]))
    model.add(Reshape((max_sequence_length, feature_dimension_size,1)))
    model.add(Conv2D(32, filter_size, padding='same', name='input',data_format='channels_last', activation='relu'))
    model.add(Conv2D(16,filter_size, padding='valid', activation='relu')) 
    model.add(Conv2D(8,filter_size, padding='valid', activation='relu'))
    model.add(MaxPooling2D(filter_size, padding='same'))
    model.add(Flatten())
    model.add(Dense(cluster , name='bottleneck'))
    
    #Unflatten 
    model.add(Reshape((1, 1, cluster)))
   
    #Decoder Part
    model.add(Conv2DTranspose(8,(int(max_sequence_length/3),100), padding='valid',activation='relu'))
    model.add(Conv2DTranspose(16,(int(max_sequence_length/3),100), padding='valid', activation='relu'))
    model.add(Conv2DTranspose(1,(int(max_sequence_length/3+(2+adjustment)),102), padding='valid', activation='sigmoid'))
    model.summary()
    model.add(Reshape((max_sequence_length, feature_dimension_size)))
    model.summary()
    
    model.compile(optimizer='adadelta', loss='mse', metrics=["accuracy"])
    #model.compile(optimizer='adadelta', loss='cosine_proximity', metrics=["accuracy"])
    expected_autoencoder_output = np.array([[embedding_matrix[word_index] for word_index in encoded_sequence] for encoded_sequence in x_train])
    model.fit(x_train, expected_autoencoder_output, epochs=5, verbose=1)
    bottleneck = model.get_layer('bottleneck')
    encoder = Model(input=model.input, output=bottleneck.output)
    
    return model, encoder


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T