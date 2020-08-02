import sys
import os
import errno

import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
import pickle
import random
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.python.keras import Sequential



def train_model(model,trainX,batch_size,size,vecsize,epochs,context,testmode):
    if testmode:
        size = int(size/100)
    model.fit(generator(trainX, batch_size, vecsize, size, context), epochs=epochs, steps_per_epoch=int(size/batch_size), verbose=1)
    return model

def get_model(input_shape):
    model = Sequential()
    #add LSTM layer
    model.add(LSTM(100, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(20))
    #output
    model.add(Dense(input_shape[1], activation='sigmoid'))
    model.compile('Adam', 'mean_squared_error', metrics='accuracy')
    return model

def get_data(size, vecsize):
    trainX = np.memmap('../data/prepared/TrainMap', dtype='float', mode='r', shape=(size, vecsize))
    return trainX

def generator(X,batch_size,vecsize,size,context):
    batchX = np.zeros((batch_size, context, vecsize))
    batchY = np.zeros((batch_size, vecsize))
    index = np.array(range(no_of_reviews - 1))

    while True:
        i = 0
        for j in range(0, size-context):
            if i < batch_size:
                batchX[i, 0:context-1, :] = X[j:j+context-1, :]
                batchY[i, :] = X[j+context, :]
                i += 1
            else:
                i = 0
                yield batchX, batchY


if __name__ == '__main__':

    #load shape of memory maps
    [no_of_reviews, size, vecsize, output_shape, context] = np.load('../data/prepared/shape.npy')

    #get datasets
    print('loading data...')
    trainX = get_data(size, vecsize)

    #determine number of label classes
    batch_size = 64
    epochs = 2



    #get model
    #context = 20
    model = get_model((context, vecsize))

    #training model
    #class_weights = compute_class_weight('balanced', np.unique(trainY), trainY)
    #d_class_weights = dict(enumerate(class_weights))
    print('training model...')
    model = train_model(model, trainX, batch_size, size, vecsize, epochs, context, testmode=True)
    model.save('../data/model/model.h5')


