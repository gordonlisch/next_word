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


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def train_model(model,trainX,trainY,testX,testY,batch_size,no_of_reviews_train,no_of_reviews_test,maxlen,vecsize,lensetY,d_class_weights,epochs):
    model.fit(generator(trainX,trainY,batch_size,vecsize,no_of_reviews_train,maxlen,lensetY),class_weight=d_class_weights, epochs=epochs, steps_per_epoch=int(no_of_reviews_train/batch_size), verbose=1, validation_data=generator(testX,testY,batch_size,vecsize,no_of_reviews_test,maxlen,lensetY), validation_steps=int(no_of_reviews_test/batch_size))
    return model,history

def get_model(input_shape,output_shape,loss,metric,maxlen):
    model = Sequential()
    model.add(Dropout(0.3))
    #add LSTM layer
    model.add(Bidirectional(LSTM(maxlen),input_shape=input_shape))
    model.add(Dropout(0.3))
    #output
    model.add(Dense(output_shape, activation='softmax'))
    model.compile('adam',loss,metrics=[metric])
    return model

def get_data(no_of_reviews_train,no_of_reviews_test,maxlen,vecsize,input):
    test_vec_path = os.path.join(input, 'testmapX')
    train_vec_path = os.path.join(input, 'trainmapX')
    test_label_path = os.path.join(input, 'testDatalabel.pickle')
    train_label_path = os.path.join(input, 'trainDatalabel.pickle')
    trainX = np.memmap(train_vec_path, dtype='float', mode='r',shape=(no_of_reviews_train,maxlen,vecsize))
    testX = np.memmap(test_vec_path, dtype='float', mode='r',shape=(no_of_reviews_test,maxlen,vecsize))
    trainYfile = open(train_label_path, 'rb')
    testYfile = open(test_label_path, 'rb')
    trainY = pickle.load(trainYfile)
    testY = pickle.load(testYfile)
    return trainX, testX, trainY, testY

def generator(X,Y,batch_size,vecsize,no_of_reviews,maxlen,lensetY):
    if lensetY == 2:
        batchX = np.zeros((batch_size,maxlen,vecsize))
        batchY = np.zeros((batch_size))
        index = np.array(range(no_of_reviews - 1))
        Y=np.array(Y)
        while True:
            random.shuffle(index)
            for i in range(int((no_of_reviews-1)/batch_size)):
                batchX[:,:,:] = X[index[i*batch_size:(i+1)*batch_size],:,:]
                batchY[:] = Y[index[i*batch_size:(i+1)*batch_size]]
                yield batchX, batchY
    else:
        bin = LabelBinarizer()
        bin.fit(Y)
        batchX = np.zeros((batch_size, maxlen, vecsize))
        batchY = np.zeros((batch_size,lensetY))
        index = np.array(range(no_of_reviews - 1))
        Y = bin.transform(Y)
        while True:
            random.shuffle(index)
            for i in range(int((no_of_reviews - 1) / batch_size)):
                batchX[:, :, :] = X[index[i * batch_size:(i + 1) * batch_size], :, :]
                batchY[:,:] = Y[index[i * batch_size:(i + 1) * batch_size],:]
                yield batchX, batchY


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.stderr.write('Arguments error. Usage:\n')
        sys.stderr.write('\tpython train.py features model\n')
        sys.exit(1)
    input = sys.argv[1]
    output = sys.argv[2]

    print('data path:', input)
    print('output path:', output)
    mkdir_p(output)
    writepath = os.path.join(output, 'model.h5')
    shape_path = os.path.join(input,'shape.npy')

    #load shape of memory maps
    [maxlen,no_of_reviews_train,no_of_reviews_test,vecsize] = np.load(shape_path)

    #get datasets
    print('loading data...')
    trainX, testX, trainY, testY = get_data(no_of_reviews_train,no_of_reviews_test,maxlen,vecsize,input)

    #determine number of label classes
    lensetY=len(set(testY))
    batch_size = 100
    epochs = 30

    #get model
    if lensetY == 2:
        loss='binary_crossentropy'
        metric='accuracy'
        print("defining model...")
        model = get_model((maxlen,vecsize), lensetY-1,loss,metric,maxlen)
    else:
        loss='CategoricalCrossentropy'
        metric='accuracy'
        print("defining model...")
        model = get_model((maxlen, vecsize), lensetY, loss, metric,maxlen)

    #training model
    class_weights = compute_class_weight('balanced', np.unique(trainY), trainY)
    d_class_weights = dict(enumerate(class_weights))
    print('training model...')
    model = train_model(model, trainX, trainY, testX, testY,batch_size,no_of_reviews_train,no_of_reviews_test,maxlen,vecsize,lensetY,d_class_weights,epochs)
    model.save(writepath)


# python3 src/prepare.py data/dataset/ data/prepared/ clothing_set.zip
# python3 src/prepare.py data/dataset data/prepared imdb_trainset.zip imdb_testset.zip
# python3 src/featurization.py data/prepared data/features
# python3 src/train.py data/features data/models
# python3 src/evaluate.py data/models data/features data/scores

