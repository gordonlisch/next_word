import sys
import os
import errno
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import pickle
import random
import keras.models




def mkdir_p(path):
    if not os.path.exists(path):
        os.makedirs(path)





if __name__ == '__main__':

    mkdir_p(sys.argv[3])
    writepath = os.path.join(sys.argv[3], 'auc.json')
    mode = 'a' if os.path.exists(writepath) else 'w'
    with open(writepath, mode) as fd:
        fd.write('0 - 100')

    os.system(' python3 src/featurization.py data/prepared data/features evaluation')


    model = sys.argv[1]
    input = sys.argv[2]
    output = sys.argv[3]

    shape_path = os.path.join(input,'eval_shape.npy')
    model_path = os.path.join(model,'model.h5')
    #load shape of memory maps
    [maxlen,no_of_reviews,vecsize] = np.load(shape_path)

    eval_vec_path = os.path.join(input, 'evalmapX')
    eval_label_path = os.path.join(input, 'EvaluationDatalabel.pickle')
    evalX = np.memmap(eval_vec_path, dtype='float', mode='r',shape=(no_of_reviews,maxlen,vecsize))
    evalYfile = open(eval_label_path, 'rb')
    evalY = pickle.load(evalYfile)

   #load model
    model = keras.models.load_model(model_path)


    #predict labels
    predEvalY = model.predict(evalX)
    print(predEvalY)

