import numpy as np
import sys
import os
import errno
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import pickle
import random
import keras.models
import re  # For regular expressions
from gensim.models import word2vec
from gensim.models import word2vec
import nltk
# Stopwords can be useful to undersand the semantics of the sentence.
# Therefore stopwords are not removed while creating the word2vec model.
# But they will be removed  while averaging feature vectors.
from nltk.corpus import stopwords
# word2vec expects a list of lists.
# Using punkt tokenizer for better splitting of a paragraph into sentences.
import csv
from tqdm import tqdm


if __name__ == '__main__':


    # Maybe implement other ratios if necessary, Test data set split ratio

    filehandler = open('../data/dataset/eval.txt', 'r')

    sentences = list()
    for line in filehandler:
        line = re.sub("[^a-zA-Z ]", "", line)
        sentences.append(line.split(' '))

    filehandler.close()

    [no_of_reviews, size, num_features, output_shape, context,vocab_size] = np.load('../data/prepared/shape.npy',allow_pickle=True)  # Word vector dimensionality


    #no_of_reviews = sentences.__len__()
    maxlen = len(max(sentences, key=len))


    model = word2vec.Word2Vec.load('../data/prepared/M2V_model')

    sentencetmp2=np.zeros((2, 20, num_features))
    for (idxSentence, sentence) in enumerate(sentences):
        sentencetmp = np.zeros((1, 20, num_features))
        for (idxWord, word) in enumerate(sentence):
            try:
                sentencetmp[0, idxWord, :] = model[word]
            except Exception:
                print('Exeption')
        sentencetmp2[idxSentence-1, :, :] = sentencetmp[0, :, :]
    print(sentencetmp2)


    model2 = keras.models.load_model('../data/model/model.h5')

    predEvalY = np.array(model2.predict(sentencetmp2))
    print(predEvalY)
    print(model.most_similar(predEvalY))


    #model.most_similar(positive=['man'],negative=['male'])
