import numpy as np
import pandas as pd
import sys
import random
import os
import errno
import zipfile
import pandas as pd
from sklearn.model_selection import train_test_split
# BeautifulSoup is used to remove html tags from the text
from bs4 import BeautifulSoup
import re  # For regular expressions
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
    sentences = list()

    for idx, filename in enumerate(os.listdir('../data/dataset/Fantasy/')):
        filehandler = open(os.path.join('../data/dataset/Fantasy/', filename), 'r')
        print(idx)
        for line in filehandler:
            line = re.sub("[^a-zA-Z .,;]", "", line)
            sentences.append(line.split(' '))
        filehandler.close()



    # Creating the model and setting values for the various parameters, To do: finetuning
    num_features = 20  # Word vector dimensionality
    min_word_count = 100  # Minimum word count
    num_workers = 6  # Number of parallel threads
    context = 5  # Context window size
    downsampling = 1e-3  # (0.001) Downsample setting for frequent words

    # Initializing the train model

    print("Training model....")
    model = word2vec.Word2Vec(
        sentences,
        workers=num_workers,
        size=num_features,
        min_count=min_word_count,
        window=context,
        sample=downsampling
    )

    # To make the model memory efficient
    model.init_sims(replace=True)

    # Saving the model for later use. Can be loaded using Word2Vec.load()

    model.save("../data/prepared/M2V_model")


    model = word2vec.Word2Vec.load('../data/prepared/M2V_model')
    number_of_sentences = sentences.__len__()
    size = 0
    for sentence in sentences:
        size += len(sentence)
    print('Total Number of words:', size, 'Total Number of Sentences', number_of_sentences)

    shape = [number_of_sentences, size, num_features, model.wv.syn0.shape, context]
    np.save('../data/prepared/shape.npy', shape)


    data = np.memmap('../data/prepared/TrainMap', dtype='float', mode='w+', shape=(size, num_features))



    dumbwords=list()
    position = 0
    for (idxSentence, sentence) in enumerate(sentences):
        sentencetmp = list()
        for (idxWord, word) in enumerate(sentence):
            try:
                sentencetmp.append(model[word])
            except Exception:
                dumbwords.append(word)
        sentencetmp = np.array(sentencetmp)
        data[position:position+len(sentencetmp), :] = sentencetmp
        print(idxSentence/number_of_sentences)
        position += len(sentencetmp)

    dumbwords=np.array(dumbwords)
    np.save('../data/prepared/dumbwords.npy', dumbwords)




