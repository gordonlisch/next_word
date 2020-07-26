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

#nltk.download('punkt')
#nltk.download('stopwords')
#tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
##stops = set(stopwords.words("english"))


def data2memmap(path,mmap,no_of_reviews,maxlen,num_features,output):
    file = csv.reader(open(path, 'rt'))
    mmap = os.path.join(output,mmap)
    data = np.memmap(mmap, dtype='float', mode='w+', shape=(no_of_reviews, maxlen, num_features))
    for (idx, row) in enumerate(file):
        review_length = len(row)
        review_length = min(review_length,maxlen)
        review = list()
        print(idx)
        if review_length == 1:
           wordtmp = row[0].replace('        ', ' ').replace('       ', ' ').replace('      ', ' ').replace('     ', ' ').replace('    ', ' ').replace('   ', ' ').replace('  ', ' ').replace(' ]', '').replace(']', '').replace('[ ', '').replace('[', '').replace('\n', '')
           wordtmp = wordtmp.split(' ')
           review.append(np.array(wordtmp).astype('float'))
        if review_length > 1:
            for i in range(review_length):
                wordtmp = row[i].replace('        ', ' ').replace('       ', ' ').replace('      ', ' ').replace('     ', ' ').replace('    ', ' ').replace('   ', ' ').replace('  ', ' ').replace(' ]', '').replace(']', '').replace('[ ', '').replace('[', '').replace('\n', '')
                wordtmp = wordtmp.split(' ')
                review.append(np.array(wordtmp).astype('float'))
        review = np.array(review)
        data[idx, (maxlen - review.shape[0]):maxlen, :] = review
    data.flush()


if __name__ == '__main__':


    # Maybe implement other ratios if necessary, Test data set split ratio

    filehandler = open('../data/dataset/lambada_development_plain_text.txt', 'r')

    sentences = list()
    for line in filehandler:
        line = re.sub("[^a-zA-Z .,?;]", "", line)
        sentences.append(line.split(' '))

    filehandler.close()
   
    # Creating the model and setting values for the various parameters, To do: finetuning
    num_features = 10  # Word vector dimensionality
    min_word_count = 1  # Minimum word count
    num_workers = 4  # Number of parallel threads
    context = 10  # Context window size
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

    #csv_file = open('../data/prepared/TrainData.csv', 'w+')
    #writer = csv.writer(csv_file, delimiter='|')

    no_of_reviews = sentences.__len__()
    #maxlen = len(max(sentences, key=len))
    maxlen=170


    data = np.memmap('../data/prepared/TrainMap', dtype='float', mode='w+', shape=(no_of_reviews, maxlen, num_features))


    for (idxSentence, sentence) in enumerate(sentences):
        sentencetmp = list()
        for (idxWord, word) in enumerate(sentence):
            sentencetmp.append(model[word])
        sentencetmp = np.array(sentencetmp)
        data[idxSentence, (maxlen - sentencetmp.shape[0]):maxlen, :] = sentencetmp
        print(idxSentence)



