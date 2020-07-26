# Firstly, please note that the performance of google word2vec is better on big datasets.
# In this example we are considering only 25000 training examples from the imdb dataset.
# Therefore, the performance is similar to the "bag of words" model.

# Importing libraries
import numpy as np
import pandas as pd
# BeautifulSoup is used to remove html tags from the text
from bs4 import BeautifulSoup
import re  # For regular expressions
from gensim.models import word2vec
import nltk
import logging
import pickle
# Stopwords can be useful to undersand the semantics of the sentence.
# Therefore stopwords are not removed while creating the word2vec model.
# But they will be removed  while averaging feature vectors.
from nltk.corpus import stopwords
import os
from os import path
import sys
import errno
# word2vec expects a list of lists.
# Using punkt tokenizer for better splitting of a paragraph into sentences.
import csv 
from tqdm import tqdm
nltk.download('punkt')
nltk.download('stopwords')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
stops = set(stopwords.words("english"))

def mkdir_p(path):
    if not os.path.exists(path):
        os.makedirs(path)


# This function converts a text to a sequence of words.
def review_wordlist(review, remove_stopwords=False):
    # 1. Removing html tags
    review_text = BeautifulSoup(review, "html.parser").get_text()
    # 2. Removing non-letter.
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    # 3. Converting to lower case and splitting
    words = review_text.lower().split()
    # 4. Optionally remove stopwords
    if remove_stopwords:
        words = [w for w in words if not w in stops]

    return (words)

# Function to average all word vectors in a paragraph
def featureVecMethod(words, model, num_features):
    # Pre-initialising empty numpy array for speed
    featureVec = np.zeros(num_features, dtype="float32")
    nwords = 0
    list = []


    for word in words:
        if word in index2word_set:
            nwords = nwords + 1
            list.append(model[word])

    return list

#Function write data into memmory maps
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

#retuns shape of the input file
def get_shape(file):
    maxlen = 0
    for (idx, row) in enumerate(file):
        i = 0
        for word in row:
            i += 1
        if i > maxlen:
            maxlen = i
    no_of_reviews = idx+1
    return no_of_reviews, maxlen


if __name__ == '__main__':
    # Read data from files
    input = sys.argv[1]
    output = sys.argv[2]

    min_word_count = 40  # Minimum word count
    num_workers = 4  # Number of parallel threads
    context = 10  # Context window size
    downsampling = 1e-3  # (0.001) Downsample setting for frequent words
    model_path = os.path.join(input, "features_model")

    if path.exists(model_path):
        model = word2vec.Word2Vec.load(os.path.join(input, "features_model"))
        num_features = model.vector_size
    else:
        print('Error: model not found')

    if sys.argv.__len__() == 4:
        print('Running evaluation featurization...')
        path_eval = os.path.join(input, 'Evaluation.csv')
        eval = pd.read_csv(path_eval)

        model.wv.syn0.shape

        # Converting Index2Word which is a list to a set for better speed in the execution.
        index2word_set = set(model.wv.index2word)

        df=pd.read_csv(path_eval)
        classes = list(df['label'])
        df_path = os.path.join(output, 'EvaluationDatalabel.pickle')
        with open(df_path, 'wb') as f:
            pickle.dump(classes, f)

        clean_train_reviews = []
        eval_path = os.path.join(output, 'EvaluationDataVec.csv')
        with open(eval_path, "w") as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            for review in tqdm(eval['text']):
                clean_train_reviews=review_wordlist(review, remove_stopwords=True)
                trainDataVecs=featureVecMethod(clean_train_reviews, model, num_features)
                writer.writerow(trainDataVecs)

        # initializing the memory maps
        print('creating memory map...')
        evalXfile = csv.reader(open(eval_path, 'rt'))
        no_of_reviews_eval, maxlen_eval = get_shape(evalXfile)
        shape = np.array([maxlen_eval,no_of_reviews_eval,num_features])
        shape_path = os.path.join(output, 'eval_shape')
        np.save(shape_path, shape)
        data2memmap(eval_path, 'evalmapX', no_of_reviews_eval, maxlen_eval, num_features, output)

    else:
        mkdir_p(sys.argv[2])
        path_train = os.path.join(input, 'Train.csv')
        path_test = os.path.join(input, 'Test.csv')
        train = pd.read_csv(path_train)
        test = pd.read_csv(path_test)
        # Creating the model and setting values for the various parameters, To do: finetuning
        min_word_count = 40  # Minimum word count
        num_workers = 4  # Number of parallel threads
        context = 10  # Context window size
        downsampling = 1e-3  # (0.001) Downsample setting for frequent words
        model_path = os.path.join(input, "features_model")
        if path.exists(model_path):
            model = word2vec.Word2Vec.load(os.path.join(input, "features_model"))
            num_features = model.vector_size
        else:
            print('Error: model not found')

        # This will give the total number of words in the vocabulary created from this dataset
        model.wv.syn0.shape

        # Converting Index2Word which is a list to a set for better speed in the execution.
        index2word_set = set(model.wv.index2word)

        # Save Labels
        df=pd.read_csv(path_train)
        classes = list(df['label'])
        df_path = os.path.join(output, 'trainDatalabel.pickle')
        with open(df_path, 'wb') as f:
            pickle.dump(classes, f)

        df=pd.read_csv(path_test)
        classes = list(df['label'])
        df_path = os.path.join(output, 'testDatalabel.pickle')
        with open(df_path, 'wb') as f:
            pickle.dump(classes, f)

        # Calculating average feature vector for training set
        clean_train_reviews = []
        train_path = os.path.join(output, 'trainDataVec.csv')
        with open(train_path, "w") as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            for review in tqdm(train['text']):
                clean_train_reviews=review_wordlist(review, remove_stopwords=True)
                trainDataVecs=featureVecMethod(clean_train_reviews, model, num_features)
                writer.writerow(trainDataVecs)

        # Calculating average feature vector for test set
        clean_test_reviews = []
        test_path = os.path.join(output, 'testDataVec.csv')
        with open(test_path, "w") as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            for review in tqdm(test['text']):
                clean_test_reviews=review_wordlist(review, remove_stopwords=True)
                testDataVecs=featureVecMethod(clean_test_reviews, model, num_features)
                writer.writerow(testDataVecs)

        # initializing the memory maps
        print('creating memory map...')
        trainXfile = csv.reader(open(train_path, 'rt'))
        testXfile = csv.reader(open(test_path, 'rt'))
        no_of_reviews_train, maxlen_train = get_shape(trainXfile)
        no_of_reviews_test, maxlen_test = get_shape(testXfile)
        maxlen = max(maxlen_test,maxlen_train)
        shape = np.array([maxlen,no_of_reviews_train,no_of_reviews_test,num_features])
        shape_path = os.path.join(output, 'shape')
        np.save(shape_path, shape)
        data2memmap(train_path, 'trainmapX', no_of_reviews_train, maxlen, num_features, output)
        data2memmap(test_path, 'testmapX', no_of_reviews_test, maxlen, num_features, output)
