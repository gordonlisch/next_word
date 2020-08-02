from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty
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


def eval(sentence):

    sentence = re.sub("[^a-zA-Z ,.;]", "", sentence)
    sentence = sentence.split(' ')
    sentence = [ele for ele in sentence if ele != ""]

    if sentence.__len__() >= 5:
        sentence = sentence[-6:-1] #this needs to be automizted


    model = word2vec.Word2Vec.load('../data/prepared/M2V_model')

    sentence = [model[ele] for ele in sentence]
    sentence = np.array(sentence)
    sentence = sentence.reshape(1, sentence.shape[0], sentence.shape[1])

    model2 = keras.models.load_model('../data/model/model.h5')

    predEvalY = np.array(model2.predict(sentence))

    return model.most_similar(predEvalY, topn=1)[0][0]

class UI(Widget):
    input = ObjectProperty(None)

    def next_word(self):
        self.input.text = self.input.text + ' ' + eval(self.input.text)

    def next_words(self):
        for i in range(15):
            self.input.text = self.input.text + ' ' + eval(self.input.text)



class UIApp(App):
    def build(self):
        return UI()


if __name__ == '__main__':
    UIApp().run()
