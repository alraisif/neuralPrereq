import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
import re

from bs4 import BeautifulSoup

import sys
import os

os.environ['KERAS_BACKEND']='theano'

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout
from keras.models import Model

MAX_WORDS = 12400
MAX_LENGTH = 2590 # maximum sequence length
EMBEDDING_DIM = 100

TRAIN_TEST_SPLIT = 0.2

courseData = pd.read_csv('data/prereqDataBinary.csv', sep=',')
print (courseData.shape)
                    
texts = []
labels = []
                    
for idx in range(courseData.textContent.shape[0]):
    text = BeautifulSoup(courseData.textContent[idx],"html5lib")
    cleanstring = text.get_text().strip().lower()
    cleanstring = cleanstring.encode( "utf-8",'ignore')
    texts.append(cleanstring)
    labels.append(courseData.prereqClass[idx])
    #print(labels)
                    
tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
                    
word_index = tokenizer.word_index
#print('%s unique tokens were found.' % len(word_index))


data = pad_sequences(sequences, maxlen=MAX_LENGTH)
                    
labels = to_categorical(np.asarray(labels))
print('Data (x) dimension:', data.shape)
print('Label (y) dimension:', labels.shape)
                    
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
testdata_size = int(TRAIN_TEST_SPLIT * data.shape[0])
                    
x_train = data[:-testdata_size]
y_train = labels[:-testdata_size]
x_test = data[-testdata_size:]
y_test = labels[-testdata_size:]
                    
print('Number of positive and negative examples in trainging and test data ')
print (y_train.sum(axis=0))
print (y_test.sum(axis=0))
                    
EMDED_DIR = "embeddings/glove"
embeddings_index = {}

f = open(os.path.join(EMDED_DIR, 'glove.6B.100d.txt'),encoding="utf-8")
for line in f:
    values = line.split()
    word = values[0]
    feature_values = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = feature_values
f.close()


#initialize embedding matrix randomly
embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # for words in V with embeddings update their initialization with embedding vectors
        embedding_matrix[i] = embedding_vector
                    
embedding_layer = Embedding(len(word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix],
                            input_length=MAX_LENGTH, trainable=True)

convolutions = []
filter_windows = [3,5]


sequence_input = Input(shape=(MAX_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

        #keras.layers.Conv1D(filters, kernel_size, strides=1, padding='valid', dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

for window in filter_windows:
    l_conv = Conv1D(nb_filter=128,filter_length=window,activation='relu')(embedded_sequences)
    l_pool = MaxPooling1D(5)(l_conv)
    convolutions.append(l_pool)
                    
l_merge = Merge(mode='concat', concat_axis=1)(convolutions)
l_cov1= Conv1D(128, 5, activation='relu')(l_merge)
l_pool1 = MaxPooling1D(5)(l_cov1)
l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)
l_pool2 = MaxPooling1D(30)(l_cov2)
l_flat = Flatten()(l_pool2)
l_dense = Dense(128, activation='relu')(l_flat)
preds = Dense(2, activation='softmax')(l_dense)
                    
model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'])
                    
print("model fitting summary: CNN with multiple filters of different window sizes")
model.summary()
model.fit(x_train, y_train, validation_data=(x_test, y_test),epochs=20, batch_size=64)
