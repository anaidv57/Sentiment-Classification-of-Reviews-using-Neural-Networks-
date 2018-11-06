
from __future__ import print_function
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D, LSTM, MaxPooling1D
from keras.datasets import imdb
from keras.utils.vis_utils import plot_model
from IPython.display import Image
from sklearn.metrics import accuracy_score
import numpy as np
from datetime import datetime 
import pandas as pd
from nltk.tokenize import word_tokenize 
import re
from sklearn.model_selection import train_test_split, cross_validate
import matplotlib.pyplot as plt
from numpy.random import seed
#seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
import tensorflow as tf
#import pydot
import numpy as np


def evaluate_classifier(model, x_test, y_test):
    scores = model.evaluate(x=x_test, y=y_test, batch_size=5)
    acc = scores[1]*100
    y_predict = model.predict_classes(x_test)
    #print("y_test", y_test)    
    return acc, y_predict


def train_classifier1(x_train, x_test, y_train, y_test, previous_weight=False):#bag of words
    #example https://cloud.google.com/blog/big-data/2017/10/intro-to-text-classification-with-keras-automatically-tagging-stack-overflow-posts
    print("classifier 1")
    batch_size = 32
    epochs = 2
    model = Sequential()    
    model.add(Dense(300, input_shape=(300,)))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))    
    model.compile(loss='binary_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])
    
    history = model.fit(x_train, y_train, 
                        batch_size=batch_size, 
                        epochs=epochs,  
                        validation_data=(x_test, y_test))
    
    acc_train = history.history['acc'][len(history.history['acc'])-1]*100#accuracy training 
    acc_test = history.history['val_acc'][len(history.history['val_acc'])-1]*100#accuracy testing
    print("Training accuracy", acc_train)
    print("Testing accuracy", acc_test) 
    #model.summary()

    return model, history, acc_train, acc_test

#base on Keras examples [https://github.com/keras-team/keras/tree/master/examples]
def train_classifier2(embedding_matrix, x_train, x_test, y_train, y_test, previous_weight=False): #embedding
    # set parameters:
    maxlen = 50
    batch_size = 50
    embedding_dims = 50
    filters = 250
    kernel_size = 3
    hidden_dims = 250
    epochs = 13  #5
    
    print('---------------Build Classifier 2...')
    print("embedding_matrix", embedding_matrix.shape)
    model = Sequential()    
    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    model.add(Embedding(embedding_matrix.shape[0],#1879
                        embedding_matrix.shape[1],
#                        embedding_dims,#10
                        weights=[embedding_matrix],
                        input_length=maxlen))
    model.add(Dropout(0.5))    
    # we add a Convolution1D, which will learn filters
    # word group filters of size filter_length:
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
    # we use max pooling:
    model.add(GlobalMaxPooling1D())   
    # We add a vanilla hidden layer:
    #model.add(Dense(hidden_dims))
    model.add(Dense(hidden_dims, kernel_initializer='random_uniform', bias_initializer='zeros'))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))    
    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(1, kernel_initializer='random_uniform', bias_initializer='zeros'))
    #model.add(Dense(1))
    model.add(Activation('sigmoid'))    
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy']) 
    history = model.fit(x_train, y_train, 
                        batch_size=batch_size, 
                        epochs=epochs,  
                        validation_data=(x_test, y_test))      
    acc_train = history.history['acc'][len(history.history['acc'])-1]*100#accuracy training 
    acc_test = history.history['val_acc'][len(history.history['val_acc'])-1]*100#accuracy testing
    print("Training accuracy", acc_train)
    print("Testing accuracy", acc_test) 


#    plot_model(model, to_file='model.png')
    return model, history, acc_train, acc_test

