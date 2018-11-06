# -*- coding: utf-8 -*-
"""
Created on Fri May  4 13:51:55 2018

@author: anaid
"""
from __future__ import print_function
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from sklearn.metrics import accuracy_score
import numpy as np
from datetime import datetime 
import pandas as pd
from nltk.tokenize import word_tokenize 
import re
from sklearn.model_selection import train_test_split, cross_validate
import matplotlib.pyplot as plt
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
import tensorflow as tf
import pydot
import numpy as np
from autocorrect import spell
import nltk
from keras.preprocessing import text, sequence
from nltk.corpus import stopwords

#import tokeniser
from os import listdir
from os.path import isfile, join, splitext, split
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import *
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
import string
from nltk.corpus import stopwords
from classifiers import train_classifier1, train_classifier2, evaluate_classifier
from nlpB import load_file, get_train_test, bag_of_words, load_bow, save_bow
from nlpB import embeddings_matrix_glove, save_embedding_matrix_x, load_embedding_matrix_x, save_file

dfBow = pd.DataFrame()
dfEmb = pd.DataFrame()
start = datetime.now()
print("Staring-> ",start) 
n = 1
rep = 1#Number of retition
for k in range(0, rep):
    ###############Load dataset#################
    dataset = 'amazon'
    #dataset ='imdb'
    #dataset = 'yelp'    
    X, y = load_file(dataset)
    textReviews = X
    ############################################
    print("--------Bag Of Words-------------")
    #Pre-processing, processing of the text,Feature extraction using Bag of words
    #When I carried out the experiment I used that bag of words that it was saved, because this process take long time
    bowReview = bag_of_words(X)
    save_bow(bowReview,dataset)
    bowReview = load_bow(dataset)
    #Split dataset: training and testing 
    x_train, x_test, y_train, y_test, textReviews_train, textReviews_test, textReviews_train, textReviews_test = get_train_test(textReviews=textReviews,X= bowReview, y=y, test_size=0.10)
#    print("x_train.shape", x_train.shape)
#    print("x_test.shape", x_test.shape)    
    #Classifier(Nnet) using the bag of words
    classifier1, history1, acc_train1, acc_test1 = train_classifier1(x_train, x_test, y_train, y_test)
    #Get accuracy, precsion and recall
    acc1, y_predict1 = evaluate_classifier(classifier1, x_test, y_test)
    recall1 = recall_score(y_test, y_predict1) * 100
    precision1 = precision_score(y_test, y_predict1) * 100
    data = pd.DataFrame({'acc': [acc1], 'recall': [recall1], 'precision': [precision1]}) 
    dfBow = dfBow.append(data, ignore_index=True)
           
    ###########################################
    print("--------Embedding-------------")
    #Pre-processing, processing of the text,Feature extraction using embedding: GloVe
    #Get embedding matrix and convert reviews to numbers 
    #When I carried out the experiment I used that embedding that it was saved, because this process take long time
    embedding_matrix, x = embeddings_matrix_glove(textReviews, y) 
    save_embedding_matrix_x(embedding_matrix, x, dataset)
    embedding_matrix, x = load_embedding_matrix_x(dataset)
    print("embedding saved")
    #Split dataset: training and testing 
    x_train, x_test, y_train, y_test, textReviews_train, textReviews_test, textReviews_train, textReviews_test = get_train_test(textReviews=textReviews,X= x, y=y, test_size=0.10)
    #Classifier(CNN) using the embedding matrix 
    classifier2, history2, acc_train2, acc_test2 = train_classifier2(embedding_matrix, x_train, x_test, y_train, y_test)
    acc2, y_predict2 = evaluate_classifier(classifier2, x_test, y_test)
    recall2 = recall_score(y_test, y_predict1) * 100
    precision2 = precision_score(y_test, y_predict1) * 100
    data = pd.DataFrame({'acc': [acc2], 'recall': [recall2], 'precision': [precision2]}) 
    dfEmb = dfEmb.append(data, ignore_index=True)
    #print("dfBow", dfBow)
    #print("dfEmb", dfEmb)
    
#save the performace    
save_file(dfBow,'Experiment'+str(n)+'BoW.xlsx')    
dfmeanB = dfBow.mean()
dfmeanB = dfmeanB.add_suffix('_mean').reset_index()
save_file(dfmeanB, 'Experiment'+str(n)+'BoWMean.xlsx')    
    
save_file(dfEmb,'Experiment'+str(n)+'Embedding.xlsx')    
dfmeanE = dfEmb.mean()
dfmeanE = dfmeanE.add_suffix('_mean').reset_index()
save_file(dfmeanE, 'Experiment'+str(n)+'EmbeddingMean.xlsx')

#print("Accuracy BoW", dfmeanB)
#print("Accuracy Embedding", dfmeanE)

print("Accuracy BoW: ", acc1)
print("Accuracy Embedding: ", acc2)

end = datetime.now() - start
print("Duration", end)
print(datetime.now())

