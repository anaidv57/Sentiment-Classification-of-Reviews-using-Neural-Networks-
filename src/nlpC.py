# -*- coding: utf-8 -*-
"""
Created on Fri May  4 11:44:55 2018

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
import nltk
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
from classifiers import train_classifier1, train_classifier2, evaluate_classifier
#import tokeniser
from os import listdir
from os.path import isfile, join, splitext, split
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import *
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
import string
from nltk.corpus import stopwords

count_spelling = 0
incorrect_words = []
count_words = 0
dfbag = pd.DataFrame() 

#save files of performance
def save_file(df, namef):
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(namef, engine='xlsxwriter')    
    # Convert the dataframe to an XlsxWriter Excel object.
    df.to_excel(writer, sheet_name='Sheet1')    
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()

#Custom tokenizer to get emoticons
def custom_tokenise(text):
    word0 = '[^A-Za-z0-9 ]{2,3}' #get emoticons
    word1 = '\w+'#words
    word4 = '\'[a-z]{1,2}'#n't    
    patterns = (word0, word4, word1)
    joint_patterns = '|'.join(patterns)
    p = re.compile(r'(%s)' % joint_patterns)
    return p.findall(text)

# Removes punctuation, parentheses, question marks, etc., and leaves only alphanumeric characters
def cleanSentences(string):
#taken from https://www.oreilly.com/learning/perform-sentiment-analysis-with-lstms-using-tensorflow
    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())

#Auto-correct spelling
def autocorrect_spell(word):
    global count_spelling, count_words 
    count_words = count_words + 1
    if re.match("[^A-Za-z0-9]+", word):
        if word == '\'t' :
            word = 'n\'t'
            return spell(word).lower()
        else:
            return word 
    else:       
        if word == '\'t' :
            word = 'n\'t'
            #print("n't")
        new_string = spell(word).lower()        
        if new_string != word:
            count_spelling = count_spelling + 1
            incorrect_words.append(new_string +','+ word)            
        return new_string 
        
#Auto-correct spelling of tokens
def autocorrect_spell_tokens(tokens): #yes
    global dfbag
    for word in tokens:
        data = pd.DataFrame({'token': [word]})      
        dfbag = dfbag.append(data, ignore_index=True)       
        word = autocorrect_spell(word)   
    return tokens

def nltk_tokenise(text):#yes
    return nltk.word_tokenize(text)

def nltk_tokenise_autorrect(text):  
    return autocorrect_spell_tokens(nltk_tokenise(text))

def cust_tokenise_autorrect(text):  #yes
    return autocorrect_spell_tokens(custom_tokenise(text))

#Read dataset: reviews of amazon products, movies, and restaurants. 
def load_file(dataset= 'imdb.txt'):    
    print("Reading file->", dataset )
    directory = "D:\Box Sync\Dataset\\"
    data = pd.read_csv(directory+dataset+'.txt', header=None, sep=r"\t", engine='python')#Score is either 1 (for positive) or 0 (for negative)
    data.columns = ['review','sentiment']
    y = data['sentiment'] #labels
    X = data['review']
    return X, y

#Load GloVe: Global Vectors for Word Representation that is a pre-trained word vectors
def load_glove(): 
#http://www.orbifold.net/default/2017/01/10/embedding-and-tokenizer-in-keras/
#glove.6B.300d.txt was download from https://nlp.stanford.edu/projects/glove/
    print("Loading embeddings_index")
    embeddings_index = {}
    glove_data = 'glove.6B.300d.txt'
    f = open(glove_data, encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        value = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = value
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index)) 
    return embeddings_index

#Bag of words using Sklearn library
#https://medium.com/tensorist/classifying-yelp-reviews-using-nltk-and-scikit-learn-c58e71e962d9
#https://machinelearningmastery.com/prepare-text-data-machine-learning-scikit-learn/ 
def bag_of_words(X):    
    #bow_transformer = CountVectorizer(analyzer='word',  lowercase='True', tokenizer=cust_tokenise_autorrect, max_features=300).fit(X) 
    #bow_transformer = CountVectorizer(analyzer='word',  lowercase='True', tokenizer=nltk_tokenise_autorrect, max_features=300).fit(X)    
    bow_transformer = CountVectorizer(analyzer='word',  lowercase='True', stop_words=list(string.punctuation), tokenizer=nltk_tokenise_autorrect, max_features=300).fit(X)       
    save_file(dfbag, 'tokens.xlsx')
    #print(len(bow_transformer.vocabulary_))
    X_25 = X[1]
    #print(X_25)
    # tokenize and build vocab
    bow_transformer.fit(X)    
    #transform our dataframe into a sparse matrix using transform
    bow = bow_transformer.transform(X)
    print("bow.shape", bow.shape)
    print("Review: ", X[1])
    print("Review BOWs", bow[1].toarray())    
    #print('Shape of Sparse Matrix: ', bow.shape)
    #print('Amount of Non-Zero occurrences: ', bow.nnz)    
    bowReview = bow.toarray()
#    print(bowReview)
    return bowReview

def embeddings_matrix_glove(X, y):
#base on http://www.orbifold.net/default/2017/01/10/embedding-and-tokenizer-in-keras/ 
#https://www.oreilly.com/learning/perform-sentiment-analysis-with-lstms-using-tensorflow
    embeddings_index = load_glove()
    df = pd.DataFrame() 
    embedding_dimension = 50    
    list_tokens = []
    xreviews = []
    unique_words = []
    #Creating list of words
    for row in X:  
        row = cleanSentences(row)
        token = nltk_tokenise(row)
        #token = custom_tokenise(row)
        xreviews.append(token)#tokens for each review
        for word in token:
            i = 0
            word = autocorrect_spell(word)
            list_tokens.append(word.lower()) 
    
    #print("All tokens", len(list_tokens)) 
    unique_tokens = set(list_tokens)
    #print("Unique tokens in this dataset ", len(unique_tokens))
    embedding_matrix = np.zeros((len(unique_tokens), embedding_dimension)) 
    i = 0
    cont_error = 0
    cont_correct = 0
    #Create the embedding_matrix 
    print("Creating embeddings matrix glove")
    for word in unique_tokens:
        word = autocorrect_spell(word)
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector[:embedding_dimension]
            cont_correct = cont_correct + 1
        else:
           cont_error = cont_error + 1 
        unique_words.append(word)
        i = i + 1
    print("embedding_matrix.shape: ", embedding_matrix.shape)
#    print("cont_error: ", cont_error)
#    print("cont_correct: ", cont_correct)
#    print("embedding_matrix[1]", embedding_matrix[1])
    print("Review: ", X[1])
    print("x1", xreviews[1])   
    words_index = []
    x = []
    cont_error = 0
    cont_correct = 0
    #Convert words to numbers using list of words in each review
    for row in xreviews:        
        for word in row:
            word = autocorrect_spell(word)
#            word = word.lower()
            data = pd.DataFrame({'token': [word]})      
            df = df.append(data, ignore_index=True)
            try:
                unique_words.index(word)
                words_index.append(unique_words.index(word))
                cont_correct = cont_correct + 1
            except (ValueError) as e:    
                cont_error = cont_error + 1
                #print(word)     
        x.append(words_index)#save in a file
        words_index = []
    save_file(df, 'tokensEmb.xlsx')       
    print("Reviews token: ", x[1])
    print("x len", len(x))
    #print("cont_error: ", cont_error)
    #print("cont_correct: ", cont_correct)
    x = pad_sequences(x, maxlen=50)
    
    return embedding_matrix, x 

#Save embedding matrix and reviews converted in numbers
def save_embedding_matrix_x(embedding_matrix, x, dataset):
    emb_name = 'embed_mtrx_' + dataset +'.npy'
    x_name = 'x_embed_' + dataset +'.npy'
    np.save(emb_name, embedding_matrix)
    np.save(x_name, x)
    
#Load embedding matrix and reviews converted in numbers
def load_embedding_matrix_x(dataset):
    emb_name = 'embed_mtrx_' + dataset +'.npy'
    x_name = 'x_embed_' + dataset +'.npy'
    embedding_matrix = np.load(emb_name)
    x = np.load(x_name)   
    return embedding_matrix, x
    
#Save BOWs
def save_bow(bowReview,dataset):
    bow_name = 'bow_' + dataset +'.npy'
    np.save(bow_name, bowReview)
    
#Load embedding matrix and reviews converted in numbers
def load_bow(dataset):
    bow_name = 'bow_' + dataset +'.npy'  
    return np.load(bow_name) 
        
def frequency_analysis(tokens):
    freq = nltk.FreqDist(tokens) #produce frequency list by counting occurence of each token
    for key,val in freq.most_common(): #for each token with frequency, most_common() provides the tokens in frequency order, highest first.
        print(str(key) + ":" + str(val))
    freq.plot(25, cumulative=False)

def get_train_test(textReviews,X, y, test_size): 
    global count_spelling, count_words
    print("X", X.shape)
    print("y", y.shape)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify = y) 
    textReviews_train, textReviews_test, textReviews_train, textReviews_test = train_test_split(textReviews, y, test_size=test_size, stratify = y)
    x_test = np.asarray(x_test) 
    x_train = np.asarray(x_train) 
    y_test = np.asarray(y_test) 
    y_train = np.asarray(y_train) 
    #print("autocorrect_spel", count_spelling)
    #print("Total words", count_words)
    count_spelling = 0
    count_words = 0
    return x_train, x_test, y_train, y_test, textReviews_train, textReviews_test, textReviews_train, textReviews_test



