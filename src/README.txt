----README----
Download glove.6B.zip from https://nlp.stanford.edu/projects/glove/ 
I am using this glove.6B.300d.txt
Put this file glove.6B.300d.txt in Code folder

There are 3 programs:
- nlp.py: this program is used to read the data, clean the data, create tokens, create BoW, load the Global Vectors for Word Representation and other.
- classifiers.py: this has the code of the NN and CNN using Keras library.
- main.py: it is the main program.

I created 3 copies of nlp.py, because It is going to be eassier for you to run each group of experiments:
- nplA.py: Group of experiment A (With punctuations + Customize tokenizer)
- nplB.py: Group of experiment B (With punctuations + nltk.word_tokenize function) 
- nplC.py: Group of experiment C (Without punctuation + nltk.word_tokenize function)
Each group of experiments takes around 5 minutes.

Steps:
1) Set the directory of the dataset on nlpA, nlpB, nlpC -> line 127. 
2) Set a dataset in main.py -> Between lines 67 and 69. It is amazon right now.
3) Set the nlp (nplA or nplB or nplC) program that you want to run in main.py -> Line 56 and 57. It nplB right now.
4) Execute the main program. This is going to execute the classifications.

Download glove.6B.zip. Put this glove.6B.300d in code folder
You should install those libraries: keras, spell, sklearn, nltk, tensorflow, numpy, pandas, datetime, pydot.











