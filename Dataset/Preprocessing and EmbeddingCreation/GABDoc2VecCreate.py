#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pickle
import os
import time
from os import listdir
import sys
import json
import gzip
from collections import defaultdict, Counter
import multiprocessing
import time
import threading
from operator import itemgetter
from random import shuffle
import pandas as pd
import numpy as np
import re
from nltk.stem import *
from nltk.stem.snowball import SnowballStemmer
import random
stemmer = SnowballStemmer("english")
import datetime
import json
from gensim.models.doc2vec import Doc2Vec, TaggedDocument



with open('gab_user_Sentenses_proc.json', 'r') as fp:
    user_sentences = json.load(fp)

chosen_users={}
for user in user_sentences:
    sentences=[]
    for sentence in user_sentences[user]:
        if sentence!=[]:
            sentences.extend(sentence)
    chosen_users[user]= list(sentences)



print("StartDoc2Vec")
documents=[TaggedDocument(doc,[i]) for i,doc in enumerate([chosen_users[user] for user in chosen_users])]
model= Doc2Vec(documents, vector_size=100, window=5, min_count=3, workers=20)

print("Start Dumping")
user_vectors={}
for user in chosen_users:
        user_vectors[user]=model.infer_vector(chosen_users[user])

dataDirectory = '../Dataset/GabData/'
with open(dataDirectory + 'GABDoc2vec100.p', 'wb') as handle:
        pickle.dump(user_vectors, handle)

model.save("GAB_doc2vec.model")


with open('twitter_user_Sentenses_proc.json', 'r') as fp:
    twitter = json.load(fp)


chosen_users={}
for user in twitter:
    sentences=[]
    for sentence in twitter[user]:
        if sentence!=[]:
            sentences.extend(sentence)
    chosen_users[user]= list(sentences)


for user in chosen_users:
    if chosen_users[user] ==[]:
        print(user)
        chosen_users[user].append('')


print("Start Dumping")
user_vectors={}
for user in chosen_users:
        user_vectors[user]=model.infer_vector(chosen_users[user])

dataDirectory = '../Dataset/TwitterData/'

with open(dataDirectory +'GABTrain_twitterDoc2vec_100.p', 'wb') as handle:
        pickle.dump(user_vectors, handle)
