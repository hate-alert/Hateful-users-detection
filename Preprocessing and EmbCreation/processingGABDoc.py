import pickle
import os
import gzip
import time
from os import listdir
from os.path import isfile, join
import sys
import json
import gzip
from collections import defaultdict, Counter
import joblib
#from klepto.archives import dir_archive
import multiprocessing
import time
import pickle
import threading
from operator import itemgetter
#import igraph as ig
from random import shuffle
import pandas as pd
import networkx as nx
import numpy as np
import re
from nltk.stem import *
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
import stop_words


stop_words=stop_words.get_stop_words('en')
stop_words_2=['i','me','we','us','you','u','she','her','his','he','him','it','they','them','who','which','whom','whose','that','this','these','those','anyone','someone','some','all','most','himself','herself','myself','itself','hers','ours','yours','theirs','to','in','at','for','from','etc',' ',',']
stop_words.extend(stop_words_2)
stop_words.extend(['with', 'at', 'from', 'into', 'during', 'including', 'until', 'against', 'among', 'throughout', 'despite', 'towards', 'upon', 'concerning', 'of', 'to', 'in', 'for', 'on', 'by', 'about', 'like', 'through', 'over', 'before', 'between', 'after', 'since', 'without', 'under', 'within', 'along', 'following', 'across', 'behind', 'beyond', 'plus', 'except', 'but', 'up', 'out', 'around', 'down', 'off', 'above', 'near', 'and', 'or', 'but', 'nor', 'so', 'for', 'yet', 'after', 'although', 'as', 'as', 'if', 'long', 'because', 'before', 'even', 'if', 'even though', 'once', 'since', 'so', 'that', 'though', 'till', 'unless', 'until', 'what', 'when', 'whenever', 'wherever', 'whether', 'while', 'the', 'a', 'an', 'this', 'that', 'these', 'those', 'my', 'yours', 'his', 'her', 'its', 'ours', 'their', 'few', 'many', 'little', 'much', 'many', 'lot', 'most', 'some', 'any', 'enough', 'all', 'both', 'half', 'either', 'neither', 'each', 'every', 'other', 'another', 'such', 'what', 'rather', 'quite'])
stop_words=list(set(stop_words))
stopword_file=open("stopword.txt",'r')
stop_words.extend([line.rstrip() for line in stopword_file])
import preprocess
import re
only_words='[^a-z0-9\' ]+'


dataDirectory = '../Dataset/GabData/'
with gzip.open(dataDirectory+'user_Sentenses_Raw.pklz','rb') as fp:
    dict_user = pickle.load(fp)

users_sentences={}

for user in  dict_user:
    users_sentences[user] =[]
    for post_text in dict_user[user]['sentences']:
        text= preprocess.tweet_preprocess2(post_text)
        text=re.sub(only_words,'',text.lower())
        users_sentences[user].append([stemmer.stem(elem) for elem in text.split(' ') if elem not in stop_words])
    #users_sentences[user] = procPosts

with open('gab_user_Sentenses_proc.json', 'w') as f:
    json.dump(users_sentences,f)
    
print("Done")
