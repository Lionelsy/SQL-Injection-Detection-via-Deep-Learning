import glob
import time
import pandas as pd
# from xml.dom import minidom
from nltk import ngrams
from nltk.tokenize import sent_tokenize
import nltk

import ssl
# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context

# nltk.download()

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
from nltk.stem import PorterStemmer
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import accuracy_score


def trans_to_vector(df, vocabulary):
    vectorizer = CountVectorizer( min_df=2, max_df=0.7, stop_words=stopwords.words('english'), vocabulary=vocabulary)
    posts = vectorizer.fit_transform(df['Sentence'].values.astype('U')).toarray()

    return posts


def model_predict(posts: np.ndarray, model):
    model.eval()
    #pred=model.predict(posts)
    print(posts.shape)
    posts = torch.from_numpy(posts)
    pred = model(posts.float())                                                      
    prediction = torch.max(F.softmax(pred, dim=1), 1)[1].detach().cpu().numpy()
    
    return prediction

def predict(posts: np.ndarray, model):
    pred = model.predict(posts)
    return pred





