#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"


# # Read Data

# In[2]:


import numpy as np 
import pandas as pd 


# In[3]:


df = pd.read_csv("/home/shuyu/others/cs315/data/all.csv")
train = pd.read_csv("/home/shuyu/others/cs315/data/train.csv")
val = pd.read_csv("/home/shuyu/others/cs315/data/val.csv")
test = pd.read_csv("/home/shuyu/others/cs315/data/test.csv")


# In[4]:


import glob
import time
import pandas as pd

from nltk import ngrams
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import PorterStemmer
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize


# # Pre-Process

# In[5]:


from sklearn.feature_extraction.text import CountVectorizer


# In[6]:


temp = CountVectorizer(min_df=2, max_df=0.7, stop_words=stopwords.words('english'))
t1 = temp.fit_transform(df['Sentence'].values.astype('U')).toarray()
vocabulary = temp.get_feature_names()
vectorizer = CountVectorizer(min_df=2, max_df=0.7, stop_words=stopwords.words('english'), vocabulary=vocabulary)


# In[7]:


train


# In[8]:


X_train = vectorizer.fit_transform(train['Sentence'].values.astype('U')).toarray()
X_val = vectorizer.fit_transform(val['Sentence'].values.astype('U')).toarray()
X_test = vectorizer.fit_transform(test['Sentence'].values.astype('U')).toarray()


# In[9]:


y_train = train['Label'].values
y_val = val['Label'].values
y_test = test['Label'].values


# In[10]:


print('X_train: ', X_train.shape)
print('X_val: ', X_val.shape)
print('X_test: ', X_test.shape)


# # Experiement

# In[11]:


import joblib


# In[12]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

def performence(y_test, y_pred):
    a = accuracy_score(y_test, y_pred)
    precision, recall, fscore, _ =         precision_recall_fscore_support(y_test, y_pred, average='weighted')
    print("Accuracy : {:.4f}".format(a))
    print("Precision : {:.4f}".format(precision))
    print("Recall : {:.4f}".format(recall))
    print("F-Score : {:.4f}".format(fscore))


# ## Linear Regression

# In[13]:


from sklearn.linear_model import LogisticRegression


# In[14]:


clf = LogisticRegression(random_state=0).fit(X_train, y_train)


# In[15]:


y_pred=clf.predict(X_test)


# In[16]:


y_test.shape


# In[17]:


performence(y_test, y_pred)


# In[18]:


joblib.dump(clf, '../model/LR.pkl')


# In[ ]:





# ## SVM

# In[19]:


from sklearn.svm import SVC


# In[20]:


svm = SVC(gamma='auto')
svm.fit(X_train, y_train)
y_pred=svm.predict(X_test)


# In[21]:


performence(y_test, y_pred)


# In[22]:


joblib.dump(svm, '../model/SVM.pkl')


# In[ ]:





# ## KNN

# In[23]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


neigh = KNeighborsClassifier(n_neighbors=2)
neigh.fit(X_train, y_train)
y_pred = neigh.predict(X_test)


# In[ ]:


performence(y_test, y_pred)


# In[ ]:


joblib.dump(neigh, '../model/KNN.pkl')


# In[ ]:





# ## Decision Tree

# In[ ]:


from sklearn import tree


# In[ ]:


dt = tree.DecisionTreeClassifier()
dt = dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)


# In[ ]:


performence(y_test, y_pred)


# In[ ]:


joblib.dump(dt, '../model/DT.pkl')


# In[ ]:




