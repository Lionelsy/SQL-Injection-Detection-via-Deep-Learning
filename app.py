from flask import Flask, render_template, request, jsonify
from util import model_predict, trans_to_vector, predict
import pandas as pd 

from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from MLP import *
from AE import *
from LSTM import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
app = Flask(__name__)
vocabulary = []
model = []

df = pd.read_csv("data/all.csv")
vectorizer = CountVectorizer( min_df=2, max_df=0.7, stop_words=stopwords.words('english'))
posts = vectorizer.fit_transform(df['Sentence'].values.astype('U')).toarray()

vocabulary = vectorizer.get_feature_names()
print(torch.__version__)

model_mlp = torch.load('model/torch_mlp.pt')
model_AE = torch.load('model/torch_ae.pt')
model_lstm = torch.load('model/torch_lstm.pt')
model_LR = joblib.load('model/LR.pkl')

@app.route('/', methods=['GET', 'POST'])
def open():
    return render_template('index.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    command = request.values.get('command')
    model_name = request.values.get('type')
    print(model_name)
    df = pd.DataFrame([{"Sentence": command}])
    posts = trans_to_vector(df, vocabulary)
    if model_name == 'LR':
        pred = predict(posts, model_LR)
    elif model_name == 'AE':
        pred = model_predict(posts, model_AE)
    elif model_name == 'MLP':
        pred = model_predict(posts, model_mlp)
    elif model_name == 'LSTM':
        pred = model_predict(posts, model_lstm)

    result = False
    if pred == 1:
        result = True

    return jsonify({"result": result})


if __name__ == '__main__':
    app.run(host='10.20.84.191', port=8950)
