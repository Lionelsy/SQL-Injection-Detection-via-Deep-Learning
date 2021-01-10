from flask import Flask, request

import json

import joblib
import numpy as np 
import pandas as pd
import torch
import torch.nn.functional as F


from torch_deploy.MLP import MLP
from torch_deploy.LSTM import LSTM
from torch_deploy.AE import AE

from nltk.corpus import stopwords 
from sklearn.feature_extraction.text import CountVectorizer


app = Flask(__name__)

df = pd.read_csv("/home/shuyu/others/cs315/data/all.csv")
temp = CountVectorizer(min_df=2, max_df=0.7, stop_words=stopwords.words('english'))
t1 = temp.fit_transform(df['Sentence'].values.astype('U')).toarray()
vocabulary = temp.get_feature_names()
vectorizer = CountVectorizer(min_df=2, max_df=0.7, stop_words=stopwords.words('english'), vocabulary=vocabulary)
lr_clf = joblib.load('../model/LR.pkl')

svm_clf = joblib.load('../model/SVM.pkl')

knn_clf = joblib.load('../model/KNN.pkl')

dt_clf = joblib.load('../model/DT.pkl')

torch_mlp = torch.load('../model/torch_mlp.pt')
torch_mlp.eval()

torch_lstm = torch.load('../model/torch_lstm.pt')
torch_lstm.eval()

torch_ae = torch.load('../model/torch_ae.pt')
torch_ae.eval()


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/demo')
def demo():
    return 'Here is Demo Page!'


@app.route('/predict', methods=['POST'])
def predict():
    ask = json.loads(request.json)
    ask_arr = vectorizer.fit_transform(ask).toarray()
    res = [i for i in range(len(ask))]
    return json.dumps(res)


@app.route('/lr', methods=['POST'])
def lr_predict():
    ask = json.loads(request.json)
    ask_arr = vectorizer.fit_transform(ask).toarray()
    res = lr_clf.predict(ask_arr)
    return json.dumps(res.tolist())


@app.route('/svm', methods=['POST'])
def svm_predict():
    ask = json.loads(request.json)
    ask_arr = vectorizer.fit_transform(ask).toarray()
    res = svm_clf.predict(ask_arr)
    return json.dumps(res.tolist())


@app.route('/knn', methods=['POST'])
def knn_predict():
    ask = json.loads(request.json)
    ask_arr = vectorizer.fit_transform(ask).toarray()
    res = knn_clf.predict(ask_arr)
    return json.dumps(res.tolist())


@app.route('/dt', methods=['POST'])
def dt_predict():
    ask = json.loads(request.json)
    ask_arr = vectorizer.fit_transform(ask).toarray()
    res = dt_clf.predict(ask_arr)
    return json.dumps(res.tolist())


@app.route('/mlp', methods=['POST'])
def mlp_predict():
    ask = json.loads(request.json)
    ask_arr = vectorizer.fit_transform(ask).toarray()
    ask_arr = torch.from_numpy(ask_arr).float()
    pred = torch_mlp(ask_arr)
    res = torch.max(F.softmax(pred, dim=1), 1)[1].detach().numpy()
    return json.dumps(res.tolist())


@app.route('/lstm', methods=['POST'])
def lstm_predict():
    ask = json.loads(request.json)
    ask_arr = vectorizer.fit_transform(ask).toarray()
    ask_arr = torch.from_numpy(ask_arr).float()
    pred = torch_lstm(ask_arr)
    res = torch.max(F.softmax(pred, dim=1), 1)[1].detach().numpy()
    return json.dumps(res.tolist())


@app.route('/ae', methods=['POST'])
def ae_predict():
    ask = json.loads(request.json)
    ask_arr = vectorizer.fit_transform(ask).toarray()
    ask_arr = torch.from_numpy(ask_arr).float()
    pred = torch_ae(ask_arr)
    res = torch.max(F.softmax(pred, dim=1), 1)[1].detach().numpy()
    return json.dumps(res.tolist())


if __name__ == '__main__':
    app.run(host='10.20.84.191', port=8950, debug=True)

