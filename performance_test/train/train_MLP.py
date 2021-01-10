# -*- coding: utf-8 -*-
# @Time    : 2020/12/07
# @Author  : Shuyu ZHANG
# @FileName: train_MLP.py
# @Software: PyCharm
# @Description: Here


import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

import numpy as np
import pandas as pd

df = pd.read_csv("/home/shuyu/others/cs315/data/all.csv")
train = pd.read_csv("/home/shuyu/others/cs315/data/train.csv")
val = pd.read_csv("/home/shuyu/others/cs315/data/val.csv")
test = pd.read_csv("/home/shuyu/others/cs315/data/test.csv")

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer


temp = CountVectorizer(min_df=2, max_df=0.7, stop_words=stopwords.words('english'))
t1 = temp.fit_transform(df['Sentence'].values.astype('U')).toarray()
vocabulary = temp.get_feature_names()
vectorizer = CountVectorizer(min_df=2, max_df=0.7, stop_words=stopwords.words('english'), vocabulary=vocabulary)

X_train = vectorizer.fit_transform(train['Sentence'].values.astype('U')).toarray()
X_val = vectorizer.fit_transform(val['Sentence'].values.astype('U')).toarray()
X_test = vectorizer.fit_transform(test['Sentence'].values.astype('U')).toarray()

y_train = train['Label'].values
y_val = val['Label'].values
y_test = test['Label'].values

print('X_train: ', X_train.shape)
print('X_val: ', X_val.shape)
print('X_test: ', X_test.shape)

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support


def performence(y_test, y_pred):
    a = accuracy_score(y_test, y_pred)
    precision, recall, fscore, _ = \
        precision_recall_fscore_support(y_test, y_pred, average='weighted')
    print("Accuracy : {:.4f}".format(a))
    print("Precision : {:.4f}".format(precision))
    print("Recall : {:.4f}".format(recall))
    print("F-Score : {:.4f}".format(fscore))


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(108)

X_train = torch.from_numpy(X_train).cuda()
X_val = torch.from_numpy(X_val).cuda()
X_test = torch.from_numpy(X_test).cuda()

y_train = torch.from_numpy(y_train).cuda()
y_val = torch.from_numpy(y_val).cuda()
# y_test = torch.from_numpy(y_test)


class MLP(nn.Module):
    def __init__(self, in_dim=10319, dropout=0.0):
        super(MLP, self).__init__()
        self.dropout = dropout

        self.mlp = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=256, out_features=64),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(in_features=64, out_features=2),
        )

    def forward(self, x):
        return self.mlp(x)


in_dim = X_train.shape[1]
dropout = 0.01
lr = 6e-6

mlp = MLP(in_dim, dropout).cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(mlp.parameters(), lr=lr,
                      momentum=0.9, weight_decay=5e-4)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

epoch = 50
batch = 100

max_loss = 1e5

for i in range(0, epoch):
    mlp.train()
    for j in range(0, batch):
        X_in = X_train[j:j + batch]
        y_true = y_train[j:j + batch]
        pred = mlp(X_in.float())
        loss = criterion(pred, y_true)
        loss.backward()
        optimizer.step()
    mlp.eval()
    pred = mlp(X_val.float())
    loss = criterion(pred, y_val)
    lo = loss.item()
    if lo < max_loss:
        max_loss = lo
        dic = mlp.state_dict()
    print('Iter:{:d}, Loss:{:.2f}'.format(i, lo))

mlp.load_state_dict(dic)
pred = mlp(X_test.float())
prediction = torch.max(F.softmax(pred, dim=1), 1)[1].detach().cpu().numpy()
performence(y_test, prediction)
mlp.cpu()
# torch.save(mlp, '../model/torch_mlp.pt')

