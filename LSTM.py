# -*- coding: utf-8 -*-
# @Time    : 2020/12/14
# @Author  : Shuyu ZHANG
# @FileName: LSTM.py
# @Software: PyCharm
# @Description: Here

import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, in_dim=10319, dropout=0.0):
        super(LSTM, self).__init__()
        self.dropout = dropout

        self.lstm = nn.LSTM(in_dim, 512, batch_first=True)

        self.mlp = nn.Sequential(
            nn.Linear(in_features=512, out_features=64),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(in_features=64, out_features=2),
        )

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        ou, _ = self.lstm(x)
        x = torch.squeeze(ou, 1)
        return self.mlp(x)