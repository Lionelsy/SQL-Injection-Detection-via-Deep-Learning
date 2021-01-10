# -*- coding: utf-8 -*-
# @Time    : 2020/12/14
# @Author  : Shuyu ZHANG
# @FileName: MLP.py
# @Software: PyCharm
# @Description: Here

import torch.nn as nn


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