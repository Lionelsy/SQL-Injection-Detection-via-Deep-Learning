# -*- coding: utf-8 -*-
# @Time    : 2020/12/14
# @Author  : Shuyu ZHANG
# @FileName: AE.py
# @Software: PyCharm
# @Description: Here

import torch.nn as nn


class AE(nn.Module):
    def __init__(self, in_dim=10319):
        super(AE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=256, out_features=64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=64, out_features=32),
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=32, out_features=64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=64, out_features=256),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=256, out_features=2),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.classifier(x)
        return x
