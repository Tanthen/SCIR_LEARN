# -*- coding: utf-8 -*-
"""
    the emotion GRU of the DialogueRNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmotionLayer(nn.Module):

    def __init__(self, input_size, hidden_size, Dl, c):
        super(EmotionLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Dl = Dl
        self.c = c
        self.gru_e = nn.GRUCell(input_size=self.input_size, hidden_size=self.hidden_size)
        self.l1 = nn.Linear(in_features=self.hidden_size, out_features=self.Dl)
        self.l2 = nn.Linear(in_features=self.Dl, out_features=self.c)

    def forward(self, e_l, q):
        e = self.gru_e(q, e_l)
        lt = F.relu(self.l1(e))
        return F.softmax(self.l2(lt), dim=-1)
