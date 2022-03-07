# -*- coding: utf-8 -*-
"""
    the listener GRU of the DialogueRNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRU_L(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(GRU_L, self).__init__()

    def forward(self, q_l, utter, c):
        # q_l:layers, batch_size, hidden_size
        # utter+c: utter_number, batch_size, utter_embedding
        # pass the listener party state
        return q_l
