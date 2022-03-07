# -*- coding: utf-8 -*-
"""
    the global GRU of the DialogueRNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRU_G(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(GRU_G, self).__init__()
        self.gru = nn.GRUCell(input_size=input_size, hidden_size=hidden_size)

    def forward(self, g_l, utter, q):
        # g_l:batch_size, hidden_size
        # utter+q:batch_size, utter_embedding + q_hidden
        g_plus = self.gru(torch.cat((utter, q), -1), g_l)
        return g_plus
