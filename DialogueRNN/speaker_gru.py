# -*- coding: utf-8 -*-
"""
    the speaker GRU of the DialogueRNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRU_S(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(GRU_S, self).__init__()
        self.gru = nn.GRUCell(input_size=input_size, hidden_size=hidden_size)

    def forward(self, q_l, utter, c):
        # q_l:batch_size, hidden_size
        # utter+c:batch_size, utter_embedding+Dg
        q_plus = self.gru(torch.cat((utter, c), -1), q_l)
        return q_plus
