# -*- coding: utf-8 -*-
"""
    the main model of the DialogueRNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from global_gru import GRU_G
from speaker_gru import GRU_S
from listen_gru import GRU_L
from emotion_layer import EmotionLayer


class DialogueRNN(nn.Module):

    def __init__(self, embedding_size=100, batch_size=1, Dg=50, Dp=50, Dep=50, lay=50, c=6):
        super(DialogueRNN, self).__init__()
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.Dg = Dg
        self.Dp = Dp
        self.Dep = Dep
        self.lay = lay
        self.c = c

        self.gru_g = GRU_G(input_size=self.embedding_size + self.Dp, hidden_size=self.Dg)
        self.W_a = nn.Parameter(torch.randn(self.embedding_size, Dg))
        self.gru_s = GRU_S(input_size=self.Dg + self.embedding_size, hidden_size=self.Dp)
        self.gru_l = GRU_L(input_size=self.embedding_size + self.Dg, hidden_size=self.Dp)
        self.e_layer = EmotionLayer(input_size=self.Dp, hidden_size=self.Dep, Dl=self.lay, c=self.c)

    def forward(self, utters, speakers):
        # utters: batch_size, dialogue_length, embedding : dialogue must to be equal
        # speakers: batch_size, dialogue_length

        utters = utters.permute(1, 0, 2)
        speakers = speakers.transpose(0, 1)
        ps1 = torch.randn(self.batch_size, self.Dp)
        ps2 = torch.randn(self.batch_size, self.Dp)  # batch_size, Dp
        g = torch.randn(self.batch_size, self.Dg)
        g_list = g.unsqueeze(1)  # (batch_size, 1(t), Dg)
        e = torch.randn(self.batch_size, self.Dep)
        emotions = []
        for utter, w_speaker in zip(utters, speakers):
            # utter: batch_size, embedding
            # speaker: batch_size

            # q = torch.Tensor([ps1[i] if speaker[i] == 'M' else ps2[i]  # 分段函数太糟糕
            #                   for i in range(len(speaker))])  # q : batch_size, Dp
            q = (w_speaker * ps1.transpose(0, 1) + (1 - w_speaker) * ps2.transpose(0, 1)).transpose(0, 1)

            g = self.gru_g(g, utter, q)
            g_list = torch.cat((g_list, g.unsqueeze(1)), dim=1)  # (batch_size, t, Dg)

            utter_s = utter.unsqueeze(1)  # batch_size, 1, embedding
            g_list_t = g_list.transpose(1, 2)  # batch_size, Dg, t
            a = F.softmax(torch.matmul(utter_s, torch.matmul(self.W_a, g_list_t)), dim=2)  # batch_size, 1, t
            ct = torch.matmul(a, g_list).squeeze(1)  # batch_size, 1, Dg -> batch_size, Dg
            ps1 = (w_speaker * self.gru_s(ps1, utter, ct).transpose(0, 1) +
                   (1 - w_speaker) * self.gru_l(ps1, utter, ct).transpose(0, 1)).transpose(0, 1)
            ps2 = (w_speaker * self.gru_s(ps2, utter, ct).transpose(0, 1) +
                   (1 - w_speaker) * self.gru_l(ps2, utter, ct).transpose(0, 1)).transpose(0, 1)

            ps = (w_speaker * ps1.transpose(0, 1) + (1 - w_speaker) * ps2.transpose(0, 1)).transpose(0, 1)
            emotion = self.e_layer(e, ps).unsqueeze(1)  # batch_size * c
            emotions.append(emotion)
        outputs = torch.cat(emotions, dim=1)  # batch_size, dialogue_length, c
        return outputs
