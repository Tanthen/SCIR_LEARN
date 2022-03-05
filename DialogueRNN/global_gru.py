# -*- coding: utf-8 -*-
"""
    the global GRU of the DialogueRNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GRU_G(nn.Module):

    def __init__(self, ):
        super(GRU_G, self).__init__()