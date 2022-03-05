# -*- coding: utf-8 -*-
"""
    load the dataset
"""
import torch

import pickle
import sys

if __name__ == '__main__':
    f = "./DialogueRNN_features/DialogueRNN_features/IEMOCAP_features/IEMOCAP_features_raw"
    a = pickle.load(open(f, 'rb'), encoding='latin1')
    print(a[4])