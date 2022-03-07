# -*- coding: utf-8 -*-
"""
    load the dataset
"""
import torch
from torch.utils.data import Dataset

import pickle
import sys


class DataLoader(Dataset):

    def __init__(self, path, train=True):
        self.ids, self.speakers, self.labels, self.text, _, _, self.sentence, \
        self.trainvid, self.testvid = pickle.load(open(path, 'rb'), encoding='latin1')
        # self.speakers = [[1 if ch == 'M' else 0 for ch in speaker] for speaker in self.speakers]
        for key in self.speakers.keys():
            self.speakers[key] = [1 if ch == 'M' else 0 for ch in self.speakers.get(key)]
        # labels: 0:hap, 1:sad, 2:neu, 3:ang, 4:exc, 5:fru
        self.keys = [x for x in (self.trainvid if train else self.testvid)]
        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.Tensor(self.text.get(vid)), \
               torch.Tensor(self.speakers.get(vid)), \
               torch.LongTensor(self.labels.get(vid))

    def __len__(self):
        return self.len


if __name__ == '__main__':
    f = "./DialogueRNN_features/DialogueRNN_features/IEMOCAP_features/IEMOCAP_features_raw"
    ids, speakers, labels, text, audio, visual, sentence, trainvid, testvid \
        = pickle.load(open(f, 'rb'), encoding='latin1')
    print(torch.FloatTensor(text.get('Ses03M_impro08b')))
    print(ids.keys())
    print(trainvid)
    condition1 = text.get('Ses03M_impro08b')
    condition2 = text.get('Ses05M_impro02')
    print(speakers.get('Ses02M_impro03'))
    print(labels.get('Ses02M_impro03'))
    print(len(condition1))
    print(len(condition1[0]))
    print(len(condition2))
    print(len(condition2[0]))
    print(text.get('Ses03M_impro08b'))
