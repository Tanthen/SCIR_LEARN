#!/bin/bash

if [ "$1" = "train" ]; then
  CUDA_VISIBLE_DEVICES=0 python run.py train \
  --dataset_path=DialogueRNN_features/DialogueRNN_features/IEMOCAP_features/IEMOCAP_features_raw \
  --dev-tgt=./en_es_data/dev.en --vocab=vocab.json --cuda
fi