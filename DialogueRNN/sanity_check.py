# -*- coding: utf-8 -*-
"""
sanity_check.py:sanity check for dialogueRNN
Usage:
    model shape check:
    -sanity_check.py 1a
    -sanity_check.py 1b
    -sanity_check.py 1c

    DialogueRNN shape check:
    -sanity_check.py 2a
"""
import torch
from global_gru import GRU_G
from speaker_gru import GRU_S
from emotion_layer import EmotionLayer
from dialogueRNN import DialogueRNN

def sanity_check_1a():
    print("-" * 80)
    print("Running Sanity Check for global GRU")
    print("-" * 80)
    batch_size = 20
    sent_length = 100
    g = 50
    q = 50
    sent = torch.zeros(batch_size, sent_length)
    q_state = torch.zeros(batch_size, q)
    g_state = torch.zeros(batch_size, g)
    gru_g = GRU_G(sent_length + q, g)
    output = gru_g.forward(g_state, sent, q_state)
    output_expected_state = [batch_size, g]
    assert (
        list(output.shape) == output_expected_state
    ), "output shape is incorret: it should be:\n {} but is:\n{}".format(
        output_expected_state, list(output.size())
    )
    print("Sanity Check Passed for global GRU!")
    print("-" * 80)


def sanity_check_1b():
    print("-" * 80)
    print("Running Sanity Check for speaker GRU")
    print("-" * 80)
    batch_size = 20
    sent_length = 100
    g = 50
    q = 50
    sent = torch.zeros(batch_size, sent_length)
    q_state = torch.zeros(batch_size, q)
    g_state = torch.zeros(batch_size, g)
    gru_g = GRU_S(sent_length + q, g)
    output = gru_g.forward(g_state, sent, q_state)
    output_expected_state = [batch_size, g]
    assert (
        list(output.shape) == output_expected_state
    ), "output shape is incorret: it should be:\n {} but is:\n{}".format(
        output_expected_state, list(output.size())
    )
    print("Sanity Check Passed for speaker GRU!")
    print("-" * 80)

def sanity_check_1c():
    print("-" * 80)
    print("Running Sanity Check for emotion layer")
    print("-" * 80)
    batch_size = 1
    e_dim = 50
    q_dim = 50
    Dl = 20
    c = 6

    e = torch.zeros(batch_size, e_dim)
    q = torch.zeros(batch_size, q_dim)

    emo = EmotionLayer(input_size=q_dim, hidden_size=e_dim, Dl=Dl, c=c)
    output = emo.forward(e, q)

    output_expected_state = [batch_size, c]
    assert (
            list(output.shape) == output_expected_state
    ), "output shape is incorret: it should be:\n {} but is:\n{}".format(
        output_expected_state, list(output.size())
    )
    print("Sanity Check Passed for emotion layer!")
    print("-" * 80)

def sanity_check_2a():
    print("-" * 80)
    print("Running Sanity Check for DialogueRNN")
    print("-" * 80)
    batch_size = 1
    embedding_size = 100
    dialogue_length = 44
    Dg = 50
    Dp = 50
    Dep = 50
    lay = 50
    c = 6

    dialogueRNN = DialogueRNN(embedding_size, batch_size, Dg, Dp, Dep, lay, c)
    utters = torch.ones(batch_size, dialogue_length, embedding_size)
    speakers = torch.ones(batch_size, dialogue_length)
    output = dialogueRNN.forward(utters, speakers)

    output_expected_state = [batch_size, dialogue_length, c]
    assert (
            list(output.shape) == output_expected_state
    ), "output shape is incorret: it should be:\n {} but is:\n{}".format(
        output_expected_state, list(output.size())
    )
    print("Sanity Check Passed for DialogueRNN!")
    print("-" * 80)

if __name__ == '__main__':
    sanity_check_1a()
    sanity_check_1b()
    sanity_check_1c()
    sanity_check_2a()