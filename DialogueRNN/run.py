from docopt import docopt
from dataloader import DataLoader
from dialogueRNN import DialogueRNN

import torch.optim as optim
import torch.nn as nn
import torch

def train(args):
    train_dataset = DataLoader(args['--dataset-path'], train=True)
    batch_size = int(args['--batch-size'])
    model = DialogueRNN(
        embedding_size=int(args['--embed-size']),
        batch_size=batch_size,
        Dg=int(args['--Dg']),
        Dp=int(args['--Dp']),
        Dep=int(args['--Dep']),
        lay=int(args['--lay']),
        c=int(args['--c'])
    )
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.03, momentum=0.9)

    for epoch in range(args['--epoches']):
        running_loss = 0.0
        i = 0
        while i < len(train_dataset):
            batch_length = min(batch_size, len(train_dataset) - i)
            utters, speakers, labels = train_dataset[i]

            i += batch_length
            utters = torch.Tensor(utters).unsqueeze(0)
            speakers = torch.Tensor(speakers).unsqueeze(0)
            # labels = onehot(torch.LongTensor(labels).unsqueeze(0), args['--c'])
            labels = torch.LongTensor(labels).unsqueeze(0)
            optimizer.zero_grad()
            outputs = model(utters, speakers)
            loss = criterion(outputs.transpose(1, 2), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f'{epoch + 1} loss: {running_loss:.3f}')
    print('Finished Training')


def onehot(labels, class_num):
    # labels: batch_size, dialogue_length
    labels = labels.resize_(labels.shape[0], labels.shape[1], 1)
    m_zeros = torch.zeros(labels.shape[0], labels.shape[1], class_num)
    onehot = m_zeros.scatter_(2, labels, 1)
    return onehot


def main():
    """
    main func
    """
    args = docopt(__doc__)
    if args['train']:
        train(args)

def main2():
    args = dict()
    args['--dataset-path'] = 'DialogueRNN_features/DialogueRNN_features/IEMOCAP_features/IEMOCAP_features_raw'
    args['--batch-size'] = 1
    args['--embed-size'] = 100
    args['--Dg'] = 50
    args['--Dp'] = 50
    args['--Dep'] = 50
    args['--lay'] = 50
    args['--c'] = 6
    args['--epoches'] = 100
    train(args)

if __name__ == '__main__':
    main2()