
import torch
import torchvision
import random
import numpy as np

from torchvision import transforms, datasets

def splitData(data):
    random.shuffle(data)

    split = 0.9
    length = len(data)
    trainset = data[:int(0.9*length)]
    testset = data[int(0.9*length):]

    return (trainset, testset)

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(96, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 27)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim = 1)

import torch.optim as optim

def one_hot(atPosition):
    ohe = torch.zeros(27).int()
    ohe[atPosition] = 1
    return ohe

one_hot_dict = { "alef": one_hot(0), "ayin": one_hot(1), "bet": one_hot(2), "dalet": one_hot(3), "gimel": one_hot(4),
                 "he": one_hot(5), "het": one_hot(6), "kaf": one_hot(7), "kaf-final": one_hot(8), "lamed": one_hot(9),
                 "mem": one_hot(10), "mem-medial": one_hot(11), "nun-final": one_hot(12), "nun-medial": one_hot(13), "pe": one_hot(14),
                 "pe-final": one_hot(15), "qof": one_hot(16), "resh": one_hot(17), "samekh": one_hot(18), "shin": one_hot(19),
                 "taw": one_hot(20), "tet": one_hot(21), "tsadi-final": one_hot(22), "tsadi-medial": one_hot(23), "waw": one_hot(24),
                 "yod": one_hot(25), "zayin": one_hot(26) }

def get_index(label):
    one_hot = one_hot_dict[label].numpy()
    for i, item in enumerate(one_hot):
        if item == 1:
            return i

def train_network(net, trainset):
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr = 0.001)

    for epoch in range(50):
        for data in trainset:
            X, y = data
            net.zero_grad()

            X = torch.tensor(X).float()
            output = net(X.view(-1, 96))

            y = get_index(y.lower())
            y = torch.tensor([y])

            loss = F.nll_loss(output, y)
            loss.backward()
            optimizer.step()
        print(loss)

def test_network(net, testset):
    correct = 0
    total = 0 

    with torch.no_grad():
        for data in testset:
            X, y = data

            X = torch.tensor(X).float()
            output = net(X.view(-1,96))

            for idx, i in enumerate(output):
                
                y = get_index(y.lower())
                y = torch.tensor([y])

                if torch.argmax(i) == y[idx]:
                    correct += 1
                total += 1

    print("Accuracy: ", round(correct/total, 3))
