
import torch
import torchvision
import random
import numpy as np

from torchvision import transforms, datasets

num_segments = 6*3
num_top_k = 8

def splitData(data):
    random.shuffle(data)

    split = 0.9
    length = len(data)
    trainset = data[:int(split*length)]
    testset = data[int(split*length):]

    return (trainset, testset)

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(num_segments * num_top_k, 64)
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

characters_lookup = ["alef", "ayin", "bet", "dalet", "gimel", "he", "het", "kaf", "kaf-final", "lamed", "mem", "mem-medial", "nun-final", "nun-medial", "pe", "pe-final", "qof", "resh", "samekh", "shin", "taw", "tet", "tsadi-final", "tsadi-medial", "waw", "yod", "zayin"]

def train_network(net, trainset, testset):
    # loss_function = nn.CrossEntropyLoss()
    loss_function = nn.NLLLoss()
    # optimizer = optim.Adam(net.parameters(), lr = 0.00005)
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.5)

    for epoch in range(500):
        for data in trainset:
            X, y = data
            net.zero_grad()

            X = torch.tensor(X).float()
            output = net(X.view(-1, num_segments * num_top_k))

            y = get_index(y.lower())
            y = torch.tensor([y])

            loss = F.nll_loss(output, y)
            loss.backward()
            optimizer.step()
        print(loss)
        test_network(net, testset)

        if (epoch % 50 == 0): 
            print("saving")
            torch.save(net.state_dict(), "/home/niek/git/HandWritingRecognition/model_dimension3_NLLL.pt")

def test_network(net, testset):
    correct = 0
    total = 0 

    with torch.no_grad():
        for data in testset:
            X, y = data

            X = torch.tensor(X).float()
            output = net(X.view(-1,num_segments * num_top_k))

            y = get_index(y.lower())
            y = torch.tensor([y])

            if torch.argmax(output) == y:
                correct += 1
            total += 1
                
    print("Accuracy: ", round(correct/total, 3))

def evaluate_directions_with_model(model_path, directions):
    net = Net()
    net.load_state_dict(torch.load(model_path))
    net.eval()

    characters = []
    with torch.no_grad():
        for data in directions:
            X = torch.tensor(data).float()
            output = net(X.view(-1,num_segments * num_top_k))
            output = torch.argmax(output)
            character_number = output.item()
            character_name = characters_lookup[character_number]
            characters.append(character_name)

    return characters