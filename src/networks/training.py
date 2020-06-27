
import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self, num_input):
        super().__init__()
        self.num_input = num_input
        self.fc1 = nn.Linear(self.num_input, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 27)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim = 1)

    def train(self, data, epochs = 100, saved_network = True):
        # loss_function = nn.CrossEntropyLoss()
        loss_function = nn.NLLLoss()
        # optimizer = optim.Adam(net.parameters(), lr = 0.00005)
        optimizer = optim.SGD(self.parameters(), lr = 0.0001, momentum = 0.5)

        for epoch in range(epochs):
            for point in data:
                X, y = point
                self.zero_grad()

                X = torch.tensor(X).float()
                output = self(X.view(-1, self.num_input))

                y = get_index(y.lower())
                y = torch.tensor([y])

                loss = F.nll_loss(output, y)
                loss.backward()
                optimizer.step()
                
            print(loss)

    def test(self, data):
        correct = 0
        total = 0 

        with torch.no_grad():
            for point in data:
                X, y = point

                X = torch.tensor(X).float()
                output = self(X.view(-1, self.num_input))

                y = get_index(y.lower())
                y = torch.tensor([y])

                if torch.argmax(output) == y:
                    correct += 1
                total += 1
                    
        print("Accuracy: ", round(correct/total, 3))

def get_index(label):
    for i, c in enumerate(characters_lookup):
        if c == label:
            return i

characters_lookup = ["alef", "ayin", "bet", "dalet", "gimel", "he", "het", "kaf", "kaf-final", "lamed", "mem", "mem-medial", "nun-final", "nun-medial", "pe", "pe-final", "qof", "resh", "samekh", "shin", "taw", "tet", "tsadi-final", "tsadi-medial", "waw", "yod", "zayin"]

