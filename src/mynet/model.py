import torch.nn as nn
import torch

class ThreeLayerFullyConnectedNetwork(nn.Module):

    def __init__(self):
        super(ThreeLayerFullyConnectedNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.linear_1 = nn.Linear(784, 32)
        self.linear_2 = nn.Linear(32, 64)
        self.linear_3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        x = self.relu(x)
        x = self.linear_3(x)
        return x
