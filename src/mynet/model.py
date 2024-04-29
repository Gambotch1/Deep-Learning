import torch.nn as nn

class ThreeLayerFullyConnectedNetwork(nn.Module):

    def __init__(self):
        super(ThreeLayerFullyConnectedNetwork, self).__init__()
        #Flatten the input images
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits