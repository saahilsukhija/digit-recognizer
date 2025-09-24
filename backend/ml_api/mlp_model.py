import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, layer_2 = 128):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(28*28, layer_2),
            nn.ReLU(),
            nn.Linear(layer_2, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        return self.layers(x)