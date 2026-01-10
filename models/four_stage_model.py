import torch
import torch.nn as nn

class FourStageModel(nn.Module):
    def __init__(self, dim=2048):
        super().__init__()
        self.dim = dim
        # Define 4 heavy stages
        self.stage1_layer = nn.Linear(dim, dim)
        self.stage2_layer = nn.Linear(dim, dim)
        self.stage3_layer = nn.Linear(dim, dim)
        self.stage4_layer = nn.Linear(dim, dim)
        self.relu = nn.ReLU()

    def stage1(self, x):
        return self.relu(self.stage1_layer(x))

    def stage2(self, x):
        return self.relu(self.stage2_layer(x))

    def stage3(self, x):
        return self.relu(self.stage3_layer(x))

    def stage4(self, x):
        return self.stage4_layer(x)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x
