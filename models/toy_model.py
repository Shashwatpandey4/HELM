
import torch
import torch.nn as nn

class ToyModel(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, layers=4):
        super().__init__()
        self.layers = nn.ModuleList()
        # Create a deep-ish network to allow splitting
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.ReLU())
        for _ in range(layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_dim, input_dim))
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def toy_model_factory():
    print("Factory: Creating Toy Model...")
    model = ToyModel()
    # Return (model, example_input)
    # Important: Input must be on CPU initially, runtime moves it.
    example_input = torch.randn(16, 128)
    return model, example_input
