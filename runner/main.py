import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.backend import helm

def main():
    print("Project Hetero: Testing Custom Backend")
    
    # Define a simple model
    def model(x, y):
        return x + y * x

    # Compile the model using our custom backend
    opt_model = torch.compile(model, backend=helm)

    # Run the model
    x = torch.randn(10, 10)
    y = torch.randn(10, 10)
    
    print("Running compiled model...")
    result = opt_model(x, y)
    print("Result shape:", result.shape)

    print("Result shape:", result.shape)
    print("Result device:", result.device)




if __name__ == "__main__":
    main()
