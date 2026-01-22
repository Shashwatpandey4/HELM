import torch
import torch._dynamo
from backend.backend import helm
from tests.visualize_passes import ToyTransformer

def test_helm_backend():
    print("Testing Helm Backend Integration...")
    
    # Define model
    model = ToyTransformer()
    example_input = torch.randn(2, 10)
    
    # Torch Compile with custom backend
    print("Compiling model with backend='helm'...")
    compiled_model = torch.compile(model, backend=helm)
    
    # Run
    print("Running forward pass...")
    # This triggers compilation
    output = compiled_model(example_input)
    print("Forward pass complete.")

if __name__ == "__main__":
    test_helm_backend()
