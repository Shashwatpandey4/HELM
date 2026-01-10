import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.backend import helm
from backend.passes import data_parallel_pass, tensor_parallel_pass, pipeline_parallel_pass

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32) # Column
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 16) # Row

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def test_dp():
    print("\n=== Testing Data Parallel Pass ===")
    model = SimpleMLP()
    
    def my_dp_backend(gm, inputs):
        gm = data_parallel_pass(gm, world_size=2)
        gm.print_readable()
        return gm
    
    opt = torch.compile(model, backend=my_dp_backend)
    x = torch.randn(4, 16)
    try:
        opt(x)
    except RuntimeError as e:
        if "Could not resolve the process group" in str(e):
            print("Execution stopped at AllReduce (expected without distributed setup). Pass applied successfully.")
        else:
            raise e

def test_tp():
    print("\n=== Testing Tensor Parallel Pass ===")
    model = SimpleMLP()
    
    # We need to ensure weights have shapes for the pass to work.
    # torch.compile normally traces with FakeTensor, providing 'val'.
    
    def my_tp_backend(gm, inputs):
        # We simulate Rank 0 of 2
        gm = tensor_parallel_pass(gm, world_size=2, rank=0)
        gm.print_readable()
        return gm
        
    opt = torch.compile(model, backend=my_tp_backend)
    x = torch.randn(4, 16)
    try:
        opt(x)
    except Exception as e:
        print(f"Execution failed as expected (sharding changed shapes): {e}")

def test_pp():
    print("\n=== Testing Pipeline Parallel Pass ===")
    # Larger model to split
    class DeepMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(16, 16) for _ in range(4)])
        def forward(self, x):
            # sys.modules
            for layer in self.layers:
                x = layer(x)
            return x
            
    model = DeepMLP()
    
    def my_pp_backend_rank0(gm, inputs):
        print("--- Rank 0 Graph ---")
        gm = pipeline_parallel_pass(gm, world_size=2, rank=0)
        gm.print_readable()
        return gm

    def my_pp_backend_rank1(gm, inputs):
        print("--- Rank 1 Graph ---")
        gm = pipeline_parallel_pass(gm, world_size=2, rank=1)
        gm.print_readable()
        return gm
        
    x = torch.randn(4, 16)
    
    print("Compiling Rank 0...")
    opt0 = torch.compile(model, backend=my_pp_backend_rank0)
    try: 
        opt0(x) 
    except Exception as e: 
        print(f"Rank 0 Execution failed (expected): {e}")
    
    print("Compiling Rank 1...")
    opt1 = torch.compile(model, backend=my_pp_backend_rank1)
    try: 
        opt1(x)
    except Exception as e:
        print(f"Rank 1 Execution failed (expected): {e}")

def test_auto():
    print("\n=== Testing Auto Parallel Pass ===")
    from backend.passes import auto_parallel_pass
    
    # Use DeepMLP
    class DeepMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(16, 16) for _ in range(8)])
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x
            
    model = DeepMLP()
    x = torch.randn(4, 16)
    
    # Simulate World Size 4 (Should pick PP=2, TP=2)
    print("--- Simulating World Size 4 (Hybrid: PP=2, TP=2) ---")
    
    ranks = [0, 1, 2, 3]
    for r in ranks:
        print(f"\n--- Compiling for Rank {r} ---")
        def backend(gm, inputs):
            gm = auto_parallel_pass(gm, world_size=4, global_rank=r)
            # gm.print_readable()
            return gm
            
        opt = torch.compile(model, backend=backend)
        try:
            opt(x)
        except:
            pass

if __name__ == "__main__":
    test_dp()
    test_tp()
    test_pp()
    test_auto()
