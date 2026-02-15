"""
Simple distributed test to verify TP/PP execution.

Run with:
    python -m helm.tools.launch_distributed tests/test_distributed_simple.py
"""

import torch
import torch.nn as nn
import os


def main():
    # Get distributed info
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    print(f"[Rank {rank}/{world_size}] Starting (Local Rank: {local_rank})")
    
    # Simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(128, 256)
            self.fc2 = nn.Linear(256, 128)
            
        def forward(self, x):
            x = self.fc1(x)
            x = torch.relu(x)
            x = self.fc2(x)
            return x
    
    model = SimpleModel()
    
    # Compile with HELM
    from helm.compiler import helm_backend
    
    compiled_model = torch.compile(
        model,
        backend=helm_backend,
        options={"tp_degree": world_size, "dtype": "fp16"}
    )
    
    # Test input
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        x = torch.randn(4, 128, device=device)
    else:
        device = torch.device("cpu")
        x = torch.randn(4, 128)
    
    # Forward pass
    print(f"[Rank {rank}] Running forward pass...")
    output = compiled_model(x)
    
    print(f"[Rank {rank}] Output shape: {output.shape}")
    print(f"[Rank {rank}] Test complete!")
    
    # Cleanup
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        print(f"[Rank {rank}] All ranks synchronized")


if __name__ == "__main__":
    main()
