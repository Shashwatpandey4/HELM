import torch
from helm import helm_backend

def simple_fn(x, y):
    # Matmul: 128*128*128 = ~4M FLOPs
    z = torch.mm(x, y)
    # Add: 128*128 = ~16K FLOPs
    return z + x

compiled_fn = torch.compile(simple_fn, backend=helm_backend)

print("Starting compilation test...")
x = torch.randn(128, 128).cuda() if torch.cuda.is_available() else torch.randn(128, 128)
y = torch.randn(128, 128).cuda() if torch.cuda.is_available() else torch.randn(128, 128)

# Execution will trigger compilation
result = compiled_fn(x, y)
print("Compilation test successful!")
print("Result shape:", result.shape)
