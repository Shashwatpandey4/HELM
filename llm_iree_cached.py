import torch
import iree.turbine.aot as aot
import iree.runtime
import numpy as np
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from kv_cache import StatefulGPT2Wrapper

# 1. Load Model
model_name = "gpt2"
print(f"Loading {model_name}...")
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

# 2. Wrapper for state
wrapped_model = StatefulGPT2Wrapper(model)

# 3. Export to IREE with State
print("Exporting model with KV Cache support...")

# We need to trace with "past_key_values".
# GPT-2 has 12 layers. Each layer has a Key and Value tensor.
# Shape: [batch, n_heads, seq_len, head_dim]
# batch=1, n_heads=12, seq_len=1 (initial?), head_dim=64
# Actually for trace, we usually provide the *max* logical size or a dynamic size.
# BUT handling complex nested tuples in IREE arguments is tricky. 
# THIS IS THE HARD PART OF MANUAL IREE: Signature management.

# Simplifying: We will try to rely on AOT's ability to flatten tuples.

# Dummy inputs for tracing
# Step 1: Prompt phase (no cache yet) - usually handled separately or we treat cache as empty.
# Step 2: Generation phase (cache exists).

# Let's try to export just the "step" function: taking 1 token and full cache.
batch = 1
n_layers = 12
n_heads = 12
head_dim = 64
seq_len = 10 # dummy cached length

# Create dummy cache (tuple of tuples)
dummy_past = []
for _ in range(n_layers):
    k = torch.randn(batch, n_heads, seq_len, head_dim)
    v = torch.randn(batch, n_heads, seq_len, head_dim)
    dummy_past.append((k, v))
dummy_past = tuple(dummy_past)

input_ids = torch.tensor([[50256]]) # single token

# Dynamic shapes are crucial here.
# input_ids: static [1, 1]
# cache: [1, 12, N, 64] where N is dynamic
# We need to tell the exporter that dim 2 is dynamic.

# This complex export often requires the Shark-Turbine "Stateless" or "Caustic" wrappers,
# doing it raw via `aot.export` might hit "complex data structure" errors.
# Let's try a naive export first.
try:
    module = aot.export(wrapped_model, args=(input_ids, dummy_past))
    print("Export successful!")
    module.save_mlir("gpt2_cached.mlir")
    
    # Just compiling to verify it works, running this is very complex logic-wise
    # because we have to manage 24 input tensors and 24 output tensors manually in Python.
    print("Compiling...")
    compiled_binary = module.compile(save_to=None)
    print("Compiled successfully. (Runtime execution requires managing 24+ tensors manually, skipping for now)")
    
except Exception as e:
    print(f"Export failed (as expected for complex tuples): {e}")
    print("Implementing full KV cache manually requires simplified I/O signatures.")
