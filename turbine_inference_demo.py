import torch
import iree.turbine
import torch
import iree.turbine
import time

from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 1. Load Model (GPT-2)
model_name = "gpt2"
print(f"Loading {model_name}...")
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

# 2. Prepare Input
prompt = "The quick brown fox"
inputs = tokenizer(prompt, return_tensors="pt")
# JIT requires fixed shapes or dynamic shape constraints usually.
# We will just pass the inputs directly.

# 3. Compile with IREE Turbine
print("\nCompiling with torch.compile(backend='turbine_cpu')...")
print("This performs JIT compilation on the first run.")
# We wrap it to return only logits because dictionary outputs can sometimes be tricky for JIT
class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model(x)[0]

wrapped_model = Wrapper(model)
opt_model = torch.compile(wrapped_model, backend="turbine_cpu")

# 4. Run Inference Loop
print("\nFirst Run (JIT Compiling)...")
start = time.time()
with torch.no_grad():
    output = opt_model(inputs["input_ids"])
end = time.time()
print(f"First run time: {end - start:.4f}s")
print(f"Output shape: {output.shape}")

print("\nSecond Run (Fast)...")
start = time.time()
with torch.no_grad():
    output = opt_model(inputs["input_ids"])
end = time.time()
print(f"Second run time: {end - start:.4f}s")

# Decode
logits_np = output.numpy()
next_token_id = logits_np[0, -1, :].argmax()
print(f"Predicted Token: '{tokenizer.decode(next_token_id)}'")
