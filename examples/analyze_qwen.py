import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from helm import helm_backend
from accelerate import init_empty_weights
import os
import contextlib

# --- MONKEY PATCH TO FIX META DEVICE CRASH ---
# transformers uses 'torch.is_autocast_enabled(device_type)' which crashes on 'meta' device.
import transformers.utils.generic

@contextlib.contextmanager
def mock_maybe_autocast(device_type, enabled=True, dtype=None, cache_enabled=True):
    yield

transformers.utils.generic.maybe_autocast = mock_maybe_autocast
print("Patched transformers.utils.generic.maybe_autocast for meta execution compatibility.")
# ---------------------------------------------

model_id = "Qwen/Qwen2.5-7B-Instruct"
print(f"Loading {model_id} Config...")
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

print("Instantiating Model using Accelerate (Meta Device)...")
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(model_id)
prompt = "Hello, world!"
inputs = tokenizer(prompt, return_tensors="pt")
meta_inputs = {k: v.to("meta") for k, v in inputs.items()}

print("Compiling with HELM backend...")
compiled_model = torch.compile(model, backend=helm_backend, fullgraph=False)

print("Running trace (use_cache=False) to generate graph...")
try:
    with torch.no_grad():
        # Disabling KV cache is crucial for stable meta tracing
        compiled_model(**meta_inputs, use_cache=False)
except Exception as e:
    print(f"Compilation/Execution failed: {e}")
    import traceback
    traceback.print_exc()

print("Done! Check helm_graph_complete.png")
