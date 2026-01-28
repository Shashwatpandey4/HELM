import torch
from transformers import AutoModelForCausalLM, AutoConfig

def load_llama_model_meta():
    """
    Loads Llama-2-7b structure on 'meta' device.
    This uses 0 VRAM, allowing compilation analysis on memory-constrained GPUs.
    """
    model_name = "meta-llama/Llama-2-7b-hf" 
    # model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # Open alternate
    print(f"Factory: Loading {model_name} on META device...")
    
    token = os.environ.get("HF_TOKEN")

    # We use AutoConfig to avoid downloading weights immediately if possible,
    # but AutoModelForCausalLM with device_map='meta' is the standard way.
    # Note: user needs HF token if accessing gated repo, but assuming env is set up or open equivalent.
    try:
        config = AutoConfig.from_pretrained(model_name, token=token)
        # Using context manager for meta device is cleaner often
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config)
            
        return model
    except Exception as e:
        print(f"Error loading Llama model: {e}")
        # Fallback to tiny random model if HF access fails?
        # Let's assume user has access or we catch it later.
        raise e
