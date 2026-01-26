import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_qwen_model():
    """
    Loads Qwen/Qwen3-4B-Instruct-2507 model and tokenizer.
    Returns:
        model: The loaded model
        example_input: A sample input tensor for tracing/compilation
    """
    model_id = "Qwen/Qwen3-4B-Instruct-2507"
    print(f"Loading {model_id}...")
    
    # Load model with trust_remote_code=True if required, though Qwen2/common usually supported.
    # Using float16/bfloat16 is typical for these models, but keeping default (float32) for simple compilation safety unless specified.
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        device_map="cpu", # Load on CPU for compiling
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    prompt = "Hello, how are you?"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    return model, inputs.input_ids
