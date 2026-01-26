import torch
import sys
import os

# Ensure we can import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.backend import helm
from models.qwen_loader import load_qwen_model

def compile_and_verify():
    print("----------------------------------------------------------------")
    print("Running Qwen-4B Compilation Verify Script")
    print("----------------------------------------------------------------")
    
    # 1. Load Model
    model, example_input = load_qwen_model()
    print(f"Model loaded. Architectures: {model.config.architectures}")
    print(f"Example input shape: {example_input.shape}")

    # 2. Compile with HELM Backend
    print("\n[Action] Compiling with torch.compile(backend=helm)...")
    
    # We use dynamic=True if we expect variable sequence lengths, but for static analysis simple is fine.
    compiled_model = torch.compile(model, backend=helm)
    
    # 3. Validating Compilation (Triggering Passes)
    print("\n[Action] Triggering compilation with forward pass...")
    try:
        # We don't care about the output value, just that the compilation passes run
        with torch.no_grad():
            _ = compiled_model(example_input)
        print("\n[Success] Compilation passes completed successfully!")
    except Exception as e:
        print(f"\n[Error] compilation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    compile_and_verify()
