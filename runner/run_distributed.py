import torch
import sys
import os

# Ensure we can import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runtime.ray_runtime import RayPipelineRuntime
from models.qwen_loader import load_qwen_model

def qwen_factory():
    # Helper to load model and simplify return if needed
    # Ray pickles this function, so imports inside might be safer if not top-level?
    # Actually, load_qwen_model is a top-level import, so it should be fine.
    print("Factory: Loading Qwen...")
    return load_qwen_model()

def main():
    print("----------------------------------------------------------------")
    print("Running Distributed Qwen via Generic Runtime")
    print("----------------------------------------------------------------")
    
    # 1. Initialize Runtime with Qwen Factory
    # Note: This will spin up Ray actors and trigger compilation on each rank.
    try:
        runtime = RayPipelineRuntime(model_factory=qwen_factory, world_size=2)
        
        # 2. Prepare Input
        # Qwen loader returns (model, input_ids)
        # We need just the input here for execution.
        # We can re-call factory or just create dummy input (load_qwen_model is heavy).
        # Let's just create a dummy input matching the one from loader?
        # Actually, let's just use a dummy tokenizer output structure if possible, or just IDs.
        # load_qwen_model returns input_ids tensor.
        print("Preparing input...")
        # (1, sequence_length)
        input_ids = torch.randint(0, 1000, (1, 10), dtype=torch.long)
        
        # 3. Run Distributed Inference
        print("\n[Action] Running Inference...")
        output = runtime.run(input_ids)
        
        print("\n[Success] Inference Completed.")
        print(f"Output: {output}")
        
    finally:
        if 'runtime' in locals():
            runtime.shutdown()

if __name__ == "__main__":
    main()
