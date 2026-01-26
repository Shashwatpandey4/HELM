import torch
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runtime.ray_runtime import RayPipelineRuntime
from models.toy_model import toy_model_factory

def main():
    print("----------------------------------------------------------------")
    print("Verifying HELM + Inductor Pipeline with Toy Model")
    print("----------------------------------------------------------------")
    
    runtime = None
    try:
        # 2 Workers (Ranks)
        runtime = RayPipelineRuntime(model_factory=toy_model_factory, world_size=2)
        
        # Input (Batch 16, Dim 128)
        x = torch.randn(16, 128)
        
        print("\n[Action] Running Inference...")
        start = time.time()
        output = runtime.run(x)
        end = time.time()
        
        print(f"\n[Success] Finished in {end-start:.4f}s")
        print(f"Output Shape: {output.shape}")
        
    except Exception as e:
        print(f"\n[Error] {e}")
        import traceback
        traceback.print_exc()
    finally:
        if runtime:
            print("Shutting down Ray...")
            runtime.shutdown()

if __name__ == "__main__":
    main()
