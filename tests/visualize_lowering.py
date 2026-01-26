import torch
import torch.nn as nn
import os
import sys

# Ensure backend can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.passes.parallelism import pipeline_parallel_pass, tensor_parallel_pass
from backend.passes.hardware_analysis import hardware_analysis_pass
from backend.passes.soft_analysis import soft_analysis_pass
from backend.passes.cost_model import cost_model_pass
from torch.fx.passes.shape_prop import ShapeProp
from tests.visualize_passes import ToyTransformer

def main():
    print("----------------------------------------------------------------")
    print("Visualizing Inductor Lowering (Code Gen)")
    print("----------------------------------------------------------------")
    
    # 1. Setup Model (Same as visualize_passes)
    model = ToyTransformer()
    gm = torch.fx.symbolic_trace(model)
    example_input = torch.randn(2, 10)
    ShapeProp(gm).propagate(example_input)
    
    # 2. Run HELM Passes
    print("[HELM] Running Analysis & Splitting...")
    gm = hardware_analysis_pass(gm)
    gm = soft_analysis_pass(gm)
    gm = cost_model_pass(gm)
    gm = pipeline_parallel_pass(gm)
    gm = tensor_parallel_pass(gm)
    
    print("\n[HELM] Graph is ready. Handing off to Inductor...")
    
    # 3. Compile with Inductor
    # We trigger compilation by running a forward pass
    # We rely on TORCH_LOGS env var to capture the output
    optimized = torch.compile(gm, backend="inductor")
    
    print("\n[Action] Triggering Compilation...")
    try:
        with torch.no_grad():
            optimized(example_input)
        print("\n[Success] Compilation triggered.")
    except Exception as e:
        print(f"\n[Error] {e}")

if __name__ == "__main__":
    main()
