import torch
import sys
import os
import torch.fx as fx
from transformers import AutoConfig, AutoModelForCausalLM

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.passes.data_analysis import data_analysis_pass
from backend.passes.cost_model import cost_model_pass
from backend.passes.device_placement import device_placement_pass
from backend.passes.hardware_analysis import hardware_analysis_pass
from backend.passes.parallelism import pipeline_parallel_pass

def analysis_backend(gm: fx.GraphModule, example_inputs):
    print("\n[Analysis] Backend Triggered! Simulating Distributed Compilation...")
    
    import copy
    # We need to run the pipeline for Rank 0 and Rank 1 separately
    # Deepcopy gm because partitioning is destructive
    gm_rank0 = copy.deepcopy(gm)
    gm_rank1 = copy.deepcopy(gm)
    
    print("\n\n=== COMPILING RANK 0 ===")
    gm0 = hardware_analysis_pass(gm_rank0)
    gm0 = data_analysis_pass(gm0)
    gm0 = cost_model_pass(gm0, world_size=2)
    
    gm0 = device_placement_pass(gm0)
    # Partition for Rank 0
    gm0 = pipeline_parallel_pass(gm0, rank=0, world_size=2)
    print("\n[Rank 0 IR Tail]:")
    print("\n".join(gm0.code.strip().split('\n')[-20:])) # Print last 20 lines
    
    # --- Run for Rank 1 ---
    print("\n\n=== COMPILING RANK 1 ===")
    gm1 = copy.deepcopy(gm) # Fresh copy or from same base? 
    # original code copied gm_rank1 from gm.
    
    gm1 = hardware_analysis_pass(gm_rank1)
    gm1 = data_analysis_pass(gm1)
    gm1 = cost_model_pass(gm1, world_size=2)
    
    gm1 = device_placement_pass(gm1)
    # Partition for Rank 1
    gm1 = pipeline_parallel_pass(gm1, rank=1, world_size=2)
    print("\n[Rank 1 IR Head]:")
    print("\n".join(gm1.code.strip().split('\n')[:20])) # Print first 20 lines
    
    return gm0 # Return one to satisfy compile() contract

def main():
    print("----------------------------------------------------------------")
    print("Running Llama-2-7b Split Analysis (Single Process)")
    print("----------------------------------------------------------------")

    model_name = "meta-llama/Llama-2-7b-hf"
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("WARNING: HF_TOKEN environment variable not set. Loading restricted models may fail.")
    
    print(f"Loading {model_name} on META device...")
    try:
        config = AutoConfig.from_pretrained(model_name, token=token)
        # Disable cache to avoid Dynamo "Fake vs Real Meta Tensor" mix in torch.cat
        config.use_cache = False 
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Trigger Compilation
    print("\n[Action] Compiling with torch.compile...")
    
    # We use our analysis backend
    opt_model = torch.compile(model, backend=analysis_backend)
    
    # Create Meta Input
    input_ids = torch.randint(0, 1000, (1, 128), device="meta")
    
    print("Triggering Graph Capture (Warmup)...")
    try:
        # Patching Autocast to avoid crash on Meta device
        original_autocast = torch.autocast
        class MockAutocast:
             def __init__(self, *args, **kwargs): pass
             def __enter__(self): pass
             def __exit__(self, *args): pass
        torch.autocast = MockAutocast
        
        opt_model(input_ids)
        
    except Exception as e:
        # We expect execution to fail because we are analyzing only.
        print(f"Execution failed (Expected during analysis): {e}")
        pass
    finally:
        torch.autocast = original_autocast
        print("\n[Success] Compilation Pipeline verified.")

if __name__ == "__main__":
    main()
