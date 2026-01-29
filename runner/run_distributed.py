import os
import sys
import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.backend import helm

def setup():
    # Environment variables are set by torchrun
    dist.init_process_group("nccl")
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    return rank, int(os.environ["WORLD_SIZE"])

def cleanup():
    dist.destroy_process_group()

def main():
    rank, world_size = setup()
    device = torch.device(f"cuda:{rank}")
    
    model_name = "meta-llama/Llama-2-7b-hf"
    token = os.environ.get("HF_TOKEN")
    
    print(f"[Rank {rank}] Loading 2-layer Llama configuration...")
    config = AutoConfig.from_pretrained(model_name, token=token)
    config.num_hidden_layers = 2
    config.use_cache = False 
    
    print(f"[Rank {rank}] Materializing model on {device}...")
    try:
        # Load directly to GPU to save memory and avoid SIGSEGV in mp.spawn
        model = AutoModelForCausalLM.from_config(config).to(torch.bfloat16).to(device)
        model.eval()
    except Exception as e:
        print(f"[Rank {rank}] Failed to load model: {e}")
        cleanup()
        return

    input_ids = torch.randint(0, 32000, (1, 128)).to(device)

    print(f"[Rank {rank}] Triggering HELM compilation...")
    def helm_backend(gm, example_inputs):
        return helm(gm, example_inputs, world_size=world_size, rank=rank)

    try:
        # We need to ensure all ranks go through the compilation process
        opt_model = torch.compile(model, backend=helm_backend)
        
        print(f"[Rank {rank}] Starting forward pass...")
        dist.barrier()
        
        with torch.no_grad():
            output = opt_model(input_ids)
            
        dist.barrier()
        print(f"[Rank {rank}] Forward pass completed successfully!")
        
        if rank == world_size - 1:
             if hasattr(output, 'logits'):
                 print(f"[Rank {rank}] Output logits shape: {output.logits.shape}")
             elif isinstance(output, torch.Tensor):
                 print(f"[Rank {rank}] Output tensor shape: {output.shape}")

    except Exception as e:
        print(f"[Rank {rank}] Execution failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup()

if __name__ == "__main__":
    main()
