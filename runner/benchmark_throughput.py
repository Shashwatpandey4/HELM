import os
import sys
import time
import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.backend import helm

def setup():
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
    
    if rank == 0:
        print(f"--- HELM Throughput Benchmark ---")
        print(f"Model: {model_name} (2 layers)")
        print(f"World Size: {world_size}")
        print(f"---------------------------------\n")

    config = AutoConfig.from_pretrained(model_name, token=token)
    config.num_hidden_layers = 2
    config.use_cache = False 
    
    model = AutoModelForCausalLM.from_config(config).to(torch.bfloat16).to(device)
    model.eval()

    # Benchmark settings
    warmup_iters = 5
    bench_iters = 20
    batch_size = 1
    seq_len = 128
    
    # Input tensor
    input_ids = torch.randint(0, 32000, (batch_size, seq_len)).to(device)

    def helm_backend(gm, example_inputs):
        return helm(gm, example_inputs, world_size=world_size, rank=rank)

    opt_model = torch.compile(model, backend=helm_backend)

    # 1. Warm-up
    if rank == 0: print(f"Starting {warmup_iters} warm-up iterations...")
    for i in range(warmup_iters):
        with torch.no_grad():
            _ = opt_model(input_ids)
        if rank == 0 and i == 0: print("   [Warmup] Compilation triggered.")

    dist.barrier()
    if rank == 0: print(f"Warm-up complete. Starting {bench_iters} steady-state iterations...")
    
    # 2. Steady-State Measurement
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    dist.barrier()
    start_event.record()
    
    for i in range(bench_iters):
        with torch.no_grad():
            _ = opt_model(input_ids)
            
    end_event.record()
    torch.cuda.synchronize()
    
    total_time_ms = start_event.elapsed_time(end_event)
    avg_latency_ms = total_time_ms / bench_iters
    
    # Calculate throughput
    # tokens = batch * seq_len * iters
    total_tokens = batch_size * seq_len * bench_iters
    tokens_per_sec = (total_tokens / total_time_ms) * 1000
    
    if rank == world_size - 1:
        print(f"\n--- Results (Rank {rank}) ---")
        print(f"Average Latency: {avg_latency_ms:.2f} ms")
        print(f"Throughput:      {tokens_per_sec:.2f} tokens/sec")
        print(f"Total Time:      {total_time_ms/1000:.2f} sec")
        print(f"----------------------------\n")

    cleanup()

if __name__ == "__main__":
    main()
