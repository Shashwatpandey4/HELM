import os
import sys
import time
import torch
import torch.distributed as dist
import argparse
from transformers import AutoConfig, AutoModelForCausalLM

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.backend import helm

def setup():
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    return rank, int(os.environ["WORLD_SIZE"])

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, default="7b", choices=["7b", "13b", "30b"])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    rank, world_size = setup()
    device = torch.device(f"cuda:{rank}")
    
    model_map = {
        "7b": "meta-llama/Llama-2-7b-hf",
        "13b": "meta-llama/Llama-2-13b-hf",
        "30b": "huggyllama/llama-30b" # Llama-1 since Llama-2 has no 30b
    }
    
    model_name = model_map[args.model_size]
    token = os.environ.get("HF_TOKEN")
    
    if rank == 0:
        print(f"\n=== HELM Full-Scale Benchmark: Llama-{args.model_size} ===")
        print(f"Model ID: {model_name}")
        print(f"World Size: {world_size}")
        print(f"--------------------------------------------\n")

    # Load config without modification (as-is)
    config = AutoConfig.from_pretrained(model_name, token=token)
    config.use_cache = False 
    
    # Materialize model
    # For 30b, we might need to load on meta device first to avoid OOM during initialization
    # but A6000 has 48GB, 30b is ~60GB. We MUST use meta materialization or sharded loading for 30b.
    
    print(f"[Rank {rank}] Materializing model...")
    if args.model_size == "30b":
        # Load on CPU first to avoid GPU OOM, then shard to GPU via HELM
        model = AutoModelForCausalLM.from_config(config).to(torch.bfloat16).to("cpu")
    else:
        # Standard load for 7b/13b which fits in 48GB
        model = AutoModelForCausalLM.from_config(config).to(torch.bfloat16).to(device)
    
    model.eval()

    # Benchmark settings
    warmup_iters = args.warmup
    bench_iters = args.iters
    batch_size = 1
    seq_len = 128
    
    # Input tensor
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)

    def helm_backend(gm, example_inputs):
        return helm(gm, example_inputs, world_size=world_size, rank=rank)

    # Trigger HELM-optimized compilation
    opt_model = torch.compile(model, backend=helm_backend)

    # 1. Warm-up
    if rank == 0: print(f"Starting {warmup_iters} warm-up iterations (Compilation)...")
    for i in range(warmup_iters):
        with torch.no_grad():
            _ = opt_model(input_ids)
        if rank == 0 and i == 0: print("   [Warmup] Compilation triggered and partition applied.")

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
    
    total_tokens = batch_size * seq_len * bench_iters
    tokens_per_sec = (total_tokens / total_time_ms) * 1000
    
    dist.barrier()
    if rank == world_size - 1:
        print(f"\n--- Results: Llama-{args.model_size} (Rank {rank}) ---")
        print(f"Layers:          {config.num_hidden_layers}")
        print(f"Average Latency: {avg_latency_ms:.2f} ms")
        print(f"Throughput:      {tokens_per_sec:.2f} tokens/sec")
        print(f"Total Time:      {total_time_ms/1000:.2f} sec")
        print(f"--------------------------------------------\n")

    cleanup()

if __name__ == "__main__":
    main()
