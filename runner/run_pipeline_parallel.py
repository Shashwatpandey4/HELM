import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.multiprocessing as mp
# import torchvision # Not needed for FourStageModel
import sys
import time

# Add current path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.four_stage_model import FourStageModel

from backend.backend import helm

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # Set device
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def run_worker(rank, world_size):
    print(f"[Rank {rank}] Initializing process group...")
    setup(rank, world_size)
    
    # Load Model (Same on all ranks)
    print(f"[Rank {rank}] Loading FourStageModel...")
    dim = 1024
    model = FourStageModel(dim=dim).cuda(rank)
    model.eval()
    
    # Input
    x = torch.randn(8, dim).cuda(rank)
    
    # Custom Backend
    def helm_backend(gm, inputs):
        # Apply Helm pipeline
        return helm(gm, inputs, world_size=world_size, rank=rank)
    
    print(f"[Rank {rank}] Compiling...")
    # Compile
    opt_model = torch.compile(model, backend=helm_backend)
    
    # Run
    # Run
    print(f"[Rank {rank}] Running Inference...")
    try:
        # Warmup and Compile
        start_compile = time.time()
        with torch.no_grad():
             out = opt_model(x)
        torch.cuda.synchronize()
        end_compile = time.time()
        print(f"[Rank {rank}] Compilation + 1st Run Time: {end_compile - start_compile:.4f}s")

        # Benchmark Runtime
        start_run = time.time()
        for _ in range(10):
            with torch.no_grad():
                 out = opt_model(x)
        torch.cuda.synchronize()
        end_run = time.time()
        avg_time = (end_run - start_run) / 10
        print(f"[Rank {rank}] Average Inference Time: {avg_time:.4f}s")
            
        print(f"[Rank {rank}] Inference Completed!")
        if rank == world_size - 1:
            print(f"[Rank {rank}] Final Output Shape: {out.shape if hasattr(out, 'shape') else out}")
            # Verify result shape (FourStageModel: [8, 1024])
            if hasattr(out, 'shape') and out.shape == (8, 1024):
                 print(f"[Rank {rank}] SUCCESS: Output shape matches FourStageModel.")
            else:
                 print(f"[Rank {rank}] Result: {out}")
        else:
            print(f"[Rank {rank}] Completed Stage (Output: {out})")
            
    except Exception as e:
        print(f"[Rank {rank}] Execution Failed: {e}")
        import traceback
        traceback.print_exc()
        
    cleanup()

def main():
    world_size = 2 # Matches your 2 GPUs
    if torch.cuda.device_count() < world_size:
        print(f"Error: Need {world_size} GPUs")
        return
        
    print(f"Spawning {world_size} workers for ResNet18 Pipeline Parallelism...")
    mp.spawn(run_worker, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
