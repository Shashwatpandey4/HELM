import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def run_worker(rank, world_size):
    print(f"[Rank {rank}] Initializing NCCL...")
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    device = torch.device(f"cuda:{rank}")
    print(f"[Rank {rank}] Testing communication...")
    
    tensor = torch.ones(2, 2).to(device) * (rank + 1)
    
    if rank == 0:
        dist.send(tensor, dst=1)
        print(f"[Rank 0] Sent tensor to Rank 1")
    else:
        recv_tensor = torch.zeros(2, 2).to(device)
        dist.recv(recv_tensor, src=0)
        print(f"[Rank 1] Received tensor from Rank 0: {recv_tensor}")
        
    dist.destroy_process_group()
    print(f"[Rank {rank}] Done.")

def main():
    world_size = torch.cuda.device_count()
    if world_size < 2:
        print("Need at least 2 GPUs.")
        return
    
    mp.spawn(run_worker, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
