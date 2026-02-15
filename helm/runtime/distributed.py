import torch
import torch.distributed as dist
import os

class DistributedManager:
    """
    Manages distributed process groups and communication backends.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DistributedManager, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance
        
    def initialize(self, rank: int, world_size: int, backend: str = "nccl"):
        if self.initialized:
            return
            
        # Set environment if not set
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500"
            
        # Initialize Process Group
        # If CPU only, force gloo
        if not torch.cuda.is_available() and backend == "nccl":
            print("[DistributedManager] CUDA not available. Forcing backend='gloo'.")
            backend = "gloo"
            
        print(f"[DistributedManager] Initializing Process Group (Backend={backend}, Rank={rank}/{world_size})...")
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        
        self.rank = rank
        self.world_size = world_size
        self.backend = backend
        self.initialized = True
        
    def get_rank(self):
        return self.rank if self.initialized else 0
        
    def get_world_size(self):
        return self.world_size if self.initialized else 1
        
    def cleanup(self):
        if self.initialized:
            dist.destroy_process_group()
            self.initialized = False
            print("[DistributedManager] Process Group Destroyed.")

# Global Accessor
def get_dist_manager():
    return DistributedManager()
