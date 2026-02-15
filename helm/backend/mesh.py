import torch
from typing import List, Tuple, Optional

class DeviceMesh:
    """
    Manages the mapping of parallel dimensions (DP, PP, TP) to physical devices.
    
    Coordinate System:
    (dp_rank, pp_rank, tp_rank) -> physical_device_id
    """
    def __init__(self, dp=1, pp=1, tp=1, device_type="cuda"):
        self.dp = dp
        self.pp = pp
        self.tp = tp
        self.world_size = dp * pp * tp
        self.device_type = device_type
        
        # Verify available devices if CUDA
        if device_type == "cuda" and torch.cuda.is_available():
            available = torch.cuda.device_count()
            if available < self.world_size:
                print(f"[DeviceMesh] Warning: Requested {self.world_size} GPUs but only {available} available.")
                print(f"[DeviceMesh] Utilizing mapping logic but execution may fail if not simulated.")
                
    def get_coordinate(self, global_rank: int) -> Tuple[int, int, int]:
        """
        Returns (dp, pp, tp) coordinates for a given global rank.
        Layour Order: DP outer, PP middle, TP inner.
        """
        # Linear Index = dp * (PP*TP) + pp * (TP) + tp
        
        tp = global_rank % self.tp
        rem = global_rank // self.tp
        
        pp = rem % self.pp
        dp = rem // self.pp
        
        return (dp, pp, tp)
        
    def get_global_rank(self, dp: int, pp: int, tp: int) -> int:
        """
        Returns global rank from coordinates.
        """
        return dp * (self.pp * self.tp) + pp * (self.tp) + tp
        
    def get_physical_device_id(self, global_rank: int) -> int:
        """
        Maps global rank to physical device ID.
        Currently assumes 1-to-1 mapping (Rank N -> Device N).
        """
        return global_rank % torch.cuda.device_count() if torch.cuda.is_available() else 0

    def get_stage_devices(self, stage_idx: int) -> List[int]:
        """
        Get all physical device IDs involved in a specific pipeline stage 
        across all DP replicas and TP shards.
        """
        pass
        
    def get_tp_group_ranks(self, global_rank: int) -> List[int]:
        """
        For a given rank, return all ranks that share the same DP and PP
        coordinates (i.e., the Tensor Parallel peer group).
        """
        dp, pp, _ = self.get_coordinate(global_rank)
        group = []
        for t in range(self.tp):
            group.append(self.get_global_rank(dp, pp, t))
        return group
        
    def get_dp_group_ranks(self, global_rank: int) -> List[int]:
        """
        For a given rank, return all ranks that share the same PP and TP
        coordinates (i.e., Data Parallel replicas of this shard).
        """
        _, pp, tp = self.get_coordinate(global_rank)
        group = []
        for d in range(self.dp):
            group.append(self.get_global_rank(d, pp, tp))
        return group
