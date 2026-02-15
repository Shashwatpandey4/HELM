import torch
import subprocess
import re
from typing import Dict, List, Tuple, Optional
import numpy as np

class GPUTopology:
    """
    Detects and represents GPU topology (NVLink/PCIe connectivity).
    
    Provides:
    - Adjacency matrix of GPU-GPU bandwidth
    - Device grouping recommendations for Tensor Parallelism
    """
    def __init__(self):
        self.num_gpus = 0
        self.bandwidth_matrix = None  # NxN matrix in GB/s
        self.device_names = []
        
    def detect(self) -> bool:
        """
        Detect GPU topology using nvidia-smi and torch APIs.
        Returns True if successful.
        """
        if not torch.cuda.is_available():
            print("[GPUTopology] No CUDA devices available.")
            return False
            
        self.num_gpus = torch.cuda.device_count()
        self.device_names = [torch.cuda.get_device_name(i) for i in range(self.num_gpus)]
        
        print(f"[GPUTopology] Detected {self.num_gpus} GPUs:")
        for i, name in enumerate(self.device_names):
            print(f"  GPU {i}: {name}")
        
        # Initialize bandwidth matrix
        self.bandwidth_matrix = np.zeros((self.num_gpus, self.num_gpus))
        
        # Try nvidia-smi topo first
        if not self._detect_via_nvidia_smi():
            # Fallback: use torch P2P capability check
            self._detect_via_torch()
        
        return True
    
    def _detect_via_nvidia_smi(self) -> bool:
        """
        Parse nvidia-smi topo -m to detect NVLink connections.
        Returns True if successful.
        """
        try:
            result = subprocess.run(
                ['nvidia-smi', 'topo', '-m'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode != 0:
                print("[GPUTopology] nvidia-smi topo failed")
                return False
            
            lines = result.stdout.strip().split('\n')
            
            # Find header line with GPU indices
            header_idx = -1
            for i, line in enumerate(lines):
                if 'GPU0' in line or 'GPU 0' in line:
                    header_idx = i
                    break
            
            if header_idx == -1:
                print("[GPUTopology] Could not find GPU header in nvidia-smi output")
                return False
            
            # Parse topology matrix
            for i in range(self.num_gpus):
                line_idx = header_idx + 1 + i
                if line_idx >= len(lines):
                    break
                
                line = lines[line_idx]
                # Split by whitespace, skip first column (GPU X)
                parts = line.split()
                if len(parts) < self.num_gpus + 1:
                    continue
                
                for j in range(self.num_gpus):
                    if i == j:
                        continue
                    
                    # Connection type is in parts[j+1] (skip GPU X column)
                    conn_type = parts[j + 1] if j + 1 < len(parts) else 'X'
                    
                    # Map connection types to bandwidth estimates
                    if 'NV' in conn_type:
                        # NVLink detected
                        if '12' in conn_type or '18' in conn_type:
                            bw = 600e9  # NVLink 4.0
                        elif '4' in conn_type or '6' in conn_type:
                            bw = 200e9  # NVLink 2.0/3.0
                        else:
                            bw = 300e9  # Default NVLink
                        self.bandwidth_matrix[i, j] = bw
                    elif conn_type == 'SYS':
                        self.bandwidth_matrix[i, j] = 16e9  # PCIe
                    elif conn_type == 'NODE':
                        self.bandwidth_matrix[i, j] = 32e9  # Same NUMA
                    else:
                        self.bandwidth_matrix[i, j] = 0
            
            print("[GPUTopology] Successfully parsed nvidia-smi topology")
            return True
            
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"[GPUTopology] nvidia-smi parsing failed: {e}")
            return False
    
    def _detect_via_torch(self):
        """
        Fallback: Use torch.cuda P2P capability to estimate bandwidth.
        """
        print("[GPUTopology] Using torch P2P capability detection...")
        
        for i in range(self.num_gpus):
            for j in range(self.num_gpus):
                if i == j:
                    # Self: use HBM bandwidth (from profiler or default)
                    self.bandwidth_matrix[i, j] = 0  # Not used
                else:
                    # Check if P2P is possible
                    can_p2p = torch.cuda.can_device_access_peer(i, j)
                    
                    if can_p2p:
                        # Assume NVLink (optimistic)
                        self.bandwidth_matrix[i, j] = 200e9  # 200 GB/s
                    else:
                        # PCIe or cross-socket
                        self.bandwidth_matrix[i, j] = 16e9  # 16 GB/s
        
        print("[GPUTopology] Bandwidth matrix (GB/s):")
        for i in range(self.num_gpus):
            row_str = f"  GPU{i}: "
            for j in range(self.num_gpus):
                if i == j:
                    row_str += "  -   "
                else:
                    row_str += f"{self.bandwidth_matrix[i, j] / 1e9:5.0f} "
            print(row_str)
    
    def find_best_tp_group(self, tp_degree: int) -> List[int]:
        """
        Find the best group of `tp_degree` GPUs with highest total bandwidth.
        
        Algorithm: Greedy clique search
        - Start with GPU 0
        - Iteratively add GPU with highest avg bandwidth to current group
        """
        if tp_degree > self.num_gpus:
            raise ValueError(f"TP degree {tp_degree} exceeds available GPUs {self.num_gpus}")
        
        if tp_degree == 1:
            return [0]
        
        # Greedy: Start with GPU 0
        group = [0]
        
        while len(group) < tp_degree:
            best_gpu = None
            best_score = -1
            
            for candidate in range(self.num_gpus):
                if candidate in group:
                    continue
                
                # Score = average bandwidth to existing group members
                score = np.mean([self.bandwidth_matrix[candidate, g] for g in group])
                
                if score > best_score:
                    best_score = score
                    best_gpu = candidate
            
            if best_gpu is not None:
                group.append(best_gpu)
            else:
                break
        
        print(f"[GPUTopology] Best TP group (degree={tp_degree}): {group}")
        return group
    
    def partition_for_pp(self, tp_degree: int, pp_degree: int) -> List[List[int]]:
        """
        Partition GPUs into PP stages, each with TP groups.
        
        Returns: List of TP groups (each group is a list of GPU IDs)
        """
        total_needed = tp_degree * pp_degree
        if total_needed > self.num_gpus:
            raise ValueError(f"Need {total_needed} GPUs but only have {self.num_gpus}")
        
        groups = []
        used_gpus = set()
        
        for stage in range(pp_degree):
            # Find best TP group from remaining GPUs
            available = [i for i in range(self.num_gpus) if i not in used_gpus]
            
            if len(available) < tp_degree:
                # Fallback: linear chunking
                group = available[:tp_degree]
            else:
                # Greedy search within available
                group = self._greedy_group(available, tp_degree)
            
            groups.append(group)
            used_gpus.update(group)
        
        print(f"[GPUTopology] PP={pp_degree} partition: {groups}")
        return groups
    
    def _greedy_group(self, available: List[int], size: int) -> List[int]:
        """Greedy clique search within available GPUs."""
        if size > len(available):
            return available[:size]
        
        group = [available[0]]
        
        while len(group) < size:
            best_gpu = None
            best_score = -1
            
            for candidate in available:
                if candidate in group:
                    continue
                
                score = np.mean([self.bandwidth_matrix[candidate, g] for g in group])
                
                if score > best_score:
                    best_score = score
                    best_gpu = candidate
            
            if best_gpu is not None:
                group.append(best_gpu)
            else:
                # Fallback: add any remaining
                for gpu in available:
                    if gpu not in group:
                        group.append(gpu)
                        if len(group) >= size:
                            break
                break
        
        return group[:size]


if __name__ == "__main__":
    topo = GPUTopology()
    if topo.detect():
        # Test: Find best TP=2 group
        if topo.num_gpus >= 2:
            group = topo.find_best_tp_group(2)
            print(f"\nBest TP=2 group: {group}")
        
        # Test: Partition for PP=2, TP=2 (needs 4 GPUs)
        if topo.num_gpus >= 4:
            partition = topo.partition_for_pp(tp_degree=2, pp_degree=2)
            print(f"\nPP=2, TP=2 partition: {partition}")
