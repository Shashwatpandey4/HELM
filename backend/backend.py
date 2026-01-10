import torch
from typing import List
from .passes import (
    hardware_analysis_pass,
    flops_analysis_pass,
    heuristic_pass,
    device_placement_pass,
    topology_pass
)

def helm(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor], world_size=None, rank=0):
    """
    Helm Backend Pipeline:
    1. Hardware Analysis
    2. FLOPs Analysis
    3. Heuristic Partitioning
    4. Device Placement
    5. Topology Transformation
    """
    print(f"\n[Helm] Starting Compilation Pipeline (Rank {rank}, World Size {world_size})...")
    
    # 1. Hardware Analysis
    gm = hardware_analysis_pass(gm)
    
    # 2. FLOPs Analysis
    gm = flops_analysis_pass(gm)
    
    # 3. Heuristic Partitioning
    gm = heuristic_pass(gm, world_size=world_size)
    
    # 4. Device Placement (Annotation)
    gm = device_placement_pass(gm)
    
    # 5. Topology (Distribution)
    if world_size and world_size > 1:
        gm = topology_pass(gm, rank=rank, world_size=world_size)
    
    return gm
