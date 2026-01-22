import torch
from typing import List
from .passes import (
    hardware_analysis_pass,
    soft_analysis_pass,
    cost_model_pass,
    pipeline_parallel_pass,
    tensor_parallel_pass,
    flops_analysis_pass
)
from torch.fx.passes.shape_prop import ShapeProp

def helm(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor], world_size=None, rank=0):
    """
    Helm Backend Pipeline:
    1. Hardware Analysis
    2. Soft Analysis (FLOPs & IO)
    3. Pipeline Parallelism
    4. Tensor Parallelism
    """
    print(f"\n[Helm] Starting Compilation Pipeline (Rank {rank}, World Size {world_size})...")
    
    # 1. Hardware Analysis
    gm = hardware_analysis_pass(gm)
    
    # Prerequisite: Shape Propagation for Soft Analysis
    if example_inputs:
        print("[Helm] Running Shape Propagation...")
        ShapeProp(gm).propagate(*example_inputs)
    else:
        print("[Helm] WARNING: No example inputs provided. Soft Analysis may be inaccurate.")
    
    # 2. Soft Analysis
    gm = soft_analysis_pass(gm)
    
    # 3. Cost-Model Partitioning (Decisions)
    gm = cost_model_pass(gm, world_size=world_size)
    
    # 4. Pipeline Parallelism (Execution)
    # Reads meta['pipeline_split'] set by cost model
    gm = pipeline_parallel_pass(gm)
    
    # 5. Tensor Parallelism (Execution)
    # Reads meta['sharding_strategy'] set by cost model
    gm = tensor_parallel_pass(gm)
    
    return gm
