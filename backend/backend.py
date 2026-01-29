import torch
from typing import List
from .passes import (
    hardware_analysis_pass,
    data_analysis_pass,
    cost_model_pass,
    pipeline_parallel_pass,
    device_placement_pass
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
    print(f"\n[Helm] Starting Compilation Pipeline (Rank {rank}, World Size {world_size})...", flush=True)
    
    # 1. Hardware Analysis
    gm = hardware_analysis_pass(gm)
    
    # Prerequisite: Shape Propagation for Soft Analysis
    if example_inputs:
        print("[Helm] Skipping ShapeProp (relying on Dynamo metadata for stability).")
        # try:
        #     ShapeProp(gm).propagate(*example_inputs)
        # except Exception as e:
        #     print(f"[Helm] WARNING: ShapeProp failed: {e}. Falling back to Dynamo metadata.")
    else:
        print("[Helm] WARNING: No example inputs provided. Soft Analysis may be inaccurate.")
    
    # 2. Data Analysis (Prev. Soft Analysis)
    gm = data_analysis_pass(gm)
    
    # 3. Cost-Model Partitioning (Decisions)
    gm = cost_model_pass(gm, world_size=world_size)

    # 3.5 Device Placement (Annotation)
    gm = device_placement_pass(gm)
    
    # 4. Pipeline Parallelism (Execution / Partitioning)
    # Splits the graph and keeps only nodes relevant to 'rank'
    gm = pipeline_parallel_pass(gm, rank=rank, world_size=world_size)
    
    # 5. Tensor Parallelism (Execution)
    # Reads meta['sharding_strategy'] set by cost model
    # gm = tensor_parallel_pass(gm)
    
    # 6. Inductor Compilation (Local Optimization)
    # print(f"[Helm] Handing off local split graph to TorchInductor (Rank {rank})...")
    
    # We re-compile the split graph using Inductor to generate efficient Triton kernels
    # for the local computation slices.
    # try:
    #     optimized_local_graph = torch.compile(gm, backend="inductor")
    #     return optimized_local_graph
    # except Exception as e:
    #     print(f"[Helm] WARNING: Inductor compilation failed: {e}. Falling back to eager execution.")
    #     return gm
    return gm

