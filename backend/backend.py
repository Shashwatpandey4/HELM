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

def parameter_materialization_pass(gm: torch.fx.GraphModule, device: torch.device):
    """
    Materializes any remaining 'meta' device parameters/buffers on the actual 'device'.
    """
    print(f"\n>> [Pass] Materializing Parameters/Buffers on {device}...")
    # Iterate over all attributes in the GraphModule
    # Instead of named_parameters(), we check all targets of get_attr nodes
    count = 0
    for node in gm.graph.nodes:
        if node.op == 'get_attr':
            attr_name = node.target
            # split and traverse
            parts = attr_name.split('.')
            parent = gm
            for part in parts[:-1]:
                parent = getattr(parent, part)
            
            leaf_name = parts[-1]
            attr = getattr(parent, leaf_name)
            
            if hasattr(attr, 'is_meta') and attr.is_meta:
                # Materialize on device
                new_attr = torch.empty_like(attr, device=device)
                # Re-register or set
                if isinstance(attr, torch.nn.Parameter):
                    setattr(parent, leaf_name, torch.nn.Parameter(new_attr))
                else:
                    setattr(parent, leaf_name, new_attr)
                count += 1
    
    print(f"   [Materialization] Total: {count} attributes materialized.")
    return gm

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
    
    # 6. Parameter Materialization (for meta-device models)
    device = torch.device(f"cuda:{rank}")
    gm = parameter_materialization_pass(gm, device)
    
    return gm

