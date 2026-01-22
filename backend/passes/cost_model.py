import torch
import torch.fx as fx
import math

def cost_model_pass(gm: fx.GraphModule, world_size: int = None):
    """
    Heuristic Cost Model:
    1. Tensor Parallelism (TP): Mark High-FLOP nodes.
    2. Pipeline Parallelism (PP): Mark Min-Data-Transfer edges for splitting.
    """
    print("\n>> [Pass] Running Cost Model (TP/PP Decisions)...")
    
    # helper to get stats
    def get_stats(node):
        res = node.meta.get('soft_analysis', {'flops': 0, 'mem_bytes': 0})
        return res['flops'], res['mem_bytes']

    nodes = list(gm.graph.nodes)
    
    # -----------------------------
    # 1. Tensor Parallelism (TP)
    # -----------------------------
    # Strategy: Find nodes with Top 10% FLOPs (or high absolute threshold)
    flops_list = []
    for node in nodes:
        flops, _ = get_stats(node)
        if flops > 0:
            flops_list.append((flops, node))
            
    if flops_list:
        # Sort desc
        flops_list.sort(key=lambda x: x[0], reverse=True)
        # Top 20%?
        cutoff_idx = max(1, int(len(flops_list) * 0.2))
        top_nodes = set(n for f, n in flops_list[:cutoff_idx])
        
        for node in top_nodes:
            # Check if it's a linear/matmul (safeguard)
            if 'linear' in node.name or 'attn' in node.name or 'ff' in node.name:
                 node.meta['sharding_strategy'] = 'col'
                 print(f"   [TP] Marked {node.name} for Tensor Parallelism (High FLOPs: {node.meta['soft_analysis']['flops']:.2e})")

    # -----------------------------
    # 2. Pipeline Parallelism (PP)
    # -----------------------------
    # Strategy: Find "bottleneck" edges with minimum data transfer.
    # We want to split the graph into N stages (if world_size > 1).
    # Simple heuristic: Find the local minimum of "active tensor memory" across topological sort?
    # Or just find edges between blocks?
    
    # For this toy demo, let's find the point with globally minimum activation usage 
    # anywhere in the middle 50% of the graph (to avoid splitting at start/end).
    
    # We iterate through nodes and check the size of values flowing *out* of them to the next nodes.
    # The "cut size" at node N is roughly the size of N's output (assuming linear chain).
    # For complex graphs, it's the size of all values crossing the cut.
    
    # Simplified: Scan nodes in topological order.
    # Calculate output size of each node.
    # Pick the node with MIN output size in the middle region.
    
    if len(nodes) > 4:
        start_search = len(nodes) // 4
        end_search = 3 * len(nodes) // 4
        
        min_bytes = float('inf')
        best_split_node = None
        
        for i in range(start_search, end_search):
            node = nodes[i]
            _, mem_bytes = get_stats(node)
            
            # Use output size estimate if available (mem_bytes is Input+Output in soft_analysis)
            # soft_analysis doesn't separate In/Out cleanly in the dict, but we can re-estimate or use mem_bytes as proxy.
            # Low total mem usage usually implies low activation size too.
            
            if mem_bytes < min_bytes and mem_bytes > 0:
                min_bytes = mem_bytes
                best_split_node = node
                
        if best_split_node:
            best_split_node.meta['pipeline_split'] = True
            print(f"   [PP] Marked split after {best_split_node.name} (Min Data Transfer: ~{min_bytes} bytes)")
            
    return gm
