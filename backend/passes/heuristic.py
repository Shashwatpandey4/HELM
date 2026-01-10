import torch
import torch.fx as fx

def heuristic_pass(gm: torch.fx.GraphModule, world_size: int = None):
    """
    Annotates nodes with 'target_rank' based on FLOPs balancing and hardware info.
    """
    print("\n>> [Pass] Running Heuristic Partitioning...")
    
    # Get hardware info if available
    hardware_info = gm.meta.get("hardware_info", {})
    device_count = hardware_info.get("device_count", 1)
    
    if world_size is None:
        world_size = device_count
        
    if world_size <= 1:
        print("   + World size is 1. All ops on Rank 0.")
        for node in gm.graph.nodes:
            node.meta['target_rank'] = 0
        return gm
        
    # Get total FLOPs
    total_flops = gm.meta.get('total_flops', 0)
    if total_flops == 0:
        # Fallback to simple ops count splitting if no FLOPs info
        print("   + No FLOPs info found. Using op count balancing.")
        computational_nodes = [n for n in gm.graph.nodes if n.op not in ('placeholder', 'output', 'get_attr')]
        total_ops = len(computational_nodes)
        ops_per_stage = total_ops / world_size
        
        current_rank = 0
        ops_in_stage = 0
        
        for node in gm.graph.nodes:
            if node in computational_nodes:
                node.meta['target_rank'] = current_rank
                ops_in_stage += 1
                if ops_in_stage > ops_per_stage and current_rank < world_size - 1:
                    current_rank += 1
                    ops_in_stage = 0
            else:
                # Placeholders on 0, Output on last? 
                # Or just assign based on users/flow?
                # Simplification: Assign non-computational to same as current
                node.meta['target_rank'] = current_rank
    else:
        print(f"   + Balancing {total_flops:.2e} FLOPs across {world_size} ranks.")
        target_flops_per_stage = total_flops / world_size
        
        current_rank = 0
        current_stage_flops = 0
        
        # Greedy partitioning
        # Note: This assumes topological sort order is execution order
        for node in gm.graph.nodes:
            flops = node.meta.get('flops', 0)
            
            # Simple greedy decision to switch stage
            if current_rank < world_size - 1:
                # If adding this node exceeds target significantly, switch BEFORE this node?
                # Or just fill up?
                # Let's switch if we are strictly OVER target, or if adding this makes us way over?
                if current_stage_flops + flops > target_flops_per_stage:
                    # Heuristic: Switch to next stage
                    # Reset
                    current_rank += 1
                    current_stage_flops = 0
            
            node.meta['target_rank'] = current_rank
            current_stage_flops += flops

            # Log boundaries for debug
            if flops > 0:
                 pass
                 
    # Validate/Smooth
    # Ensure all nodes have target_rank
    for node in gm.graph.nodes:
        if 'target_rank' not in node.meta:
            node.meta['target_rank'] = 0
            
    return gm
