import torch
import torch.fx as fx

def device_placement_pass(gm: fx.GraphModule):
    """
    Annotates nodes with 'placement' metadata (rank ID).
    Reads 'pipeline_split' metadata set by the Cost Model.
    """
    print("\n>> [Pass] Running Device Placement...")
    
    current_rank = 0
    # Assuming 2 stages for now based on current cost model
    # We trace the graph in topological order. 
    # Everything is Rank 0 until we hit the split node.
    # The split node ITSELF is the last node of Rank 0 (or first of Rank 1? Cost model says "split after").
    # Cost model log: "Marked split at node: X". Usually implies X is the last node of Stage 0.
    
    # We need to map nodes to ranks.
    # Using a simple state machine.
    
    nodes_count = {0: 0, 1: 0}
    
    for node in gm.graph.nodes:
        # Assign current rank
        node.meta['placement'] = current_rank
        nodes_count[current_rank] += 1
        
        # Check if this node is the split point
        if node.meta.get('pipeline_split', False):
            print(f"   [Placement] Found Split Point at '{node.name}'. Switching to Next Rank.", flush=True)
            current_rank += 1
            
    print(f"   [Placement] Summary: Rank 0: {nodes_count.get(0,0)} nodes, Rank 1: {nodes_count.get(1,0)} nodes.", flush=True)
    
    # Attach global placement info if needed
    gm.meta['device_placement_summary'] = nodes_count
    
    return gm
