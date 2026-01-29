import torch
import torch.fx as fx

def device_placement_pass(gm: fx.GraphModule):
    """
    Annotates nodes with 'placement' metadata (rank ID).
    Reads 'split_config' metadata set by the Cost Model.
    Uses structural analysis (SDPA traceback) to map logical layer split to graph nodes.
    """
    print("\n>> [Pass] Running Device Placement...")
    
    split_config = gm.meta.get('split_config')
    split_layer_idx = 0
    if split_config:
        split_layer_idx = split_config.get('split_k', 0)
        
    print(f"   [Placement] Target Split Layer: {split_layer_idx}")
    
    # 1. Identify Split Point (Node)
    split_node = None # The node AFTER which rank changes (i.e. last node of Rank 0)
    
    if split_layer_idx > 0:
        # Strategy: Find k-th SDPA. Trace inputs to find QKV Projections.
        # The split is BEFORE those projections.
        
        sdpa_nodes = []
        for node in gm.graph.nodes:
            if node.op == 'call_function' and 'scaled_dot_product_attention' in str(node.target):
                sdpa_nodes.append(node)
                
        if split_layer_idx < len(sdpa_nodes):
            target_sdpa = sdpa_nodes[split_layer_idx]
            
            # Trace inputs (Q, K, V are usually 0, 1, 2)
            # We want to find the Linear/Linear-like ops feeding these.
            
            projections = []
            for arg in target_sdpa.args[:3]: # q, k, v
                # Traverse up through views/rotary to find the Source (Linear)
                curr = arg
                while isinstance(curr, fx.Node):
                    # Check if curr is a Linear/MatMul
                    target_str = str(curr.target)
                    if 'linear' in target_str or 'mm' in target_str or 'addmm' in target_str:
                         projections.append(curr)
                         break
                    # If not, go up. Assumes single input chain per branch roughly.
                    if len(curr.args) > 0 and isinstance(curr.args[0], fx.Node):
                        curr = curr.args[0]
                    else:
                        break
            
            # We found the projections (e.g. q_proj, k_proj, v_proj) for Layer K.
            # We want to split BEFORE these.
            # So the split node is the Node that feeds these projections?
            # Or simpler: The earliest of these projections is the FIRST node of Rank 1.
            # So the node topologically before it is the LAST node of Rank 0.
            
            if projections:
                # Iterate graph to find the earliest projection node in topological order
                projections_set = set(projections)
                for node in gm.graph.nodes:
                     if node in projections_set:
                         # Found the first projection for this layer.
                         # The node immediately before it is our split point.
                         split_node = node.prev
                         print(f"   [Placement] Found Split Boundary. Last Rank 0 Node: '{split_node.name}'. First Rank 1 Node: '{node.name}'")
                         break
            else:
                 print("   [Placement] WARNING: Traceback from SDPA failed to find projections.")
        else:
             print("   [Placement] WARNING: Split Layer Index out of range.")

    # 2. Assign Ranks
    current_rank = 0
    
    # First Pass: Forward assignments based on split node
    for node in gm.graph.nodes:
        node.meta['placement'] = current_rank
        if node == split_node:
            current_rank = 1
            
    # Second Pass: Fix Weights (get_attr) and Constants
    # If a weight is only used by Rank 1 nodes, it should also be on Rank 1.
    # Otherwise it stays on Rank 0 and gets sent (bad, but correct for shared).
    
    for node in gm.graph.nodes:
        if node.op in ['get_attr', 'placeholder']: # Parameters/Buffers/Inputs
             # Check users
             users = list(node.users.keys())
             if not users: continue 
             
             # If all users are Rank 1, move this node to Rank 1
             all_rank_1 = True
             for user in users:
                 if user.meta.get('placement', 0) != 1:
                     all_rank_1 = False
                     break
             
             if all_rank_1:
                 node.meta['placement'] = 1

    # Summarize
    nodes_count = {0: 0, 1: 0}
    for node in gm.graph.nodes:
        r = node.meta.get('placement', 0)
        nodes_count[r] = nodes_count.get(r, 0) + 1
            
    print(f"   [Placement] Summary: Rank 0: {nodes_count.get(0,0)} nodes, Rank 1: {nodes_count.get(1,0)} nodes.", flush=True)
    gm.meta['device_placement_summary'] = nodes_count
    
    return gm
