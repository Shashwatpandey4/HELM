import torch
import torch.fx as fx
from .common import dist_send, dist_recv

def topology_pass(gm: torch.fx.GraphModule, rank: int, world_size: int):
    """
    Determines the topology and cuts the graph for the specific 'rank'.
    Inserts communication primitives.
    """
    if world_size <= 1:
        return gm
        
    print(f"\n>> [Pass] Applying Topology (Rank {rank}/{world_size})...")
    
    graph = gm.graph
    nodes = list(graph.nodes)
    
    # Identify my nodes
    my_nodes = set()
    for node in nodes:
        node_rank = node.meta.get('target_rank', 0)
        if node_rank == rank:
             my_nodes.add(node)
             
    # Clean up non-computational inclusions/exclusions?
    # Placeholders/Outputs might have been assigned rank 0 or last.
    # We need strict handling.
    
    # 1F1B / Pipeline Logic
    # Scan for boundaries.
    
    boundary_inputs = set() # (Node, SourceRank)
    boundary_outputs = set() # (Node, DestRank)
    
    # Check inputs to my nodes
    for node in my_nodes:
        for arg in node.args:
            if isinstance(arg, fx.Node):
                arg_rank = arg.meta.get('target_rank', 0)
                if arg_rank != rank and arg.op != 'placeholder' and arg.op != 'get_attr':
                    # It's a dependency from another rank
                    # We need to RECV it.
                    boundary_inputs.add((arg, arg_rank))
                    
    # Check outputs from my nodes
    for node in my_nodes:
        for user in node.users:
            user_rank = user.meta.get('target_rank', 0)
            if user_rank != rank:
                # It's used by another rank (future or past?)
                # We need to SEND it.
                boundary_outputs.add((node, user_rank))
                
    # Modify Graph
    
    # Insert RECVs
    # We insert them at the start of the graph (after placeholders) or just before first use?
    # Before first use is safer.
    
    # Group by (Node, SourceRank) to avoid multiple recvs for same node?
    # Yes, one recv per unique tensor.
    
    recv_replacements = {}
    
    for inp_node, src_rank in boundary_inputs:
        # Insert Recv before the first user in my_nodes
        first_user = None
        for user in inp_node.users:
            if user in my_nodes:
                # Find topologically first? 
                # Heuristic: just pick the first one we find for finding insertion point
                first_user = user
                break
        
        if first_user:
            with graph.inserting_before(first_user):
                 val = inp_node.meta.get('val', inp_node.meta.get('example_value'))
                 shape = val.shape if val is not None and isinstance(val, torch.Tensor) else None
                 
                 recv_node = graph.call_function(dist_recv, args=(src_rank, shape))
                 recv_replacements[inp_node] = recv_node
                 
    # Replace inputs
    for node in my_nodes:
        new_args = []
        for arg in node.args:
            if isinstance(arg, fx.Node) and arg in recv_replacements:
                new_args.append(recv_replacements[arg])
            else:
                new_args.append(arg)
        node.args = tuple(new_args)
        
    # Insert SENDs
    # Insert after the node production
    for out_node, dst_rank in boundary_outputs:
        with graph.inserting_after(out_node):
            graph.call_function(dist_send, args=(out_node, dst_rank))
            
    # Prune nodes
    to_delete = []
    
    # Define what to keep:
    # 1. My nodes
    # 2. Placeholders (inputs) ?? If I don't need them, I can drop them or ignore.
    # 3. Get_attr (params) ?? If I don't use them, auto-pruning usually handles it.
    # 4. Output ?? 
    
    # Keep placeholders and get_attrs initially, rely on erase_node usage checks?
    # If I erase a node that assumes I am rank X, its users on rank Y are gone (from my graph).
    # So usages should be 0.
    
    # Wait, the graph currently contains ALL nodes.
    # If I delete a node assigned to Rank 1 (and I am Rank 0), 
    # its input is a node on Rank 0. That node on Rank 0 still serves it.
    # I replaced the usage on Rank 1 with a Recv.
    # So the link Rank0_Node -> Rank1_Node is broken?
    # No, I am modifying MY copy of the graph.
    
    # For Rank 0:
    # I keep Rank 0 nodes.
    # I see Rank 1 nodes. I want to delete them.
    # Rank 1 nodes use Rank 0 nodes.
    # If I delete Rank 1 node, Rank 0 node has one less user.
    # Perfect.
    
    for node in nodes:
        if node in my_nodes:
            continue
            
        # Don't delete inserted communication
        if node.target in (dist_send, dist_recv):
            continue
            
        # Handle IO
        if node.op == 'placeholder':
             # Keep placeholders for now? Or check if used.
             # If unused, we can delete later.
             continue
        if node.op == 'get_attr':
             continue
        if node.op == 'output':
             # If I am not the last rank (or the rank owning output), I shouldn't yield meaningful data?
             # For simplicity, if I am not the rank owning the nominal output elements, I return empty.
             # Who owns the output? The users of the output node?
             # The output node has args. Those args are nodes.
             # If those nodes are on my rank, I return them.
             
             # Let's inspect output args
             out_args = node.args[0]
             if not isinstance(out_args, (list, tuple)):
                 out_args = (out_args,)
                 
             new_out = []
             # If I own the node, I return it. If not, I return nothing?
             # Or I return None?
             # Existing pipeline logic: Last rank returns.
             # Here: dynamic.
             
             has_content = False
             
             # For now, just keep output but prune args if I don't own them?
             # If I don't own the arg, it's deleted. So I can't return it.
             # So I must clear it.
             continue # handle specifically?
             
        to_delete.append(node)

    # Pruning Loop
    for node in reversed(to_delete):
        if len(node.users) == 0:
            graph.erase_node(node)
        else:
             # It acts as a dependency for something we kept?
             # If we did our job (inserted Recv), dependencies should be broken.
             # Except Output node?
             
             # Check if users are only the Output node?
             pass
             
    # Fix Output Node
    # Any arg pointing to a deleted node must be removed/none'd.
    out_node = [n for n in graph.nodes if n.op == 'output'][0]
    curr_out_args = out_node.args[0] if out_node.args else ()
    if not isinstance(curr_out_args, (list, tuple)):
        curr_out_args = (curr_out_args,)
        
    filtered_out = []
    
    # If I am Rank X, and the final output of the model is produced by Rank Y.
    # If X != Y, I return nothing.
    # If X == Y, I return result.
    
    for arg in curr_out_args:
        if isinstance(arg, fx.Node):
            # If the node is still in graph (not deleted), keep it.
            # But wait, checking existence in 'graph.nodes' is O(N).
            # We can check if it's in 'my_nodes'?
            if arg in my_nodes:
                filtered_out.append(arg)
            else:
                # Node was presumably deleted or belonged to another rank
                pass
        else:
            filtered_out.append(arg)
            
    # Update output
    # If filtered_out is empty -> return ()
    # If partial -> return partial
    if not filtered_out:
        out_node.args = ((),)
    else:
        if len(filtered_out) == 1:
            out_node.args = (filtered_out[0],)
        else:
            out_node.args = (tuple(filtered_out),)

    graph.lint()
    gm.recompile()
    return gm
